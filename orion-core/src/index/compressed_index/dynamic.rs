/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Orion query preparation and exact native-storage search.

use crate::{Error, Result};
use bytemuck::Pod;
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::{DistanceKernelBackend, L2F32Distance, L2U8Distance};
use rayon::prelude::*;

use super::{CompressedIndex, IndexData};
use crate::algorithm::search::calibrate::CalibrationMetric;
use crate::algorithm::search::diagnostics::{NeighborMode, NoopObserver, SearchObserver};
use crate::model::dataset::quantized_dataset::exact::ExactSession;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u8 {}
}

pub trait DynamicElement: sealed::Sealed + Pod + Send + Sync {
    type Kernel: DistanceKernelBackend<Storage = Self>;
    fn valid_coordinate(value: Self) -> bool;
    fn supports_calibration(metric: CalibrationMetric) -> bool;
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        scratch: &mut [f32],
    ) -> Result<f32>;
}
impl DynamicElement for f32 {
    type Kernel = L2F32Distance;
    fn supports_calibration(_: CalibrationMetric) -> bool {
        true
    }
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        _: &mut [f32],
    ) -> Result<f32> {
        super::calibration::exact_distance(metric, query, row)
    }
    fn valid_coordinate(value: Self) -> bool {
        value.is_finite()
    }
}
impl DynamicElement for u8 {
    type Kernel = L2U8Distance;
    fn supports_calibration(metric: CalibrationMetric) -> bool {
        metric == CalibrationMetric::L2
    }
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        scratch: &mut [f32],
    ) -> Result<f32> {
        for (out, &v) in scratch.iter_mut().zip(row) {
            *out = f32::from(v);
        }
        super::calibration::exact_distance(metric, query, scratch)
    }
    fn valid_coordinate(_: Self) -> bool {
        true
    }
}

/// Explicit controls for the dynamic exact-L2 path. Defaults disable
/// convergence switching/early exit; they are not calibrated parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicSearchConfig {
    pub k: usize,
    pub search_list_size: usize,
    pub window_size: usize,
    pub epsilon: f32,
    pub early_exit_limit: usize,
}
impl Default for DynamicSearchConfig {
    fn default() -> Self {
        Self {
            k: 10,
            search_list_size: 64,
            window_size: 3,
            epsilon: 0.,
            early_exit_limit: usize::MAX,
        }
    }
}

fn error(message: impl Into<String>) -> Error {
    Error::invalid_argument(message)
}

impl CompressedIndex<PaddedDataset<u8>> {
    /// Accept byte-valued f32 queries from the shared CLI/Python interface.
    /// Prepare each query on its search worker, directly in its final padded
    /// buffer. This avoids a serial whole-batch conversion and a second copy.
    pub fn search_batch_l2_byte_queries(
        &self,
        queries: &[f32],
        config: DynamicSearchConfig,
    ) -> Result<crate::SearchResult> {
        self.validate_search(config)?;
        let dim = self.dataset.dimension();
        if queries.len() % dim != 0 {
            return Err(Error::packed_shape("queries", dim, queries.len()).at("read_queries"));
        }
        queries
            .par_chunks(dim)
            .map(|query| {
                let started = crate::algorithm::search::metrics::SetupTimer::now();
                let mut padded = PaddedDataset::new(1, self.dataset.layout()).map_err(error)?;
                for (out, &value) in padded.row_mut(0).unwrap().iter_mut().zip(query) {
                    // Conversion must remain exact: this is native byte search,
                    // not a query quantizer that may round or clamp values.
                    if !value.is_finite() || value < 0. || value > 255. || value.fract() != 0. {
                        return Err(error(
                            "native u8 query must contain integer coordinates in 0..=255",
                        ));
                    }
                    *out = value as u8;
                }
                self.search_prepared(&padded, 0, config, started)
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|rows| crate::SearchResult::from_rows(rows, config.k, self.num_nodes))
    }
}
impl<T: DynamicElement> CompressedIndex<PaddedDataset<T>> {
    pub(super) fn validate_search(&self, config: DynamicSearchConfig) -> Result<()> {
        self.validate_dataset_graph()?;
        if config.k == 0
            || config.k > self.num_nodes
            || config.search_list_size < config.k
            || !(1..=64).contains(&config.window_size)
            || !config.epsilon.is_finite()
            || !(0.0..=1.0).contains(&config.epsilon)
        {
            return Err(error(
                "invalid k, search list size, convergence window or epsilon",
            ));
        }
        let layout = self.dataset.layout();
        if layout.logical_dim() > T::Kernel::MAX_COORDINATES
            || layout.compute_bytes() % T::Kernel::CHUNK_BYTES != 0
        {
            return Err(error(
                "dataset layout is incompatible with the exact-L2 kernel",
            ));
        }
        Ok(())
    }

    /// Search a logical, unpadded query. Padding is prepared once before the
    /// existing graph loop. u8 callers keep native bytes throughout.
    pub fn search_l2(&self, query: &[T], config: DynamicSearchConfig) -> Result<Vec<u32>> {
        self.search_l2_observed(query, config, &mut NoopObserver)
    }

    /// The same native L2 path with query-local diagnostics; no distance is recomputed.
    pub fn search_l2_observed<O: SearchObserver>(
        &self,
        query: &[T],
        config: DynamicSearchConfig,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        self.validate_search(config)?;
        if query.len() != self.dataset.dimension() {
            return Err(Error::dimension(
                "queries.dimension",
                self.dataset.dimension(),
                query.len(),
            )
            .at("read_queries"));
        }
        if query.iter().any(|&x| !T::valid_coordinate(x)) {
            return Err(error("query contains a non-finite coordinate"));
        }
        let t_setup = crate::algorithm::search::metrics::SetupTimer::now();
        let queries = PaddedDataset::from_packed(query, self.dataset.layout()).map_err(error)?;
        self.search_prepared_observed(&queries, 0, config, t_setup, observer)
    }

    /// Batch input is packed row-major. One preparation pass pads all queries;
    /// Rayon and the existing scratch pool handle scheduling and scratch reuse.
    pub fn search_batch_l2(
        &self,
        queries: &[T],
        config: DynamicSearchConfig,
    ) -> Result<crate::SearchResult> {
        self.validate_search(config)?;
        let dim = self.dataset.dimension();
        if queries.len() % dim != 0 || queries.iter().any(|&x| !T::valid_coordinate(x)) {
            return Err(error("invalid packed query shape or non-finite coordinate"));
        }
        if queries.is_empty() {
            return crate::SearchResult::from_rows(Vec::new(), config.k, self.num_nodes);
        }
        let padded = PaddedDataset::from_packed(queries, self.dataset.layout()).map_err(error)?;
        (0..padded.len())
            .into_par_iter()
            .map(|id| {
                self.search_prepared(
                    &padded,
                    id,
                    config,
                    crate::algorithm::search::metrics::SetupTimer::now(),
                )
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|rows| crate::SearchResult::from_rows(rows, config.k, self.num_nodes))
    }

    fn search_prepared(
        &self,
        queries: &PaddedDataset<T>,
        query_id: usize,
        config: DynamicSearchConfig,
        started: crate::algorithm::search::metrics::SetupTimer,
    ) -> Result<Vec<u32>> {
        self.search_prepared_observed(queries, query_id, config, started, &mut NoopObserver)
    }

    fn search_prepared_observed<O: SearchObserver>(
        &self,
        queries: &PaddedDataset<T>,
        query_id: usize,
        config: DynamicSearchConfig,
        started: crate::algorithm::search::metrics::SetupTimer,
        observer: &mut O,
    ) -> Result<Vec<u32>> {
        let session = ExactSession::<T, T::Kernel>::new(&self.dataset, queries, query_id);
        let mode = NeighborMode::default();
        self.search_sessions(
            config.k,
            config.search_list_size,
            config.window_size,
            config.epsilon,
            config.early_exit_limit,
            None,
            &session,
            mode,
            observer,
            started,
            |scratch, _| {
                scratch
                    .pq
                    .neighbors()
                    .iter()
                    .take(config.k)
                    .map(|n| n.id)
                    .collect()
            },
        )
    }
}

//! Orion query preparation and exact native-storage search.

use bytemuck::Pod;
use diskann::{
    common::{ANNError, ANNResult},
    model::PaddedDataset,
};
use rayon::prelude::*;
use vector::{DistanceFn, L2F32Distance, L2U8Distance};

use super::{CompressedIndex, IndexData};
use crate::algorithm::search::calibrate::CalibrationMetric;
use crate::algorithm::search::{
    diagnostics::{NeighborMode, NoopObserver},
    stage::AdmissionSession,
};
use crate::model::Neighbor;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u8 {}
}

pub trait DynamicElement: sealed::Sealed + Pod + Send + Sync {
    type Kernel: DistanceFn<Storage = Self>;
    fn valid_coordinate(value: Self) -> bool;
    fn supports_calibration(metric: CalibrationMetric) -> bool;
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        scratch: &mut [f32],
    ) -> ANNResult<f32>;
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
    ) -> ANNResult<f32> {
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
    ) -> ANNResult<f32> {
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
#[derive(Clone, Copy, Debug)]
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

fn error(message: impl Into<String>) -> ANNError {
    ANNError::log_index_error(message.into())
}

struct ExactSession<'a, T: DynamicElement> {
    base: &'a PaddedDataset<T>,
    queries: &'a PaddedDataset<T>,
    query_id: usize,
}

impl CompressedIndex<PaddedDataset<u8>> {
    /// Accept byte-valued f32 queries from the shared CLI/Python interface.
    /// Prepare each query on its search worker, directly in its final padded
    /// buffer. This avoids a serial whole-batch conversion and a second copy.
    pub fn search_batch_l2_byte_queries(
        &self,
        queries: &[f32],
        config: DynamicSearchConfig,
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.validate_search(config)?;
        let dim = self.dataset.dimension();
        if queries.len() % dim != 0 {
            return Err(error("invalid packed query shape"));
        }
        queries
            .par_chunks(dim)
            .map(|query| {
                let started = std::time::Instant::now();
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
            .collect()
    }
}
impl<T: DynamicElement> AdmissionSession for ExactSession<'_, T> {
    fn entry_distance(&self, id: u32) -> f32 {
        self.base
            .distance::<T::Kernel>(id, self.queries, self.query_id)
            .expect("validated dynamic entry and layout")
    }

    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        near: usize,
    ) -> usize {
        let mut count = 0;
        // Search preparation checks the kernel/layout once; the graph loop
        // supplies valid row IDs under AdmissionSession's safety contract.
        unsafe {
            self.base
                .stream_unchecked::<T::Kernel>(self.queries, self.query_id, ids, near)
        }
        .run(|id, distance| {
            unsafe {
                out.add(count).write(Neighbor::new(id, distance));
            }
            count += (distance < cutoff) as usize;
        });
        count
    }
}

impl<T: DynamicElement> CompressedIndex<PaddedDataset<T>> {
    pub(super) fn validate_search(&self, config: DynamicSearchConfig) -> ANNResult<()> {
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
    pub fn search_l2(&self, query: &[T], config: DynamicSearchConfig) -> ANNResult<Vec<u32>> {
        self.validate_search(config)?;
        if query.len() != self.dataset.dimension() || query.iter().any(|&x| !T::valid_coordinate(x))
        {
            return Err(error("query dimension mismatch or non-finite coordinate"));
        }
        let t_setup = std::time::Instant::now();
        let queries = PaddedDataset::from_packed(query, self.dataset.layout()).map_err(error)?;
        self.search_prepared(&queries, 0, config, t_setup)
    }

    /// Batch input is packed row-major. One preparation pass pads all queries;
    /// Rayon and the existing scratch pool handle scheduling and scratch reuse.
    pub fn search_batch_l2(
        &self,
        queries: &[T],
        config: DynamicSearchConfig,
    ) -> ANNResult<Vec<Vec<u32>>> {
        self.validate_search(config)?;
        let dim = self.dataset.dimension();
        if queries.len() % dim != 0 || queries.iter().any(|&x| !T::valid_coordinate(x)) {
            return Err(error("invalid packed query shape or non-finite coordinate"));
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let padded = PaddedDataset::from_packed(queries, self.dataset.layout()).map_err(error)?;
        (0..padded.len())
            .into_par_iter()
            .map(|id| self.search_prepared(&padded, id, config, std::time::Instant::now()))
            .collect()
    }

    fn search_prepared(
        &self,
        queries: &PaddedDataset<T>,
        query_id: usize,
        config: DynamicSearchConfig,
        started: std::time::Instant,
    ) -> ANNResult<Vec<u32>> {
        let session = ExactSession {
            base: &self.dataset,
            queries,
            query_id,
        };
        let mode = if crate::algorithm::search::include_extras() {
            NeighborMode::LocalExtra
        } else {
            NeighborMode::FullNeighbor
        };
        self.search_sessions(
            config.k,
            config.search_list_size,
            config.window_size,
            config.epsilon,
            config.early_exit_limit,
            None,
            &session,
            mode,
            &mut NoopObserver,
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

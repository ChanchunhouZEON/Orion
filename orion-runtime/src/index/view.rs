/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Search backend and immutable identities selected for this session.
//! ReadView owns the engine; Engine defines its operations; DynamicEngine
//! connects those operations to the selected core search backend.
use super::query::{QueryBatch, QueryValues, SearchOptions};
use super::snapshot::Payloads;
use crate::{config::ResolvedRunConfig, search::SearchBackend, Result};
use orion_ann_diagnostics::RecordingObserver;
use std::path::PathBuf;
/// Objects and engine captured together at load time. Search and save use the
/// same generation even if another process replaces the working lookup record.
pub(super) struct ReadView {
    pub(super) engine: Box<dyn Engine>,
    pub(super) object_store: PathBuf,
    pub(super) payloads: Payloads,
    pub(super) graph_bytes: u64,
    pub(super) metadata_bytes: u64,
}

pub(super) trait Engine: Send + Sync {
    fn release_auto_pin(&mut self);
    fn mapped_sidecar_bytes(&self) -> usize;
    fn mapped_base_bytes(&self) -> usize;
    fn mapped_graph_bytes(&self) -> usize;
    fn pin_report(&self) -> &crate::PinReport;
    fn set_pq_code_storage(&mut self, policy: crate::PqCodeStorage) -> Result<()>;
    fn pin<'a>(
        &'a self,
        config: &ResolvedRunConfig,
        options: &crate::PinOptions,
    ) -> Result<crate::PinnedDatasets<'a>>;
    fn pin_hints(&self, config: &ResolvedRunConfig) -> Result<Vec<crate::PinHint>>;

    fn prepare_search(&mut self, config: &ResolvedRunConfig) -> Result<()>;
    fn calibrate(
        &self,
        queries: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
    ) -> Result<orion_ann_core::CalibratedParams>;
    fn search(
        &self,
        queries: QueryBatch<'_>,
        config: &ResolvedRunConfig,
        options: &SearchOptions,
        params: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult>;
    fn search_observed(
        &self,
        query: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        params: orion_ann_core::CalibratedParams,
        observer: &mut RecordingObserver,
    ) -> Result<Vec<u32>>;
    fn sidecars(&self) -> Result<crate::sidecars::SnapshotSidecars>;
}

pub(super) struct DynamicEngine<B: SearchBackend> {
    // Drop locks before the index's heap allocations, including on unwind.
    // All allocation-replacing operations must release this token first.
    pub(super) auto_pin: orion_ann_platform::residency::OwnedPinGuard,
    pub(super) pin_report: crate::PinReport,
    pub(super) receipts: crate::sidecars::Receipts,
    pub(super) index: orion_ann_core::Orion<B::Element>,
}

impl<B: SearchBackend> Engine for DynamicEngine<B> {
    fn mapped_base_bytes(&self) -> usize {
        if self.index.dataset().placement() == orion_ann_platform::region::Placement::Mapped {
            std::mem::size_of_val(self.index.dataset().as_slice())
        } else {
            0
        }
    }
    fn mapped_graph_bytes(&self) -> usize {
        if self.index.graph().placement() == orion_ann_platform::region::Placement::Mapped {
            self.index.graph().buffer_bytes().len()
        } else {
            0
        }
    }
    fn mapped_sidecar_bytes(&self) -> usize {
        use crate::QuantizedKind::*;
        [
            L2U8,
            L2U16,
            MipsI8,
            MipsI16,
            Pq,
            L2Kt,
            JlSparse,
            JlSparseMips,
            JlHadamard,
            Rabitq,
            Rabitq4,
        ]
        .into_iter()
        .filter(|&k| self.index.sidecar_is_mapped(k))
        .map(|k| {
            let regions = self.index.resident_sidecar_regions(k).unwrap_or_default();
            if k == Pq {
                self.index.mapped_pq_bytes()
            } else {
                regions.iter().map(|r| r.len()).sum::<usize>()
            }
        })
        .sum()
    }

    fn release_auto_pin(&mut self) {
        self.auto_pin = Default::default();
        self.pin_report = Default::default();
    }
    fn pin_report(&self) -> &crate::PinReport {
        &self.pin_report
    }
    fn prepare_search(&mut self, config: &ResolvedRunConfig) -> Result<()> {
        self.release_auto_pin();
        let guard = B::prepare_search(&self.index, config)?;
        // SAFETY: base, graph and prepared sidecars own stable heap buffers.
        // The token is dropped before index (field order). Calibration and PQ
        // replacement release it first; shared searches never replace buffers.
        let (locks, report) = unsafe { guard.into_owned() };
        self.auto_pin = locks;
        self.pin_report = report;
        Ok(())
    }
    fn set_pq_code_storage(&mut self, policy: crate::PqCodeStorage) -> Result<()> {
        self.release_auto_pin();
        self.index.set_pq_code_storage(policy)
    }
    fn pin<'a>(
        &'a self,
        config: &ResolvedRunConfig,
        options: &crate::PinOptions,
    ) -> Result<crate::PinnedDatasets<'a>> {
        B::pin_search_storage(&self.index, config, options)
    }
    fn pin_hints(&self, config: &ResolvedRunConfig) -> Result<Vec<crate::PinHint>> {
        B::pin_hints(&self.index, config)
    }

    fn calibrate(
        &self,
        q: &[f32],
        c: &ResolvedRunConfig,
        k: usize,
        beam: usize,
    ) -> Result<orion_ann_core::CalibratedParams> {
        self.index.calibrate(
            q,
            beam,
            c.window_size,
            orion_ann_core::CalibrationConfig {
                k,
                metric: c.metric.calibration_metric(),
            },
        )
    }
    fn search(
        &self,
        q: QueryBatch<'_>,
        c: &ResolvedRunConfig,
        options: &SearchOptions,
        params: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        if options.scores {
            return match q.values {
                QueryValues::F32(values) => B::search_batch_scored(
                    &self.index,
                    values,
                    c,
                    options.k,
                    options.search_list_size,
                    params,
                ),
                QueryValues::U8(values) => B::search_batch_u8_scored(
                    &self.index,
                    values,
                    c,
                    options.k,
                    options.search_list_size,
                    params,
                ),
            };
        }
        match q.values {
            QueryValues::F32(values) => B::search_batch(
                &self.index,
                values,
                c,
                options.k,
                options.search_list_size,
                params,
            ),
            QueryValues::U8(values) => B::search_batch_u8(
                &self.index,
                values,
                c,
                options.k,
                options.search_list_size,
                params,
            ),
        }
    }
    fn search_observed(
        &self,
        query: &[f32],
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        params: orion_ann_core::CalibratedParams,
        observer: &mut RecordingObserver,
    ) -> Result<Vec<u32>> {
        B::search_observed(&self.index, query, config, k, beam, params, observer)
    }
    fn sidecars(&self) -> Result<crate::sidecars::SnapshotSidecars> {
        crate::sidecars::materialized(&self.index, &self.receipts)
    }
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Storage and immutable identities selected for this session.
//! ReadView owns the engine; Engine defines its operations; DynamicEngine
//! connects those operations to one core search interface for every element type.
use super::query::{QueryBatch, SearchOptions};
use super::snapshot::Payloads;
use crate::{config::ResolvedRunConfig, data::BaseElement, Result};
use orion_ann_diagnostics::RecordingObserver;
use std::path::PathBuf;
/// Objects and engine captured together at load time. Search and save use the
/// same generation even if another process replaces the working lookup record.
pub(super) struct ReadView {
    /// Search implementation and allocations belonging to this captured generation.
    pub(super) engine: Box<dyn Engine>,

    /// Immutable payload store used by later saves of this generation.
    pub(super) object_store: PathBuf,

    /// Content identities captured with the engine, independent of mutable lookup records.
    pub(super) payloads: Payloads,

    /// Serialized graph length retained for snapshot metadata validation.
    pub(super) graph_bytes: u64,

    /// Serialized graph-metadata length retained for snapshot validation.
    pub(super) metadata_bytes: u64,
}

/// Element-erased runtime adapter over the shared core search and storage contracts.
pub(super) trait Engine: Send + Sync {
    /// Reports allocation-domain ownership and limits without sampling process RSS.
    fn storage_accounting(&self) -> serde_json::Value;

    /// Materializes required native auxiliaries and publishes missing immutable receipts.
    fn prepare_auxiliary(
        &mut self,
        config: &ResolvedRunConfig,
        store: &std::path::Path,
    ) -> Result<()>;

    /// Releases owned page locks before any allocation-replacing operation.
    fn release_auto_pin(&mut self);

    /// Counts mapped sidecar and norm spans, not currently resident pages.
    fn mapped_sidecar_bytes(&self) -> usize;

    /// Counts the mapped base span, or zero for resident backing.
    fn mapped_base_bytes(&self) -> usize;

    /// Counts the mapped graph span, or zero for resident backing.
    fn mapped_graph_bytes(&self) -> usize;

    /// Borrows the latest automatic page-lock outcome without attempting another lock.
    fn pin_report(&self) -> &crate::PinReport;

    /// Changes PQ code backing after releasing locks that could reference replaced buffers.
    fn set_pq_code_storage(&mut self, policy: crate::PqCodeStorage) -> Result<()>;

    /// Locks selected prepared payloads for the lifetime of the returned borrowing guard.
    fn pin<'a>(
        &'a self,
        config: &ResolvedRunConfig,
        options: &crate::PinOptions,
    ) -> Result<crate::PinnedDatasets<'a>>;

    /// Reports eligible prepared payload sizes without materializing or locking them.
    fn pin_hints(&self, config: &ResolvedRunConfig) -> Result<Vec<crate::PinHint>>;

    /// Prepares search requirements and refreshes automatic page locks under exclusive access.
    fn prepare_search(&mut self, config: &ResolvedRunConfig) -> Result<()>;

    /// Measures adaptive controls for one k using the already prepared core index.
    fn calibrate(
        &self,
        queries: orion_ann_core::search::query::QueryInput<'_>,
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
    ) -> Result<orion_ann_core::CalibratedParams>;

    /// Executes a query batch through the shared core cascade with supplied adaptive controls.
    fn search(
        &self,
        queries: QueryBatch<'_>,
        config: &ResolvedRunConfig,
        options: &SearchOptions,
        params: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult>;

    /// Scores an allowed internal-ID subset using the original metric and optional result scores.
    fn search_subset(
        &self,
        queries: QueryBatch<'_>,
        ids: &[u32],
        metric: crate::SearchMetric,
        k: usize,
        scores: bool,
    ) -> Result<orion_ann_core::SearchResult>;

    /// Runs one query through the ordinary traversal while emitting bounded diagnostic events.
    fn search_observed(
        &self,
        query: orion_ann_core::search::query::QueryInput<'_>,
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        params: orion_ann_core::CalibratedParams,
        observer: &mut RecordingObserver,
    ) -> Result<Vec<u32>>;

    /// Collects materialized sidecar identities for a snapshot without rebuilding representations.
    fn sidecars(&self) -> Result<crate::sidecars::SnapshotSidecars>;
}

/// Typed core index plus runtime-owned pin guards, receipts and allocation accounting.
pub(super) struct DynamicEngine<T: BaseElement> {
    // Drop locks before the index's heap allocations, including on unwind.
    // All allocation-replacing operations must release this token first.
    /// Owned locks released before index buffers are dropped or replaced.
    pub(super) auto_pin: orion_ann_platform::residency::OwnedPinGuard,

    /// Outcome of the most recent automatic page-lock attempt.
    pub(super) pin_report: crate::PinReport,

    /// Shared identities of published sidecars used by working-cache and snapshot saves.
    pub(super) receipts: crate::sidecars::Receipts,

    /// Core graph and datasets whose allocation lifetimes determine budget usage.
    pub(super) index: orion_ann_core::Orion<T>,
    // Observer only: leases live inside the underlying buffers, not this handle.
    resources: orion_ann_platform::allocation_budget::AllocationBudget,
}

impl<T: BaseElement> Engine for DynamicEngine<T>
where
    orion_ann_diskann::model::PaddedDataset<T>: orion_ann_core::model::dataset::rows::CascadeData,
{
    /// Implements [`Engine::storage_accounting`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn storage_accounting(&self) -> serde_json::Value {
        crate::placement::report(&self.resources)
    }

    /// Implements [`Engine::prepare_auxiliary`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn prepare_auxiliary(
        &mut self,
        config: &ResolvedRunConfig,
        store: &std::path::Path,
    ) -> Result<()> {
        if crate::placement::native_norm_bytes(config, self.index.len())? == 0 {
            return Ok(());
        }
        crate::search::prepare_datasets(&self.index, config)?;
        let mut receipts = self
            .receipts
            .lock()
            .map_err(|_| crate::Error::invalid_state("sidecar receipt lock poisoned"))?;
        self.index.visit_sidecars(|codec, encode| {
            if !receipts.contains_key(codec) {
                let record = crate::storage::put(store, codec, |writer| encode(writer))?;
                receipts.insert(codec.to_owned(), record);
            }
            Ok(())
        })
    }

    /// Implements [`Engine::mapped_base_bytes`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn mapped_base_bytes(&self) -> usize {
        if self.index.dataset().placement() == orion_ann_platform::region::Placement::Mapped {
            std::mem::size_of_val(self.index.dataset().as_slice())
        } else {
            0
        }
    }

    /// Implements [`Engine::mapped_graph_bytes`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn mapped_graph_bytes(&self) -> usize {
        if self.index.graph().placement() == orion_ann_platform::region::Placement::Mapped {
            self.index.graph().buffer_bytes().len()
        } else {
            0
        }
    }

    /// Implements [`Engine::mapped_sidecar_bytes`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
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
        .sum::<usize>()
            + self
                .index
                .native_norms_region()
                .filter(|(_, placement)| {
                    *placement == orion_ann_platform::region::Placement::Mapped
                })
                .map_or(0, |(bytes, _)| bytes.len())
    }

    /// Implements [`Engine::release_auto_pin`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn release_auto_pin(&mut self) {
        self.auto_pin = Default::default();
        self.pin_report = Default::default();
    }

    /// Implements [`Engine::pin_report`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn pin_report(&self) -> &crate::PinReport {
        &self.pin_report
    }

    /// Implements [`Engine::prepare_search`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn prepare_search(&mut self, config: &ResolvedRunConfig) -> Result<()> {
        self.release_auto_pin();
        let guard = crate::search::prepare_search(&self.index, config)?;
        // SAFETY: base, graph and prepared sidecars own stable heap buffers.
        // The token is dropped before index (field order). Calibration and PQ
        // replacement release it first; shared searches never replace buffers.
        let (locks, report) = unsafe { guard.into_owned() };
        self.auto_pin = locks;
        self.pin_report = report;
        Ok(())
    }

    /// Implements [`Engine::set_pq_code_storage`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn set_pq_code_storage(&mut self, policy: crate::PqCodeStorage) -> Result<()> {
        self.release_auto_pin();
        if self.index.prepared_pq_code_storage() == Some(policy) {
            return Ok(());
        }
        self.index.set_pq_code_storage(policy)
    }

    /// Implements [`Engine::pin`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn pin<'a>(
        &'a self,
        config: &ResolvedRunConfig,
        options: &crate::PinOptions,
    ) -> Result<crate::PinnedDatasets<'a>> {
        crate::search::pin_search_storage(&self.index, config, options)
    }

    /// Implements [`Engine::pin_hints`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn pin_hints(&self, config: &ResolvedRunConfig) -> Result<Vec<crate::PinHint>> {
        crate::search::pin_hints(&self.index, config)
    }

    /// Implements [`Engine::calibrate`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn calibrate(
        &self,
        q: orion_ann_core::search::query::QueryInput<'_>,
        c: &ResolvedRunConfig,
        k: usize,
        beam: usize,
    ) -> Result<orion_ann_core::CalibratedParams> {
        self.index.calibrate_queries(
            q,
            beam,
            c.window_size,
            orion_ann_core::CalibrationConfig {
                k,
                metric: c.metric.calibration_metric(),
            },
        )
    }

    /// Implements [`Engine::search`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn search(
        &self,
        q: QueryBatch<'_>,
        c: &ResolvedRunConfig,
        options: &SearchOptions,
        params: orion_ann_core::CalibratedParams,
    ) -> Result<orion_ann_core::SearchResult> {
        let input = q.values;
        crate::search::search_batch(
            &self.index,
            input,
            c,
            options.k,
            options.search_list_size,
            params,
            options.scores,
        )
    }

    /// Implements [`Engine::search_subset`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn search_subset(
        &self,
        q: QueryBatch<'_>,
        ids: &[u32],
        metric: crate::SearchMetric,
        k: usize,
        scores: bool,
    ) -> Result<orion_ann_core::SearchResult> {
        let metric = match metric {
            crate::SearchMetric::SquaredL2 => crate::ScoreMetric::SquaredL2,
            crate::SearchMetric::InnerProduct => crate::ScoreMetric::InnerProduct,
            crate::SearchMetric::Cosine => crate::ScoreMetric::CosineSimilarity,
        };
        self.index
            .search_subset_queries(q.values, ids, k, metric, scores)
    }

    /// Implements [`Engine::search_observed`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn search_observed(
        &self,
        query: orion_ann_core::search::query::QueryInput<'_>,
        config: &ResolvedRunConfig,
        k: usize,
        beam: usize,
        params: orion_ann_core::CalibratedParams,
        observer: &mut RecordingObserver,
    ) -> Result<Vec<u32>> {
        crate::search::search_observed(&self.index, query, config, k, beam, params, observer)
    }

    /// Implements [`Engine::sidecars`] for the typed core index.
    ///
    /// Storage ownership remains with the underlying core buffers.
    fn sidecars(&self) -> Result<crate::sidecars::SnapshotSidecars> {
        crate::sidecars::materialized(&self.index, &self.receipts)
    }
}

impl<T: BaseElement> DynamicEngine<T>
where
    orion_ann_diskann::model::PaddedDataset<T>: orion_ann_core::model::dataset::rows::CascadeData,
{
    /// Wraps a prepared core index and joins its existing allocation domain without pinning it.
    pub(super) fn new(
        index: orion_ann_core::Orion<T>,
        receipts: crate::sidecars::Receipts,
        config: &ResolvedRunConfig,
    ) -> Result<Self> {
        let resources = crate::placement::budget(config, 0).clone();
        Ok(Self {
            auto_pin: Default::default(),
            pin_report: Default::default(),
            receipts,
            index,
            resources,
        })
    }

    /// Restores sidecar objects named by a snapshot using their recorded content identities.
    pub(super) fn restore_snapshot(
        &mut self,
        store: &std::path::Path,
        records: &crate::sidecars::SnapshotSidecars,
    ) -> Result<()> {
        crate::sidecars::restore_snapshot(&mut self.index, store, records)
    }
}

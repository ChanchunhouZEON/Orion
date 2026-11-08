/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Explicit imports for research executables. Implementation modules stay private.
pub use crate::data::{inspect_vector_file, query::VectorReader, LoadedDataset};
pub use crate::execution::{
    F32CascadeBackend, GroundTruthReader, NativeU8L2Backend, QuerySource, ResolvedRunConfig,
    SearchBackend,
};
/// Run the same batch pipeline with convergence switching and early exit disabled.
/// This baseline requires no calibration and still uses approximate graph search.
pub fn search_non_adaptive(
    index: &mut crate::Index,
    queries: Vec<f32>,
    k: usize,
    beam: usize,
) -> crate::Result<crate::SearchResult> {
    index.search_batch(
        crate::QueryBatch::from_f32(&queries, index.dimension())?,
        &crate::SearchOptions::new(k, beam),
        false,
    )
}

pub mod config {
    pub use crate::config::cascade::{
        AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric,
    };
    pub use crate::config::execution::{
        validate_positive_threads, validate_threads, validate_window,
    };
    pub use crate::config::plan::IndexPlan;
    pub use crate::config::resources::validate_memory_budget;
    pub use crate::config::{
        GraphSource, GraphSourcePolicy, PreflightFormat, ResolvedSweep, VectorStorageKind,
    };
    pub use crate::search::utils::validate_search_list_sizes;
}
pub mod cascade {
    pub use crate::search::cascade::{
        build_admission, build_prefilter, build_rerank, search_batch_compose, search_compose,
        AdmissionChoice, PrefilterChoice, RerankChoice, SearchMetric,
    };
}
pub mod graph {
    pub use crate::graph::load_index;
    pub use crate::graph::persistence::{load_working, save_working};
}
pub mod parlayann {
    pub use crate::graph::parlayann::load_from_staged_file;
}
pub mod memory {
    pub use crate::execution::resources::{enforce_budget, preflight_report, print_report};
}
pub mod utils {
    pub use crate::search::utils::{
        flush_cache, mlock_bytes, result_path, set_thread_qos_user_interactive,
        validate_ground_truth,
    };
}

/// Dataset preparation and residency for controlled experiments.
pub mod residency {
    use crate::config::cascade::{
        AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric,
    };
    use crate::sidecars::requirements::DatasetRequirement;
    use orion_ann_core::{
        model::dataset::rows::{CascadeData, F32Rows},
        CompressedIndex, QuantizedDatasetShape,
    };
    #[derive(Debug, Clone)]
    pub struct DatasetRequirements(Vec<DatasetRequirement>);
    impl DatasetRequirements {
        pub fn from_cascade(
            prefilter: PrefilterChoice,
            admission: AdmissionChoice,
            rerank: RerankChoice,
        ) -> crate::Result<Self> {
            let cascade = Cascade {
                prefilter,
                admission,
                rerank,
            };
            let metric = if matches!(
                admission,
                AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16 | AdmissionChoice::PqIp
            ) {
                SearchMetric::InnerProduct
            } else {
                SearchMetric::L2
            };
            cascade.validate_cascade_options(metric)?;
            Ok(Self(cascade.dataset_requirements()))
        }
    }
    pub fn prepare<D: F32Rows>(
        index: &CompressedIndex<D>,
        requirements: &DatasetRequirements,
    ) -> crate::Result<()> {
        let sidecars: Vec<_> = requirements
            .0
            .iter()
            .filter_map(|r| match r {
                DatasetRequirement::Sidecar(s) => Some(*s),
                _ => None,
            })
            .collect();
        crate::sidecars::requirements::prepare(index, &sidecars)
    }
    /// Use the same resource detection, bounded budget and RAII locks as Index.
    pub fn auto_pin<'a, D: CascadeData>(
        index: &'a CompressedIndex<D>,
        requirements: &DatasetRequirements,
        enabled: bool,
        budget_gib: Option<f64>,
    ) -> crate::Result<crate::PinnedDatasets<'a>> {
        crate::residency::automatic_pin(
            index,
            bytemuck::cast_slice(index.dataset().allocation()),
            QuantizedDatasetShape {
                rows: F32Rows::len(index.dataset()),
                dimension: F32Rows::dimension(index.dataset()),
            },
            &requirements.0,
            enabled,
            budget_gib,
        )
    }

    pub fn pin<'a, D: CascadeData>(
        index: &'a CompressedIndex<D>,
        requirements: &DatasetRequirements,
        options: &crate::PinOptions,
    ) -> crate::Result<crate::PinnedDatasets<'a>> {
        crate::residency::pin(
            index,
            bytemuck::cast_slice(index.dataset().allocation()),
            QuantizedDatasetShape {
                rows: F32Rows::len(index.dataset()),
                dimension: F32Rows::dimension(index.dataset()),
            },
            &requirements.0,
            options,
        )
    }
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Explicit imports for research executables. Implementation modules stay private.
pub use crate::data::{inspect_vector_file, query::VectorReader, LoadedDataset};
pub use crate::driver::{
    Args, F32CascadeBackend, GroundTruthReader, NativeU8L2Backend, QuerySource, ResolvedRunConfig,
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
    pub use crate::driver::presets::{
        load_dataset_config, load_dataset_config_by_dim, load_sweep_config, ResolvedSweep,
        SweepOverrides,
    };
}
pub mod cascade {
    pub use crate::driver::pinning::{pin_admission, pin_cascade};
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
    pub use crate::driver::resources::{enforce_budget, preflight_report, print_report};
}
pub mod utils {
    pub use crate::search::utils::{
        flush_cache, mlock_bytes, result_path, set_thread_qos_user_interactive,
        validate_ground_truth,
    };
}

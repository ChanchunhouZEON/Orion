/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Dispatch base element types and graph sources into the same owned read view.
//! Source-specific persistence is separate; neither this dispatcher nor loading
//! changes the core search kernels or their SIMD/prefetch schedule.
use super::super::{base::Rows, view::ReadView};
use super::{snapshot, source::GraphInput, working};
use crate::{
    config::ResolvedRunConfig,
    data::{BaseElement, LoadedBase},
    graph::cache::CachePlan,
    Result,
};

pub(super) fn prepare(
    config: &ResolvedRunConfig,
    rows: Rows,
    source: GraphInput,
    base_sha256: &str,
) -> Result<ReadView> {
    log::info!(
        "Search dispatch: {}",
        config.cascade().dispatch_description(config.storage())
    );
    match rows {
        Rows::F32(rows) => {
            prepare_typed(config, rows, source, base_sha256, crate::graph::prepare_f32)
        }
        Rows::U8(rows) => {
            prepare_typed(config, rows, source, base_sha256, crate::graph::prepare_u8)
        }
    }
}

fn prepare_typed<T: BaseElement + 'static>(
    config: &ResolvedRunConfig,
    data: LoadedBase<T>,
    source: GraphInput,
    base_sha256: &str,
    load: impl FnOnce(
        &ResolvedRunConfig,
        &mut LoadedBase<T>,
        CachePlan,
    ) -> Result<crate::graph::LoadedGraph<T>>,
) -> Result<ReadView>
where
    orion_ann_diskann::model::PaddedDataset<T>: orion_ann_core::model::dataset::rows::CascadeData,
{
    match source {
        GraphInput::Snapshot(saved) => snapshot::restore(config, data, saved),
        GraphInput::Working(plan) => working::prepare(config, data, plan, base_sha256, load),
        #[cfg(feature = "experimental-graph")]
        GraphInput::Supplied(graph) => {
            super::super::supplier::prepare(config, data, graph, base_sha256)
        }
    }
}

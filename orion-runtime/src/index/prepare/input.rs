/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve the base allocation and original input identity before normalization.
//! Application IDs reserve budget first; graph selection and supplier checks use
//! the resulting row count and digest. Decoding remains in `index::base`.
use super::super::{base::PreparedBase, ids::IdMap, snapshot::Snapshot, BaseInput};
use crate::{
    config::{GraphSourcePolicy, ResolvedRunConfig},
    Error, IndexOptions, Result,
};
use std::path::PathBuf;

pub(super) struct LoadedInput {
    pub(super) base: PreparedBase,
    pub(super) config: ResolvedRunConfig,
    pub(super) snapshot_graph: Option<PathBuf>,
}

pub(super) fn load(
    input: BaseInput,
    options: &IndexOptions,
    snapshot: Option<&Snapshot>,
    external_ids: Option<&IdMap>,
    #[cfg(feature = "experimental-graph")] supplied: Option<&super::super::supplier::SuppliedGraph>,
) -> Result<LoadedInput> {
    let mut allocation_options = options.clone();
    // Reserve application identity alongside graph/base rather than silently
    // spending the entire user budget on the core representation.
    if let (Some(ids), Some(budget)) = (external_ids, options.memory_budget_gib) {
        allocation_options.memory_budget_gib =
            Some(budget - ids.0.len() as f64 * 8.0 / 1073741824.0);
    }
    let snapshot_graph = snapshot
        .map(|s| crate::storage::object(&s.object_store, &s.payloads.graph_sha256, "pgraph"))
        .transpose()?;
    let (base, mut config) =
        input.validate_and_load(&allocation_options, snapshot_graph.as_deref())?;
    if config.graph_source_policy == GraphSourcePolicy::Auto {
        config.select_automatic_graph(true)?;
    }

    let points = base.rows.len();
    if config.graph_source == crate::config::GraphSource::Supplied {
        config.graph_selection_reason = "supplier partitions; no builder fallback";
    }
    #[cfg(feature = "experimental-graph")]
    if let Some(graph) = supplied {
        graph.validate_base(&config, points, &base.sha256)?;
    }
    // Imported identities use content digests, never the supplier's display name.
    if let Some(saved) = snapshot {
        super::source::bind_supplier_identity(&mut config, &saved.payloads)?;
    }
    if external_ids.is_some_and(|ids| ids.0.len() != points) {
        return Err(Error::invalid_argument(
            "external ID count must equal indexed point count",
        ));
    }
    Ok(LoadedInput {
        base,
        config,
        snapshot_graph,
    })
}

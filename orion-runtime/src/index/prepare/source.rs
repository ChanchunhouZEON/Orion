/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Validate the graph request, verify snapshot provenance, and select a budgeted
//! source. A snapshot never falls back to rebuilding a missing or invalid graph.
use super::super::{
    snapshot::{CheckedManifest, Payloads, Snapshot},
    Calibration,
};
use crate::{
    config::{GraphSourcePolicy, ResolvedRunConfig},
    graph::cache::CachePlan,
    preflight as resources, Error, IndexOptions, Result,
};
use std::{collections::BTreeMap, path::Path};

pub(super) enum GraphInput {
    Snapshot(CheckedManifest),
    Working(CachePlan),
    #[cfg(feature = "experimental-graph")]
    Supplied(super::super::supplier::SuppliedGraph),
}

/// Reject an unavailable supplier path before acquiring a lease or loading rows.
pub(super) fn validate_request(
    options: &IndexOptions,
    snapshot: Option<&Snapshot>,
    #[cfg(feature = "experimental-graph")] supplied: Option<&super::super::supplier::SuppliedGraph>,
) -> Result<()> {
    if snapshot.is_none() && options.graph.source == GraphSourcePolicy::Supplied {
        #[cfg(feature = "experimental-graph")]
        let missing = supplied.is_none();
        #[cfg(not(feature = "experimental-graph"))]
        let missing = true;
        if missing {
            return Err(Error::invalid_argument(
                "use the experimental supplier import API for supplied graphs",
            ));
        }
    }
    Ok(())
}

/// Verify snapshot identity and budget the selected source before graph loading.
/// Payload checks remain in the source-specific loader. Calibration belongs to
/// the snapshot, and is empty for every fresh build.
pub(super) fn resolve(
    config: &ResolvedRunConfig,
    points: usize,
    base_sha256: &str,
    snapshot: Option<Snapshot>,
    snapshot_graph: Option<&Path>,
    #[cfg(feature = "experimental-graph")] supplied: Option<super::super::supplier::SuppliedGraph>,
) -> Result<(GraphInput, BTreeMap<usize, Calibration>)> {
    let saved = snapshot
        .map(|manifest| manifest.verify(config, points, base_sha256))
        .transpose()?;
    let (source, calibrations) = match saved {
        Some(mut manifest) => {
            resources::check_index_budget(config, points, snapshot_graph)?;
            let calibrations = std::mem::take(&mut manifest.calibrations);
            (GraphInput::Snapshot(manifest), calibrations)
        }
        None => {
            #[cfg(feature = "experimental-graph")]
            if let Some(graph) = supplied {
                resources::check_index_budget(config, points, None)?;
                (GraphInput::Supplied(graph), BTreeMap::new())
            } else {
                working_input(config, points)?
            }
            #[cfg(not(feature = "experimental-graph"))]
            {
                working_input(config, points)?
            }
        }
    };
    Ok((source, calibrations))
}

fn working_input(
    config: &ResolvedRunConfig,
    points: usize,
) -> Result<(GraphInput, BTreeMap<usize, Calibration>)> {
    let plan = CachePlan::resolve(config, points)?;
    resources::check_selected_budget(config, points, &plan)?;
    Ok((GraphInput::Working(plan), BTreeMap::new()))
}

/// Supplier cache identity is content based. Snapshots need it before manifest
/// verification; fresh imports can bind it only after producing their payloads.
pub(super) fn bind_supplier_identity(
    config: &mut ResolvedRunConfig,
    payloads: &Payloads,
) -> Result<()> {
    if config.graph_source == crate::config::GraphSource::Supplied {
        config.cache_namespace =
            super::super::snapshot::manifest::supplier_namespace(config, payloads)?;
    }
    Ok(())
}

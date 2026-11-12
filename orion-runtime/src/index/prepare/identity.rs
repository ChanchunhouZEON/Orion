/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Restore/publish application IDs and bind label state to this exact generation.
//! ID objects are immutable and saved once. Label restoration verifies both the
//! base and ID-map identity before any public search can observe the index.
use super::super::{ids::IdMap, snapshot::Snapshot, view::ReadView};
use crate::{labels::LabelStore, Result};

pub(super) fn restore_ids(
    snapshot: Option<&Snapshot>,
    memory_budget_gib: Option<f64>,
) -> Result<Option<IdMap>> {
    snapshot
        .and_then(|s| s.payloads.external_ids.as_ref().map(|id| (s, id)))
        .map(|(s, id)| IdMap::load(&s.object_store, &id.sha256, id.points, memory_budget_gib))
        .transpose()
}

pub(super) fn attach(
    view: &mut ReadView,
    external_ids: Option<&IdMap>,
    points: usize,
) -> Result<LabelStore> {
    // Publish application identities once. Later calibration-only saves
    // reference this immutable object without re-encoding the mapping.
    if view.payloads.external_ids.is_none() {
        if let Some(ids) = external_ids {
            view.payloads.external_ids = Some(super::super::snapshot::ExternalIdObject {
                sha256: ids.save(&view.object_store)?,
                points: ids.0.len(),
            });
        }
    }
    let identity = crate::labels::identity(
        &view.payloads.base_sha256,
        points,
        view.payloads
            .external_ids
            .as_ref()
            .map(|ids| ids.sha256.as_str()),
    );
    let labels = if let Some(object) = &view.payloads.labels {
        crate::labels::LabelStore::from_snapshot(crate::labels::LabelSnapshot::load(
            &view.object_store,
            object,
            &identity,
            points,
        )?)
    } else {
        crate::labels::LabelStore::empty(identity, points)
    };
    Ok(labels)
}

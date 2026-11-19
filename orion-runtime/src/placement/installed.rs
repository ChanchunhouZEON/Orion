/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Reuse is validation, not replanning. A bound core policy is immutable and may
//! already have live buffers; a phase-local binding cannot masquerade as complete.
use super::{StorageMode, StoragePlan};
use crate::{Error, Result};
use orion_ann_platform::region::{Placement, RegionOptions};
use std::collections::BTreeMap;

/// Reuses installed placements only when they cover every requirement and agree with explicit overrides.
///
/// The report is updated to actual choices; live resident buffers are never silently
/// reclassified as mappings to make a new estimate fit.
pub(super) fn validate(
    plan: &mut StoragePlan,
    installed: &BTreeMap<String, RegionOptions>,
) -> Result<()> {
    let missing = plan
        .objects
        .iter()
        .filter(|object| !installed.contains_key(object.name))
        .map(|object| object.object.to_string())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Error::invalid_state(
            "installed storage plan does not cover all required objects",
        )
        .with_detail("missing_objects", missing)
        .with_detail(
            "action",
            "resolve all phase requirements and overrides before binding the core storage plan",
        )
        .at("plan_storage"));
    }
    plan.resident_bytes = 0;
    plan.mapped_bytes = 0;
    for object in &mut plan.objects {
        let placement = installed[object.name].placement;
        let selected = match placement {
            Placement::Resident => StorageMode::Resident,
            Placement::Mapped => StorageMode::Mapped,
        };
        if object.requested != StorageMode::Auto && object.requested != selected {
            return Err(Error::invalid_state(
                "requested placement conflicts with the installed storage plan",
            )
            .with_detail("object", object.object.to_string())
            .at("plan_storage"));
        }
        object.placement = selected;
        let total = match placement {
            Placement::Resident => &mut plan.resident_bytes,
            Placement::Mapped => &mut plan.mapped_bytes,
        };
        *total = total
            .checked_add(object.bytes)
            .ok_or_else(|| Error::invalid_argument("installed storage size overflow"))?;
    }
    // The caller enforces these actual choices, not a newly forecast Auto layout.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ResolvedRunConfig;
    use crate::{CascadeConfig, IndexOptions, VectorFormat};
    use orion_ann_core::{Orion, PaddedDataset};

    fn setup() -> (Orion, ResolvedRunConfig) {
        let base = PaddedDataset::from_packed(
            &[1.0f32; 64],
            orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(8)
                .unwrap(),
        )
        .unwrap();
        let graph = (0..8)
            .map(|i| (vec![(i + 1) % 8], vec![], vec![]))
            .collect::<Vec<_>>();
        let index = Orion::from_graph_partitions(base, &graph, 0, 1, 0).unwrap();
        let config = IndexOptions {
            auto_pin: false,
            cascade: CascadeConfig {
                prefilter: Some("none".into()),
                admission: Some("l2-u8".into()),
                rerank: Some("none".into()),
            },
            ..Default::default()
        }
        .resolve("unused.fbin".into(), VectorFormat::Fbin, 8, 8, None)
        .unwrap();
        (index, config)
    }

    #[test]
    fn phase_local_binding_does_not_bypass_missing_requirements() {
        let (index, config) = setup();
        index.configure_sidecar_storage(BTreeMap::new()).unwrap();
        let error = super::super::configure_storage_policy(&index, &config, std::iter::empty())
            .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::InvalidState);
        assert_eq!(
            error.details()["missing_objects"],
            serde_json::json!(["l2-u8"])
        );
        assert!(index.sidecar_storage_plan().unwrap().is_empty());
    }

    #[test]
    fn installed_plan_validates_snapshot_extras_before_returning() {
        let (index, config) = setup();
        super::super::configure_storage_policy(&index, &config, std::iter::empty()).unwrap();
        let before = index.sidecar_storage_plan().unwrap().clone();
        let error =
            super::super::configure_storage_policy(&index, &config, ["jls".into()].into_iter())
                .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::InvalidState);
        assert_eq!(
            error.details()["missing_objects"],
            serde_json::json!(["jl-sparse"])
        );
        assert!(super::super::configure_storage_policy(
            &index,
            &config,
            ["unknown-codec".into()].into_iter()
        )
        .is_err());
        assert_eq!(index.sidecar_storage_plan().unwrap(), &before);
        super::super::configure_storage_policy(&index, &config, std::iter::empty()).unwrap();
    }

    #[test]
    fn installed_resident_objects_cannot_be_silently_replanned_as_mappings() {
        let (index, mut config) = setup();
        let fixed =
            std::mem::size_of_val(index.dataset().as_slice()) + index.graph().buffer_bytes().len();
        config.memory_budget_gib = Some((fixed + 1) as f64 / (1u64 << 30) as f64);
        index
            .configure_sidecar_storage([("qds".into(), RegionOptions::default())].into())
            .unwrap();
        let error = super::super::configure_storage_policy(&index, &config, std::iter::empty())
            .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        assert_eq!(
            index.sidecar_storage_plan().unwrap()["qds"].placement,
            Placement::Resident
        );
    }

    #[test]
    fn changed_explicit_override_is_rejected_without_rebinding() {
        let (index, mut config) = setup();
        index
            .configure_sidecar_storage([("qds".into(), RegionOptions::default())].into())
            .unwrap();
        config
            .storage_options
            .objects
            .insert(super::super::StorageObject::L2U8, StorageMode::Mapped);
        let error = super::super::configure_storage_policy(&index, &config, std::iter::empty())
            .unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::InvalidState);
    }
}

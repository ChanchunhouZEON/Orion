/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Bind object identities to allocation-owned leases. No buffer inventory or
//! reconciliation is required: successful allocations and Drop update counters.
use crate::{config::ResolvedRunConfig, Result};
use orion_ann_platform::{
    allocation_budget::AllocationBudget, region::RegionOptions, resources::MemoryResources,
};

/// Initializes the shared allocation ledger once using the effective resource limit.
///
/// `input_credit` accounts for storage already owned by the caller during this first decision.
pub(crate) fn budget(config: &ResolvedRunConfig, input_credit: u64) -> &AllocationBudget {
    config.allocation_budget.get_or_init(|| {
        AllocationBudget::new(super::effective_limit(
            config,
            MemoryResources::detect(),
            input_credit,
        ))
    })
}

/// Attaches a named allocation account to region options before any buffer is created.
pub(super) fn bind(
    config: &ResolvedRunConfig,
    name: &str,
    mut options: RegionOptions,
) -> Result<RegionOptions> {
    let object = super::StorageObject::from_codec(name)?;
    options.accounting = Some(budget(config, 0).account(&object.to_string())?);
    Ok(options)
}

/// Snapshots allocation-owned resident and mapped bytes for diagnostics, independently of RSS.
pub(crate) fn report(budget: &AllocationBudget) -> serde_json::Value {
    let usage = budget.usage();
    // Only registry bookkeeping holds this lock; no user code runs under it.
    let objects = budget.objects().expect("allocation registry poisoned");
    serde_json::json!({
        "objects": objects.into_iter().map(|(name, bytes)| serde_json::json!({
            "object": name,
            "resident_bytes": bytes.resident_bytes,
            "mapped_bytes": bytes.mapped_bytes,
        })).collect::<Vec<_>>(),
        "resident_payload_bytes": usage.resident_bytes,
        "mapped_payload_bytes": usage.mapped_bytes,
        "limit_bytes": usage.limit_bytes,
        "available_bytes": usage.limit_bytes.map(|l| l.saturating_sub(usage.resident_bytes)),
        "scope": "allocation-owned persistent payloads; not RSS or peak build memory",
    })
}

#[cfg(test)]
mod tests {
    use crate::sidecars::requirements::{prepare, SidecarRecipe};
    use orion_ann_core::{Orion, PaddedDataset, QuantizedDatasetShape};
    use orion_ann_platform::{
        allocation_budget::AllocationBudget,
        region::{Placement, RegionOptions},
    };

    fn index() -> Orion {
        let values: Vec<f32> = (0..7 * 33).map(|i| (i % 43) as f32 / 7.0 - 2.0).collect();
        let base = PaddedDataset::from_packed(
            &values,
            orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(33)
                .unwrap(),
        )
        .unwrap();
        let partitions = (0..7)
            .map(|i| (vec![(i + 1) % 7], vec![], vec![]))
            .collect::<Vec<_>>();
        Orion::from_graph_partitions(base, &partitions, 0, 1, 0).unwrap()
    }

    #[test]
    fn every_sidecar_buffer_matches_estimate_and_repeated_preparation_shares_lease() {
        for placement in [Placement::Resident, Placement::Mapped] {
            for kind in super::super::KINDS {
                let index = index();
                let requirement = kind.requirement();
                let expected = requirement
                    .estimate(QuantizedDatasetShape {
                        rows: 7,
                        dimension: 33,
                    })
                    .unwrap()
                    .total_bytes() as u64;
                let budget = AllocationBudget::new(Some(expected));
                index
                    .configure_sidecar_storage(
                        [(
                            super::super::codec(kind).into(),
                            RegionOptions {
                                placement,
                                accounting: Some(budget.account("sidecar").unwrap()),
                                ..Default::default()
                            },
                        )]
                        .into(),
                    )
                    .unwrap();
                prepare(&index, &[requirement, requirement]).unwrap();
                let usage = budget.usage();
                assert_eq!(
                    usage.resident_bytes + usage.mapped_bytes,
                    expected,
                    "{kind:?} {placement:?}"
                );
                assert_eq!(
                    usage.resident_bytes,
                    if placement == Placement::Resident {
                        expected
                    } else {
                        0
                    }
                );
                prepare(&index, &[requirement]).unwrap();
                assert_eq!(budget.usage(), usage);
                assert_eq!(budget.objects().unwrap().len(), 1);
                drop(index);
                assert_eq!(
                    budget.usage().resident_bytes + budget.usage().mapped_bytes,
                    0
                );
            }
        }
    }

    #[test]
    fn failure_after_first_buffer_rolls_back_entire_unpublished_sidecar() {
        let index = index();
        let budget = AllocationBudget::new(Some(7 * 64));
        index
            .configure_sidecar_storage(
                [(
                    "qdsl2kt".into(),
                    RegionOptions {
                        accounting: Some(budget.account("l2-kt").unwrap()),
                        ..Default::default()
                    },
                )]
                .into(),
            )
            .unwrap();
        // Codes fit exactly; allocation of per-row norms fails after codes exist.
        let error = index.try_ensure_quantized_dataset_l2_kt().unwrap_err();
        assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
        assert_eq!(budget.usage().resident_bytes, 0);
        assert!(budget.objects().unwrap().is_empty());
    }
}

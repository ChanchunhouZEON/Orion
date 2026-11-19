/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::*;
fn config() -> ResolvedRunConfig {
    crate::IndexOptions::default()
        .resolve(
            "unused.fbin".into(),
            crate::VectorFormat::Fbin,
            128,
            1_000_000,
            Some("fixture".into()),
        )
        .unwrap()
}
#[test]
fn auto_moves_only_enough_unique_payloads_and_preserves_fixed_storage() {
    let mut c = config();
    let shape = QuantizedDatasetShape {
        rows: 1_000_000,
        dimension: 128,
    };
    let sidecar_requirements = [
        SidecarRequirement::L2U8,
        SidecarRequirement::L2U8,
        SidecarRequirement::L2U16,
    ];
    c.memory_budget_gib = Some(0.9);
    let resources = MemoryResources {
        available_bytes: Some(1 << 30),
        ..Default::default()
    };
    let p =
        plan_with_resources(&c, shape, &sidecar_requirements, 700_000_000, 0, resources).unwrap();
    assert_eq!(p.mapped_bytes, 256_000_000);
    assert_eq!(p.resident_bytes, 128_000_000);
    assert_eq!(p.options.len(), 1);
    assert!(p.options.contains_key("qds6"));
    c.storage_options.mode = StorageMode::Resident;
    let p =
        plan_with_resources(&c, shape, &sidecar_requirements, 700_000_000, 0, resources).unwrap();
    assert_eq!(p.resident_bytes, 384_000_000);
    assert!(p.options.is_empty());
    assert!(700_000_000 + p.resident_bytes > p.limit_bytes.unwrap());
}
#[test]
fn unknown_observations_do_not_invent_pressure_and_live_base_is_not_counted_twice() {
    let c = config();
    let shape = QuantizedDatasetShape {
        rows: 1_000_000,
        dimension: 128,
    };
    let sidecar_requirements = [SidecarRequirement::L2U8];
    let p = plan_with_resources(
        &c,
        shape,
        &sidecar_requirements,
        700_000_000,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    assert!(p.options.is_empty());
    let p = plan_with_resources(
        &c,
        shape,
        &sidecar_requirements,
        700_000_000,
        700_000_000,
        MemoryResources {
            available_bytes: Some(300_000_000),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(p.options.is_empty());
}
#[test]
fn fitting_owned_payload_uses_only_its_actual_bytes() {
    let c = config();
    let p = plan_with_resources(
        &c,
        QuantizedDatasetShape {
            rows: 1_000_000,
            dimension: 128,
        },
        &[SidecarRequirement::L2U8],
        700_000_000,
        0,
        MemoryResources {
            available_bytes: Some(830_000_000),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(p.options.is_empty());
}
#[test]
fn all_sidecar_payloads_including_codebooks_can_be_mapped() {
    let mut c = config();
    c.storage_options.mode = StorageMode::Mapped;
    let shape = QuantizedDatasetShape {
        rows: 1000,
        dimension: 128,
    };
    let sidecar_requirements = [SidecarKind::Pq.requirement(), SidecarRequirement::L2Kt];
    let p = plan_with_resources(
        &c,
        shape,
        &sidecar_requirements,
        1000,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    assert_eq!(
        p.mapped_bytes,
        estimate_sidecars(shape, &sidecar_requirements)
            .unwrap()
            .total_bytes() as u64
    );
    assert_eq!(p.resident_bytes, 0);
    assert_eq!(p.options.len(), 2);
}
#[test]
fn whole_index_plan_maps_base_and_graph_below_their_owned_size() {
    let mut c = config();
    c.memory_budget_gib = Some(0.1);
    let shape = QuantizedDatasetShape {
        rows: 1_000_000,
        dimension: 128,
    };
    let p = complete_plan(
        &c,
        shape,
        &c.run_phase().sidecar_requirements(c.index_plan()),
        512_000_000,
        384_000_000,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    assert!(p.options.contains_key("base"));
    assert!(p.options.contains_key("graph"));
    assert_eq!(p.resident_bytes, 0);
    enforce(&p, 0).unwrap();
    c.memory_budget_gib = Some(0.01);
    let p = complete_plan(
        &c,
        shape,
        &c.run_phase().sidecar_requirements(c.index_plan()),
        512_000_000,
        384_000_000,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    // Mapping all candidates does not consume a synthetic resident allowance.
    enforce(&p, 0).unwrap();
    // Real fixed resident allocations must still fit.
    assert!(enforce(&p, p.limit_bytes.unwrap() + 1).is_err());
}

#[test]
fn explicit_cold_resident_is_reserved_before_hot_auto_and_shared_sidecars_count_once() {
    let mut c = config();
    c.storage_options
        .objects
        .insert(StorageObject::L2U8, StorageMode::Resident);
    c.memory_budget_gib = Some(128_000_000f64 / (1u64 << 30) as f64);
    let p = complete_plan(
        &c,
        QuantizedDatasetShape {
            rows: 1_000_000,
            dimension: 128,
        },
        &[SidecarRequirement::L2U8, SidecarRequirement::L2U8],
        512_000_000,
        100_000_000,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    assert_eq!(p.resident_bytes, 128_000_000);
    assert_eq!(p.objects.len(), 3);
    assert!(!p.options.contains_key("qds"));
    assert!(p.options.contains_key("graph"));
    assert!(p.options.contains_key("base"));
    enforce(&p, 0).unwrap();
    c.memory_budget_gib = Some(127_999_999f64 / (1u64 << 30) as f64);
    let p = complete_plan(
        &c,
        QuantizedDatasetShape {
            rows: 1_000_000,
            dimension: 128,
        },
        &[SidecarRequirement::L2U8],
        512_000_000,
        100_000_000,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    assert_eq!(p.resident_bytes, 128_000_000);
    assert!(enforce(&p, 0).is_err());
}

#[test]
fn object_names_capabilities_and_inactive_overrides_are_explicit() {
    for name in [
        "base",
        "graph",
        "norms",
        "labels",
        "l2-u8",
        "l2-u16",
        "mips-i8",
        "mips-i16",
        "pq",
        "l2-kt",
        "jl-sparse",
        "jl-sparse-mips",
        "jl-hadamard",
        "rabitq",
        "rabitq4",
    ] {
        let object: StorageObject = name.parse().unwrap();
        assert_eq!(object.to_string(), name);
    }
    for name in ["qds", "admission", "invalid"] {
        assert!(name.parse::<StorageObject>().is_err());
    }
    assert!("ram".parse::<StorageMode>().is_err());
    let mut c = config();
    c.storage_options
        .objects
        .insert(StorageObject::Labels, StorageMode::Mapped);
    assert!(c.storage_options.validate().is_err());
    c.storage_options.objects.clear();
    c.storage_options
        .objects
        .insert(StorageObject::Pq, StorageMode::Resident);
    let p = plan_with_resources(
        &c,
        QuantizedDatasetShape {
            rows: 64,
            dimension: 128,
        },
        &[SidecarRequirement::L2U8],
        0,
        0,
        MemoryResources::default(),
    )
    .unwrap();
    assert_eq!(p.inactive_overrides, [StorageObject::Pq]);
    assert_eq!(p.objects.len(), 1);
}

#[test]
fn every_sidecar_override_reaches_backend_and_unknown_fields_fail() {
    let mut c = config();
    c.storage_options.mode = StorageMode::Resident;
    for kind in KINDS {
        let name = codec(kind);
        let object = StorageObject::from_codec(name).unwrap();
        c.storage_options.objects = [(object, StorageMode::Mapped)].into();
        let p = plan_with_resources(
            &c,
            QuantizedDatasetShape {
                rows: 64,
                dimension: 128,
            },
            &[kind.requirement()],
            0,
            0,
            MemoryResources::default(),
        )
        .unwrap();
        assert_eq!(p.objects.len(), 1);
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.options[name].placement, Placement::Mapped);
        assert_eq!(p.objects[0].object, object);
    }
    assert!(
        serde_json::from_str::<StorageOptions>(r#"{"objects":{"not-an-object":"resident"}}"#)
            .is_err()
    );
    assert!(serde_json::from_str::<StorageOptions>(r#"{"objects":{"graph":"memory"}}"#).is_err());
    assert!(serde_json::from_str::<StorageOptions>(r#"{"object":{"graph":"resident"}}"#).is_err());
}

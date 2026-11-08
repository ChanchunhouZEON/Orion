/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use super::*;
use crate::sidecars::requirements::{prepare, SidecarKind as K, SidecarRequirement as S};
use orion_ann_core::{Orion, PaddedDataset, PaddedLayout};

fn index() -> Orion {
    let flat: Vec<_> = (0..20 * 33)
        .map(|i| ((i * 31) % 101) as f32 / 10. - 5.)
        .collect();
    let data = PaddedDataset::from_packed(
        &flat,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(33).unwrap(),
    )
    .unwrap();
    let graph: Vec<_> = (0..20u32)
        .map(|id| (vec![(id + 1) % 20], vec![], vec![]))
        .collect();
    Orion::from_graph_partitions(data, &graph, 0, 1, 0).unwrap()
}
fn shape() -> QuantizedDatasetShape {
    QuantizedDatasetShape {
        rows: 20,
        dimension: 33,
    }
}

#[test]
fn search_preparation_checks_budget_then_automatically_pins() {
    use crate::search::{F32CascadeBackend, SearchBackend};
    let index = index();
    // Resident preparation must not open input/cache files or require a CLI feature.
    let directory = std::env::temp_dir().join(format!(
        "orion-auto-pin-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let options = crate::IndexOptions {
        cache_dir: directory.join("cache"),
        graph: crate::GraphConfig {
            source: crate::GraphSourcePolicy::Rust,
            degree: 1,
            max_extra: 0,
            ..Default::default()
        },
        memory_budget_gib: Some(1e-12),
        ..Default::default()
    };
    let mut config = options
        .resolve(
            "in-memory".into(),
            crate::VectorFormat::Fbin,
            33,
            20,
            Some("a".repeat(64)),
        )
        .unwrap();
    config.prepare_only = false;
    let error = F32CascadeBackend::prepare_search(&index, &config).unwrap_err();
    assert_eq!(error.code(), crate::ErrorCode::ResourceExhausted);
    assert!(index.resident_sidecar_regions(K::L2U8).is_err());

    config.memory_budget_gib = Some(1.);
    let guard = F32CascadeBackend::prepare_search(&index, &config).unwrap();
    let report = guard.report();
    assert!(report.requested_datasets > 0);
    assert_eq!(
        report.requested_datasets,
        report.pinned_datasets + report.skipped.len()
    );
    assert_eq!(report.automatic.as_ref().unwrap()["enabled"], true);
    drop(guard);
    config.auto_pin = false;
    // No source I/O even when the original staged export no longer exists.
    config.graph_source = crate::config::GraphSource::Parlayann;
    config.staged_file_explicit = true;
    config.staged_file = Some(directory.join("missing.staged"));
    let off = F32CascadeBackend::prepare_search(&index, &config).unwrap();
    assert_eq!(off.pinned_bytes(), 0);
    assert_eq!(off.report().automatic.as_ref().unwrap()["enabled"], false);
    assert!(index.resident_sidecar_regions(K::L2U8).is_ok());
    assert!(!directory.exists());
}

#[test]
fn pinning_never_materializes_an_unprepared_sidecar() {
    let index = index();
    let requirements = [DatasetRequirement::Sidecar(S::L2U8)];
    let result = pin(&index, &[], shape(), &requirements, &PinOptions::default());
    assert!(result.is_err());
    assert!(index.resident_sidecar_regions(K::L2U8).is_err());
    let off = pin(
        &index,
        &[],
        shape(),
        &requirements,
        &PinOptions {
            mode: PinMode::Off,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(off.report().requested_datasets, 0);
}

#[test]
fn stacked_requirements_deduplicate_and_cover_all_search_buffers() {
    let index = index();
    let kinds = [
        K::L2U8,
        K::L2U16,
        K::MipsI8,
        K::MipsI16,
        K::L2Kt,
        K::JlSparse,
        K::JlSparseMips,
        K::JlHadamard,
        K::Rabitq,
        K::Rabitq4,
    ];
    let sidecars: Vec<_> = kinds.into_iter().map(SidecarRecipe::requirement).collect();
    prepare(&index, &sidecars).unwrap();
    let requirements: Vec<_> = sidecars
        .iter()
        .chain(&sidecars)
        .map(|s| DatasetRequirement::Sidecar(*s))
        .collect();
    let hints = hints(&index, &[], shape(), &requirements).unwrap();
    assert_eq!(hints.len(), 10);
    let expected = crate::sidecars::requirements::estimate_sidecars(shape(), &sidecars).unwrap();
    assert_eq!(
        hints.iter().map(|h| h.bytes).sum::<usize>(),
        expected.total_bytes()
    );
    for (kind, requirement) in kinds.into_iter().zip(sidecars) {
        let hint = hints
            .iter()
            .find(|h| h.dataset == DatasetPin::Quantized(kind))
            .unwrap();
        assert_eq!(
            hint.bytes,
            requirement.estimate(shape()).unwrap().total_bytes()
        );
    }
}

#[test]
fn byte_base_is_borrowed_without_an_exact_sidecar() {
    let data = PaddedDataset::from_packed(
        &[1u8; 66],
        PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(33).unwrap(),
    )
    .unwrap();
    let index = Orion::from_graph_partitions(
        data,
        &[(vec![1], vec![], vec![]), (vec![0], vec![], vec![])],
        0,
        1,
        0,
    )
    .unwrap();
    let base = index.dataset().as_slice();
    let hints = hints(
        &index,
        base,
        QuantizedDatasetShape {
            rows: 2,
            dimension: 33,
        },
        &[DatasetRequirement::Base, DatasetRequirement::Base],
    )
    .unwrap();
    assert_eq!(
        hints,
        [PinHint {
            dataset: DatasetPin::Base,
            bytes: base.len()
        }]
    );
    assert!(index.resident_sidecar_regions(K::L2U8).is_err());
}

#[test]
fn policies_are_bounded_selective_and_do_not_require_lock_privilege() {
    let index = index();
    prepare(&index, &[S::L2U8]).unwrap();
    let base = bytemuck::cast_slice(index.dataset().as_slice());
    let requirements = [
        DatasetRequirement::Base,
        DatasetRequirement::Sidecar(S::L2U8),
    ];
    let options = PinOptions {
        budget_bytes: 0,
        datasets: Some(vec![DatasetPin::Quantized(K::L2U8)]),
        ..Default::default()
    };
    let pinned = pin(&index, base, shape(), &requirements, &options).unwrap();
    assert_eq!(pinned.report().requested_datasets, 1);
    assert_eq!(pinned.report().pinned_datasets, 0);
    assert_eq!(pinned.report().skipped.len(), 1);
    assert_eq!(pinned.pinned_bytes(), 0);
    assert!(pin(
        &index,
        base,
        shape(),
        &requirements,
        &PinOptions {
            mode: PinMode::Required,
            ..options.clone()
        }
    )
    .is_err());
    assert!(pin(
        &index,
        base,
        shape(),
        &requirements,
        &PinOptions {
            datasets: Some(vec![DatasetPin::Quantized(K::MipsI8)]),
            ..options.clone()
        }
    )
    .is_err());
    let empty = pin(
        &index,
        base,
        shape(),
        &requirements,
        &PinOptions {
            datasets: Some(vec![]),
            ..options
        },
    )
    .unwrap();
    assert_eq!(empty.report().requested_datasets, 0);
}

#[test]
fn different_transform_recipe_cannot_pin_the_canonical_slot() {
    let index = index();
    prepare(&index, &[K::JlSparse.requirement()]).unwrap();
    let S::JlSparse(seed) = K::JlSparse.requirement() else {
        unreachable!()
    };
    assert!(hints(
        &index,
        &[],
        shape(),
        &[DatasetRequirement::Sidecar(S::JlSparse(seed + 1))]
    )
    .is_err());
}

#[test]
fn selecting_base_does_not_require_unselected_sidecars() {
    let index = index();
    let base = bytemuck::cast_slice(index.dataset().as_slice());
    let result = pin(
        &index,
        base,
        shape(),
        &[
            DatasetRequirement::Base,
            DatasetRequirement::Sidecar(S::L2U8),
        ],
        &PinOptions {
            budget_bytes: 0,
            datasets: Some(vec![DatasetPin::Base]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.report().requested_datasets, 1);
    assert_eq!(result.report().skipped[0].0, DatasetPin::Base);
    assert!(index.resident_sidecar_regions(K::L2U8).is_err());
}

#[test]
fn os_lock_denial_falls_back_without_changing_prepared_storage() {
    let index = index();
    let base = bytemuck::cast_slice(index.dataset().as_slice());
    let requirements = [DatasetRequirement::Base];
    let options = PinOptions {
        budget_bytes: usize::MAX,
        ..Default::default()
    };
    let mut calls = 0;
    let result = pin_with(&index, base, shape(), &requirements, &options, |_, _, _| {
        calls += 1;
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "test memlock refusal",
        ))
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(result.pinned_bytes(), 0);
    assert_eq!(result.report().skipped[0].0, DatasetPin::Base);
    assert!(result.report().skipped[0]
        .1
        .contains("test memlock refusal"));
    assert_eq!(
        base,
        bytemuck::cast_slice::<_, u8>(index.dataset().as_slice())
    );
    assert!(pin_with(
        &index,
        base,
        shape(),
        &requirements,
        &PinOptions {
            mode: PinMode::Required,
            ..options
        },
        |_, _, _| Err(std::io::Error::other("denied"))
    )
    .is_err());
}

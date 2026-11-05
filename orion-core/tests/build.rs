/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_core::{
    BuildBackend, BuildConfig, BuildMetric, ErrorCode, Orion, PaddedDataset, PaddedLayout,
};
use orion_ann_vector::L2F32Distance;
#[cfg(feature = "parlay")]
use orion_ann_vector::L2U8Distance;

fn config() -> BuildConfig {
    BuildConfig {
        degree: 4,
        build_l: 12,
        max_extra: 3,
        threads: 2,
        ..Default::default()
    }
}

#[test]
fn invalid_inputs_are_structured_errors() {
    for (values, dimension) in [(&[][..], 2), (&[1.0, 2.][..], 0), (&[1.][..], 2)] {
        assert_eq!(
            Orion::<f32>::build_from_slice(values, dimension, config())
                .err()
                .unwrap()
                .code(),
            ErrorCode::InvalidArgument
        );
    }
    for (field, cfg) in [
        (
            "degree",
            BuildConfig {
                degree: 0,
                ..config()
            },
        ),
        (
            "build_l",
            BuildConfig {
                build_l: 1,
                ..config()
            },
        ),
        (
            "alpha",
            BuildConfig {
                alpha: f32::NAN,
                ..config()
            },
        ),
        (
            "alpha",
            BuildConfig {
                alpha: 0.9,
                ..config()
            },
        ),
        (
            "threads",
            BuildConfig {
                threads: 0,
                ..config()
            },
        ),
        (
            "threads",
            BuildConfig {
                threads: 65,
                ..config()
            },
        ),
        (
            "max_extra",
            BuildConfig {
                max_extra: usize::MAX,
                ..config()
            },
        ),
        (
            "local_pct",
            BuildConfig {
                local_pct: 0,
                ..config()
            },
        ),
    ] {
        let error = Orion::<f32>::build_from_slice(&[0.; 8], 2, cfg)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert_eq!(error.details()["field"], field);
        assert_eq!(error.operation(), Some("build_graph"));
    }
    let error = Orion::<f32>::build_from_slice(&[f32::NAN; 8], 2, config())
        .err()
        .unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidArgument);
}

#[test]
fn rust_builder_preserves_owned_base_and_requires_explicit_save() {
    let values: Vec<f32> = (0..64 * 13)
        .map(|i| ((i * 71 + i / 13) % 251) as f32)
        .collect();
    let data = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<L2F32Distance>(13).unwrap(),
    )
    .unwrap();
    let ptr = data.as_slice().as_ptr();
    let index = Orion::<f32>::build(
        data,
        BuildConfig {
            backend: BuildBackend::DiskAnn,
            ..config()
        },
    )
    .unwrap();
    assert_eq!(index.dataset().as_slice().as_ptr(), ptr);
    assert_eq!(index.len(), 64);
    assert!(index.graph().neighbors(0).len() <= 4);
}

#[test]
fn explicit_backends_never_silently_change_metric_or_storage() {
    let cfg = BuildConfig {
        backend: BuildBackend::DiskAnn,
        ..config()
    };
    assert_eq!(
        Orion::<u8>::build_from_slice(&[1; 32], 2, cfg.clone())
            .err()
            .unwrap()
            .code(),
        ErrorCode::UnsupportedConfiguration
    );
    assert_eq!(
        Orion::<f32>::build_from_slice(
            &[1.; 32],
            2,
            BuildConfig {
                metric: BuildMetric::InnerProduct,
                ..cfg
            }
        )
        .err()
        .unwrap()
        .code(),
        ErrorCode::UnsupportedConfiguration
    );
}

#[cfg(feature = "parlay")]
#[test]
fn default_u8_builder_preserves_allocation_extras_and_thread_independence() {
    let values: Vec<u8> = (0..96 * 13)
        .map(|i| ((i * 19 + i / 13) % 251) as u8)
        .collect();
    let data = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<L2U8Distance>(13).unwrap(),
    )
    .unwrap();
    let ptr = data.as_slice().as_ptr();
    let index = Orion::<u8>::build(data, config()).unwrap();
    assert_eq!(index.dataset().as_slice().as_ptr(), ptr);
    assert!((0..96).any(|id| !index.graph().extra_candidates(id).is_empty()));
    let other = Orion::<u8>::build_from_slice(
        &values,
        13,
        BuildConfig {
            threads: 1,
            ..config()
        },
    )
    .unwrap();
    assert_eq!(index.graph().buffer_bytes(), other.graph().buffer_bytes());
    assert_eq!(index.entry_point(), other.entry_point());
    // Return a usable exact-search index without silently calibrating a cascade.
    let ids = index
        .search_l2(
            &values[..13],
            orion_ann_core::DynamicSearchConfig {
                k: 1,
                search_list_size: 32,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids, vec![0]);
}

#[cfg(not(feature = "parlay"))]
#[test]
fn without_parlay_auto_is_f32_only_and_explicit_pa_errors() {
    let values: Vec<f32> = (0..64 * 13).map(|i| (i % 251) as f32).collect();
    assert!(Orion::<f32>::build_from_slice(&values, 13, config()).is_ok());
    assert_eq!(
        Orion::<u8>::build_from_slice(&[1; 32], 2, config())
            .err()
            .unwrap()
            .code(),
        ErrorCode::UnsupportedConfiguration
    );
    assert_eq!(
        Orion::<f32>::build_from_slice(
            &values,
            13,
            BuildConfig {
                backend: BuildBackend::ParlayAnn,
                ..config()
            }
        )
        .err()
        .unwrap()
        .code(),
        ErrorCode::UnsupportedConfiguration
    );
}

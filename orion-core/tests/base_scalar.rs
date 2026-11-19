/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Storage and calibration contracts for built-in base scalars.

use orion_ann_core::index::{BaseScalar, CalibrationScalar};
use orion_ann_core::{CalibrationConfig, Orion, PaddedDataset, ScoreMetric};
use orion_ann_vector::{
    DistanceKernelBackend, IpU8Distance, L2F32Distance, L2U8Distance, PaddedLayout,
};

/// Verifies physical compatibility with existing caches, including the byte L2 limit exception.
#[test]
fn legacy_layouts_are_unchanged() {
    for dim in [1, 25, 32, 33, 128, 960, 1536] {
        assert_eq!(
            PaddedLayout::for_base_scalar::<f32>(dim).unwrap(),
            PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap()
        );
        assert_eq!(
            PaddedLayout::for_base_scalar::<u8>(dim).unwrap(),
            PaddedLayout::for_kernel::<IpU8Distance>(dim).unwrap()
        );
    }
    let wide = L2U8Distance::MAX_COORDINATES + 1;
    assert!(PaddedLayout::for_kernel::<L2U8Distance>(wide).is_err());
    assert!(PaddedLayout::for_base_scalar::<u8>(wide).is_ok());
    assert!(PaddedLayout::for_base_scalar::<u8>(0).is_err());
    assert!(PaddedLayout::for_base_scalar::<f32>(usize::MAX).is_err());
}

/// Constructs integer storage through the shared graph boundary and exercises calibration.
fn check_integer<T: BaseScalar + CalibrationScalar>(values: &[T]) {
    let layout = PaddedLayout::for_base_scalar::<T>(2).unwrap();
    assert_eq!(layout.element_bytes(), std::mem::size_of::<T>());
    let data = PaddedDataset::from_packed(values, layout).unwrap();
    let parts = [
        (vec![1, 2], vec![], vec![]),
        (vec![0, 2], vec![], vec![]),
        (vec![0, 1], vec![], vec![]),
    ];
    let index = Orion::from_graph_partitions(data, &parts, 0, 2, 0).unwrap();
    let params = index
        .calibrate(
            &[1., 2.],
            3,
            3,
            CalibrationConfig {
                k: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(params.threshold.is_finite());
    let results = index
        .search_subset_exact(&[1., 2.], &[0, 1, 2], 3, ScoreMetric::SquaredL2, true)
        .unwrap();
    assert_eq!(results.counts, [3]);
    assert_eq!(results.ids[0], 0);
}

/// Signed and unsigned widths share storage and exact search without float base copies.
#[test]
fn integer_storage_calibration_and_subset_search() {
    check_integer(&[1u16, 2, u16::MAX, 4, 5, 6]);
    check_integer(&[1u32, 2, 16_777_217, u32::MAX, 5, 6]);
    check_integer(&[1i8, 2, -3, 4, 5, -6]);
    check_integer(&[1i16, 2, i16::MIN, 4, 5, i16::MAX]);
    check_integer(&[1i32, 2, 16_777_217, -16_777_217, i32::MIN, i32::MAX]);
}

/// Distances distinguish adjacent i32 values that would collapse after an f32 conversion.
#[test]
fn i32_exact_subset_preserves_original_coordinates() {
    let data = PaddedDataset::from_packed(
        &[16_777_217i32, 16_777_216],
        PaddedLayout::for_base_scalar::<i32>(1).unwrap(),
    )
    .unwrap();
    let parts = [(vec![1], vec![], vec![]), (vec![0], vec![], vec![])];
    let index = Orion::from_graph_partitions(data, &parts, 0, 1, 0).unwrap();
    let result = index
        .search_subset_exact(&[16_777_216.], &[0, 1], 2, ScoreMetric::SquaredL2, true)
        .unwrap();
    assert_eq!(result.ids, [1, 0]);
    assert_eq!(result.scores.unwrap().values, [0., 1.]);
}

/// Adjacent u32 values remain distinguishable even at the maximum coordinate.
#[test]
fn unsigned_calibration_and_exact_subset_preserve_wide_coordinates() {
    use orion_ann_core::search::query::QueryInput;
    let data = PaddedDataset::from_packed(
        &[u32::MAX - 1, u32::MAX],
        PaddedLayout::for_base_scalar::<u32>(1).unwrap(),
    )
    .unwrap();
    let parts = [(vec![1], vec![], vec![]), (vec![0], vec![], vec![])];
    let index = Orion::from_graph_partitions(data, &parts, 0, 1, 0).unwrap();
    let query = [u32::MAX];
    let result = index
        .search_subset_queries(
            QueryInput::U32(&query),
            &[0, 1],
            2,
            ScoreMetric::SquaredL2,
            true,
        )
        .unwrap();
    assert_eq!(result.ids, [1, 0]);
    assert_eq!(result.scores.unwrap().values, [0., 1.]);
    let params = index
        .calibrate_queries(
            QueryInput::U32(&query),
            2,
            2,
            CalibrationConfig {
                k: 1,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(params.threshold.is_finite());
    assert_eq!(
        u16::compatible_distance(
            orion_ann_core::CalibrationMetric::SquaredL2,
            &[65534.],
            &[u16::MAX],
            &mut [0.]
        )
        .unwrap(),
        1.
    );
}

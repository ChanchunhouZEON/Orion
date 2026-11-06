/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::*;
use orion_ann_core::{PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;
fn base(rows: usize, dimension: usize) -> PaddedDataset<f32> {
    let values: Vec<_> = (0..rows * dimension)
        .map(|i| ((i * 13) % 43) as f32 / 7.0 - 2.0)
        .collect();
    PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<L2F32Distance>(dimension).unwrap(),
    )
    .unwrap()
}
#[test]
fn stacked_memory_matches_distinct_owned_datasets() {
    let input_shape = QuantizedDatasetShape {
        rows: 7,
        dimension: 33,
    };
    let base = base(input_shape.rows, input_shape.dimension);
    let jl = JLSparseDataset::<1024, 9>::build(&base, &42).unwrap();
    let kt = L2KTDataset::build(&base, &()).unwrap();
    let u16 = ScalarQuantizedDataset::<L2U16>::build(&base, &()).unwrap();
    let request = [
        SidecarRequirement::JlSparse(42),
        SidecarRequirement::L2Kt,
        SidecarRequirement::L2U16,
        SidecarRequirement::L2U16,
        SidecarRequirement::JlSparse(42),
    ];
    let report = estimate_sidecars(input_shape, &request).unwrap();
    assert_eq!(report.allocations().len(), 3);
    // Independently sum exposed buffers, including JL's projection and KT's norms.
    let actual = jl.codes().len()
        + std::mem::size_of_val(jl.indices())
        + kt.data().len()
        + std::mem::size_of_val(kt.norms_sq())
        + std::mem::size_of_val(u16.data());
    assert_eq!(report.total_bytes(), actual);
    assert_eq!(report.allocations()[0].requirement, request[0]);
    assert_eq!(
        estimate_sidecars(input_shape, &[]).unwrap().total_bytes(),
        0
    );
}
#[test]
fn distinct_recipes_with_equal_size_do_not_alias() {
    let input_shape = QuantizedDatasetShape {
        rows: 7,
        dimension: 33,
    };
    let a = SidecarRequirement::JlSparse(42);
    let b = SidecarRequirement::JlSparse(43);
    assert_eq!(
        a.estimate(input_shape).unwrap(),
        b.estimate(input_shape).unwrap()
    );
    let report = estimate_sidecars(input_shape, &[a, b, a]).unwrap();
    assert_eq!(report.allocations().len(), 2);
    assert_eq!(
        report.total_bytes(),
        a.estimate(input_shape).unwrap().total_bytes() * 2
    );
    let a = SidecarRequirement::JlHadamard(HadamardOptions {
        seed: 42,
        padded_dimension: Some(64),
    });
    let b = SidecarRequirement::JlHadamard(HadamardOptions {
        seed: 42,
        padded_dimension: Some(128),
    });
    assert_eq!(
        estimate_sidecars(input_shape, &[a, b])
            .unwrap()
            .allocations()
            .len(),
        2
    );
}
#[test]
fn invalid_transform_and_large_matrix_fail_before_allocation() {
    let input_shape = QuantizedDatasetShape {
        rows: 1,
        dimension: 33,
    };
    for padded in [0, 32, 63, usize::MAX] {
        assert!(SidecarRequirement::JlHadamard(HadamardOptions {
            seed: 0,
            padded_dimension: Some(padded)
        })
        .estimate(input_shape)
        .is_err());
    }
    assert!(SidecarKind::Rabitq
        .requirement()
        .estimate(QuantizedDatasetShape {
            rows: 1,
            dimension: u32::MAX as usize
        })
        .is_err());
    assert!(JLSparseDataset::<7, 9>::estimate_resident_memory(input_shape, &0).is_err());
    assert!(JLSparseDataset::<128, 0>::estimate_resident_memory(input_shape, &0).is_err());
}

#[test]
fn canonical_list_initializes_exactly_the_requested_lazy_slots() {
    let rows = 8;
    let dimension = 33;
    let parts: Vec<_> = (0..rows as u32)
        .map(|id| (vec![(id + 1) % rows as u32], vec![], vec![]))
        .collect();
    let index = orion_ann_core::CompressedIndex::from_graph_partitions(
        base(rows, dimension),
        &parts,
        0,
        4,
        0,
    )
    .unwrap();
    let kinds = [
        SidecarKind::Pq,
        SidecarKind::L2U8,
        SidecarKind::L2U16,
        SidecarKind::MipsI8,
        SidecarKind::MipsI16,
        SidecarKind::L2Kt,
        SidecarKind::JlSparse,
        SidecarKind::JlSparseMips,
        SidecarKind::JlHadamard,
        SidecarKind::Rabitq,
        SidecarKind::Rabitq4,
    ];
    let requirements: Vec<_> = kinds.into_iter().map(SidecarKind::requirement).collect();
    prepare(&index, &requirements).unwrap();
    prepare(&index, &requirements).unwrap();
    let mut codecs = std::collections::HashSet::new();
    index
        .visit_sidecars(|codec, _| {
            assert!(codecs.insert(codec));
            Ok(())
        })
        .unwrap();
    assert_eq!(codecs.len(), kinds.len());
    let report =
        estimate_sidecars(QuantizedDatasetShape { rows, dimension }, &requirements).unwrap();
    let resident = [
        index
            .try_ensure_quantized_dataset_pq()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_l2_u16()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_mips()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_mips_i16()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_l2_kt()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_jl()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_jl_mips()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_jl_hadamard()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_rabitq()
            .unwrap()
            .resident_memory(),
        index
            .try_ensure_quantized_dataset_rabitq_b4()
            .unwrap()
            .resident_memory(),
    ];
    assert_eq!(
        report.total_bytes(),
        resident.into_iter().map(|m| m.total_bytes()).sum::<usize>()
    );
    let SidecarRequirement::JlHadamard(mut options) = SidecarKind::JlHadamard.requirement() else {
        unreachable!()
    };
    options.padded_dimension = Some(1024);
    prepare(&index, &[SidecarRequirement::JlHadamard(options)]).unwrap();
    // An unsupported recipe must not silently reuse the canonical cache slot.
    assert!(prepare(&index, &[SidecarRequirement::JlSparse(123)]).is_err());
}

#[test]
fn implicit_and_explicit_equivalent_widths_share_storage() {
    let input_shape = QuantizedDatasetShape {
        rows: 7,
        dimension: 33,
    };
    let implicit = SidecarRequirement::JlHadamard(HadamardOptions {
        seed: 42,
        padded_dimension: None,
    });
    let explicit = SidecarRequirement::JlHadamard(HadamardOptions {
        seed: 42,
        padded_dimension: Some(1024),
    });
    let report = estimate_sidecars(input_shape, &[implicit, explicit]).unwrap();
    assert_eq!(report.allocations().len(), 1);
    assert_eq!(
        report.total_bytes(),
        implicit.estimate(input_shape).unwrap().total_bytes()
    );
}

#[test]
fn pq_options_resolve_before_deduplication_and_match_actual_allocation() {
    use orion_ann_core::model::dataset::quantized_dataset::pq::{PqBuildOptions, PqDataset};
    let shape = QuantizedDatasetShape {
        rows: 23,
        dimension: 33,
    };
    let canonical = PqBuildOptions::default();
    let explicit = canonical.resolved(shape.dimension).unwrap();
    let requirements = [
        SidecarRequirement::Pq(canonical),
        SidecarRequirement::Pq(explicit),
    ];
    let report = estimate_sidecars(shape, &requirements).unwrap();
    assert_eq!(report.allocations().len(), 1);
    let data = PqDataset::build(&base(shape.rows, shape.dimension), &canonical).unwrap();
    assert_eq!(report.total_bytes(), data.resident_memory().total_bytes());
    let different = SidecarRequirement::Pq(PqBuildOptions {
        seed: 43,
        ..canonical
    });
    assert_eq!(
        estimate_sidecars(shape, &[requirements[0], different])
            .unwrap()
            .allocations()
            .len(),
        2
    );
}

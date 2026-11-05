/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Allocation contracts and stacked identity, independent of cascade choices.
use orion_ann_core::model::dataset::{
    quantized_dataset::HadamardOptions, rabitq_dataset::RabitQDataset, *,
};
use orion_ann_core::{
    PaddedDataset, PaddedLayout, PreparedQuery, QuantizedDataset, QuantizedDatasetShape,
};
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
fn check<D: QuantizedDataset>(options: D::BuildOptions) {
    for dimension in [1, 17, 31, 33, 65, 100, 128, 129] {
        let input_shape = QuantizedDatasetShape { rows: 7, dimension };
        let base = base(input_shape.rows, input_shape.dimension);
        let estimate = D::estimate_resident_memory(input_shape, &options).unwrap();
        let dataset = D::build(&base, &options).unwrap();
        assert_eq!(
            estimate,
            dataset.resident_memory(),
            "{} dim={dimension}",
            std::any::type_name::<D>()
        );
        let query = vec![0.125; dimension];
        let prepared = dataset.prepare_query(&query).unwrap();
        for id in 0..input_shape.rows as u32 {
            assert!(prepared.distance(id).unwrap().is_finite());
        }
        assert!(prepared.distance(input_shape.rows as u32).is_err());
        assert!(dataset.prepare_query(&query[..dimension - 1]).is_err());
        let mut nonfinite = query;
        nonfinite[0] = f32::NAN;
        assert!(dataset.prepare_query(&nonfinite).is_err());
    }
    for input_shape in [
        QuantizedDatasetShape {
            rows: 0,
            dimension: 32,
        },
        QuantizedDatasetShape {
            rows: 1,
            dimension: 0,
        },
        QuantizedDatasetShape {
            rows: usize::MAX,
            dimension: 32,
        },
        QuantizedDatasetShape {
            rows: 1,
            dimension: usize::MAX,
        },
    ] {
        assert!(D::estimate_resident_memory(input_shape, &options).is_err());
    }
}
#[test]
fn every_representation_accounts_for_its_actual_allocations() {
    check::<ScalarQuantizedDataset<L2U8>>(());
    check::<ScalarQuantizedDataset<L2U16>>(());
    check::<ScalarQuantizedDataset<MipsI8>>(());
    check::<ScalarQuantizedDataset<MipsI16>>(());
    check::<L2KTDataset>(());
    check::<JLSparseDataset<1024, 9>>(42);
    check::<JLSparseDatasetMips<1024, 9>>(42);
    // Noncanonical projection shapes must obey the same storage contract.
    check::<JLSparseDataset<128, 3>>(7);
    check::<JLSparseDatasetMips<256, 5>>(7);
    check::<JlHadamardDataset>(JlHadamardDataset::default_build_options());
    check::<JlHadamardDataset>(HadamardOptions {
        seed: 7,
        padded_dimension: Some(256),
    });
    check::<RabitQDataset>(42);
    check::<RabitQ4Dataset>(42);
}

#[test]
fn public_memory_debug_contains_only_the_total() {
    let memory = L2KTDataset::estimate_resident_memory(
        QuantizedDatasetShape {
            rows: 7,
            dimension: 33,
        },
        &(),
    )
    .unwrap();
    assert_eq!(
        format!("{memory:?}"),
        format!("ResidentMemory {{ total_bytes: {} }}", memory.total_bytes())
    );
}

#[test]
fn prepared_query_owns_its_encoding_and_borrows_only_the_dataset() {
    let dataset = ScalarQuantizedDataset::<L2U8>::build(&base(7, 33), &()).unwrap();
    let prepared = {
        let input = vec![0.125; 33];
        dataset.prepare_query(&input).unwrap()
    };
    assert!(prepared.distance(0).unwrap().is_finite());
}

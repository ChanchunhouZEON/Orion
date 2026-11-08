/* Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License. */
use orion_ann_core::model::dataset::quantized_dataset::{
    pq::{PqBuildOptions, PqDataset, PqMetric},
    scalar::{
        ScalarQuantizedDataset,
        spec::{L2U8, L2U16, MipsI8, MipsI16, QuantSpec},
    },
};
use orion_ann_core::{PaddedDataset, PaddedLayout};
use orion_ann_platform::region::{Placement, RegionOptions};

fn scalar<Q: QuantSpec>(dim: usize) {
    let values: Vec<f32> = (0..83 * dim)
        .map(|i| ((i * 17) % 101) as f32 / 50. - 1.)
        .collect();
    let base = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(dim).unwrap(),
    )
    .unwrap();
    let owned = ScalarQuantizedDataset::<Q>::try_from_f32_dataset(&base).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let options = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
    };
    let mapped = ScalarQuantizedDataset::<Q>::build_with_storage(&base, &options).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(owned.data()),
        bytemuck::cast_slice::<_, u8>(mapped.data())
    );
    let mut encoded = vec![];
    owned.encode(&mut encoded).unwrap();
    let decoded = ScalarQuantizedDataset::<Q>::decode_with_storage(
        &mut encoded.as_slice(),
        encoded.len() as u64,
        &options,
    )
    .unwrap();
    assert_eq!(decoded.placement(), Placement::Mapped);
    // Mutating/truncating the original source cannot invalidate the private mapping.
    encoded.clear();
    let q = owned.try_quantize_query(&values[..dim], true).unwrap();
    for id in [0, 1, 17, 82] {
        assert_eq!(
            owned.distance(id, &q).unwrap(),
            decoded.distance(id, &q).unwrap()
        );
    }
    std::thread::scope(|s| {
        s.spawn(|| {
            assert_eq!(
                decoded.distance(82, &q).unwrap(),
                mapped.distance(82, &q).unwrap()
            )
        });
    });
}
#[test]
fn scalar_build_decode_padding_and_lifetime() {
    for dim in [1, 25, 32, 33, 128, 129, 960] {
        scalar::<L2U8>(dim);
        scalar::<L2U16>(dim);
        scalar::<MipsI8>(dim);
        scalar::<MipsI16>(dim);
    }
}
#[test]
fn pq_mapped_build_and_decode_match_owned() {
    let dim = 25;
    let values: Vec<f32> = (0..128 * dim).map(|i| ((i * 17) % 251) as f32).collect();
    let base = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(dim).unwrap(),
    )
    .unwrap();
    let recipe = PqBuildOptions {
        chunks: Some(4),
        training_rows: 128,
        iterations: 2,
        ..Default::default()
    };
    let owned = PqDataset::build_with_storage(&base, &recipe, &RegionOptions::default()).unwrap();
    let options = RegionOptions {
        placement: Placement::Mapped,
        ..Default::default()
    };
    let mapped = PqDataset::build_with_storage(&base, &recipe, &options).unwrap();
    assert_eq!(owned.codes(), mapped.codes());
    let mut bytes = vec![];
    owned.encode(&mut bytes).unwrap();
    let restored =
        PqDataset::decode_with_options(&mut bytes.as_slice(), bytes.len() as u64, &options)
            .unwrap();
    bytes.clear();
    for metric in [PqMetric::L2, PqMetric::InnerProduct] {
        let mut expected = vec![];
        let mut actual = vec![];
        owned
            .query(&values[..dim], metric)
            .unwrap()
            .stream(&[127, 0, 55, 55, 1], 32, |id, d| expected.push((id, d)))
            .unwrap();
        restored
            .query(&values[..dim], metric)
            .unwrap()
            .stream(&[127, 0, 55, 55, 1], 32, |id, d| actual.push((id, d)))
            .unwrap();
        assert_eq!(actual, expected);
    }
    assert!(PqDataset::decode_with_options(&mut &[0u8; 32][..], 32, &options).is_err());
}

#[test]
fn projected_sidecars_preserve_encoding_scores_and_private_lifetime() {
    use orion_ann_core::model::dataset::quantized_dataset::{
        jl::{
            hadamard::JlHadamardDataset,
            sparse::{JLSparseDataset, JLSparseDatasetMips},
        },
        kernel_trick::L2KTDataset,
        rabitq::{binary::RabitQDataset, multibit::RabitQ4Dataset},
    };
    use orion_ann_core::{PreparedQuery, QuantizedDataset, QuantizedDatasetCompute};
    let dim = 25;
    let values: Vec<f32> = (0..33 * dim)
        .map(|i| ((i * 17) % 101) as f32 / 50. - 1.)
        .collect();
    let base = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(dim).unwrap(),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let options = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
    };
    macro_rules! verify {
        ($ty:ty, $owned:expr, $mapped:expr) => {{
            let owned: $ty = $owned.unwrap();
            let mapped: $ty = $mapped.unwrap();
            let mut original = vec![];
            let mut actual = vec![];
            owned.encode(&mut original).unwrap();
            mapped.encode(&mut actual).unwrap();
            assert_eq!(original, actual);
            assert_eq!(owned.resident_memory(), mapped.resident_memory());
            let decoded =
                <$ty>::decode_with_storage(&mut actual.as_slice(), actual.len() as u64, &options)
                    .unwrap();
            actual.fill(0);
            drop(mapped);
            let query = &values[..dim];
            let expected = owned.prepare_query(query).unwrap();
            let q = decoded.prepare_query(query).unwrap();
            for id in 0..33 {
                assert_eq!(expected.distance(id).unwrap(), q.distance(id).unwrap());
            }
            let mut encoded = vec![];
            decoded.encode(&mut encoded).unwrap();
            assert_eq!(encoded, original);
            assert!(
                <$ty>::decode_with_storage(
                    &mut original[..original.len() - 1].as_ref(),
                    original.len() as u64,
                    &options
                )
                .is_err()
            );
        }};
    }
    verify!(
        L2KTDataset,
        L2KTDataset::try_build_from(&base),
        L2KTDataset::try_build_from_with_storage(&base, &options)
    );
    verify!(
        JLSparseDataset,
        JLSparseDataset::try_build_from(&base, 42),
        JLSparseDataset::try_build_from_with_storage(&base, 42, &options)
    );
    verify!(
        JLSparseDatasetMips,
        JLSparseDatasetMips::try_build_from(&base, 42),
        JLSparseDatasetMips::try_build_from_with_storage(&base, 42, &options)
    );
    verify!(
        JlHadamardDataset,
        JlHadamardDataset::try_build_from(&base, 42),
        JlHadamardDataset::try_build_from_with_storage(&base, 42, &options)
    );
    verify!(
        RabitQDataset,
        RabitQDataset::try_build_from(&base, 42),
        RabitQDataset::try_build_from_with_storage(&base, 42, &options)
    );
    verify!(
        RabitQ4Dataset,
        RabitQ4Dataset::try_build_from(&base, 42),
        RabitQ4Dataset::try_build_from_with_storage(&base, 42, &options)
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

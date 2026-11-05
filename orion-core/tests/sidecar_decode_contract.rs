/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Persisted metadata must be checked before it determines allocations or SIMD shape.
use orion_ann_core::model::dataset::*;
use orion_ann_core::{PaddedDataset, PaddedLayout, PreparedQuery, QuantizedDataset, Result};
use std::io::{Cursor, Read};

fn base() -> PaddedDataset<f32> {
    let values: Vec<_> = (0..3 * 17).map(|i| (i % 19) as f32 - 8.).collect();
    PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(17).unwrap(),
    )
    .unwrap()
}
fn check<D: QuantizedDataset + std::fmt::Debug>(
    original: D,
    encode: impl Fn(&D, &mut dyn std::io::Write) -> Result<()>,
    decode: impl Fn(&mut dyn Read, u64) -> Result<D>,
    zero_fields: &[usize],
) {
    let mut good = vec![];
    encode(&original, &mut good).unwrap();
    let restored = decode(&mut Cursor::new(&good), good.len() as u64).unwrap();
    assert_eq!(restored.resident_memory(), original.resident_memory());
    let query = vec![0.25; 17];
    let a = original.prepare_query(&query).unwrap();
    let b = restored.prepare_query(&query).unwrap();
    for id in 0..3 {
        assert_eq!(a.distance(id).unwrap(), b.distance(id).unwrap());
    }
    assert!(
        format!("{restored:?}").len() < 300,
        "Debug must stay bounded"
    );
    // Different metadata invariants fail independently, not all at the magic check.
    for &offset in zero_fields {
        let mut bad = good.clone();
        bad[offset..offset + 4].fill(0);
        assert!(
            decode(&mut Cursor::new(&bad), bad.len() as u64).is_err(),
            "zero field {offset}"
        );
    }
    for end in [0, 3, good.len() / 2, good.len() - 1] {
        assert!(
            decode(&mut Cursor::new(&good[..end]), end as u64).is_err(),
            "truncated at {end}"
        );
        assert!(
            decode(&mut Cursor::new(&good[..end]), good.len() as u64).is_err(),
            "reader shorter than declared length at {end}"
        );
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(decode(&mut Cursor::new(&trailing), trailing.len() as u64).is_err());
}

#[test]
fn scalar_headers_and_lengths_are_validated_for_every_codec() {
    macro_rules! scalar {
        ($spec:ty) => {
            check(
                ScalarQuantizedDataset::<$spec>::build(&base(), &()).unwrap(),
                ScalarQuantizedDataset::<$spec>::encode,
                ScalarQuantizedDataset::<$spec>::decode,
                &[0, 4, 8, 12, 16],
            );
        };
    }
    scalar!(L2U8);
    scalar!(L2U16);
    scalar!(MipsI8);
    scalar!(MipsI16);
}
#[test]
fn sparse_projection_headers_and_lengths_are_validated() {
    check(
        JLSparseDataset::<1024, 9>::build(&base(), &42).unwrap(),
        JLSparseDataset::<1024, 9>::encode,
        JLSparseDataset::<1024, 9>::decode,
        &[0, 4, 8, 12, 16, 20, 24],
    );
    check(
        JLSparseDatasetMips::<1024, 9>::build(&base(), &42).unwrap(),
        JLSparseDatasetMips::<1024, 9>::encode,
        JLSparseDatasetMips::<1024, 9>::decode,
        &[0, 4, 8, 12, 16, 20, 24],
    );
}
#[test]
fn hadamard_and_kernel_trick_headers_and_lengths_are_validated() {
    check(
        JlHadamardDataset::build(&base(), &JlHadamardDataset::default_build_options()).unwrap(),
        JlHadamardDataset::encode,
        JlHadamardDataset::decode,
        &[0, 4, 8, 12, 16, 20],
    );
    check(
        L2KTDataset::build(&base(), &()).unwrap(),
        L2KTDataset::encode,
        L2KTDataset::decode,
        &[0, 4, 8, 12, 16, 24],
    );
}
#[test]
fn rabitq_headers_lengths_and_corrections_are_validated() {
    use rabitq_dataset::RabitQDataset;
    check(
        RabitQDataset::build(&base(), &42).unwrap(),
        RabitQDataset::encode,
        RabitQDataset::decode,
        &[0, 4, 8, 12, 16, 20],
    );
    check(
        RabitQ4Dataset::build(&base(), &42).unwrap(),
        RabitQ4Dataset::encode,
        RabitQ4Dataset::decode,
        &[0, 4, 8, 12, 16],
    );
    let mut bytes = vec![];
    RabitQDataset::build(&base(), &42)
        .unwrap()
        .encode(&mut bytes)
        .unwrap();
    let correction_offset = 32 + 17 * 17 * 4 + 3 * 4;
    bytes[correction_offset..correction_offset + 4].fill(0);
    assert!(RabitQDataset::decode(&mut Cursor::new(&bytes), bytes.len() as u64).is_err());
    bytes.clear();
    RabitQ4Dataset::build(&base(), &42)
        .unwrap()
        .encode(&mut bytes)
        .unwrap();
    bytes[correction_offset..correction_offset + 4].fill(0);
    assert!(RabitQ4Dataset::decode(&mut Cursor::new(&bytes), bytes.len() as u64).is_err());
}

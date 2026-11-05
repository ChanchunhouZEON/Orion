/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use orion_ann_core::{Orion, PaddedDataset, PaddedLayout};
use std::{collections::BTreeMap, io::Cursor};

fn index() -> Orion {
    let values: Vec<_> = (0..4 * 17).map(|i| ((i * 13) % 47) as f32 - 12.).collect();
    let base = PaddedDataset::from_packed(
        &values,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(17).unwrap(),
    )
    .unwrap();
    let parts: Vec<_> = (0..4)
        .map(|i| ((0..4).filter(|j| *j != i).collect(), vec![], vec![]))
        .collect();
    Orion::from_graph_partitions(base, &parts, 0, 4, 0).unwrap()
}
fn encoded(index: &Orion) -> BTreeMap<String, Vec<u8>> {
    let mut records = BTreeMap::new();
    index
        .visit_sidecars(|name, encode| {
            let mut bytes = vec![];
            encode(&mut bytes)?;
            assert!(records.insert(name.to_owned(), bytes).is_none());
            Ok(())
        })
        .unwrap();
    records
}
#[test]
fn all_lazy_sidecars_restore_without_rebuilding_or_changing_encoded_bytes() {
    let source = index();
    assert!(encoded(&source).is_empty());
    source.try_ensure_quantized_dataset().unwrap();
    source.try_ensure_quantized_dataset_l2_u16().unwrap();
    source.try_ensure_quantized_dataset_mips().unwrap();
    source.try_ensure_quantized_dataset_mips_i16().unwrap();
    source.try_ensure_quantized_dataset_l2_kt().unwrap();
    source.try_ensure_quantized_dataset_jl().unwrap();
    source.try_ensure_quantized_dataset_jl_mips().unwrap();
    source.try_ensure_quantized_dataset_jl_hadamard().unwrap();
    source.try_ensure_quantized_dataset_rabitq().unwrap();
    source.try_ensure_quantized_dataset_rabitq_b4().unwrap();
    let records = encoded(&source);
    assert_eq!(records.len(), 10);
    let mut destination = index();
    for (name, bytes) in &records {
        destination
            .restore_sidecar(name, |decode| {
                decode(&mut Cursor::new(bytes), bytes.len() as u64)
            })
            .unwrap();
    }
    assert_eq!(encoded(&destination), records);
    for (name, bytes) in &records {
        assert!(
            destination
                .restore_sidecar(name, |decode| decode(
                    &mut Cursor::new(bytes),
                    bytes.len() as u64
                ))
                .is_err()
        );
    }
    assert!(
        destination
            .restore_sidecar("unknown-codec", |_| panic!(
                "unknown codec must not open input"
            ))
            .is_err()
    );
    assert_eq!(encoded(&destination), records);
    assert!(format!("{destination:?}").len() < 1500);
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![cfg(feature = "experimental-graph")]
use orion_ann_core::{Orion, PaddedDataset, PaddedLayout, supplier::*};
use orion_ann_platform::region::{Placement, RegionOptions};
use orion_ann_vector::L2F32Distance;
fn spec() -> PartitionSpec {
    PartitionSpec {
        origin: PartitionOrigin {
            name: "custom/graph".into(),
            revision: Some("r1".into()),
        },
        points: 4,
        dimension: 2,
        element: PartitionElement::F32,
        metric: PartitionMetric::SquaredL2,
        entry: 2,
        max_degree: 2,
        max_extra: 1,
        base_sha256: None,
    }
}
fn rows() -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    vec![
        (vec![2], vec![1], vec![3]),
        (vec![3], vec![0], vec![2]),
        (vec![0], vec![3], vec![1]),
        (vec![1], vec![2], vec![0]),
    ]
}
fn data() -> PaddedDataset<f32> {
    PaddedDataset::from_packed(
        &[0., 1., 2., 3., 4., 5., 6., 7.],
        PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap(),
    )
    .unwrap()
}
fn encode(spec: &PartitionSpec) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_staged(&mut bytes, spec, &mut SlicePartitions::new(&rows())).unwrap();
    bytes
}
#[test]
fn memory_and_staged_preserve_order_entry_and_native_bytes() {
    for placement in [Placement::Resident, Placement::Mapped] {
        let s = spec();
        let r = rows();
        let options = RegionOptions {
            placement,
            directory: None,
        };
        let memory =
            Orion::from_partition_source(data(), &s, &mut SlicePartitions::new(&r), &options)
                .unwrap();
        let bytes = encode(&s);
        let mut reader = StagedReader::new(&bytes[..], bytes.len() as u64).unwrap();
        assert_eq!(reader.spec(), &s);
        let staged = Orion::from_partition_source(data(), &s, &mut reader, &options).unwrap();
        assert_eq!(memory.graph().buffer_bytes(), staged.graph().buffer_bytes());
        assert_eq!(staged.graph().placement(), placement);
        for (i, (l, r, e)) in r.iter().enumerate() {
            assert_eq!(staged.graph().local_neighbors(i), l);
            assert_eq!(staged.graph().remote_neighbors(i), r);
            assert_eq!(staged.graph().extra_candidates(i), e);
        }
        // Entry lives in native metadata, not in the graph slab.
        let legacy =
            Orion::from_graph_partitions(data(), &r, s.entry, s.max_degree, s.max_extra as usize)
                .unwrap();
        let native = |index: &Orion| {
            let mut meta = Vec::new();
            let mut graph = std::io::BufWriter::new(Vec::new());
            index.encode_native_graph(&mut meta, &mut graph).unwrap();
            (meta, graph.into_inner().unwrap())
        };
        assert_eq!(native(&staged), native(&legacy));
    }
}
#[test]
fn rejects_duplicates_self_bounds_capacity_shape_and_row_count() {
    let s = spec();
    let mut cases = Vec::new();
    for id in [0, 2, 4, u32::MAX] {
        let mut r = rows();
        r[0].2 = vec![id];
        cases.push(r);
    }
    let mut over = rows();
    over[0].0.push(3);
    cases.push(over);
    let mut missing = rows();
    missing.pop();
    cases.push(missing);
    let mut extra = rows();
    extra.push((vec![], vec![], vec![]));
    cases.push(extra);
    for r in cases {
        assert!(
            Orion::from_partition_source(
                data(),
                &s,
                &mut SlicePartitions::new(&r),
                &Default::default()
            )
            .is_err()
        );
        assert!(write_staged(Vec::new(), &s, &mut SlicePartitions::new(&r)).is_err());
    }
    for field in 0..6 {
        let mut s = spec();
        match field {
            0 => s.entry = 4,
            1 => s.dimension = 3,
            2 => s.points = 0,
            3 => s.origin.name = " ".into(),
            4 => s.element = PartitionElement::U8,
            _ => s.base_sha256 = Some("bad".into()),
        };
        assert!(
            Orion::from_partition_source(
                data(),
                &s,
                &mut SlicePartitions::new(&rows()),
                &Default::default()
            )
            .is_err()
        );
    }
}
#[test]
fn reader_rejects_truncation_trailing_bytes_unbounded_metadata_and_counts() {
    let s = spec();
    let bytes = encode(&s);
    for len in 0..bytes.len() {
        let result = StagedReader::new(&bytes[..len], len as u64).and_then(|mut r| {
            Orion::from_partition_source(data(), &s, &mut r, &Default::default())
        });
        assert!(result.is_err(), "accepted prefix {len}");
    }
    let mut tail = bytes.clone();
    tail.push(0);
    let mut r = StagedReader::new(&tail[..], tail.len() as u64).unwrap();
    assert!(Orion::from_partition_source(data(), &s, &mut r, &Default::default()).is_err());
    let mut huge = bytes.clone();
    huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(StagedReader::new(&huge[..], huge.len() as u64).is_err());
    let offset = 12 + u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut huge = bytes;
    huge[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut r = StagedReader::new(&huge[..], huge.len() as u64).unwrap();
    assert!(r.next_partition().is_err());
}
#[test]
fn legacy_adapter_requires_matching_entry_and_preserves_partitions() {
    let s = spec();
    let mut bytes = Vec::new();
    for word in [0x53544147, 3, 4, 2, 1, 2] {
        bytes.extend_from_slice(&u32::to_le_bytes(word));
    }
    for (l, r, e) in rows() {
        for n in [l.len(), r.len(), e.len()] {
            bytes.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for id in l.into_iter().chain(r).chain(e) {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
    }
    let mut r = StagedReader::legacy_v3(&bytes[..], bytes.len() as u64, s.clone()).unwrap();
    let index = Orion::from_partition_source(data(), &s, &mut r, &Default::default()).unwrap();
    assert_eq!(index.graph().extra_candidates(0), &[3]);
    let mut wrong = s;
    wrong.entry = 0;
    assert!(StagedReader::legacy_v3(&bytes[..], bytes.len() as u64, wrong).is_err());
}

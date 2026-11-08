/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use super::super::PhasedGraph;
use orion_ann_platform::region::{Placement, RegionOptions};

#[test]
fn mapped_slab_build_import_decode_preserve_partitions_and_lifetime() {
    let dir = tempfile::tempdir().unwrap();
    let storage = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
    };
    let mut staged = Vec::new();
    for word in [0x53544147u32, 3, 4, 2, 1, 0] {
        staged.extend(word.to_le_bytes());
    }
    for node in 0..4 {
        for word in [1u32, 1, 1, (node + 1) % 4, (node + 2) % 4, (node + 3) % 4] {
            staged.extend(word.to_le_bytes());
        }
    }
    let owned = PhasedGraph::decode_staged(staged.as_slice(), staged.len() as u64, 4, 2).unwrap();
    let mapped = PhasedGraph::decode_staged_with_storage(
        staged.as_slice(),
        staged.len() as u64,
        4,
        2,
        &storage,
    )
    .unwrap();
    assert_eq!(mapped.placement(), Placement::Mapped);
    assert_eq!(mapped.buffer_bytes(), owned.buffer_bytes());
    let built = PhasedGraph::try_build_with_storage(
        4,
        2,
        1,
        |slab, stride| {
            assert_eq!(stride, owned.stride);
            slab.copy_from_slice(&owned.buffer);
            Ok(())
        },
        &storage,
    )
    .unwrap();
    assert_eq!(built.buffer_bytes(), mapped.buffer_bytes());
    let mut file = Vec::new();
    mapped
        .encode(&mut std::io::BufWriter::new(&mut file))
        .unwrap();
    let decoded =
        PhasedGraph::decode_with_storage(file.as_slice(), file.len() as u64, &storage).unwrap();
    file.fill(0);
    staged.clear();
    drop(mapped);
    drop(built);
    std::thread::scope(|s| {
        for _ in 0..4 {
            let g = &decoded;
            let expected = &owned;
            s.spawn(move || {
                assert_eq!(g.buffer_bytes(), expected.buffer_bytes());
                for n in 0..4 {
                    assert_eq!(g.neighbors(n), expected.neighbors(n));
                    assert_eq!(g.extra_candidates(n), expected.extra_candidates(n));
                }
            });
        }
    });
    assert!(
        PhasedGraph::try_build_with_storage(
            4,
            2,
            1,
            |slab, _| {
                slab[0] = 99;
                Ok(())
            },
            &storage
        )
        .is_err()
    );
    drop(decoded);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

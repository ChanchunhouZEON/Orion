/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Malformed input contracts. No benchmark data or network access is needed.
use super::support;
use crate::{ErrorCode, PhasedGraph};
use support::{Rng, TempDir};

#[test]
fn invalid_updates_do_not_mutate_or_poison_the_graph() {
    let mut graph = PhasedGraph::try_empty(4, 2, 1).unwrap();
    graph.set_partition(0, &[1], &[2], &[3]).unwrap();
    let before = graph.buffer_bytes().to_vec();
    let invalid = [
        (4, vec![1], vec![], vec![]),
        (usize::MAX, vec![], vec![], vec![]),
        (0, vec![1, 2, 3], vec![], vec![]),
        (0, vec![1, 2], vec![3], vec![]),
        (0, vec![], vec![], vec![1, 2]),
        (0, vec![u32::MAX], vec![], vec![]),
        (0, vec![], vec![4], vec![]),
        (0, vec![], vec![], vec![4]),
    ];
    for (node, local, remote, extra) in invalid {
        assert_eq!(
            graph
                .set_partition(node, &local, &remote, &extra)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidData
        );
        assert_eq!(graph.buffer_bytes(), before);
    }
    graph.set_partition(0, &[3], &[], &[]).unwrap();
    assert_eq!(graph.neighbors(0), &[3]);
}

#[test]
fn constructors_reject_bad_shapes_counts_and_ids() {
    for (nodes, degree, extra) in [
        (0, 1, 0),
        (2, 0, 0),
        (u32::MAX as usize, 1, 0),
        (2, 1, usize::MAX),
        (2, u32::MAX, 1),
    ] {
        assert!(PhasedGraph::try_empty(nodes, degree, extra).is_err());
    }
    for parts in [
        vec![(vec![2], vec![], vec![]), (vec![], vec![], vec![])],
        vec![(vec![0, 0], vec![], vec![])],
        vec![(vec![], vec![], vec![0, 0])],
    ] {
        assert!(PhasedGraph::try_from_partitions(&parts, 1, 1).is_err());
    }
    for (word, value) in [(0, u32::MAX), (1, u32::MAX), (2, 2), (3, 1), (4, 2)] {
        assert!(
            PhasedGraph::try_build_with(2, 1, 1, |words, _| {
                words[0] = 1;
                words[word] = value;
                Ok(())
            })
            .is_err()
        );
    }
}

#[test]
fn native_cache_rejects_truncation_headers_and_malformed_rows() {
    let dir = TempDir::new();
    let path = dir.0.join("graph.pgraph");
    let graph = PhasedGraph::try_from_partitions(
        &[(vec![1], vec![], vec![]), (vec![0], vec![], vec![])],
        1,
        1,
    )
    .unwrap();
    graph.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    for len in [0, 1, 15, 16, bytes.len() - 1] {
        std::fs::write(&path, &bytes[..len]).unwrap();
        assert!(PhasedGraph::load(&path).is_err(), "truncation at {len}");
    }
    // Both header and record fields must be rejected before any unsafe search.
    for (word, value) in [
        (0, 0),
        (0, u32::MAX),
        (1, u32::MAX),
        (2, 0),
        (2, u32::MAX),
        (3, 2),
        (4, 2),
        (5, u32::MAX),
        (6, 2),
        (7, 1),
        (8, 2),
    ] {
        let mut bad = bytes.clone();
        bad[word * 4..word * 4 + 4].copy_from_slice(&value.to_ne_bytes());
        std::fs::write(&path, bad).unwrap();
        assert!(PhasedGraph::load(&path).is_err(), "word {word} = {value}");
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    std::fs::write(&path, trailing).unwrap();
    assert!(PhasedGraph::load(&path).is_err());
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(
        PhasedGraph::load(&path).unwrap().buffer_bytes(),
        graph.buffer_bytes()
    );
}

#[test]
fn staged_import_rejects_invalid_counts_ids_and_trailing_bytes() {
    let dir = TempDir::new();
    let path = dir.0.join("graph.staged");
    let valid = [0x53544147u32, 3, 2, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0];
    let write = |words: &[u32]| {
        std::fs::write(
            &path,
            words
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    write(&valid);
    assert!(PhasedGraph::load_staged(&path, 2, 1).is_ok());
    assert!(PhasedGraph::load_staged(&path, 3, 1).is_err());
    assert!(PhasedGraph::load_staged(&path, 2, 2).is_err());
    for (word, value) in [
        (0, 0),
        (1, 2),
        (2, u32::MAX),
        (3, u32::MAX),
        (4, u32::MAX),
        (5, 2),
        (6, 2),
        (7, u32::MAX),
        (9, 2),
    ] {
        let mut bad = valid;
        bad[word] = value;
        write(&bad);
        assert!(
            PhasedGraph::load_staged(&path, 2, 1).is_err(),
            "word {word}"
        );
    }
    write(&valid[..13]);
    assert!(PhasedGraph::load_staged(&path, 2, 1).is_err());
    let mut trailing = valid.to_vec();
    trailing.push(0);
    write(&trailing);
    assert!(PhasedGraph::load_staged(&path, 2, 1).is_err());
}

#[test]
fn mutated_callback_records_are_either_rejected_or_validated() {
    let mut rng = Rng(0x84f312ab);
    for case in 0..4096 {
        let graph = PhasedGraph::try_build_with(8, 4, 2, |words, stride| {
            let row = &mut words[rng.below(8) * stride..][..stride];
            for word in row.iter_mut() {
                *word = rng.next() as u32;
            }
            // Regularly generate almost-valid inputs to reach ID checks.
            if case % 2 == 0 {
                row[..4].copy_from_slice(&[4, 2, 2, 0]);
            }
            Ok(())
        });
        assert!(graph.is_err(), "mutation {case} unexpectedly valid");
    }
}

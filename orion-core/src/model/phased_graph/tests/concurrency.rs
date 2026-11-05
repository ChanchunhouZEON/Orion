/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Scoped readers are joined before exclusive updates; no shared-reference writes.
use crate::{ErrorCode, PhasedGraph};

#[test]
fn readers_finish_before_exclusive_mutation_and_keep_original_layout() {
    let mut graph = PhasedGraph::try_empty(4, 2, 1).unwrap();
    graph.set_partition(0, &[1], &[2], &[3]).unwrap();
    let address = graph.buffer_bytes().as_ptr().addr();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let graph = &graph;
            scope.spawn(move || {
                for _ in 0..32 {
                    assert_eq!(graph.local_neighbors(0), &[1]);
                    assert_eq!(graph.remote_neighbors(0), &[2]);
                    assert_eq!(graph.extra_candidates(0), &[3]);
                }
            });
        }
    });
    graph.set_partition(0, &[3], &[1], &[2]).unwrap();
    assert_eq!(graph.local_neighbors(0), &[3]);
    assert_eq!(graph.remote_neighbors(0), &[1]);
    assert_eq!(graph.buffer_bytes().as_ptr().addr(), address);
}

#[test]
fn cloned_graph_and_rejected_writes_preserve_the_original() {
    let mut graph = PhasedGraph::try_empty(4, 2, 1).unwrap();
    graph.set_partition(0, &[1], &[], &[2]).unwrap();
    let copy = graph.clone();
    assert_eq!(
        graph.set_partition(0, &[4], &[], &[]).unwrap_err().code(),
        ErrorCode::InvalidData
    );
    assert_eq!(graph.buffer_bytes(), copy.buffer_bytes());
    graph.set_partition(0, &[3], &[], &[]).unwrap();
    assert_eq!(copy.neighbors(0), &[1]);
}

#[test]
fn exclusive_build_callback_can_split_rows_between_workers() {
    let graph = PhasedGraph::try_build_with(4, 1, 0, |words, stride| {
        std::thread::scope(|scope| {
            for (id, row) in words.chunks_exact_mut(stride).enumerate() {
                scope.spawn(move || {
                    row[..5].copy_from_slice(&[1, 0, 1, 0, ((id + 1) % 4) as u32]);
                });
            }
        });
        Ok(())
    })
    .unwrap();
    for id in 0..4 {
        assert_eq!(graph.neighbors(id), &[((id + 1) % 4) as u32]);
    }
}

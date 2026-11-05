/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Public graph imports validate topology before exposing an immutable view.
use orion_ann_core::{ErrorCode, Orion, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;

fn dataset() -> PaddedDataset<f32> {
    PaddedDataset::from_packed(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap(),
    )
    .unwrap()
}

#[test]
fn read_only_view_preserves_navigation_and_reranking_partitions() {
    let parts = [
        (vec![1], vec![2], vec![0]),
        (vec![0], vec![], vec![2]),
        (vec![], vec![1], vec![]),
    ];
    let index = Orion::from_graph_partitions(dataset(), &parts, 0, 2, 1).unwrap();
    let graph = index.graph();
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(graph.neighbors(0), &[1, 2]);
    assert_eq!(graph.local_neighbors(0), &[1]);
    assert_eq!(graph.remote_neighbors(0), &[2]);
    assert_eq!(graph.extra_candidates(0), &[0]);
    assert!(graph.contains_edge(0, 1));
    assert!(!graph.contains_edge(0, 0));
    assert_eq!(
        graph.buffer_bytes().as_ptr(),
        index.graph().buffer_bytes().as_ptr()
    );
}

#[test]
fn public_import_rejects_mismatched_shapes_entries_and_edges() {
    let parts = vec![(vec![], vec![], vec![]); 3];
    for (rows, entry) in [(0, 0), (2, 0), (3, 3), (3, u32::MAX)] {
        let error =
            Orion::from_graph_partitions(dataset(), &parts[..rows], entry, 2, 1).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
    }
    let mut invalid = parts;
    invalid[0].0.push(3);
    assert_eq!(
        Orion::from_graph_partitions(dataset(), &invalid, 0, 2, 1)
            .unwrap_err()
            .code(),
        ErrorCode::InvalidData,
    );
}

#[test]
#[should_panic(expected = "graph node ID out of range")]
fn public_inspection_rejects_ids_before_offset_arithmetic() {
    let index =
        Orion::from_graph_partitions(dataset(), &vec![(vec![], vec![], vec![]); 3], 0, 2, 1)
            .unwrap();
    index.graph().neighbors(usize::MAX);
}

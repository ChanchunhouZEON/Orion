/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
mod support;
use orion_ann_diskann::model::InMemoryGraph;
use support::TestDir;

#[test]
fn graph_conversions_preserve_directed_edges_and_vectors() {
    let dir = TestDir::new();
    let edges = [(0, vec![1, 2]), (1, vec![0]), (2, vec![]), (99, vec![0])]
        .into_iter()
        .collect();
    let graph = InMemoryGraph::from_hashmap(&edges, 3, 4);
    assert!(graph.contains_edge(0, 1).unwrap());
    assert!(!graph.contains_edge(2, 0).unwrap());
    assert_eq!(graph.to_neighbor_vec(0).unwrap(), [1, 2]);
    let expected = graph.to_hashmap();
    let path = dir.file("graph");
    graph.save_graph_mmap(&path).unwrap();
    let restored = InMemoryGraph::load_graph_mmap(&path).unwrap();
    assert_eq!(restored.max_degree(), 4);
    assert_eq!(restored.to_hashmap(), expected);
    let values = [1., 2., 3., 4., 5., 6.];
    graph
        .save_graph_mmap_with_data(&path, &values, 2, 0)
        .unwrap();
    let mapped = orion_ann_platform::MmapGraph::open(&path, 0).unwrap();
    for i in 0..3 {
        assert_eq!(mapped.vector_f32(i), &values[i as usize * 2..][..2]);
    }
    assert!(InMemoryGraph::load_graph_mmap(&dir.file("missing")).is_err());
    assert!(graph.save_graph_mmap(&dir.file("missing/output")).is_err());
}

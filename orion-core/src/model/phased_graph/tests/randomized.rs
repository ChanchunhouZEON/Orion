/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Explicit stress suite: `cargo test -p orion-ann --release --lib model::phased_graph::tests::randomized -- --ignored`.
//! Fixed seeds make failures reproducible. CI runs this independently of unit tests.
use super::support;
use crate::{ErrorCode, PhasedGraph};
use support::{Partition, Rng, TempDir, check_node};

#[test]
#[ignore = "large deterministic model test; run by graph-safety CI"]
fn million_updates_match_an_independent_partition_model() {
    const STEPS: usize = 16_384;
    for seed in 0..64u64 {
        eprintln!("graph randomized seed={seed}, steps={STEPS}");
        let mut rng = Rng(seed ^ 0x713fbb4a);
        let nodes = 8 + rng.below(121);
        let degree = 1 + rng.below(32);
        let extras = rng.below(17);
        let mut model: Vec<Partition> = (0..nodes)
            .map(|_| rng.partition(nodes, degree, extras))
            .collect();
        let mut graph = PhasedGraph::try_from_partitions(&model, degree as u32, extras).unwrap();
        for step in 0..STEPS {
            let node = rng.below(nodes);
            let update = rng.partition(nodes, degree, extras);
            graph
                .set_partition(node, &update.0, &update.1, &update.2)
                .unwrap();
            model[node] = update;
            assert_eq!(
                graph
                    .set_partition(node, &[nodes as u32], &[], &[])
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidData
            );
            check_node(&graph, node, &model[node]);
            if step % 127 == 0 {
                let clone = graph.clone();
                for (node, expected) in model.iter().enumerate() {
                    check_node(&clone, node, expected);
                }
            }
        }
        let dir = TempDir::new();
        let path = dir.0.join("graph.pgraph");
        graph.save(&path).unwrap();
        let restored = PhasedGraph::load(&path).unwrap();
        for (node, expected) in model.iter().enumerate() {
            check_node(&restored, node, expected);
        }
    }
}

#[test]
#[ignore = "large deterministic read/publication stress; dedicated CI"]
fn readers_observe_complete_versions_across_exclusive_publications() {
    let mut graph = PhasedGraph::try_empty(128, 4, 4).unwrap();
    for generation in 0..128u32 {
        graph
            .set_partition(0, &[generation; 4], &[], &[generation; 4])
            .unwrap();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let graph = &graph;
                scope.spawn(move || {
                    for _ in 0..2048 {
                        assert!(
                            graph
                                .local_neighbors(0)
                                .iter()
                                .chain(graph.extra_candidates(0))
                                .all(|&v| v == generation)
                        );
                    }
                });
            }
        });
    }
}

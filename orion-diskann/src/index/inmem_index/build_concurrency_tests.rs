/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Force reciprocal updates to meet at the lock boundary, without timing races.
//! The hook exists only in unit-test builds and is local to each test thread.
use super::InmemIndex;
use crate::model::configuration::index_write_parameters::IndexWriteParametersBuilder;
use crate::model::{InMemQueryScratch, InMemoryGraph, IndexConfiguration};
use orion_ann_vector::Metric;
use std::cell::RefCell;
use std::sync::{Arc, Barrier};

type ReverseInsertHook = Box<dyn Fn(&InMemoryGraph, u32)>;

thread_local! {
    static BEFORE_REVERSE_INSERT: RefCell<Option<ReverseInsertHook>> = const { RefCell::new(None) };
}

pub(super) fn before_reverse_insert(graph: &InMemoryGraph, source: u32) {
    BEFORE_REVERSE_INSERT.with_borrow(|hook| {
        if let Some(hook) = hook {
            hook(graph, source);
        }
    });
}

#[test]
fn reciprocal_updates_release_source_locks_before_reverse_edges() {
    let params = IndexWriteParametersBuilder::new(8, 4)
        .with_num_threads(2)
        .build();
    let config = IndexConfiguration::new(Metric::L2, 32, 32, 4, false, 0, false, 0, 1.0, params);
    let mut index = InmemIndex::<f32, 32>::new(config).unwrap();
    index.num_active_pts = 4;
    for node in 0..4 {
        index.dataset.data[node * 32..(node + 1) * 32].fill(node as f32);
    }
    // 0 <-> 1 creates the old lock cycle. The other outgoing edges verify
    // actual insertion (2 -> 0 and 3 -> 1), not just duplicate-edge handling.
    for (source, neighbors) in [(0, vec![1, 2]), (1, vec![0, 3])] {
        let mut node = index
            .final_graph
            .write_vertex_and_neighbors(source)
            .unwrap();
        #[cfg(feature = "orion")]
        {
            let distances = neighbors
                .iter()
                .map(|&target| index.get_distance(source, target).unwrap())
                .collect();
            node.set_neighbors_sorted(neighbors.into(), distances);
        }
        #[cfg(not(feature = "orion"))]
        node.set_neighbors(neighbors.into());
    }

    let before_probe = Arc::new(Barrier::new(2));
    let after_probe = Arc::new(Barrier::new(2));
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for source in 0..2 {
            let index = &index;
            let before_probe = before_probe.clone();
            let after_probe = after_probe.clone();
            workers.push(scope.spawn(move || {
                let mut scratch =
                    InMemQueryScratch::new(8, &index.configuration.index_write_parameter, false)
                        .unwrap();
                BEFORE_REVERSE_INSERT.with_borrow_mut(|slot| {
                    *slot = Some(Box::new(move |graph, source| {
                        before_probe.wait();
                        // Both threads have taken their snapshots. Neither can
                        // start reverse insertion until both probes finish, so
                        // a failure means our own source guard is still alive.
                        // Drop this temporary write guard before rendezvousing.
                        let released = graph.final_graph[source as usize].try_write().is_ok();
                        after_probe.wait();
                        assert!(
                            released,
                            "source {source} read lock spans reverse insertion"
                        );
                    }));
                });
                index
                    .update_neighbors_of_vertex(source, &mut scratch)
                    .unwrap();
                BEFORE_REVERSE_INSERT.with_borrow_mut(|slot| *slot = None);
            }));
        }
        // Join both even if the old lock discipline makes one probe panic.
        let outcomes: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
        assert!(outcomes.into_iter().all(|outcome| outcome.is_ok()));
    });

    for (node, expected) in [(0, vec![1, 2]), (1, vec![0, 3]), (2, vec![0]), (3, vec![1])] {
        assert_eq!(index.final_graph.to_neighbor_vec(node).unwrap(), expected);
    }
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(feature = "benchmark")]
use orion_ann_core::{
    NaiveOrion,
    model::Neighbor,
    model::scratch::InMemSearchScratch,
    search::diagnostics::{MatchedStateTrace, SearchObserver},
};
use orion_ann_diskann::model::{CsrGraph, InmemDataset};

#[test]
fn naive_prefix_baseline_preserves_order_and_matches_exhaustive_search() {
    let count = 12;
    let mut dataset = InmemDataset::<f32, 32>::new(count, 1.).unwrap();
    let rows: Vec<[f32; 32]> = (0..count)
        .map(|i| std::array::from_fn(|d| ((i * 17 + d * 3 + i * d) % 97) as f32))
        .collect();
    for (i, row) in rows.iter().enumerate() {
        dataset.data[i * 32..][..32].copy_from_slice(row);
    }
    let edges: Vec<Vec<u32>> = (0..count as u32)
        .map(|i| (0..count as u32).filter(|j| *j != i).collect())
        .collect();
    let index = NaiveOrion::new(
        dataset,
        CsrGraph::from_adjacency_list(edges.clone(), 12),
        0,
        2,
        3,
    );
    for (i, row) in edges.iter().enumerate() {
        assert_eq!(index.graph.neighbors(i), row);
        assert_eq!(index.graph.compressed_neighbors(i), &row[..6]);
    }
    let expected: Vec<Vec<u32>> = rows
        .iter()
        .map(|query| {
            let mut ids: Vec<_> = (0..count as u32).collect();
            ids.sort_by(|a, b| {
                let distance = |i: u32| {
                    query
                        .iter()
                        .zip(&rows[i as usize])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                };
                distance(*a).total_cmp(&distance(*b))
            });
            ids.truncate(4);
            ids
        })
        .collect();
    for epsilon in [0., 1.] {
        // The first full expansion sees every candidate. Subsequent convergence
        // changes neighbor selection but cannot change the exact top four.
        assert_eq!(
            index.search_batch(&rows, 4, 16, 2, epsilon).unwrap(),
            expected
        );
        for (i, query) in rows.iter().enumerate() {
            assert_eq!(index.search(query, 4, 24, 2, epsilon).unwrap(), expected[i]);
        }
    }
    assert!(index.search_batch(&[], 4, 16, 2, 0.).unwrap().is_empty());
    assert!(format!("{index:?}").contains("max_pruned_degree: 6"));
}

#[test]
fn matched_state_trace_keeps_first_checkpoint_across_reversals() {
    let scratch = InMemSearchScratch::new(8);
    let mut trace = MatchedStateTrace::new(&[0, 1, 2]);
    trace.entry(0);
    trace.before_expansion(0, false, &scratch, 1, 0);
    assert!(trace.checkpoint.is_none());
    trace.expansion(false, &[1]);
    trace.prefilter(2, 1);
    trace.admission(1, &[Neighbor::new(1, 4.)]);
    trace.after_flush(&[Neighbor::new(1, 4.)]);
    assert!(trace.retained_after_switch.is_empty());
    trace.before_expansion(1, true, &scratch, 0, 2);
    trace.expansion(true, &[2]);
    trace.prefilter(3, 1);
    trace.admission(1, &[Neighbor::new(2, 9.)]);
    trace.after_flush(&[Neighbor::new(1, 4.), Neighbor::new(2, 9.)]);
    trace.before_expansion(2, false, &scratch, 1, 0);
    trace.expansion(false, &[]);
    trace.rerank(2);
    trace.early_exit();
    let checkpoint = trace.checkpoint.as_ref().unwrap();
    assert_eq!(
        (
            checkpoint.node,
            checkpoint.step,
            checkpoint.previous_admits,
            checkpoint.hops_since_flush
        ),
        (1, 2, 0, 2)
    );
    assert!(checkpoint.beam.is_empty());
    assert!(checkpoint.seen.is_empty());
    assert_eq!(trace.tail_nodes, [(1, true), (2, false)]);
    assert_eq!(
        trace.admitted_after_switch.into_iter().collect::<Vec<_>>(),
        [2]
    );
    assert_eq!(
        trace.retained_after_switch.into_iter().collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(trace.work.before.ndc(), 5);
    assert_eq!(trace.work.after.ndc(), 5);
    assert_eq!(trace.work.first_converged_step, Some(2));
    assert_eq!(trace.work.rerank_candidates, 2);
    assert!(trace.work.early_exited);
    assert_eq!(trace.discovery.first_discovery, [Some(0), Some(1), Some(2)]);
    assert_eq!(trace.discovery.first_admission, [Some(0), Some(1), Some(2)]);
    assert_eq!(trace.discovery.reversals, 1);
}

#[test]
fn benchmark_compatibility_search_matches_validated_cascade() {
    use orion_ann_core::search::cascade::stage::{
        Prefilter, admission::L2U8Admission, rerank::F32Rerank,
    };
    use orion_ann_core::{DynamicCascade, DynamicSearchConfig, StaticOrion};
    let mut base = InmemDataset::<f32, 32>::new(12, 1.).unwrap();
    for i in 0..12 {
        base.data[i * 32..][..32].fill(i as f32);
    }
    let parts: Vec<_> = (0..12)
        .map(|i| ((0..12).filter(|j| *j != i).collect(), vec![], vec![]))
        .collect();
    let index = StaticOrion::from_graph_partitions(base, &parts, 0, 12, 0).unwrap();
    let admission = L2U8Admission::new(index.ensure_quantized_dataset());
    let rerank = F32Rerank::new(index.dataset());
    let query = [5.1; 32];
    let config = DynamicSearchConfig {
        k: 3,
        search_list_size: 16,
        window_size: 2,
        epsilon: 0.,
        early_exit_limit: usize::MAX,
    };
    let expected = index
        .search_observed(
            orion_ann_core::search::query::QueryInput::F32(&query),
            config,
            DynamicCascade {
                prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                admission: &admission,
                rerank: &rerank,
            },
            orion_ann_core::search::diagnostics::NeighborMode::default(),
            &mut orion_ann_core::search::diagnostics::NoopObserver,
        )
        .unwrap();
    assert_eq!(expected, [5, 6, 4]);
    assert_eq!(
        index
            .search_unified(
                &query,
                3,
                16,
                2,
                0.,
                usize::MAX,
                None::<&Prefilter>,
                &admission,
                &rerank
            )
            .unwrap(),
        expected
    );
    assert_eq!(
        index
            .search_batch_unified(
                &[query, query],
                3,
                16,
                2,
                0.,
                usize::MAX,
                None::<&Prefilter>,
                &admission,
                &rerank
            )
            .unwrap(),
        [expected.clone(), expected]
    );
}

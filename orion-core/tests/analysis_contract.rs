/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![cfg(feature = "benchmark")]
use orion_ann_core::algorithm::analysis::*;

#[test]
fn cliff_rank_summary_reports_a_known_top_k_boundary() {
    let stats = vec![NodeCliffStats {
        node: 0,
        degree: 3,
        sorted_neighbors: vec![
            SortedNeighbor {
                id: 1,
                distance: 1.,
                bf_rank: Some(0),
            },
            SortedNeighbor {
                id: 2,
                distance: 4.,
                bf_rank: Some(1),
            },
            SortedNeighbor {
                id: 3,
                distance: 100.,
                bf_rank: Some(9),
            },
        ],
        gap_ratios: vec![4., 25.],
        cliff_pos: 1,
        cliff_ratio: 25.,
        min_distance: 1.,
        max_distance: 100.,
    }];
    let summary = summarize_cliff_ranks(&stats, &[1, 2, 10]);
    assert_eq!(summary.num_sampled, 1);
    assert_eq!(summary.avg_rank_before_cliff, 0.5);
    assert_eq!(summary.avg_rank_after_cliff, 9.);
    assert_eq!(summary.cliff_separates_topk, [(1, 0.), (2, 1.), (10, 0.)]);
    assert_eq!(summary.rank_at_cliff_percentiles, [1.; 5]);
    assert_eq!(summary.rank_after_cliff_percentiles, [9.; 5]);
    let formatted = summary.to_string();
    assert!(formatted.contains("1 sampled nodes"));
    assert!(formatted.contains("100.0% of nodes"));
    assert!(!formatted.contains("NaN"));
    let empty = summarize_cliff_ranks(&[], &[10]);
    assert_eq!(empty.num_sampled, 0);
    assert_eq!(empty.cliff_separates_topk, [(10, 0.)]);
    assert!(!empty.to_string().contains("NaN"));
}

#[test]
fn rank_precision_aggregation_handles_different_degrees_and_empty_samples() {
    let profiles = compute_rank_precision_profiles(
        &[vec![1, 9, 2], vec![3]],
        &[0, 1],
        &[vec![1, 2, 3], vec![3]],
    );
    assert_eq!(profiles[0].precision, [1., 0.5, 2. / 3.]);
    assert_eq!(profiles[0].cliff_pos, 0);
    assert_eq!(profiles[0].cliff_drop, 0.5);
    let summary = summarize_rank_precision(&profiles);
    assert_eq!(summary.num_sampled, 2);
    assert_eq!(summary.avg_precision_by_pos, [1., 0.5, 2. / 3.]);
    assert_eq!(summary.cliff_pos_histogram, [2]);
    assert!(summary.to_string().contains("2 nodes"));
    assert!(!summary.to_string().contains("NaN"));
    let empty = summarize_rank_precision(&[]);
    assert!(empty.avg_precision_by_pos.is_empty());
    assert_eq!(empty.mean_cliff_pos, 0.);
}

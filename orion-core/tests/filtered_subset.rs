/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Validate the core exact-subset boundary independently of runtime label checks.
use orion_ann_core::{BuildBackend, BuildConfig, ErrorCode, Orion, ScoreMetric};

fn index() -> Orion<f32> {
    Orion::<f32>::build_from_slice(
        &[
            0., 1., 0., -1., 1., 0., -1., 0., 0., 1., 0., -1., 1., 0., -1., 0.,
        ],
        2,
        BuildConfig {
            backend: BuildBackend::DiskAnn,
            degree: 4,
            build_l: 8,
            threads: 1,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn exact_subset_rejects_invalid_inputs_without_calibration() {
    let index = index();
    for (query, ids, k, metric) in [
        (&[1.][..], &[0][..], 1, ScoreMetric::SquaredL2),
        (&[f32::NAN, 1.], &[0], 1, ScoreMetric::SquaredL2),
        (&[f32::INFINITY, 1.], &[0], 1, ScoreMetric::SquaredL2),
        (&[1., 1.], &[0], 0, ScoreMetric::SquaredL2),
        (&[1., 1.], &[0], 9, ScoreMetric::SquaredL2),
        (&[1., 1.], &[8], 1, ScoreMetric::SquaredL2),
        (&[1., 1.], &[0, 0], 1, ScoreMetric::SquaredL2),
        (&[1., 1.], &[2, 1], 1, ScoreMetric::SquaredL2),
        (&[0., 0.], &[0], 1, ScoreMetric::CosineSimilarity),
        (&[f32::MAX, f32::MAX], &[0], 1, ScoreMetric::SquaredL2),
    ] {
        for scores in [false, true] {
            assert_eq!(
                index
                    .search_subset_exact(query, ids, k, metric, scores)
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidArgument,
            );
        }
    }
}

#[test]
fn exact_subset_ranking_ties_partial_results_and_empty_batches() {
    let index = index();
    let all = [0, 1, 2, 3, 4, 5, 6, 7];
    let tied = index
        .search_subset_exact(&[0., 0.], &all, 3, ScoreMetric::SquaredL2, true)
        .unwrap();
    assert_eq!(tied.ids, [0, 1, 2]);
    assert_eq!(tied.scores.unwrap().values, [1., 1., 1.]);
    for metric in [ScoreMetric::InnerProduct, ScoreMetric::CosineSimilarity] {
        let result = index
            .search_subset_exact(&[1., 0.], &all, 8, metric, true)
            .unwrap();
        assert_eq!(result.ids, [2, 6, 0, 1, 4, 5, 3, 7]);
        assert_eq!(
            result.scores.unwrap().values,
            [1., 1., 0., 0., 0., 0., -1., -1.]
        );
    }
    let partial = index
        .search_subset_exact(&[0., 0.], &[3], 3, ScoreMetric::SquaredL2, true)
        .unwrap();
    assert_eq!(partial.counts, [1]);
    assert_eq!(partial.ids[0], 3);
    assert!(
        partial.scores.unwrap().values[1..]
            .iter()
            .all(|s| s.is_nan())
    );
    let ids_only = index
        .search_subset_exact(&[0., 0.], &[3], 3, ScoreMetric::SquaredL2, false)
        .unwrap();
    assert_eq!(ids_only.ids, partial.ids);
    assert!(ids_only.scores.is_none());
    let empty = index
        .search_subset_exact(&[1., 0.], &[], 3, ScoreMetric::SquaredL2, true)
        .unwrap();
    assert_eq!(empty.counts, [0]);
    let no_queries = index
        .search_subset_exact(&[], &all, 3, ScoreMetric::SquaredL2, false)
        .unwrap();
    assert!(no_queries.ids.is_empty());
    assert!(no_queries.counts.is_empty());
}

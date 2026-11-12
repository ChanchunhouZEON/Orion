/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Exact scoring of selected results when no matching rerank score is available.
use crate::{Result, ScoreMetric, SearchResult};

/// Used for final candidate scores and the exact subset baseline. Accumulate in f64 to
/// avoid intermediate overflow; reject unrepresentable f32 outputs at assembly.
pub(crate) fn exact_score<T: Copy + Into<f32>>(
    query: &[f32],
    row: &[T],
    metric: ScoreMetric,
) -> f32 {
    crate::search::query::numeric::score(query, row, metric)
}

/// Internal per-query output; omitted scores do not allocate a score vector.
pub(super) type SearchRow = (Vec<u32>, Option<Vec<f32>>);

/// Assemble only complete batches. All row cardinality, ID and score checks
/// remain in SearchResult; neither this step nor scoring changes selection.
pub(super) fn assemble(
    rows: Vec<SearchRow>,
    k: usize,
    points: usize,
    scores: Option<ScoreMetric>,
) -> Result<SearchResult> {
    match scores {
        Some(metric) => SearchResult::from_scored_rows(
            rows.into_iter()
                .map(|(ids, scores)| {
                    (
                        ids,
                        scores.expect("scored plan must produce one score per ID"),
                    )
                })
                .collect(),
            k,
            points,
            metric,
        ),
        None => SearchResult::from_rows(rows.into_iter().map(|(ids, _)| ids).collect(), k, points),
    }
}

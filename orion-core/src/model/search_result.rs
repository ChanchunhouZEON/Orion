/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Owned batch results shared by core, Rust services and language adapters.
use crate::{Error, Result};

/// Neighbors in query order. Each row occupies `k` IDs; only its `counts`
/// prefix is valid. Remaining entries contain `u32::MAX`. Scores are not
/// computed unless explicitly requested. Scoring never changes the selected IDs.
#[derive(Clone, PartialEq)]
pub struct SearchResult {
    /// Packed row-major neighbor IDs, including padding.
    pub ids: Vec<u32>,
    /// Number of valid neighbors for each query.
    pub counts: Vec<u32>,
    /// Requested maximum number of neighbors per query.
    pub k: usize,
    /// Optional row-major scores. Only the `counts` prefix of each row is valid.
    pub scores: Option<SearchScores>,
}

/// Meaning of scores computed against the full-precision resident base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreMetric {
    /// Sum of squared coordinate differences; smaller is better.
    SquaredL2,
    /// Unnegated dot product; larger is better.
    InnerProduct,
    /// Dot product divided by both vector norms; larger is better.
    CosineSimilarity,
}

impl ScoreMetric {
    /// Stable label shared by the CLI and Python adapters.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SquaredL2 => "squared_l2",
            Self::InnerProduct => "inner_product",
            Self::CosineSimilarity => "cosine_similarity",
        }
    }
}

/// Scores of the returned candidates, not a guarantee of exact global top-k.
/// Padding is NaN and must be ignored using the corresponding result counts.
#[derive(Clone, PartialEq)]
pub struct SearchScores {
    pub metric: ScoreMetric,
    pub values: Vec<f32>,
}

impl std::fmt::Debug for SearchScores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchScores")
            .field("metric", &self.metric)
            .field("value_count", &self.values.len())
            .finish_non_exhaustive()
    }
}

impl SearchResult {
    pub(crate) fn from_rows(rows: Vec<Vec<u32>>, k: usize, points: usize) -> Result<Self> {
        if k == 0 {
            return Err(Error::invalid_argument("result width must be nonzero"));
        }
        let capacity = rows
            .len()
            .checked_mul(k)
            .ok_or_else(|| Error::invalid_argument("result size overflow"))?;
        let mut ids = vec![u32::MAX; capacity];
        let mut counts = Vec::with_capacity(rows.len());
        for (out, row) in ids.chunks_exact_mut(k).zip(rows) {
            if row.len() > k || row.iter().any(|&id| id as usize >= points) {
                return Err(Error::internal("engine returned invalid result IDs").at("search"));
            }
            out[..row.len()].copy_from_slice(&row);
            counts.push(row.len() as u32);
        }
        Ok(Self {
            ids,
            counts,
            k,
            scores: None,
        })
    }

    pub(crate) fn from_scored_rows(
        rows: Vec<(Vec<u32>, Vec<f32>)>,
        k: usize,
        points: usize,
        metric: ScoreMetric,
    ) -> Result<Self> {
        if k == 0 || rows.len().checked_mul(k).is_none() {
            return Err(Error::invalid_argument("invalid scored result size"));
        }
        let mut scores = Vec::new();
        let mut ids = Vec::with_capacity(rows.len());
        for (row, values) in rows {
            if row.len() != values.len() || values.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid_argument(
                    "exact scores overflowed or do not match result IDs",
                )
                .at("search"));
            }
            let valid = values.len();
            if valid > k {
                return Err(Error::internal("too many scores"));
            }
            scores.extend(values);
            scores.resize(scores.len() + k - valid, f32::NAN);
            ids.push(row);
        }
        let mut result = Self::from_rows(ids, k, points)?;
        result.scores = Some(SearchScores {
            metric,
            values: scores,
        });
        Ok(result)
    }

    /// Number of query rows, including rows without neighbors.
    pub fn len(&self) -> usize {
        self.counts.len()
    }
    /// Whether the input batch contained no queries.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }
    /// Valid neighbor prefixes, excluding padding and preserving query order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[u32]> {
        self.ids
            .chunks(self.k.max(1))
            .zip(&self.counts)
            .map(|(row, &count)| &row[..count as usize])
    }
}

impl std::fmt::Debug for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchResult")
            .field("query_count", &self.counts.len())
            .field("id_count", &self.ids.len())
            .field("k", &self.k)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scored_rows_preserve_partial_prefixes_and_reject_bad_scores() {
        let result = SearchResult::from_scored_rows(
            vec![(vec![1], vec![3.0]), (vec![], vec![])],
            2,
            3,
            ScoreMetric::SquaredL2,
        )
        .unwrap();
        assert_eq!(result.counts, [1, 0]);
        let scores = result.scores.unwrap();
        assert_eq!(scores.values[0], 3.0);
        assert!(scores.values[1..].iter().all(|v| v.is_nan()));
        for rows in [
            vec![(vec![1], vec![])],
            vec![(vec![1], vec![f32::INFINITY])],
            vec![(vec![3], vec![0.0])],
        ] {
            assert!(SearchResult::from_scored_rows(rows, 2, 3, ScoreMetric::SquaredL2).is_err());
        }
    }

    #[test]
    fn valid_prefixes_preserve_partial_empty_and_full_rows() {
        let result = SearchResult::from_rows(vec![vec![3], vec![], vec![2, 1]], 2, 4).unwrap();
        assert_eq!(result.ids, [3, u32::MAX, u32::MAX, u32::MAX, 2, 1]);
        assert_eq!(result.counts, [1, 0, 2]);
        assert_eq!(
            result.rows().collect::<Vec<_>>(),
            [vec![3], vec![], vec![2, 1]]
        );
        assert!(SearchResult::from_rows(vec![], 2, 4).unwrap().is_empty());
        assert!(SearchResult::from_rows(vec![vec![4]], 2, 4).is_err());
        assert!(SearchResult::from_rows(vec![vec![0, 1, 2]], 2, 4).is_err());
        assert!(SearchResult::from_rows(vec![], 0, 4).is_err());
        assert!(SearchResult::from_rows(vec![vec![], vec![]], usize::MAX, 4).is_err());
    }
}

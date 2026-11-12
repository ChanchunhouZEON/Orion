/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Exact subset baseline for filtered search. Does not change ANN traversal.
use super::{CompressedIndex, DynamicElement};
use crate::{Error, Result, ScoreMetric, SearchResult};
use orion_ann_diskann::model::PaddedDataset;
use rayon::prelude::*;
use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Clone, Copy)]
struct Candidate {
    rank: f32,
    score: f32,
    id: u32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank
            .total_cmp(&other.rank)
            .then(self.id.cmp(&other.id))
    }
}
impl<T: DynamicElement + Into<f32>> CompressedIndex<PaddedDataset<T>> {
    /// Exhaustively rank sorted, unique internal IDs using the original metric.
    /// No graph or quantized admission is involved. Ties use ascending row ID.
    /// Memory is O(k) per query worker; result rows may contain fewer than k IDs.
    ///
    /// # Errors
    /// Rejects invalid k, query shape/coordinates, zero-norm cosine queries,
    /// unordered/duplicate/out-of-range IDs and non-finite scores. This method
    /// does not require adaptive calibration. The caller owns candidate budgets.
    pub fn search_subset_exact(
        &self,
        queries: &[f32],
        ids: &[u32],
        k: usize,
        metric: ScoreMetric,
        scores: bool,
    ) -> Result<SearchResult> {
        let dataset = self.dataset();
        let dim = dataset.layout().logical_dim();
        if k == 0
            || k > dataset.len()
            || !queries.len().is_multiple_of(dim)
            || queries.iter().any(|x| !x.is_finite())
            || ids.iter().any(|&id| id as usize >= dataset.len())
            || ids.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::invalid_argument(
                "invalid exact subset search shape, k, coordinates or sorted IDs",
            ));
        }
        if queries
            .len()
            .checked_div(dim)
            .and_then(|n| n.checked_mul(k))
            .is_none()
        {
            return Err(Error::invalid_argument("filtered result size overflow"));
        }
        if metric == ScoreMetric::CosineSimilarity
            && queries
                .chunks_exact(dim)
                .any(|q| q.iter().all(|&x| x == 0.))
        {
            return Err(Error::invalid_argument(
                "cosine requires nonzero query vectors",
            ));
        }
        let rows: Result<Vec<_>> = queries
            .par_chunks_exact(dim)
            .map(|query| {
                let mut best = BinaryHeap::with_capacity(k.min(ids.len()));
                for &id in ids {
                    let row = dataset.row(id as usize).expect("validated subset ID");
                    let score =
                        crate::algorithm::search::scoring::exact_score(query, &row[..dim], metric);
                    if !score.is_finite() {
                        return Err(Error::invalid_argument("non-finite exact subset score"));
                    }
                    let rank = if metric == ScoreMetric::SquaredL2 {
                        score
                    } else {
                        -score
                    };
                    // Signed zeros are equal metric scores; total_cmp alone
                    // would order -0 before +0 instead of breaking the tie by ID.
                    let rank = if rank == 0.0 { 0.0 } else { rank };
                    let candidate = Candidate { rank, score, id };
                    if best.len() < k {
                        best.push(candidate);
                    } else if best.peek().is_some_and(|worst| candidate < *worst) {
                        *best.peek_mut().expect("nonempty heap") = candidate;
                    }
                }
                let selected = best.into_sorted_vec();
                Ok((
                    selected.iter().map(|v| v.id).collect(),
                    selected.iter().map(|v| v.score).collect(),
                ))
            })
            .collect();
        let rows = rows?;
        if scores {
            SearchResult::from_scored_rows(rows, k, dataset.len(), metric)
        } else {
            SearchResult::from_rows(
                rows.into_iter().map(|(ids, _)| ids).collect(),
                k,
                dataset.len(),
            )
        }
    }
}

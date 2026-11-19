/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Application results retain internal identities for diagnostics and evaluation.
use orion_ann_core::SearchScores;

/// Row-major results. `ids` always contains internal u32 IDs; `external_ids`
/// optionally supplies application u64 identities in the same order. Only each
/// row's `counts` prefix is valid, including when a business ID equals u64::MAX.
#[derive(Clone, PartialEq)]
pub struct SearchResult {
    /// Packed internal IDs, with u32::MAX padding.
    pub ids: Vec<u32>,
    /// Valid prefix length per query.
    pub counts: Vec<u32>,
    /// Width of each result row.
    pub k: usize,
    /// Exact selected-candidate scores when requested; padding is NaN.
    pub scores: Option<SearchScores>,
    /// Optional business IDs in matching row order; only counts define validity.
    pub external_ids: Option<Vec<u64>>,
}
impl From<orion_ann_core::SearchResult> for SearchResult {
    /// Moves core result buffers into the application result without copying their payloads.
    fn from(value: orion_ann_core::SearchResult) -> Self {
        Self {
            ids: value.ids,
            counts: value.counts,
            k: value.k,
            scores: value.scores,
            external_ids: None,
        }
    }
}
impl SearchResult {
    /// Returns the number of query rows, including rows with no valid neighbors.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Reports whether there are no query rows, rather than whether every row has zero hits.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Iterates valid internal-ID prefixes in query order, excluding padding slots.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &[u32]> {
        self.ids
            .chunks(self.k.max(1))
            .zip(&self.counts)
            .map(|(row, &n)| &row[..n as usize])
    }
}
impl std::fmt::Debug for SearchResult {
    /// Formats result shape and score availability without printing large result buffers.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchResult")
            .field("query_count", &self.len())
            .field("k", &self.k)
            .field("score_metric", &self.scores.as_ref().map(|s| s.metric))
            .field("has_external_ids", &self.external_ids.is_some())
            .finish_non_exhaustive()
    }
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Owned batch results shared by core, Rust services and language adapters.
use crate::{Error, Result};

/// Neighbors in query order. Each row occupies `k` IDs; only its `counts`
/// prefix is valid. Remaining entries contain `u32::MAX`. Scores are not
/// exported; constructing results does not perform an extra scoring pass.
#[derive(Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Packed row-major neighbor IDs, including padding.
    pub ids: Vec<u32>,
    /// Number of valid neighbors for each query.
    pub counts: Vec<u32>,
    /// Requested maximum number of neighbors per query.
    pub k: usize,
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
        Ok(Self { ids, counts, k })
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

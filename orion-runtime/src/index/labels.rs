/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Index facade: row identity conversion and one-revision filtered requests.
use super::{Index, QueryBatch};
use crate::{
    Error, ExternalLabelUpdate, FilterSearchOptions, FilteredSearchResult, LabelFilter,
    LabelUpdate, Result,
};

impl Index {
    /// Current in-process label revision, initially zero. Versions are scoped to
    /// this index's row identity; independent restored copies may branch.
    pub fn label_revision(&self) -> Result<u64> {
        Ok(self.labels.snapshot()?.revision)
    }

    /// Atomically apply all label changes if the current revision equals expected.
    /// No-op batches keep the revision. A successful update is visible in memory;
    /// use `save` for durability. Existing searches keep their original snapshot.
    ///
    /// # Errors
    /// Returns InvalidState on revision conflict; InvalidArgument on invalid keys,
    /// row IDs, overlapping add/remove sets or limits. Limits: 4096 labels and
    /// 1,000,000 memberships per state; 1,000,000 supplied IDs per batch. Failures
    /// publish nothing. No implicit graph change, calibration or disk write occurs.
    pub fn update_labels(&self, expected_revision: u64, changes: &[LabelUpdate]) -> Result<u64> {
        self.labels.update(expected_revision, changes)
    }

    /// Atomically update labels using business IDs instead of internal row IDs.
    ///
    /// Resolves the complete batch in one scan of the immutable business-ID map,
    /// then uses the same publication transaction as [`Self::update_labels`].
    /// No permanent reverse map is allocated. A concurrent writer may publish
    /// during conversion; the revision is checked under the writer lock before
    /// applying this batch, so stale conversions never overwrite newer labels.
    /// Success is in-memory visibility; call [`Self::save`] for durability.
    ///
    /// # Errors
    /// Requires a business-ID mapping, including for an empty batch. Unknown add
    /// or remove IDs, invalid labels, overlapping changes, exceeded batch limits
    /// or revision conflicts publish nothing. Limits and no-op semantics match
    /// [`Self::update_labels`]. This does not change the graph or ID mapping.
    pub fn update_labels_external(
        &self,
        expected_revision: u64,
        changes: &[ExternalLabelUpdate],
    ) -> Result<u64> {
        crate::labels::update::check_batch_limits(
            changes.len(),
            changes.iter().map(|c| (c.add.len(), c.remove.len())),
        )?;
        for change in changes {
            change.label.validate()?;
        }
        // Gather once: scanning the N-row mapping for each label would multiply
        // conversion cost by the number of changes. Preserve per-list order so
        // splitting below is lossless, including duplicate input IDs.
        let business_ids: Vec<_> = changes
            .iter()
            .flat_map(|c| c.add.iter().chain(&c.remove))
            .copied()
            .collect();
        let rows = self.resolve_external_ids(&business_ids)?;
        let mut remaining = rows.as_slice();
        let internal: Vec<_> = changes
            .iter()
            .map(|change| {
                let (add, tail) = remaining.split_at(change.add.len());
                let (remove, tail) = tail.split_at(change.remove.len());
                remaining = tail;
                LabelUpdate {
                    label: change.label.clone(),
                    add: add.to_vec(),
                    remove: remove.to_vec(),
                }
            })
            .collect();
        self.update_labels(expected_revision, &internal)
    }

    /// Resolve business IDs without allocating a permanent reverse map. Input
    /// order is preserved. A batch performs one scan of the resident ID mapping.
    ///
    /// # Errors
    /// Requires `new_with_ids`/a restored ID map, known IDs and <= 1,000,000 inputs.
    pub fn resolve_external_ids(&self, ids: &[u64]) -> Result<Vec<u32>> {
        if ids.len() > crate::labels::MAX_MEMBERSHIPS {
            return Err(Error::invalid_argument("too many external IDs"));
        }
        let mapping = self
            .external_ids
            .as_ref()
            .ok_or_else(|| Error::invalid_state("index has no business ID mapping"))?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut positions: std::collections::HashMap<u64, u32> =
            ids.iter().map(|&id| (id, u32::MAX)).collect();
        for (row, id) in mapping.0.iter().enumerate() {
            if let Some(slot) = positions.get_mut(id) {
                *slot = row as u32;
            }
        }
        ids.iter()
            .map(|id| {
                let row = positions[id];
                if row == u32::MAX {
                    Err(Error::invalid_argument("unknown external ID")
                        .with_detail("external_id", *id))
                } else {
                    Ok(row)
                }
            })
            .collect()
    }

    /// Resolve labels once and exhaustively search matching rows. One immutable
    /// label revision is held for the entire batch; writers cannot mix its rows.
    /// This initial filtered path requires no adaptive calibration and may return
    /// fewer than k results. It is a bounded exact baseline, not filtered ANN.
    ///
    /// # Errors
    /// Rejects invalid query/k/filter input, non-finite scores or a matching set
    /// exceeding max_candidates. Unknown labels are empty sets, never wildcards.
    pub fn search_filtered(
        &self,
        queries: QueryBatch<'_>,
        filter: &LabelFilter,
        options: &FilterSearchOptions,
    ) -> Result<FilteredSearchResult> {
        if options.k == 0 || options.k > self.len() {
            return Err(Error::invalid_argument("require 1 <= k <= num_points"));
        }
        let prepared = self.prepare_queries(queries)?;
        let snapshot = self.labels.snapshot()?;
        let ids = crate::filter::resolve(&snapshot, filter, options.max_candidates)?;
        let mut result = self
            .pool
            .install(|| {
                self.view.engine.search_subset(
                    prepared,
                    &ids,
                    self.config.metric,
                    options.k,
                    options.scores,
                )
            })?
            .into();
        self.attach_external_ids(&mut result);
        Ok(FilteredSearchResult {
            result,
            label_revision: snapshot.revision,
        })
    }
}

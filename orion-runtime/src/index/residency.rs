/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Scoped residency of an already-prepared search generation.
use super::Index;
use crate::{PinHint, PinOptions, PinnedDatasets, Result};

impl Index {
    /// CLI-only scoped policy after `Index::load` has checked the budget and
    /// restored prepared snapshot requirements. Embedded callers use `pin`.
    #[cfg(feature = "cli")]
    pub(crate) fn pin_for_driver(&self) -> Result<PinnedDatasets<'_>> {
        self.pin(&crate::residency::automatic_options(
            self.config.memory_budget_gib,
        )?)
    }

    /// Select owned-memory or on-demand private-file storage for prepared PQ codes.
    /// The fitted codebook remains best-effort pinned. This deployment setting does
    /// not change calibration, scores or snapshot contents. Call after calibration
    /// or snapshot open; drop any borrowed pin guards before switching placement.
    ///
    /// # Errors
    /// Returns InvalidState if the active index has no prepared PQ sidecar, or an
    /// I/O error if the private temporary code file cannot be created/written.
    pub fn set_pq_code_storage(&mut self, policy: crate::PqCodeStorage) -> Result<()> {
        self.view.engine.set_pq_code_storage(policy)
    }

    /// List the total payload for each prepared search requirement.
    /// # Errors
    /// Returns InvalidState for sidecars not yet prepared by calibration.
    pub fn pinning_hints(&self) -> Result<Vec<PinHint>> {
        self.view.engine.pin_hints(&self.config)
    }

    /// Keep selected prepared datasets in RAM until the returned guard drops.
    /// This is optional and does not affect calibration, search results or
    /// snapshot identities. It cannot guarantee CPU-cache residency. The
    /// default best-effort policy has a 64 MiB page-rounded budget.
    ///
    /// # Errors
    /// Rejects unused/unprepared selections. Required mode reports budget or
    /// OS failures and releases all locks acquired by this request.
    ///
    /// ```no_run
    /// # fn example(index: &orion_ann::Index) -> orion_ann::Result<()> {
    /// let residency = index.pin(&orion_ann::PinOptions::default())?;
    /// println!("{} bytes pinned", residency.pinned_bytes());
    /// // Search through `index` while keeping `residency` in this scope.
    /// # Ok(()) }
    /// ```
    pub fn pin(&self, options: &PinOptions) -> Result<PinnedDatasets<'_>> {
        self.view.engine.pin(&self.config, options)
    }
}

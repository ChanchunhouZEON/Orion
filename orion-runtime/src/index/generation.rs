/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Whole-index publication; no lock is held during graph traversal.
use crate::{Error, Index, Result};
use std::sync::{Arc, RwLock};

/// A deployment slot for complete, calibrated index generations.
/// Build, calibrate and explicitly warm up the replacement before `replace`.
/// `current` clones one Arc under a short lock; that single generation owns all
/// vectors, graph, sidecars and business IDs for the request. In-flight readers
/// continue using the old generation. This does not implement node mutations.
#[derive(Debug)]
pub struct IndexSlot {
    current: RwLock<Arc<Index>>,
    required_ks: Vec<usize>,
}
impl IndexSlot {
    /// Require the application's supported k values on every published index.
    /// Empty/duplicate requirements and absent calibrations return InvalidArgument
    /// or InvalidState. No calibration or warmup is performed implicitly.
    pub fn new(index: Index, required_ks: &[usize]) -> Result<Self> {
        let mut ks = required_ks.to_vec();
        ks.sort_unstable();
        if ks.is_empty() || ks.contains(&0) || ks.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::invalid_argument(
                "generation requires distinct positive k values",
            ));
        }
        check(&index, &ks)?;
        Ok(Self {
            current: RwLock::new(Arc::new(index)),
            required_ks: ks,
        })
    }

    /// Hold this handle for the complete request, including result interpretation.
    pub fn current(&self) -> Result<Arc<Index>> {
        self.current
            .read()
            .map(|index| Arc::clone(&index))
            .map_err(|_| Error::invalid_state("generation slot is poisoned"))
    }

    /// Publish a fully prepared replacement, returning the old generation.
    /// Dimension, metric and storage must match the current request contract.
    /// Validation failures leave the active index unchanged. Retain the returned
    /// Arc until in-flight readers drain, then drop it on a maintenance thread
    /// if releasing a large index on a query thread would be undesirable.
    pub fn replace(&self, index: Index) -> Result<Arc<Index>> {
        check(&index, &self.required_ks)?;
        let replacement = Arc::new(index);
        let mut current = self
            .current
            .write()
            .map_err(|_| Error::invalid_state("generation slot is poisoned"))?;
        if current.dimension() != replacement.dimension()
            || current.config.metric != replacement.config.metric
            || current.storage() != replacement.storage()
        {
            return Err(Error::invalid_argument(
                "replacement dimension, metric and storage must match",
            ));
        }
        Ok(std::mem::replace(&mut *current, replacement))
    }
}
fn check(index: &Index, ks: &[usize]) -> Result<()> {
    for &k in ks {
        if index.calibration(k).is_none() {
            return Err(
                Error::invalid_state("generation is missing required calibration")
                    .with_detail("k", k),
            );
        }
    }
    Ok(())
}

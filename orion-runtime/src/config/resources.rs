/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Pure budget rules; detection and lock planning remain in platform/residency.
use crate::{Error, Result};

pub(crate) const BYTES_PER_GIB: f64 = 1_073_741_824.0;

pub fn validate_memory_budget(budget: Option<f64>) -> Result<()> {
    if budget.is_some_and(|gib| !gib.is_finite() || gib <= 0.) {
        return Err(Error::configuration(
            "memory_budget_gib",
            "memory_budget_gib must be finite and positive",
        ));
    }
    Ok(())
}

// Keep the floating-point result for preflight comparisons. Lock planning casts
// to u64 only at its integer budget boundary (saturating for enormous limits).
pub(crate) fn gib_to_bytes(gib: f64) -> f64 {
    gib * BYTES_PER_GIB
}

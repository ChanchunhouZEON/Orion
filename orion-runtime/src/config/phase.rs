/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Select preparation dependencies without changing the validated search recipe.
use super::{plan::IndexPlan, ResolvedRunConfig};
use crate::sidecars::requirements::SidecarRequirement;

/// Target of preparation, not the name of the function performing it.
/// Native cosine norms remain a base auxiliary object in either phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunPhase {
    /// Prepare base and graph without fitting search sidecars.
    GraphOnly,
    /// Include the dependencies needed by calibration and search.
    SearchReady,
}

impl ResolvedRunConfig {
    /// Interpret the external prepare-only flag once at the runtime boundary.
    pub(crate) fn run_phase(&self) -> RunPhase {
        if self.prepare_only {
            RunPhase::GraphOnly
        } else {
            RunPhase::SearchReady
        }
    }
}

impl RunPhase {
    /// Both preflight and placement consume this phase-selected list. Keep
    /// duplicate consumers here; the dataset estimator deduplicates allocations.
    pub(crate) fn sidecar_requirements(self, plan: IndexPlan) -> Vec<SidecarRequirement> {
        match self {
            Self::GraphOnly => Vec::new(),
            Self::SearchReady => plan.search_sidecar_requirements(),
        }
    }
}

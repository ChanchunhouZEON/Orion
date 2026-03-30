/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::collections::VecDeque;

const MINIMUM_DENOMINATOR: f32 = 1e-12;

/// Sliding-window convergence detector for the two-phase search.
/// When the relative range of recent best distances drops below epsilon,
/// the search switches from full graph to compressed graph.
pub struct DistanceConvergenceChecker {
    window_size: usize,
    epsilon: f32,
    history: VecDeque<f32>,
    has_converged: bool,
}

impl DistanceConvergenceChecker {
    pub fn new(window_size: usize, epsilon: f32) -> Self {
        Self {
            window_size,
            epsilon,
            history: VecDeque::with_capacity(window_size + 1),
            has_converged: false,
        }
    }

    /// Reset the checker for reuse (scratch pattern).
    pub fn reset(&mut self) {
        self.history.clear();
        self.has_converged = false;
    }

    /// Returns whether convergence has already been detected.
    pub fn has_converged(&self) -> bool {
        self.has_converged
    }

    /// Add a new distance value and check for convergence.
    /// Returns true if converged (relative range of window < epsilon).
    pub fn update(&mut self, dist: f32) -> bool {
        if self.has_converged {
            return true;
        }

        self.history.push_back(dist);

        if self.history.len() > self.window_size {
            self.history.pop_front();
        }

        if self.history.len() < self.window_size {
            return false;
        }

        let (min_val, max_val) = self
            .history
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let range = max_val - min_val;
        let denominator = min_val.max(MINIMUM_DENOMINATOR);

        self.has_converged = (range / denominator) < self.epsilon;
        self.has_converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_window_not_converged() {
        let mut checker = DistanceConvergenceChecker::new(3, 0.01);
        assert!(!checker.update(1.0));
        assert!(!checker.update(1.0));
        // Not enough history yet (need 3)
    }

    #[test]
    fn test_stable_convergence() {
        let mut checker = DistanceConvergenceChecker::new(3, 0.01);
        checker.update(1.0);
        checker.update(1.0);
        let converged = checker.update(1.0);
        assert!(converged); // all same → range=0 → converged
    }

    #[test]
    fn test_varying_non_convergence() {
        let mut checker = DistanceConvergenceChecker::new(3, 0.01);
        checker.update(1.0);
        checker.update(2.0);
        let converged = checker.update(3.0);
        assert!(!converged); // range=2.0, denom=1.0, ratio=2.0 >> 0.01
    }

    #[test]
    fn test_stays_converged() {
        let mut checker = DistanceConvergenceChecker::new(2, 0.01);
        checker.update(1.0);
        checker.update(1.0); // converged
        assert!(checker.update(100.0)); // once converged, stays converged
    }

    #[test]
    fn test_reset() {
        let mut checker = DistanceConvergenceChecker::new(2, 0.01);
        checker.update(1.0);
        checker.update(1.0); // converged
        checker.reset();
        assert!(!checker.update(1.0)); // need full window again
    }

    #[test]
    fn test_has_converged_getter() {
        let mut checker = DistanceConvergenceChecker::new(2, 0.01);
        assert!(!checker.has_converged()); // not yet
        checker.update(1.0);
        assert!(!checker.has_converged()); // still not enough history
        checker.update(1.0); // converges here
        assert!(checker.has_converged()); // getter reflects convergence
        checker.reset();
        assert!(!checker.has_converged()); // reset clears it
    }

    #[test]
    fn test_sliding_window() {
        let mut checker = DistanceConvergenceChecker::new(3, 0.01);
        checker.update(1.0);
        checker.update(10.0);
        assert!(!checker.update(100.0)); // window=[1,10,100], range=99, not converged

        // Window slides: [10,100,100]
        assert!(!checker.update(100.0));

        // Window slides: [100,100,100] → range=0, converged
        assert!(checker.update(100.0));
    }
}

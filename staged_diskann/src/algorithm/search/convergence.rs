/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

const MINIMUM_DENOMINATOR: f32 = 1e-12;

/// Maximum supported window size for the stack-allocated ring buffer.
const MAX_WINDOW_SIZE: usize = 64;

/// Sliding-window convergence detector for the two-phase search.
/// When the relative range of recent best distances drops below epsilon,
/// the search switches from full graph to compressed graph.
///
/// Uses a stack-allocated fixed ring buffer (`[f32; 64]`) instead of
/// `VecDeque` to avoid heap allocations on the hot search path.
pub struct DistanceConvergenceChecker {
    window_size: usize,
    epsilon: f32,
    /// Fixed ring buffer stored entirely on the stack.
    buf: [f32; MAX_WINDOW_SIZE],
    /// Write head: index of the next slot to write into (wraps around).
    head: usize,
    /// Number of elements currently stored (saturates at `window_size`).
    count: usize,
    has_converged: bool,
}

impl DistanceConvergenceChecker {
    pub fn new(window_size: usize, epsilon: f32) -> Self {
        debug_assert!(
            window_size <= MAX_WINDOW_SIZE,
            "window_size ({}) exceeds MAX_WINDOW_SIZE ({})",
            window_size,
            MAX_WINDOW_SIZE,
        );
        Self {
            window_size,
            epsilon,
            buf: [0.0_f32; MAX_WINDOW_SIZE],
            head: 0,
            count: 0,
            has_converged: false,
        }
    }

    /// Reset the checker for reuse (scratch pattern).
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.has_converged = false;
    }

    /// Reconfigure window size and epsilon, then reset state.
    /// Use this when the scratch-pooled checker needs different parameters per query.
    pub fn reconfigure(&mut self, window_size: usize, epsilon: f32) {
        debug_assert!(
            window_size <= MAX_WINDOW_SIZE,
            "window_size ({}) exceeds MAX_WINDOW_SIZE ({})",
            window_size,
            MAX_WINDOW_SIZE,
        );
        self.window_size = window_size;
        self.epsilon = epsilon;
        self.head = 0;
        self.count = 0;
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

        // Write into the ring buffer at the current head position.
        self.buf[self.head] = dist;
        self.head = (self.head + 1) % self.window_size;

        if self.count < self.window_size {
            self.count += 1;
        }

        if self.count < self.window_size {
            return false;
        }

        // Scan the ring buffer to find min and max.
        // The window is always exactly `window_size` elements at this point,
        // occupying indices 0..window_size in `buf` (since head wraps within
        // window_size). Iterating the full window_size slice is correct and
        // fast for <=64 stack-local floats.
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..self.window_size {
            let val = self.buf[i];
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

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

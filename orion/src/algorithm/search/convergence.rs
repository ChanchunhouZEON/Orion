/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Maximum supported window size for the stack-allocated ring buffer.
const MAX_WINDOW_SIZE: usize = 64;

/// Admission-rate convergence detector for the two-phase search.
///
/// Tracks how many of the last `window_size` expansion steps had at least
/// one candidate admitted to the priority queue. When this fraction drops
/// below `threshold`, the search is considered converged.
///
/// Convergence is **reversible**: a burst of admissions pushes the rate
/// back up and exits convergence.
pub struct SearchConvergenceChecker {
    window_size: usize,
    /// Converge when admission fraction < threshold (e.g., 0.2 = 20%).
    threshold: f32,
    /// Ring buffer: 1 = at least one admission this step, 0 = none.
    buf: [u8; MAX_WINDOW_SIZE],
    head: usize,
    count: usize,
    /// Running sum of admissions in the window.
    admit_sum: usize,
    total_steps: usize,
    min_steps: usize,
}

impl SearchConvergenceChecker {
    pub fn new(window_size: usize, threshold: f32) -> Self {
        debug_assert!(window_size <= MAX_WINDOW_SIZE);
        Self {
            window_size,
            threshold,
            buf: [0u8; MAX_WINDOW_SIZE],
            head: 0,
            count: 0,
            admit_sum: 0,
            total_steps: 0,
            min_steps: window_size * 2,
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
        self.admit_sum = 0;
        self.total_steps = 0;
    }

    pub fn reconfigure(&mut self, window_size: usize, threshold: f32) {
        debug_assert!(window_size <= MAX_WINDOW_SIZE);
        self.window_size = window_size;
        self.threshold = threshold;
        self.min_steps = window_size * 2;
        self.reset();
    }

    /// Feed whether admitted happens from this expansion step.
    /// Returns true if currently converged (low admission rate).
    #[inline]
    pub fn update(&mut self, is_admitted: usize) -> bool {
        self.total_steps += 1;

        let val = if is_admitted > 0 { 1u8 } else { 0u8 };

        // Evict oldest entry if window is full.
        if self.count >= self.window_size {
            let oldest = self.buf[self.head] as usize;
            self.admit_sum -= oldest;
        } else {
            self.count += 1;
        }

        self.buf[self.head] = val;
        self.head = (self.head + 1) % self.window_size;
        self.admit_sum += val as usize;

        if self.count < self.window_size || self.total_steps < self.min_steps {
            return false;
        }

        let rate = self.admit_sum as f32 / self.window_size as f32;
        rate < self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_window_not_converged() {
        let mut c = SearchConvergenceChecker::new(3, 0.2);
        assert!(!c.update(0));
        assert!(!c.update(0));
    }

    #[test]
    fn test_min_steps() {
        // ws=3, min_steps=6. Even all-zero window won't converge before 6 steps.
        let mut c = SearchConvergenceChecker::new(3, 0.5);
        for _ in 0..5 {
            assert!(!c.update(0));
        }
        // 6th step: now eligible, window=[0,0,0], rate=0 < 0.5 → converged.
        assert!(c.update(0));
    }

    #[test]
    fn test_high_admission_no_convergence() {
        let mut c = SearchConvergenceChecker::new(3, 0.2);
        for _ in 0..10 {
            assert!(!c.update(3)); // every step admits → rate=1.0 >> 0.2
        }
    }

    #[test]
    fn test_reversible() {
        let mut c = SearchConvergenceChecker::new(3, 0.3);
        // Fill min_steps(6) with no admissions.
        for _ in 0..6 {
            c.update(0);
        }
        assert!(c.update(0)); // converged: rate=0/3=0 < 0.3

        // One admission: window=[0,0,1], rate=1/3=0.33 > 0.3 → exits convergence.
        assert!(!c.update(2));

        // Need 3 more zeros to flush the admission out of the window.
        c.update(0); // window=[0,1,0], rate=1/3=0.33 → not converged
        c.update(0); // window=[1,0,0], rate=1/3=0.33 → not converged
        assert!(c.update(0)); // window=[0,0,0], rate=0 < 0.3 → converged again
    }

    #[test]
    fn test_convergence_and_exit() {
        let mut c = SearchConvergenceChecker::new(4, 0.3);
        // Fill min_steps(8) with zeros.
        for _ in 0..8 {
            c.update(0);
        }
        assert!(c.update(0)); // converged

        // Single admission: window=[0,0,0,1], rate=1/4=0.25 < 0.3 → still converged.
        assert!(c.update(1));

        // Two admissions needed to exit: window=[0,0,1,1], rate=2/4=0.5 > 0.3.
        assert!(!c.update(1));
    }

    #[test]
    fn test_reset() {
        let mut c = SearchConvergenceChecker::new(2, 0.3);
        for _ in 0..4 {
            c.update(0);
        }
        assert!(c.update(0)); // converged
        c.reset();
        assert!(!c.update(0)); // reset, need full window + min_steps again
    }
}

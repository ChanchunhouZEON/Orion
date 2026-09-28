/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Early exit checker for two-phase search.
///
/// After the search enters the converged (reranking) phase, tracks
/// consecutive expansion steps with zero PQ admissions. When this
/// count exceeds `limit`, the search should terminate — remaining
/// unvisited PQ candidates are unlikely to improve results.
pub struct EarlyExitChecker {
    limit: usize,
    consecutive_no_admit: usize,
}

impl EarlyExitChecker {
    pub fn diagnostic_state(&self) -> (usize, usize) {
        (self.limit, self.consecutive_no_admit)
    }

    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            consecutive_no_admit: 0,
        }
    }

    pub fn reset(&mut self) {
        self.consecutive_no_admit = 0;
    }

    pub fn reconfigure(&mut self, limit: usize) {
        self.limit = limit;
        self.consecutive_no_admit = 0;
    }

    /// Update after an expansion step.
    /// - `converged`: whether the DCC is in converged state.
    /// - `num_admitted`: how many candidates were admitted this step.
    ///
    /// Returns `true` if the search should terminate.
    #[inline]
    pub fn should_exit(&mut self, converged: bool, num_admitted: usize) -> bool {
        if !converged {
            self.consecutive_no_admit = 0;
            return false;
        }

        if num_admitted == 0 {
            self.consecutive_no_admit += 1;
            self.consecutive_no_admit >= self.limit
        } else {
            self.consecutive_no_admit = 0;
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_exit_during_navigation() {
        let mut checker = EarlyExitChecker::new(3);
        // Not converged — never exits.
        for _ in 0..100 {
            assert!(!checker.should_exit(false, 0));
        }
    }

    #[test]
    fn test_exit_after_limit() {
        let mut checker = EarlyExitChecker::new(3);
        assert!(!checker.should_exit(true, 0)); // 1
        assert!(!checker.should_exit(true, 0)); // 2
        assert!(checker.should_exit(true, 0)); // 3 → exit
    }

    #[test]
    fn test_admission_resets_counter() {
        let mut checker = EarlyExitChecker::new(3);
        checker.should_exit(true, 0); // 1
        checker.should_exit(true, 0); // 2
        assert!(!checker.should_exit(true, 1)); // admission → reset
        assert!(!checker.should_exit(true, 0)); // 1 again
        assert!(!checker.should_exit(true, 0)); // 2
        assert!(checker.should_exit(true, 0)); // 3 → exit
    }

    #[test]
    fn test_navigation_resets_counter() {
        let mut checker = EarlyExitChecker::new(2);
        checker.should_exit(true, 0); // 1
        assert!(!checker.should_exit(false, 0)); // nav → reset
        assert!(!checker.should_exit(true, 0)); // 1 again
        assert!(checker.should_exit(true, 0)); // 2 → exit
    }

    #[test]
    fn test_reset() {
        let mut checker = EarlyExitChecker::new(2);
        checker.should_exit(true, 0);
        checker.should_exit(true, 0); // would exit on next
        checker.reset();
        assert!(!checker.should_exit(true, 0)); // reset, start over
    }
}

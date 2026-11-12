/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Explicit controls for the shared search executor. Defaults disable
/// convergence switching/early exit; they are not calibrated parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicSearchConfig {
    pub k: usize,
    pub search_list_size: usize,
    pub window_size: usize,
    pub epsilon: f32,
    pub early_exit_limit: usize,
}
impl Default for DynamicSearchConfig {
    fn default() -> Self {
        Self {
            k: 10,
            search_list_size: 64,
            window_size: 3,
            epsilon: 0.,
            early_exit_limit: usize::MAX,
        }
    }
}

impl DynamicSearchConfig {
    pub(crate) fn validate(self, points: usize) -> crate::Result<()> {
        if self.k == 0
            || self.k > points
            || self.search_list_size < self.k
            || !(1..=64).contains(&self.window_size)
            || !self.epsilon.is_finite()
            || !(0.0..=1.0).contains(&self.epsilon)
        {
            return Err(crate::Error::invalid_argument(
                "invalid k, search list size, convergence window or epsilon",
            ));
        }
        Ok(())
    }
}

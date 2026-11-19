/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::time::{Duration, Instant};

/// Monotonic elapsed-time checkpoint for build and I/O instrumentation.
#[derive(Clone)]
pub struct Timer {
    check_point: Instant,
}

impl Default for Timer {
    /// Starts a timer at construction, equivalent to [`Self::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    /// Captures the current monotonic instant as the timing origin.
    pub fn new() -> Timer {
        Timer {
            check_point: Instant::now(),
        }
    }

    /// Moves the timing origin to now without retaining earlier elapsed time.
    pub fn reset(&mut self) {
        self.check_point = Instant::now();
    }

    /// Returns time since the last construction or reset.
    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.check_point)
    }

    /// Returns elapsed monotonic time as fractional seconds.
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    /// Formats a named operation's elapsed time without resetting the checkpoint.
    pub fn elapsed_seconds_for_step(&self, step: &str) -> String {
        format!("Time for {}: {:.3} seconds", step, self.elapsed_seconds(),)
    }
}

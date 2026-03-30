/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Timer {
    check_point: Instant,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Timer {
        Timer {
            check_point: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.check_point = Instant::now();
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.check_point)
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    pub fn elapsed_seconds_for_step(&self, step: &str) -> String {
        format!("Time for {}: {:.3} seconds", step, self.elapsed_seconds(),)
    }
}

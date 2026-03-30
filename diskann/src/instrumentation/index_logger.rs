/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::common::ANNResult;
use crate::utils::Timer;

pub struct IndexLogger {
    items_processed: AtomicUsize,
    timer: Timer,
    range: usize,
}

impl IndexLogger {
    pub fn new(range: usize) -> Self {
        Self {
            items_processed: AtomicUsize::new(0),
            timer: Timer::new(),
            range,
        }
    }

    pub fn vertex_processed(&self) -> ANNResult<()> {
        let count = self.items_processed.fetch_add(1, Ordering::Relaxed);
        if count % 100_000 == 0 {
            let percentage = (100_f32 * count as f32) / (self.range as f32);
            let elapsed = self.timer.elapsed().as_secs_f32();
            log::info!(
                "Index construction: {:.1}% complete, {:.1}s elapsed",
                percentage,
                elapsed,
            );
        }

        Ok(())
    }
}

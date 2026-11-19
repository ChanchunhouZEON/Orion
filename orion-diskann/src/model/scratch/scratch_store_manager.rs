/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::common::ANNResult;

use super::ArcConcurrentBoxedQueue;
use super::scratch_traits::Scratch;
use std::time::Duration;

/// Borrowed scratch owner that returns its item to the pool on drop.
pub struct ScratchStoreManager<T: Scratch> {
    scratch: Option<Box<T>>,
    scratch_pool: ArcConcurrentBoxedQueue<T>,
}

impl<T: Scratch> ScratchStoreManager<T> {
    /// Waits until scratch can be borrowed from the shared pool and owns its return obligation.
    ///
    /// The wait interval is a retry period, not a total acquisition deadline.
    pub fn new(scratch_pool: ArcConcurrentBoxedQueue<T>, wait_time: Duration) -> ANNResult<Self> {
        let mut scratch = scratch_pool.pop()?;
        while scratch.is_none() {
            scratch_pool.wait_for_push_notify(wait_time)?;
            scratch = scratch_pool.pop()?;
        }

        Ok(ScratchStoreManager {
            scratch,
            scratch_pool,
        })
    }

    /// Lends mutable access to the scratch value exclusively held by this manager.
    pub fn scratch_space(&mut self) -> Option<&mut T> {
        self.scratch.as_deref_mut()
    }
}

impl<T: Scratch> Drop for ScratchStoreManager<T> {
    /// Clears and returns borrowed scratch to the pool; a return error cannot be propagated from Drop.
    fn drop(&mut self) {
        if let Some(mut scratch) = self.scratch.take() {
            scratch.clear();
            let _ = self.scratch_pool.push(scratch);
        }
    }
}

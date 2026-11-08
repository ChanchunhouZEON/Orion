/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Best-effort resource observations. Missing observations stay unknown, never unlimited.
//! These are snapshots, not reservations: a later allocation or lock may still fail.

/// Host memory, deployment headroom and process lock allowance, all in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryResources {
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub lock_limit_bytes: Option<u64>,
    pub locked_bytes: Option<u64>,
}

impl MemoryResources {
    /// Observe this process's environment without changing limits or allocating payloads.
    pub fn detect() -> Self {
        #[allow(unused_mut)] // Unsupported targets intentionally return unknown observations.
        let mut result = Self::default();
        #[cfg(target_os = "linux")]
        linux::observe(&mut result);
        #[cfg(target_os = "macos")]
        macos::observe(&mut result);
        #[cfg(unix)]
        {
            let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
            // SAFETY: getrlimit initializes the structure on success.
            if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, limit.as_mut_ptr()) } == 0 {
                let limit = unsafe { limit.assume_init() }.rlim_cur;
                #[allow(clippy::unnecessary_cast)] // rlim_t can be 32-bit on other Unix targets.
                let bytes = limit as u64;
                result.lock_limit_bytes = Some(if limit == libc::RLIM_INFINITY {
                    u64::MAX
                } else {
                    bytes
                });
            }
        }
        result
    }
}

#[cfg(any(target_os = "linux", test))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

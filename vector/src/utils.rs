/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Prefetch the given vector in chunks of 64 bytes (cache line size).
/// NOTE: good efficiency when total_vec_size is integral multiple of 64
#[inline]
pub fn prefetch_vector<T>(vec: &[T]) {
    let vec_ptr = vec.as_ptr() as *const i8;
    let vecsize = std::mem::size_of_val(vec);
    let max_prefetch_size = (vecsize / 64) * 64;

    for d in (0..max_prefetch_size).step_by(64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(vec_ptr.add(d), std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            // NEON prefetch using inline assembly
            let addr = vec_ptr.add(d);
            std::arch::asm!("prfm pldl1keep, [{x}]", x = in(reg) addr);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = d;
            // No prefetch available on this architecture
        }
    }
}

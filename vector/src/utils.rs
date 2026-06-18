/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Cache-line size for the supported architectures. **Apple Silicon
/// (M1 / M2 / M3) has 128-byte cache lines**; x86-64 has 64. Stepping
/// `prefetch_vector` at this granularity issues at most one `prfm`
/// per real cache line — no duplicates on aarch64, and the trailing
/// partial line is still covered because we round up.
#[cfg(target_arch = "aarch64")]
pub const CACHE_LINE_BYTES: usize = 128;
#[cfg(target_arch = "x86_64")]
pub const CACHE_LINE_BYTES: usize = 64;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const CACHE_LINE_BYTES: usize = 64;

/// Prefetch every cache line that backs `vec`, including the trailing
/// partial line. The previous implementation rounded *down*
/// (`(vecsize / 64) * 64`) and stepped by 64 bytes, which on M2
/// (128-byte lines) had two problems: (a) for vectors whose size
/// wasn't a multiple of 128, the trailing partial line was never
/// prefetched — a 400-byte f32 vector at N=100 only got 3 of its 4
/// lines touched, leaving a ~100 ns demand-fetch miss inside the
/// compute loop; (b) consecutive `prfm` ops at offsets 0 and 64 hit
/// the same 128-byte line on aarch64, wasting issue slots. The fixed
/// version rounds *up* and steps by `CACHE_LINE_BYTES`, so every line
/// is touched exactly once.
#[inline]
pub fn prefetch_vector<T>(vec: &[T]) {
    let vec_ptr = vec.as_ptr() as *const i8;
    let vecsize = std::mem::size_of_val(vec);
    if vecsize == 0 {
        return;
    }
    // Round-up: cover every line that holds at least one byte of `vec`.
    let n_lines = vecsize.div_ceil(CACHE_LINE_BYTES);
    for i in 0..n_lines {
        let off = i * CACHE_LINE_BYTES;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(vec_ptr.add(off), std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let addr = vec_ptr.add(off);
            std::arch::asm!("prfm pldl1keep, [{x}]", x = in(reg) addr);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = off;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_line_bytes_is_power_of_two() {
        assert!(CACHE_LINE_BYTES.is_power_of_two());
        assert!(CACHE_LINE_BYTES >= 64);
    }

    #[test]
    fn prefetch_empty_slice_is_noop() {
        let empty: &[f32] = &[];
        // Just verify it returns without panic / UB.
        prefetch_vector(empty);
    }

    #[test]
    fn prefetch_small_slice() {
        let v = [1.0f32; 8];
        prefetch_vector(&v);
    }

    #[test]
    fn prefetch_partial_trailing_line() {
        // 400 bytes (100 × f32) — covers a partial trailing line at
        // both 64- and 128-byte cache line sizes.
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        prefetch_vector(&v);
    }
}

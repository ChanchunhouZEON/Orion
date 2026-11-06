/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Shared arithmetic driver. Kernels determine compute granularity; the
//! inlined callback advances the stream's near-prefetch cursor by one line.
use crate::DistanceKernelBackend;

/// Internal static callback. A concrete implementation can force inlining of
/// the entire address resolver; FnMut's compiler-generated adapter cannot.
pub(crate) trait LinePrefetch {
    fn advance(&mut self);
}
#[cfg(test)]
impl<F: FnMut()> LinePrefetch for F {
    #[inline(always)]
    fn advance(&mut self) {
        self();
    }
}

/// SLOT is an unrolled position, not a runtime loop counter. For the built-in
/// 32/64-byte kernels and 64/128-byte cache lines, four steps end on a line
/// boundary. LLVM folds the divisions and removes the zero-request callbacks.
/// The fallback also supports custom kernels whose four-step group is unaligned.
#[inline(always)]
unsafe fn step<K: DistanceKernelBackend, P: LinePrefetch, const LINE: usize, const SLOT: usize>(
    accumulator: &mut K::Acc,
    base: *const u8,
    query: *const u8,
    group_start: usize,
    prefetch: &mut P,
) {
    let offset = (group_start + SLOT) * K::CHUNK_BYTES;
    K::step(
        accumulator,
        base.add(offset).cast(),
        query
            .add((group_start + SLOT) * K::QUERY_CHUNK_BYTES)
            .cast(),
    );

    let start = if (4 * K::CHUNK_BYTES).is_multiple_of(LINE) {
        SLOT * K::CHUNK_BYTES
    } else {
        offset
    };
    let completed_lines = (start + K::CHUNK_BYTES) / LINE - start / LINE;
    for _ in 0..completed_lines {
        prefetch.advance();
    }
}

#[inline(always)]
pub(crate) unsafe fn compute_vertex<
    K: DistanceKernelBackend,
    P: LinePrefetch,
    const LINE: usize,
>(
    base: *const K::Storage,
    query: *const K::Storage,
    compute_bytes: usize,
    mut prefetch: P,
) -> f32 {
    let chunks = compute_bytes / K::CHUNK_BYTES;
    let main_end = chunks & !3;
    let base = base.cast::<u8>();
    let query = query.cast::<u8>();
    let mut acc = K::init();
    let mut acc1 = K::init();
    let mut acc2 = K::init();
    let mut acc3 = K::init();
    let mut c = 0;
    while c < main_end {
        step::<K, P, LINE, 0>(&mut acc, base, query, c, &mut prefetch);
        step::<K, P, LINE, 1>(&mut acc1, base, query, c, &mut prefetch);
        step::<K, P, LINE, 2>(&mut acc2, base, query, c, &mut prefetch);
        step::<K, P, LINE, 3>(&mut acc3, base, query, c, &mut prefetch);
        c += 4;
    }

    let remaining = chunks - main_end;
    // Preserve the existing two independent tail accumulators on x86, and
    // preserve reduction order. Prefetch cadence no longer depends on this ILP
    // choice: each step reports only the cache lines it actually completes.
    #[cfg(target_arch = "x86_64")]
    if remaining >= 2 {
        step::<K, P, LINE, 0>(&mut acc, base, query, main_end, &mut prefetch);
        step::<K, P, LINE, 1>(&mut acc1, base, query, main_end, &mut prefetch);
    }
    K::merge(&mut acc, acc1);
    K::merge(&mut acc2, acc3);
    K::merge(&mut acc, acc2);

    #[cfg(target_arch = "x86_64")]
    if remaining == 1 {
        step::<K, P, LINE, 0>(&mut acc, base, query, main_end, &mut prefetch);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if remaining >= 1 {
            step::<K, P, LINE, 0>(&mut acc, base, query, main_end, &mut prefetch);
        }
        if remaining >= 2 {
            step::<K, P, LINE, 1>(&mut acc, base, query, main_end, &mut prefetch);
        }
    }
    if remaining == 3 {
        step::<K, P, LINE, 2>(&mut acc, base, query, main_end, &mut prefetch);
    }

    // A short vector still consumes one logical line. Never carry this debt
    // into the next random vertex: two short vectors need not share a line.
    if !compute_bytes.is_multiple_of(LINE) {
        prefetch.advance();
    }
    K::reduce(acc)
}

#[cfg(test)]
mod tests;

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Direct slice vs immediate read-lease acquisition around the same SIMD stream.
//! Synthetic SIFT/GIST-shaped rows isolate boundary overhead, not ANN recall.
//! Run with `cargo run --release -p orion-ann-platform --example read_path`.
use orion_ann_platform::{
    reading::{ObjectReader, ReadRequest},
    region::ArrayView,
};
use orion_ann_vector::{DistanceStream, L2F32Distance};
use std::{
    future::Future,
    hint::black_box,
    pin::pin,
    task::{Context, Poll, Waker},
    time::{Duration, Instant},
};

/// Runs the unchanged f32 distance stream over valid base rows for read-abstraction overhead measurement.
#[inline(always)]
fn compute(base: *const f32, query: &[f32], ids: &[u32], stride: usize) -> f32 {
    let mut sum = 0.0;
    // SAFETY: both wrappers receive rows initialized below, with padded queries,
    // a full SIMD tail, and candidate IDs within those rows. The lease/view stays
    // alive for the entire stream; the computation and prefetch logic are shared.
    unsafe {
        DistanceStream::<L2F32Distance>::new(
            base.cast(),
            stride * 4,
            stride * 4,
            query.as_ptr().cast(),
            ids,
            32,
        )
        .run(|_, distance| sum += distance);
    }
    sum
}

/// Scores candidates through the direct array pointer for the benchmark control arm.
#[inline(never)]
#[unsafe(no_mangle)]
fn read_path_direct(base: &ArrayView<f32>, query: &[f32], ids: &[u32], stride: usize) -> f32 {
    compute(base.as_ptr(), query, ids, stride)
}

/// Obtains an immediately ready region lease and scores the same candidates through its typed slice.
#[inline(never)]
#[unsafe(no_mangle)]
fn read_path_unified(base: &ArrayView<f32>, query: &[f32], ids: &[u32], stride: usize) -> f32 {
    let request = ReadRequest::new(0, base.region().len_bytes()).unwrap();
    let mut future = pin!(base.region().read(request));
    let Poll::Ready(result) = future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    else {
        panic!("resident reader unexpectedly suspended");
    };
    let lease = result.unwrap();
    compute(
        lease.as_slice::<f32>().unwrap().as_ptr(),
        query,
        ids,
        stride,
    )
}

type Run = fn(&ArrayView<f32>, &[f32], &[u32], usize) -> f32;

/// Repeats one benchmark arm for at least one second and reports elapsed nanoseconds per vector.
fn measure(run: Run, base: &ArrayView<f32>, query: &[f32], ids: &[u32], stride: usize) -> f64 {
    let started = Instant::now();
    let mut passes = 0;
    while started.elapsed() < Duration::from_secs(1) {
        for group in ids.chunks_exact(64) {
            black_box(run(
                black_box(base),
                black_box(query),
                black_box(group),
                black_box(stride),
            ));
        }
        passes += 1;
    }
    started.elapsed().as_secs_f64() * 1e9 / (passes * ids.len()) as f64
}

/// Compares direct and unified resident reads for SIFT- and GIST-width vectors without changing their kernels.
fn main() {
    println!("dimension,round,direct_ns_per_vector,unified_ns_per_vector,change_percent");
    for stride in [128, 960] {
        let rows = 16384;
        let values: Vec<f32> = (0..rows * stride + 32)
            .map(|i| ((i * 17 % 251) as f32) / 251.0)
            .collect();
        let base = ArrayView::copy_from(&values, &Default::default()).unwrap();
        let query = vec![0.375; stride];
        let mut ids: Vec<u32> = (0..rows as u32).collect();
        let mut state = 42u64;
        for i in (1..ids.len()).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ids.swap(i, (state >> 32) as usize % (i + 1));
        }
        for group in ids.chunks_exact(64) {
            assert_eq!(
                read_path_direct(&base, &query, group, stride),
                read_path_unified(&base, &query, group, stride)
            );
        }
        measure(read_path_direct, &base, &query, &ids, stride);
        measure(read_path_unified, &base, &query, &ids, stride);
        for round in 0..7 {
            let (direct, unified) = if round % 2 == 0 {
                (
                    measure(read_path_direct, &base, &query, &ids, stride),
                    measure(read_path_unified, &base, &query, &ids, stride),
                )
            } else {
                let unified = measure(read_path_unified, &base, &query, &ids, stride);
                (
                    measure(read_path_direct, &base, &query, &ids, stride),
                    unified,
                )
            };
            println!(
                "{stride},{},{direct:.4},{unified:.4},{:.4}",
                round + 1,
                (unified / direct - 1.0) * 100.0
            );
        }
    }
}

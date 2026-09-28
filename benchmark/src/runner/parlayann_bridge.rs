/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Load a ParlayANN-built Orion index (`.staged` v3) and pass the
//! `(local, remote, extra)` partitions through to `Orion::new`.
//!
//! File format (from `ParlayANN/algorithms/utils/staged_export.h`, v3):
//!   [u32 magic=0x53544147 "STAG"][u32 version=3]
//!   [u32 n][u32 max_deg][u32 max_extra][u32 entry_point]
//!   per node i:
//!     [u32 local_cnt][u32 remote_cnt][u32 extra_cnt]
//!     [u32×local_cnt local_ids]   (sorted by dist asc; bidir + promoted-remote)
//!     [u32×remote_cnt remote_ids] (sorted by dist asc; unidir not promoted)
//!     [u32×extra_cnt extra_ids]   (sorted by dist asc; pruned-candidate extras)
//!
//! v3 vs v2: PA C++ now does the bidir test + dist-merge (extras vs remote)
//! up-front and writes 3 partitions directly. The Rust side just reads
//! them through — no bidir lookup, no distance recompute.
//!
//! Entry point: the file header's `entry_point` is PA's hard-coded 0; we
//! recompute medoid over a stride sample instead.

use diskann::common::{ANNError, ANNResult};
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub type Partition = (Vec<u32>, Vec<u32>, Vec<u32>);

#[allow(dead_code)]
pub struct StagedInput {
    pub partitions: Vec<Partition>,
    pub entry_point: u32,
    pub num_nodes: usize,
    pub max_deg: u32,
    pub max_extra: u32,
}

const MAGIC: u32 = 0x53544147;
const VERSION: u32 = 3;

/// Read an u32 at the given byte offset from `buf`.
#[inline]
fn read_u32_at(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Read `n` u32s starting at `off` into a fresh `Vec<u32>` (little-endian).
#[inline]
fn read_u32_vec_at(buf: &[u8], off: usize, n: usize) -> Vec<u32> {
    let mut v = vec![0u32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    bytes.copy_from_slice(&buf[off..off + n * 4]);
    v
}

fn dist_l2<T: Copy + Into<f32>>(a: &[T], b: &[T]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x).into() - (*y).into();
        s += d * d;
    }
    s
}

pub fn load_from_staged_file<P: AsRef<Path>>(
    path: P,
    base_flat: &[f32],
    dim: usize,
) -> ANNResult<StagedInput> {
    // Slurp the whole file (.staged is ~ 1× graph size — fits comfortably
    // alongside the dataset). With the file in RAM we can do an O(n)
    // sequential offset-scan, then parse partitions in parallel via
    // `into_par_iter()`. Streaming reads can't parallelize since each
    // per-node block's start depends on previous block's sizes.
    let mut buf = Vec::new();
    File::open(&path)?.read_to_end(&mut buf)?;
    if buf.len() < 24 {
        return Err(ANNError::log_index_error(format!(
            "staged file too short ({} bytes)",
            buf.len()
        )));
    }

    let magic = read_u32_at(&buf, 0);
    if magic != MAGIC {
        return Err(ANNError::log_index_error(format!(
            "staged file magic mismatch: got 0x{:08x}, expected 0x{:08x}",
            magic, MAGIC
        )));
    }
    let version = read_u32_at(&buf, 4);
    if version != VERSION {
        return Err(ANNError::log_index_error(format!(
            "staged file version mismatch: got {}, expected {}",
            version, VERSION
        )));
    }
    let n = read_u32_at(&buf, 8) as usize;
    let max_deg = read_u32_at(&buf, 12);
    let max_extra = read_u32_at(&buf, 16);
    let _entry_hint = read_u32_at(&buf, 20);

    log::info!(
        "ParlayANN staged v3: n={}, max_deg={}, max_extra={}",
        n,
        max_deg,
        max_extra
    );

    assert_eq!(base_flat.len(), n * dim, "base_flat size mismatch");

    // Pass 1 (sequential, O(n)): scan per-node block offsets.
    // Each block: 3×u32 header + (lc+rc+ec)×4 bytes of IDs.
    let mut offsets: Vec<usize> = Vec::with_capacity(n);
    let mut p = 24usize;
    for _ in 0..n {
        offsets.push(p);
        let lc = read_u32_at(&buf, p) as usize;
        let rc = read_u32_at(&buf, p + 4) as usize;
        let ec = read_u32_at(&buf, p + 8) as usize;
        p += 12 + (lc + rc + ec) * 4;
    }

    // Pass 2 (parallel): parse each block at its offset.
    let partitions: Vec<Partition> = offsets
        .into_par_iter()
        .map(|off| {
            let lc = read_u32_at(&buf, off) as usize;
            let rc = read_u32_at(&buf, off + 4) as usize;
            let ec = read_u32_at(&buf, off + 8) as usize;
            let mut q = off + 12;
            let local = read_u32_vec_at(&buf, q, lc);
            q += lc * 4;
            let remote = read_u32_vec_at(&buf, q, rc);
            q += rc * 4;
            let extra = read_u32_vec_at(&buf, q, ec);
            (local, remote, extra)
        })
        .collect();

    let entry_point = sampled_medoid(base_flat, dim);

    let total_local: usize = partitions.iter().map(|(l, _, _)| l.len()).sum();
    let total_remote: usize = partitions.iter().map(|(_, r, _)| r.len()).sum();
    let total_extra: usize = partitions.iter().map(|(_, _, e)| e.len()).sum();
    log::info!(
        "ParlayANN bridge: entry={}, mean local={:.1}, remote={:.1}, extra={:.1}",
        entry_point,
        total_local as f64 / n as f64,
        total_remote as f64 / n as f64,
        total_extra as f64 / n as f64,
    );

    Ok(StagedInput {
        partitions,
        entry_point,
        num_nodes: n,
        max_deg,
        max_extra,
    })
}

/// Preserve the historical PA importer entry selection without owning partitions.
pub fn sampled_medoid<T: Copy + Sync + Into<f32>>(base_flat: &[T], dim: usize) -> u32 {
    let n = base_flat.len() / dim;
    // Recompute medoid: smallest sum-of-distances to a stride sample.
    let sample_size = 1024.min(n);
    let step = (n / sample_size).max(1);
    let sample: Vec<usize> = (0..sample_size).map(|k| k * step).collect();
    let candidate_stride = (n / 10_000).max(1);
    (0..n)
        .into_par_iter()
        .step_by(candidate_stride)
        .map(|i| {
            let i_vec = &base_flat[i * dim..(i + 1) * dim];
            let sum: f32 = sample
                .iter()
                .map(|&s| dist_l2(i_vec, &base_flat[s * dim..(s + 1) * dim]))
                .sum();
            (sum, i as u32)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, i)| i)
        .unwrap_or(0)
}

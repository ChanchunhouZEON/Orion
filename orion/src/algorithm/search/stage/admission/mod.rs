/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Admission stage
//!
//! The PQ-ranking distance metric. Every graph candidate that passes
//! the prefilter gets its admission distance computed; if that
//! distance is below the current PQ tail (`pq_worst`), it gets cmov-
//! compacted into the hop's staging buffer for batched merge into the
//! PQ.
//!
//! The admission stage is the **dominant DRAM consumer** on most
//! workloads — at GIST L=192 it accounts for ~70% of per-query cache
//! lines. The exact choice of admission tier defines the cascade's
//! shape:
//!
//! * **u8 L2** ([`l2u8::L2U8Admission`]) — direct kernel, 8 cache
//!   lines/vert on GIST. Cheapest admission compute; no sink-side
//!   reconstruction.
//!
//! * **u16 L2** ([`l2u16::L2U16Admission`]) — PA's `-quantize_bits 16`
//!   recipe. 15 cache lines per GIST vertex (2× the DRAM of u8) but
//!   richer per-cmp signal; PA's actual GIST production setting.
//!
//! * **L2 kernel trick** ([`l2kt::L2KTAdmission`]) — i8 base + per-
//!   vert `‖x‖²` slab; uses sdot for IP then reconstructs L2 via
//!   `‖q‖² + ‖x‖² − 2·IP`. Current `search_l2_u8_q` default.
//!
//! * **i8 MIPS** ([`mips_i8::MipsI8Admission`]) — `sdot` for unit-
//!   normalised data (glove). 1 cache line at low D.
//!
//! * **i16 MIPS** ([`mips_i16::MipsI16Admission`]) — twin of i8 for
//!   high-recall MIPS workloads.

pub mod ads_f32;
pub mod l2kt;
pub mod l2u16;
pub mod l2u8;
pub mod mips_i16;
pub mod mips_i8;

pub use ads_f32::AdsF32Admission;
pub use l2kt::L2KTAdmission;
pub use l2u8::L2U8Admission;
pub use l2u16::L2U16Admission;
pub use mips_i8::MipsI8Admission;
pub use mips_i16::MipsI16Admission;

use crate::model::Neighbor;

/// PQ admission distance kernel.
///
/// Stage is a **factory** for per-query [`AdmissionSession`]s. The
/// session owns its per-query state (padded SIMD query, reconstruction
/// constants like `q_norm_sq`, etc.) and exposes the hot-loop methods.
/// Trait shape is object-safe so the runner can build the cascade via
/// `Box<dyn AdmissionStage<N>>` without naming the concrete type.
pub trait AdmissionStage<const N: usize>: Send + Sync {
    /// Materialise a per-query session. Called once per search;
    /// drives the quantization + any reconstruction-constant setup
    /// (e.g. `‖q‖²` for kernel-trick).
    fn open<'a>(&'a self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'a>;
}

/// Per-query admission state. Created by [`AdmissionStage::open`];
/// lives for the duration of a single search call.
pub trait AdmissionSession: Send + Sync {
    /// Single-vertex distance computed via the same kernel as
    /// [`admit_stream`](Self::admit_stream). Used at search start to
    /// insert the entry vertex into the PQ before the main loop
    /// begins.
    fn entry_distance(&self, vid: u32) -> f32;

    /// Stream distances over `id_scratch`, write surviving
    /// `(id, distance)` pairs into `out` via cmov-compact. Returns
    /// the number of survivors written.
    ///
    /// The unified search loop owns the staging-buffer offset; this
    /// method writes into the slice starting at `out`.
    ///
    /// # Safety
    /// - `out` must point to at least `id_scratch.len()` `Neighbor`
    ///   slots (the upper bound for cmov-compact admits).
    /// - `lookahead_lines` should match `dstream_la_q()`.
    unsafe fn admit_stream(
        &self,
        id_scratch: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        lookahead_lines: usize,
    ) -> usize;
}

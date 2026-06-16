/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Three-stage cascade traits for unified beam search
//!
//! The legacy `search_l2_u8_q` / `search_mips_q` / `search_l2` etc.
//! all reimplement the same beam-search skeleton with different
//! distance kernels inlined at every stage. That made it impossible
//! to A/B test e.g. "JL Sparse → u16 admission → f32 rerank" (PA's
//! GIST recipe) without a copy-paste port of the entire beam loop.
//!
//! This module factors the cascade into **three composable stage
//! traits**, each in its own sub-module:
//!
//! * [`prefilter::PrefilterStage`] — optional cheap rejection tier.
//!   Compacts a candidate id list in place by computing a low-
//!   precision distance (JL Hamming, u8 adaptive, RaBitQ binary
//!   code, ...) and dropping anything above a running-mean threshold.
//!   `Option<&P>` at the dispatch site makes "no prefilter" a zero-
//!   cost no-op.
//!
//! * [`admission::AdmissionStage`] — the distance metric the beam
//!   ranks on. Computes per-candidate distance against the PQ tail,
//!   cmov-compacts survivors into a per-hop staging buffer. PA's
//!   reference uses **u16** here (`-quantize_bits 16`); our legacy
//!   path uses u8 with a kernel-trick reconstruction.
//!
//! * [`rerank::RerankStage`] — final precision pass on the top
//!   `k · rerank_factor` PQ entries. Either f32 truth (our default)
//!   or u16 truth (PA's recipe). Can be skipped if the admission
//!   stage already ranks at sufficient precision.
//!
//! Each sub-module holds the trait definition (`mod.rs`) plus one
//! file per concrete implementation:
//!
//! ```text
//! stage/
//! ├── mod.rs                  ← re-exports
//! ├── prefilter/
//! │   ├── mod.rs              ← PrefilterStage trait + NoPrefilter
//! │   ├── jl.rs               ← JL Sparse 1024-bit Hamming filter
//! │   └── ... (u8_adaptive, rabitq when added)
//! ├── admission/
//! │   ├── mod.rs              ← AdmissionStage trait
//! │   ├── l2u8.rs             ← direct u8 L2 distance
//! │   ├── l2u16.rs            ← u16 L2 distance (PA's recipe)
//! │   ├── l2kt.rs             ← i8 + kernel-trick reconstruction
//! │   ├── mips_i8.rs          ← i8 sdot for unit-normalised data
//! │   └── mips_i16.rs         ← i16 IP for high-recall MIPS
//! └── rerank/
//!     ├── mod.rs              ← RerankStage trait + NoRerank
//!     ├── f32_truth.rs        ← f32 base rerank (default)
//!     └── u16_truth.rs        ← u16 sidecar rerank (PA's recipe)
//! ```
//!
//! ## Dispatch
//!
//! All three traits are **statically dispatched** via generics —
//! never `dyn Trait`. The hot loop closures embed the stage methods
//! and DistanceStream<DistanceFn> kernels inline; virtual calls would
//! cost ~10 cy each per candidate (50-200 µs/query). The recipe enum
//! at the top of `search_unified` fires the appropriate
//! monomorphisation:
//!
//! ```ignore
//! match recipe {
//!     Recipe::SiftL2 => self.search_unified::<NoPrefilter, L2U8Admission, F32Rerank, N>(...),
//!     Recipe::GistPaAligned => self.search_unified::<JlPrefilter, L2U16Admission, F32Rerank, N>(...),
//!     // ...
//! }
//! ```
//!
//! Each branch monomorphises to a fully specialised beam loop with
//! all three kernels inlined. We expect ~5-7 active recipes, so the
//! code-bloat budget is ~20 KiB of compiled text — trivial.

pub mod admission;
pub mod prefilter;
pub mod rerank;

pub use admission::{AdmissionSession, AdmissionStage};
pub use prefilter::{NoPrefilter, PrefilterSession, PrefilterStage};
pub use rerank::RerankStage;

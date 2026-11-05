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
//! * [`Prefilter`] — optional cheap rejection tier.
//!   Compacts a candidate id list in place by computing a low-
//!   precision distance (JL Hamming, u8 adaptive, RaBitQ binary
//!   code, ...) and dropping anything above a running-mean threshold.
//!   `Option<&P>` at the dispatch site makes "no prefilter" a zero-
//!   cost no-op.
//!
//! * [`Admission`] — the distance metric the beam
//!   ranks on. Computes per-candidate distance against the PQ tail,
//!   cmov-compacts survivors into a per-hop staging buffer. PA's
//!   reference uses **u16** here (`-quantize_bits 16`); our legacy
//!   path uses u8 with a kernel-trick reconstruction.
//!
//! * [`Rerank`] — final precision pass on the top
//!   `k · rerank_factor` PQ entries. Either f32 truth (our default)
//!   or u16 truth (PA's recipe). Can be skipped if the admission
//!   stage already ranks at sufficient precision.
//!
//! Each stage has a private implementation contract and a public handle that
//! can refer to an Orion implementation. This boundary also applies when
//! `benchmark` is enabled.
//!
//! ```compile_fail,E0603
//! use orion_ann_core::algorithm::search::stage::admission::AdmissionStage;
//! ```
//! ```compile_fail,E0603
//! use orion_ann_core::algorithm::search::stage::prefilter::PrefilterStage;
//! ```
//! ```compile_fail,E0603
//! use orion_ann_core::algorithm::search::stage::rerank::RerankStage;
//! ```
//! ```compile_fail,E0404
//! use orion_ann_core::algorithm::search::stage::Admission;
//! struct Custom;
//! impl Admission<'static> for Custom {}
//! ```

pub mod admission;
pub mod prefilter;
pub mod rerank;

pub use admission::Admission;
#[cfg(feature = "benchmark")]
pub use admission::AdmissionQuery;
pub(crate) use admission::{AdmissionSession, AdmissionStage};
pub use prefilter::{NoPrefilter, Prefilter};
pub(crate) use prefilter::{PrefilterSession, PrefilterStage};
pub use rerank::Rerank;
pub(crate) use rerank::RerankStage;

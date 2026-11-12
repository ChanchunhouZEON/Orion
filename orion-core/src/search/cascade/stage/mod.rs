/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Three-stage cascade traits for unified beam search
//!
//! Search uses one graph traversal for f32 and native-u8 base rows. A recipe
//! selects three composable stages, each defined in its own submodule:
//!
//! * [`Prefilter`] — optional cheap rejection tier.
//!   Compacts a candidate id list in place by computing a low-
//!   precision distance (JL Hamming, u8 adaptive, RaBitQ binary
//!   code, ...) and dropping anything above a running-mean threshold.
//!   Its shared session operation maintains the threshold; `NoPrefilter`
//!   performs a strict no-op through the same session interface.
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
//! use orion_ann_core::search::cascade::stage::admission::AdmissionStage;
//! ```
//! ```compile_fail,E0603
//! use orion_ann_core::search::cascade::stage::prefilter::PrefilterStage;
//! ```
//! ```compile_fail,E0603
//! use orion_ann_core::search::cascade::stage::rerank::RerankStage;
//! ```
//! ```compile_fail,E0404
//! use orion_ann_core::search::cascade::stage::Admission;
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

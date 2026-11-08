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
//!
//! * **native u8**([`native_u8::NativeU8Admission`]) - u8 native kernel,
//!   only apply to large scale `SIFT` currently. Differ from **l2 u8** as
//!   it directly use original dataset in referenced form without introducing
//!   quantized dataset sidecar. The rest is all the same.

#[cfg(feature = "benchmark")]
pub mod ads_f32;
pub mod l2kt;
pub mod l2u16;
pub mod l2u8;
pub mod mips_i16;
pub mod mips_i8;
mod native_u8;

#[cfg(feature = "benchmark")]
pub use ads_f32::AdsF32Admission;
pub use l2kt::L2KTAdmission;
pub use l2u8::L2U8Admission;
mod pq;
pub use l2u16::L2U16Admission;
pub use mips_i8::MipsI8Admission;
pub use mips_i16::MipsI16Admission;
pub use native_u8::NativeU8Admission;
pub use pq::PqAdmission;

use crate::model::Neighbor;

mod contract;
pub(crate) use contract::{AdmissionSession, AdmissionStage};

/// An official admission stage.
pub type Admission<'a> = dyn AdmissionStage + 'a;
/// An official admission query session.
#[cfg(feature = "benchmark")]
pub type AdmissionQuery<'a> = dyn AdmissionSession + 'a;

#[cfg(feature = "benchmark")]
pub use crate::model::dataset::quantized_dataset::exact::reference::ExactAdmission;

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Quantized base-vector stores used as Stage-1 prefilter beams in the
//! staged search paths. Each submodule provides one quantization
//! family; the parent `model` module re-exports the public types so
//! callers can keep writing `crate::model::QuantizedDataset` etc.
//!
//! ## Members
//!
//! - [`quantized_dataset`] — generic `QuantizedDataset<Q, N>` over the
//!   [`QuantSpec`] trait. Three impls ship today: [`L2U8`] (SIFT/GIST
//!   L2-u8 prefilter), [`MipsI8`] (glove100 MIPS-Q i8 beam), and
//!   [`MipsI16`] (PA-style i16 beam). All three share the same slab
//!   layout, sidecar load/build, and NEON kernel dispatch flow.
//!
//! - [`rabitq_dataset`] — 1-bit-per-dim RaBitQ quantization (Gao &
//!   Long, SIGMOD 2024). Replaces the u8/i8 Stage-1 beam with a sign
//!   code per dimension after a random orthogonal rotation; targets
//!   high-dim datasets (GIST and above) where the bandwidth saving
//!   moves the entire quantized base into L1 cache. Work in progress.

pub mod jl_hadamard_dataset;
pub mod jl_sparse_dataset;
pub mod l2_kt_dataset;
pub mod quantized_dataset;
pub mod rabitq_b4_dataset;
pub mod rabitq_dataset;

pub use jl_hadamard_dataset::{JL_HADAMARD_MAGIC, JlHadamardDataset, jl_hadamard_stride};
pub use jl_sparse_dataset::{
    JL_SPARSE_MAGIC, JL_SPARSE_MIPS_MAGIC, JLSparseDataset, JLSparseDatasetMips, jl_sparse_stride,
};
pub use l2_kt_dataset::{L2_KT_MAGIC, L2KTDataset, l2_kt_stride};
pub use quantized_dataset::{
    L2U8, L2U16, MipsI8, MipsI16, QuantParamsL2, QuantParamsMips, QuantSpec, QuantizedDataset,
};
pub use rabitq_b4_dataset::{RABITQ_B4_MAGIC, RabitQ4Dataset, rabitq_b4_code_stride};

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Immutable scalar, projection and rotation sidecars with a common build and memory contract.

pub mod quantized_dataset;
// Stable import paths; each implementation lives under its representation family.
pub use quantized_dataset::{
    jl::hadamard as jl_hadamard_dataset, jl::sparse as jl_sparse_dataset,
    kernel_trick as l2_kt_dataset, rabitq::binary as rabitq_dataset,
    rabitq::multibit as rabitq_b4_dataset, scalar as scalar_dataset,
    scalar::spec as quantized_spec,
};

pub mod rows;

mod validation;

pub use jl_hadamard_dataset::{JL_HADAMARD_MAGIC, JlHadamardDataset, jl_hadamard_stride};
pub use jl_sparse_dataset::{
    JL_SPARSE_MAGIC, JL_SPARSE_MIPS_MAGIC, JLSparseDataset, JLSparseDatasetMips, jl_sparse_stride,
};
pub use l2_kt_dataset::{L2_KT_MAGIC, L2KTDataset, l2_kt_stride};
pub use quantized_dataset::{
    PreparedQuery, QuantizedDataset, QuantizedDatasetCompute, QuantizedDatasetShape, ResidentMemory,
};
pub use quantized_spec::{L2U8, L2U16, MipsI8, MipsI16, QuantParamsL2, QuantParamsMips, QuantSpec};
pub use rabitq_b4_dataset::{RABITQ_B4_MAGIC, RabitQ4Dataset, rabitq_b4_code_stride};
pub use scalar_dataset::ScalarQuantizedDataset;

#[doc(hidden)]
pub mod native_u8;

pub mod base;
pub mod store;

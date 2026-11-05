/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Immutable scalar, projection and rotation sidecars with a common build and memory contract.
pub mod jl_hadamard_dataset;
pub mod jl_sparse_dataset;
pub mod l2_kt_dataset;
pub mod quantized_dataset;
pub mod quantized_spec;
pub mod rabitq_b4_dataset;
pub mod rabitq_dataset;
pub mod rows;
pub mod scalar_dataset;
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

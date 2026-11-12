/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Common contract for immutable quantized and projected datasets.
//!
//! Estimates count payload spans (including row padding, transform tables
//! and norms). Allocator metadata, inline fields and build/query scratch are not
//! resident dataset payloads. Computation reuses the existing kernels; no trait
//! object is introduced into streaming search.
//!
//! ```
//! use orion_ann_core::{QuantizedDatasetCompute, QuantizedDatasetShape, QuantizedDataset, PreparedQuery};
//! use orion_ann_core::model::dataset::{ScalarQuantizedDataset, L2U8};
//! use orion_ann_core::{PaddedDataset, PaddedLayout};
//! use orion_ann_vector::L2F32Distance;
//!
//! let input_shape = QuantizedDatasetShape { rows: 2, dimension: 3 };
//! let expected = ScalarQuantizedDataset::<L2U8>::estimate_resident_memory(input_shape, &())?;
//! let base = PaddedDataset::from_packed(
//!     &[1.0f32, 2., 3., 4., 5., 6.], PaddedLayout::for_kernel::<L2F32Distance>(3)?,
//! )?;
//! let sidecar = ScalarQuantizedDataset::<L2U8>::build(&base, &())?;
//! assert_eq!(expected, sidecar.resident_memory());
//! let query = sidecar.prepare_query(&[1., 2., 3.])?;
//! assert_eq!(query.distance(0)?, 0.0);
//! # Ok::<(), orion_ann_core::Error>(())
//! ```
use crate::model::dataset::{rows::Rows, validation};
use crate::{Error, Result};

pub(crate) mod sealed {
    pub trait Sealed {}
}
#[doc(hidden)]
pub mod capabilities;
mod compute;
pub use capabilities::QuantizedKind;
pub(crate) mod exact;
pub mod jl;
pub mod kernel_trick;
pub mod pq;
pub mod rabitq;
pub(crate) mod residency;
pub mod scalar;

/// Logical input to quantized dataset construction: rows and original dimension.
/// This is not the encoded stride or the projected output width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedDatasetShape {
    pub rows: usize,
    pub dimension: usize,
}
impl QuantizedDatasetShape {
    pub(crate) fn validate(self) -> Result<()> {
        validation::shape(self.rows, self.dimension, u32::MAX as usize)
    }
}

/// Total dataset payload in bytes, including padding and transform storage.
/// For mapped payloads this counts the mapped span, not instantaneous RSS.
///
/// Allocator metadata, inline fields and build/query scratch are excluded.
/// Storage components are implementation details and are not part of this API.
///
/// ```compile_fail
/// fn components(memory: orion_ann_core::ResidentMemory) {
///     let _ = memory.code_bytes();
/// }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentMemory {
    total_bytes: usize,
}
impl ResidentMemory {
    pub(crate) fn new(codes: usize, auxiliary: usize, transforms: usize) -> Result<Self> {
        let total_bytes = codes
            .checked_add(auxiliary)
            .and_then(|v| v.checked_add(transforms))
            .ok_or_else(|| Error::invalid_argument("sidecar resident memory overflow"))?;
        Ok(Self { total_bytes })
    }
    /// Total allocated dataset payload; this is not a peak RSS estimate.
    pub fn total_bytes(self) -> usize {
        self.total_bytes
    }
}

/// Checked query preparation for an official dataset.
///
/// The returned opaque query remains bound to its encoder. Its storage and
/// unbound scoring operations are private implementation details.
///
/// ```compile_fail
/// use orion_ann_core::QuantizedDatasetCompute;
/// fn unbound<D: QuantizedDatasetCompute>(dataset: &D, values: &[f32]) {
///     let _ = dataset.encode_query_checked(values);
/// }
/// ```
/// ```compile_fail
/// use orion_ann_core::QuantizedDatasetCompute;
/// fn raw_query<D: QuantizedDatasetCompute>(_: <D as QuantizedDatasetCompute>::Query) {}
/// ```
pub trait QuantizedDatasetCompute: sealed::Sealed {
    /// Encode and bind a query to this dataset, without exposing its buffers.
    ///
    /// # Errors
    /// Rejects dimension mismatches and non-finite coordinates.
    fn prepare_query(&self, query: &[f32]) -> Result<impl PreparedQuery + '_>;
}

/// A prepared query whose encoder and encoded storage cannot be replaced.
/// Implemented only by the library's opaque query handles.
pub trait PreparedQuery: sealed::Sealed + std::fmt::Debug {
    /// Score one row in the original dataset.
    ///
    /// # Errors
    /// Rejects a row ID outside the dataset.
    fn distance(&self, id: u32) -> Result<f32>;
}

/// Build and account for any official sidecar, independently of cascade stages.
///
/// Runtime requirement registries and internal query storage are not public APIs.
/// ```compile_fail
/// use orion_ann_core::model::dataset::requirements::SidecarRequirement;
/// ```
/// ```compile_fail
/// use orion_ann_core::model::dataset::quantized_dataset::RotatedQuery;
/// ```
///
/// Implementations are sealed because padded storage, encoding and computation
/// must agree. No implementation may omit an owned resident buffer from its
/// estimate. Serialization is a separate internal contract.
pub trait QuantizedDataset: QuantizedDatasetCompute + Sized + Send + Sync {
    type BuildOptions: Clone + std::fmt::Debug + PartialEq + Eq;

    /// Canonical recipe used by the index's lazy sidecar slot.
    fn default_build_options() -> Self::BuildOptions;

    /// Build from finite logical f32 rows.
    ///
    /// # Errors
    /// Rejects invalid shape, options or coordinates; propagates allocation errors.
    fn build(base: &impl Rows, options: &Self::BuildOptions) -> Result<Self>;

    /// Estimate resident payload without reading or allocating the base.
    ///
    /// # Errors
    /// Rejects unsupported shapes/options and checked allocation-size overflow.
    fn estimate_resident_memory(
        input_shape: QuantizedDatasetShape,
        options: &Self::BuildOptions,
    ) -> Result<ResidentMemory>;

    /// Inspect actual allocated payload, including spare Vec capacity.
    fn resident_memory(&self) -> ResidentMemory;
}

/// Hadamard construction parameters. `None` uses the canonical padded width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HadamardOptions {
    pub seed: u64,
    pub padded_dimension: Option<usize>,
}
impl HadamardOptions {
    pub(crate) fn padded_dimension(self, dimension: usize) -> Result<usize> {
        let padded = match self.padded_dimension {
            Some(value) => value,
            None => dimension
                .max(1024)
                .checked_next_power_of_two()
                .ok_or_else(|| Error::invalid_argument("Hadamard dimension overflow"))?,
        };
        if !padded.is_power_of_two() || padded < dimension || padded > u32::MAX as usize {
            return Err(Error::invalid_argument(
                "Hadamard padding must be a representable power of two >= dimension",
            ));
        }
        Ok(padded)
    }
}

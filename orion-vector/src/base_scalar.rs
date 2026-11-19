/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Coordinate validity and physical row layouts for supported base scalar types.
//!
//! Query preparation and metric arithmetic are defined by the consuming search layer.
//! Selecting a layout here does not select a search metric or SIMD backend.

use crate::{DistanceKernelBackend, IpU8Distance, L2F32Distance};
use bytemuck::Pod;

pub(crate) mod sealed {
    /// Restricts stored base representations to the implementations audited here.
    pub trait Sealed {
        /// Keeps the established layout range without exposing a sizing kernel.
        const MAX_DIMENSION: usize;

        /// Supplies physical block width for a row of the given byte length.
        fn block_bytes(row_bytes: usize) -> usize;
    }
}

/// A supported base coordinate with validated, per-row padded storage.
///
/// This sealed trait describes f32, u8/u16/u32 and i8/i16/i32 storage, not query
/// types or cascade stages. Queries may use a different scalar type. Layout
/// construction does not allocate memory or choose the distance used by search.
/// Supporting a storage scalar does not imply that all builders, quantizers or
/// language bindings accept it; those entry points state their own capabilities.
///
/// ```
/// use orion_ann_vector::PaddedLayout;
/// let layout = PaddedLayout::for_base_scalar::<u8>(25)?;
/// assert_eq!(layout.logical_dim(), 25);
/// assert_eq!(layout.element_bytes(), 1);
/// assert!(layout.stride_bytes() >= 25);
/// # Ok::<(), String>(())
/// ```
pub trait BaseScalar: sealed::Sealed + Pod + Send + Sync + Into<f64> {
    /// Checks a stored coordinate before it enters an index or quantizer.
    ///
    /// All integer values are valid; floating coordinates must be finite.
    fn valid_coordinate(value: Self) -> bool;
}

// One table owns the sealed set, coordinate validation and layout mapping.
// A sizing kernel is an implementation detail, not a metric or backend choice.
macro_rules! define_base_scalars {
    ($($scalar:ty => ($layout:ty, $valid:expr)),+ $(,)?) => {$(
        impl sealed::Sealed for $scalar {
            const MAX_DIMENSION: usize = <$layout as DistanceKernelBackend>::MAX_COORDINATES;

            /// Retains the existing stream row width without selecting compute arithmetic.
            fn block_bytes(row_bytes: usize) -> usize {
                <$layout as DistanceKernelBackend>::stream_block_bytes(row_bytes)
            }
        }

        impl BaseScalar for $scalar {
            /// Applies this storage scalar's validity rule before index preparation.
            fn valid_coordinate(value: Self) -> bool {
                ($valid)(value)
            }
        }
    )+};
}

// u8 retains IP's wider layout limit instead of the legacy L2 accumulator bound.
// i32/u32 share f32's four-byte row sizing, without converting stored values.
define_base_scalars! {
    f32 => (L2F32Distance, f32::is_finite),
    u8 => (IpU8Distance, |_| true),
    i8 => (crate::IpI8Distance, |_| true),
    i16 => (crate::IpI16Distance, |_| true),
    i32 => (L2F32Distance, |_| true),
    u16 => (crate::L2U16Distance, |_| true),
    u32 => (L2F32Distance, |_| true),
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Borrowed logical coordinates, independent of base storage and stage encoding.
use crate::{Error, Result, ScoreMetric};
use std::{borrow::Cow, ops::Range};

/// A borrowed coordinate view plus a query-only normalization request.
/// Original coordinates remain available for exact output scores. Slices inherit
/// the request; no batch-wide normalized copy is allocated.
#[derive(Clone, Copy)]
pub struct QueryInput<'a> {
    values: QueryCoordinates<'a>,
    unit_norm: bool,
}

mod sealed {
    pub trait Sealed {}
}

/// Supported input scalars. Sealed so validation and conversion remain audited.
pub trait QueryElement: sealed::Sealed + Copy + Send + Sync + 'static {
    fn view(values: &[Self]) -> QueryInput<'_>;
    fn owned(values: Vec<Self>) -> QueryBuffer;
}

// One scalar table defines transport variants and their mechanical operations.
// Keep numerical policy (validation, lossless conversion and normalization)
// outside the macro so its safety/precision decisions remain directly readable.
// Expansion produces ordinary matches; it adds no dispatch inside SIMD kernels.
macro_rules! define_query_elements {
    ($float:ident: $float_ty:ty; $($variant:ident: $scalar:ty),+ $(,)?) => {
        /// Original coordinates, borrowed without conversion or padding.
        #[derive(Clone, Copy)]
        enum QueryCoordinates<'a> {
            $float(&'a [$float_ty]),
            $($variant(&'a [$scalar]),)+
        }

        /// Owned stream/FFI transport, not padded or quantized query storage.
        /// Borrow it with `as_input` for the duration of an operation.
        #[derive(Clone)]
        pub enum QueryBuffer {
            $float(Vec<$float_ty>),
            $($variant(Vec<$scalar>),)+
        }

        define_query_elements!(@impl $float: $float_ty, $($variant: $scalar),+);

        impl<'a> QueryInput<'a> {
            /// Approximate transport for lossy encoders. Exact scoring does not
            /// narrow u32 coordinates that cannot be represented exactly as f32.
            pub fn to_f32(self) -> Cow<'a, [f32]> {
                match self.values {
                    QueryCoordinates::$float(q) => Cow::Borrowed(q),
                    $(QueryCoordinates::$variant(q) =>
                        Cow::Owned(q.iter().map(|&x| x as f32).collect()),)+
                }
            }
        }
    };
    (@impl $($variant:ident: $scalar:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $scalar {}
            impl QueryElement for $scalar {
                fn view(values: &[Self]) -> QueryInput<'_> { QueryInput::$variant(values) }
                fn owned(values: Vec<Self>) -> QueryBuffer { QueryBuffer::$variant(values) }
            }
            impl<'a> From<&'a [$scalar]> for QueryInput<'a> {
                fn from(values: &'a [$scalar]) -> Self { Self::$variant(values) }
            }
        )+
        impl<'a> QueryInput<'a> {
            $(
                // Retain the existing QueryInput::F32 / U8 construction syntax.
                #[allow(non_snake_case)]
                pub fn $variant(values: &'a [$scalar]) -> Self {
                    Self { values: QueryCoordinates::$variant(values), unit_norm: false }
                }
            )+
            pub fn len(self) -> usize {
                match self.values { $(QueryCoordinates::$variant(q) => q.len(),)+ }
            }
            pub fn element_type(self) -> &'static str {
                match self.values { $(QueryCoordinates::$variant(_) => stringify!($scalar),)+ }
            }
            pub fn is_zero(self) -> bool {
                match self.values {
                    $(QueryCoordinates::$variant(q) => q.iter().all(|&x| x == 0 as $scalar),)+
                }
            }
            /// Borrow a coordinate range; callers validate packed shape first.
            /// Panics if the range is reversed or exceeds the input length.
            pub fn slice(self, range: Range<usize>) -> Self {
                let values = match self.values {
                    $(QueryCoordinates::$variant(q) => QueryCoordinates::$variant(&q[range]),)+
                };
                Self { values, unit_norm: self.unit_norm }
            }
            fn to_f64(self) -> Vec<f64> {
                match self.values {
                    $(QueryCoordinates::$variant(q) => q.iter().map(|&x| f64::from(x)).collect(),)+
                }
            }
            /// Original-coordinate scoring: integer inputs reach f64 without
            /// narrowing; the public score retains the existing f32 contract.
            pub(crate) fn score<T: Copy + Into<f32>>(self, row: &[T], metric: ScoreMetric) -> f32 {
                match self.values {
                    $(QueryCoordinates::$variant(q) => super::numeric::score(q, row, metric),)+
                }
            }
        }
        impl QueryBuffer {
            pub fn as_input(&self) -> QueryInput<'_> {
                match self { $(Self::$variant(q) => QueryInput::$variant(q),)+ }
            }
            pub fn clear(&mut self) {
                match self { $(Self::$variant(q) => q.clear(),)+ }
            }
            /// Append a batch of the same dtype; a mismatch leaves self intact.
            pub fn extend(&mut self, other: &Self) -> Result<()> {
                match (self, other) {
                    $((Self::$variant(a), Self::$variant(b)) => a.extend_from_slice(b),)+
                    _ => return Err(Error::invalid_argument("query stream changed element type")),
                }
                Ok(())
            }
        }
    };
}

define_query_elements! { F32: f32; U8: u8, I8: i8, U16: u16, U32: u32 }

impl QueryInput<'_> {
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl<'a> QueryInput<'a> {
    /// Request per-row normalization for stage transport, without changing the
    /// original coordinates used for returned metric scores.
    pub fn with_unit_norm(mut self) -> Self {
        self.unit_norm = true;
        self
    }
    pub(crate) fn needs_unit_norm(self) -> bool {
        self.unit_norm
    }
    pub fn validate(self) -> Result<()> {
        if let QueryCoordinates::F32(q) = self.values
            && q.iter().any(|v| !v.is_finite())
        {
            return Err(Error::invalid_argument(
                "query contains a non-finite coordinate",
            ));
        }
        Ok(())
    }
    pub(crate) fn lossless_f32(self) -> bool {
        match self.values {
            QueryCoordinates::U32(q) => q.iter().all(|&x| f64::from(x as f32) == f64::from(x)),
            _ => true,
        }
    }
    pub(crate) fn to_u8(self) -> Option<Cow<'a, [u8]>> {
        match self.values {
            QueryCoordinates::U8(q) => Some(Cow::Borrowed(q)),
            QueryCoordinates::F32(q) => q
                .iter()
                .all(|&x| x.is_finite() && (0.0..=255.0).contains(&x) && x.fract() == 0.0)
                .then(|| Cow::Owned(q.iter().map(|&x| x as u8).collect())),
            QueryCoordinates::I8(q) => q
                .iter()
                .all(|&x| x >= 0)
                .then(|| Cow::Owned(q.iter().map(|&x| x as u8).collect())),
            QueryCoordinates::U16(q) => q
                .iter()
                .all(|&x| x <= 255)
                .then(|| Cow::Owned(q.iter().map(|&x| x as u8).collect())),
            QueryCoordinates::U32(q) => q
                .iter()
                .all(|&x| x <= 255)
                .then(|| Cow::Owned(q.iter().map(|&x| x as u8).collect())),
        }
    }
    /// Shared wide compute representation for calibration and exact sessions.
    pub(crate) fn prepared_wide(self) -> Result<Vec<f64>> {
        let mut values = self.to_f64();
        if self.needs_unit_norm() {
            let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm == 0.0 || !norm.is_finite() {
                return Err(Error::invalid_argument(
                    "cosine requires finite nonzero queries",
                ));
            }
            for value in &mut values {
                *value /= norm;
            }
        }
        Ok(values)
    }
}
impl QueryBuffer {
    pub fn len(&self) -> usize {
        self.as_input().len()
    }
    pub fn is_empty(&self) -> bool {
        self.as_input().is_empty()
    }
}
impl Default for QueryBuffer {
    fn default() -> Self {
        Self::F32(Vec::new())
    }
}
impl std::fmt::Debug for QueryInput<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryInput")
            .field("type", &self.element_type())
            .field("elements", &self.len())
            .finish()
    }
}
impl std::fmt::Debug for QueryBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_input().fmt(f)
    }
}

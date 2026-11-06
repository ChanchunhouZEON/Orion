/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Representation capabilities, separate from storage width and search position.

/// Official auxiliary representations. Build options belong to requirements;
/// this identifier alone does not identify a fitted dataset or a cache object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantizedKind {
    Pq,
    L2U8,
    L2U16,
    MipsI8,
    MipsI16,
    L2Kt,
    JlSparse,
    JlSparseMips,
    JlHadamard,
    Rabitq,
    Rabitq4,
}
impl QuantizedKind {
    pub(crate) fn is_sketch(self) -> bool {
        matches!(
            self,
            Self::JlSparse | Self::JlSparseMips | Self::JlHadamard | Self::Rabitq
        )
    }
    fn is_mips(self) -> bool {
        matches!(self, Self::JlSparseMips | Self::MipsI8 | Self::MipsI16)
    }
}

/// Internal scoring precision. Native u8 is exact, despite using one byte per
/// coordinate. No total ordering is assigned to unrelated quantizers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum Precision {
    Exact,
    ProductQuantized { inner_product: bool },
    Quantized(QuantizedKind),
}

impl Precision {
    fn follows(self, previous: Self) -> bool {
        use QuantizedKind::*;
        match (previous, self) {
            (_, Self::Exact) => true,
            (Self::Exact, _) => false,
            (Self::Quantized(k), Self::ProductQuantized { inner_product }) => {
                k.is_sketch() && k.is_mips() == inner_product
            }
            (
                Self::ProductQuantized { inner_product: a },
                Self::ProductQuantized { inner_product: b },
            ) => a == b,
            (Self::ProductQuantized { .. }, _) => false,
            (Self::Quantized(a), Self::Quantized(b)) => {
                if a.is_mips() != b.is_mips() {
                    return false;
                }
                a == b
                    || (a.is_sketch() && !b.is_sketch())
                    || matches!((a, b), (L2U8 | L2Kt, L2U16) | (MipsI8, MipsI16))
            }
        }
    }
}

#[doc(hidden)]
pub fn validate_precision(
    prefilter: Option<Precision>,
    admission: Precision,
    rerank: Option<Precision>,
) -> crate::Result<()> {
    if let Some(p) = prefilter
        && !admission.follows(p)
    {
        return Err(crate::Error::invalid_argument(format!(
            "cascade precision cannot refine prefilter {p:?} into admission {admission:?}"
        )));
    }
    if let Some(r) = rerank {
        if matches!(r, Precision::Quantized(k) if k.is_sketch()) {
            return Err(crate::Error::invalid_argument(
                "binary sketches cannot be used for reranking",
            ));
        }
        if !r.follows(admission) {
            return Err(crate::Error::invalid_argument(format!(
                "cascade precision cannot refine admission {admission:?} into rerank {r:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precision_is_a_partial_order_not_a_storage_bit_count() {
        use Precision::{Exact, Quantized as Q};
        use QuantizedKind::*;
        for (p, a, r) in [
            (Some(Q(JlSparse)), Q(L2Kt), Some(Exact)),
            (None, Q(L2U8), Some(Q(L2U16))),
            (None, Q(L2U16), Some(Q(L2U16))),
            (Some(Q(JlSparseMips)), Q(MipsI8), Some(Q(MipsI16))),
            (None, Exact, None),
        ] {
            validate_precision(p, a, r).unwrap();
        }
        for (p, a, r) in [
            (None, Exact, Some(Q(L2U16))),
            (None, Q(L2U16), Some(Q(L2U8))),
            (None, Q(Rabitq4), Some(Q(L2U8))),
            (Some(Q(JlSparseMips)), Q(L2U8), None),
            (None, Q(JlSparse), Some(Q(JlHadamard))),
            (None, Q(Rabitq), Some(Q(Rabitq))),
        ] {
            assert!(validate_precision(p, a, r).is_err(), "{p:?} {a:?} {r:?}");
        }
    }
}

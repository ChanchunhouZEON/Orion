/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Metric and precision-order compatibility, independent of storage planning.
use super::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice, SearchMetric};
use crate::Result;

impl Cascade {
    /// Validates that all cascade stages are compatible with `metric`.
    ///
    /// This checks semantic compatibility rather than tuning quality:
    ///
    /// - L2 metrics must use L2-oriented admission and rerank stages.
    /// - Inner-product / cosine metrics must use MIPS-oriented stages.
    /// - Prefilters without a MIPS-aware adapter are rejected for
    ///   inner-product-style search.
    ///
    /// `RerankChoice::None` is valid for either metric family because it simply
    /// disables the final precision stage.
    pub fn validate_cascade_options(self, metric: SearchMetric) -> Result<()> {
        // Cosine shares the same stage family as inner-product search; any
        // normalization needed for cosine is handled outside this compatibility
        // check.
        let ip = metric != SearchMetric::L2;

        // These prefilters currently assume an L2-compatible geometry and do
        // not provide the adapter required by the MIPS/cosine pipeline.
        if ip
            && matches!(
                self.prefilter,
                PrefilterChoice::JlHadamard | PrefilterChoice::Rabitq
            )
        {
            return Err("this prefilter has no MIPS-aware adapter; use none or jl".into());
        }

        // Admission implementations are metric-specific. Mixing an L2
        // admission rule with an IP/cosine objective (or vice versa) would make
        // calibrated thresholds semantically invalid.
        if matches!(
            self.admission,
            AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16 | AdmissionChoice::PqIp
        ) != ip
        {
            return Err("admission does not match --metric; override --admission or choose the correct metric".into());
        }

        // Reranking may be disabled entirely; otherwise its exact-distance
        // implementation must belong to the same metric family.
        let rerank_matches = match self.rerank {
            RerankChoice::None => true,
            RerankChoice::IpF32 => ip,
            RerankChoice::F32 | RerankChoice::U16 => !ip,
        };

        if !rerank_matches {
            return Err("rerank does not match --metric; override --rerank".into());
        }
        use orion_ann_core::model::dataset::quantized_dataset::{
            capabilities::{validate_precision, Precision as P},
            QuantizedKind as K,
        };
        let prefilter = match self.prefilter {
            PrefilterChoice::None => None,
            PrefilterChoice::Jl => {
                Some(P::Quantized(if ip { K::JlSparseMips } else { K::JlSparse }))
            }
            PrefilterChoice::JlHadamard => Some(P::Quantized(K::JlHadamard)),
            PrefilterChoice::Rabitq => Some(P::Quantized(K::Rabitq)),
        };
        let admission = match self.admission {
            AdmissionChoice::PqL2 => P::ProductQuantized {
                inner_product: false,
            },
            AdmissionChoice::PqIp => P::ProductQuantized {
                inner_product: true,
            },
            AdmissionChoice::L2U8 => P::Quantized(K::L2U8),
            AdmissionChoice::L2U16 => P::Quantized(K::L2U16),
            AdmissionChoice::L2Kt => P::Quantized(K::L2Kt),
            AdmissionChoice::MipsI8 => P::Quantized(K::MipsI8),
            AdmissionChoice::MipsI16 => P::Quantized(K::MipsI16),
            AdmissionChoice::NativeL2U8 => P::Exact,
            #[cfg(feature = "benchmark")]
            AdmissionChoice::AdsF32 => P::Exact,
        };
        let rerank = match self.rerank {
            RerankChoice::None => None,
            RerankChoice::F32 | RerankChoice::IpF32 => Some(P::Exact),
            RerankChoice::U16 => Some(P::Quantized(K::L2U16)),
        };
        validate_precision(prefilter, admission, rerank)
    }
}

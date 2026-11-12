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
    /// - Inner-product and cosine require their own semantic arms, even though
    ///   both use IP arithmetic internally.
    /// - Prefilters without a MIPS-aware adapter are rejected for
    ///   inner-product-style search.
    ///
    /// `RerankChoice::None` is valid for either metric family because it simply
    /// disables the final precision stage.
    pub fn validate_cascade_options(self, metric: SearchMetric) -> Result<()> {
        let ip = metric != SearchMetric::SquaredL2;
        if self.admission.metric() != metric {
            return Err(crate::Error::configuration(
                "admission",
                format!(
                    "{} requires metric {:?}, but configured metric is {metric:?}; cosine uses cosine-i8, cosine-i16, pq-cosine or native-cosine-u8",
                    self.admission,
                    self.admission.metric(),
                ),
            ));
        }

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

        // Reranking may be disabled entirely; otherwise its exact-distance
        // implementation must belong to the same metric family.
        let rerank_matches = match self.rerank {
            RerankChoice::None => true,
            RerankChoice::IpF32 => metric == SearchMetric::InnerProduct,
            RerankChoice::CosineF32 => metric == SearchMetric::Cosine,
            RerankChoice::F32 | RerankChoice::U16 => !ip,
        };

        if !rerank_matches {
            return Err(crate::Error::configuration(
                "rerank",
                format!(
                    "{} does not match metric {metric:?}; choose l2-f32, ip-f32 or cosine-f32 for the corresponding metric",
                    self.rerank
                ),
            ));
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
            AdmissionChoice::PqIp | AdmissionChoice::PqCosine => P::ProductQuantized {
                inner_product: true,
            },
            AdmissionChoice::L2U8 => P::Quantized(K::L2U8),
            AdmissionChoice::L2U16 => P::Quantized(K::L2U16),
            AdmissionChoice::L2Kt => P::Quantized(K::L2Kt),
            AdmissionChoice::IpI8 | AdmissionChoice::CosineI8 => P::Quantized(K::MipsI8),
            AdmissionChoice::IpI16 | AdmissionChoice::CosineI16 => P::Quantized(K::MipsI16),
            AdmissionChoice::NativeL2U8
            | AdmissionChoice::NativeIpU8
            | AdmissionChoice::NativeCosineU8 => P::Exact,
            #[cfg(feature = "benchmark")]
            AdmissionChoice::AdsF32 => P::Exact,
        };
        let rerank = match self.rerank {
            RerankChoice::None => None,
            RerankChoice::F32 | RerankChoice::IpF32 | RerankChoice::CosineF32 => Some(P::Exact),
            RerankChoice::U16 => Some(P::Quantized(K::L2U16)),
        };
        validate_precision(prefilter, admission, rerank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arms_enforce_objective_independently_of_shared_kernels() {
        use AdmissionChoice as A;
        use RerankChoice as R;
        use SearchMetric as M;
        for (metric, admissions, rerank) in [
            (
                M::SquaredL2,
                vec![A::L2U8, A::L2U16, A::L2Kt, A::PqL2],
                R::F32,
            ),
            (M::InnerProduct, vec![A::IpI8, A::IpI16, A::PqIp], R::IpF32),
            (
                M::Cosine,
                vec![A::CosineI8, A::CosineI16, A::PqCosine],
                R::CosineF32,
            ),
        ] {
            for admission in admissions {
                for configured in [M::SquaredL2, M::InnerProduct, M::Cosine] {
                    for selected_rerank in [R::F32, R::IpF32, R::CosineF32, R::None] {
                        let cascade = Cascade {
                            prefilter: PrefilterChoice::None,
                            admission,
                            rerank: selected_rerank,
                        };
                        assert_eq!(
                            cascade.validate_cascade_options(configured).is_ok(),
                            configured == metric
                                && (selected_rerank == rerank || selected_rerank == R::None),
                            "{cascade:?}, metric={configured:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cosine_reuses_storage_without_hiding_normalization() {
        for (ip, cosine) in [
            (AdmissionChoice::IpI8, AdmissionChoice::CosineI8),
            (AdmissionChoice::IpI16, AdmissionChoice::CosineI16),
            (AdmissionChoice::PqIp, AdmissionChoice::PqCosine),
        ] {
            for prefilter in [PrefilterChoice::None, PrefilterChoice::Jl] {
                let raw = Cascade {
                    prefilter,
                    admission: ip,
                    rerank: RerankChoice::IpF32,
                };
                let normalized = Cascade {
                    prefilter,
                    admission: cosine,
                    rerank: RerankChoice::CosineF32,
                };
                assert_eq!(
                    raw.dataset_requirements(),
                    normalized.dataset_requirements()
                );
                normalized
                    .validate_cascade_options(SearchMetric::Cosine)
                    .unwrap();
                let report = normalized.dispatch_description(crate::VectorStorageKind::F32);
                assert!(report.contains("L2-normalized"));
                assert!(report.contains("cosine-f32"));
                assert!(report.contains("cosine similarity, higher is better"));
            }
        }
    }
}

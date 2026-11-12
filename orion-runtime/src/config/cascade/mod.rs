/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Cascade defaults and override composition. Parsing, compatibility and
//! dataset dependencies are separate concerns in the sibling modules.
mod choices;
mod requirements;
mod validation;

pub use super::data::SearchMetric;
use crate::Result;
pub use choices::*;
use serde::{Deserialize, Serialize};

/// Overrides use the same vocabulary and validation as the CLI cascade.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CascadeConfig {
    /// Prefilter identifier in the shared CLI vocabulary.
    pub prefilter: Option<String>,
    /// Admission identifier compatible with metric and resident storage.
    pub admission: Option<String>,
    /// Reranking identifier; native exact admission defaults to none.
    pub rerank: Option<String>,
}

/// Dimension at which the default L2 cascade enables the JL prefilter.
///
/// This is a practical default heuristic rather than an optimality claim for
/// arbitrary datasets or vector distributions.
pub const HIGH_DIMENSION_L2: usize = 512;

/// Three-stage precision cascade used during graph search.
///
/// Candidate evaluation proceeds, when enabled, through:
///
/// 1. [`PrefilterChoice`] — a cheap rejection stage,
/// 2. [`AdmissionChoice`] — the primary approximate / reduced-precision test,
/// 3. [`RerankChoice`] — an optional higher-precision final evaluation.
///
/// The exact stage implementations depend on the configured [`SearchMetric`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Cascade {
    pub prefilter: PrefilterChoice,
    pub admission: AdmissionChoice,
    pub rerank: RerankChoice,
}

impl Cascade {
    /// Chooses the default cascade for a given dimensionality and metric.
    ///
    /// For L2, higher-dimensional vectors enable the JL prefilter by default;
    /// lower-dimensional vectors skip prefiltering and use byte-based L2
    /// admission directly.
    ///
    /// Inner-product and cosine use distinct semantic arms while sharing IP
    /// kernels. Cosine prepares normalized base/query rows before quantization. Dimension alone is not used to enable an additional
    /// prefilter because it does not establish sketch quality for MIPS.
    pub fn default_for_specified_dimension_and_metric(
        dimension: usize,
        metric: SearchMetric,
    ) -> Self {
        match metric {
            SearchMetric::SquaredL2 if dimension >= HIGH_DIMENSION_L2 => Self {
                prefilter: PrefilterChoice::Jl,
                admission: AdmissionChoice::L2Kt,
                rerank: RerankChoice::F32,
            },
            SearchMetric::SquaredL2 => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::L2U8,
                rerank: RerankChoice::F32,
            },
            // Dimension alone does not establish sketch quality for MIPS.
            SearchMetric::InnerProduct => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::IpI8,
                rerank: RerankChoice::IpF32,
            },
            SearchMetric::Cosine => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::CosineI8,
                rerank: RerankChoice::CosineF32,
            },
        }
    }

    #[cfg(any(feature = "cli", feature = "benchmark"))]
    pub fn label(self) -> String {
        format!(
            "{} -> {} -> {}",
            self.prefilter, self.admission, self.rerank
        )
    }
}

impl Cascade {
    pub(crate) fn native_u8(metric: SearchMetric) -> Self {
        Self {
            prefilter: PrefilterChoice::None,
            admission: match metric {
                SearchMetric::SquaredL2 => AdmissionChoice::NativeL2U8,
                SearchMetric::InnerProduct => AdmissionChoice::NativeIpU8,
                SearchMetric::Cosine => AdmissionChoice::NativeCosineU8,
            },
            rerank: RerankChoice::None,
        }
    }

    pub fn with_overrides(
        mut self,
        prefilter: Option<PrefilterChoice>,
        admission: Option<AdmissionChoice>,
        rerank: Option<RerankChoice>,
    ) -> Self {
        if let Some(value) = prefilter {
            self.prefilter = value;
        }
        if let Some(value) = admission {
            self.admission = value;
        }
        if let Some(value) = rerank {
            self.rerank = value;
        }
        self
    }

    /// Parse each supplied axis before composition, including overridden YAML
    /// values. This preserves the driver's fail-fast handling of invalid presets.
    pub fn with_named_overrides(
        self,
        prefilter: Option<&str>,
        admission: Option<&str>,
        rerank: Option<&str>,
    ) -> Result<Self> {
        Ok(self.with_overrides(
            prefilter.map(str::parse).transpose()?,
            admission.map(str::parse).transpose()?,
            rerank.map(str::parse).transpose()?,
        ))
    }
}

impl CascadeConfig {
    pub(crate) fn apply(&self, defaults: Cascade) -> Result<Cascade> {
        defaults.with_named_overrides(
            self.prefilter.as_deref(),
            self.admission.as_deref(),
            self.rerank.as_deref(),
        )
    }
}

impl Cascade {
    /// Describe preparation and arithmetic without adding another stored plan.
    pub(crate) fn dispatch_description(self, storage: super::VectorStorageKind) -> String {
        let normalization = match (self.admission.metric(), storage) {
            (SearchMetric::Cosine, super::VectorStorageKind::F32) => {
                "L2-normalized f32 base/query; quantized norms are not corrected"
            }
            (SearchMetric::Cosine, super::VectorStorageKind::U8) => {
                "native u8 coordinates preserved; sidecars/query views are L2-normalized; exact cosine uses squared norms"
            }
            _ => "coordinates preserved; no normalization",
        };
        let kernel = match self.admission {
            AdmissionChoice::IpI8 | AdmissionChoice::CosineI8 => "IpI8Distance",
            AdmissionChoice::IpI16 | AdmissionChoice::CosineI16 => "IpI16Distance",
            AdmissionChoice::PqIp | AdmissionChoice::PqCosine => "PQ negative-IP ADC",
            AdmissionChoice::PqL2 => "PQ squared-L2 ADC",
            AdmissionChoice::L2U8 | AdmissionChoice::NativeL2U8 => "L2U8Distance",
            AdmissionChoice::L2U16 => "L2U16Distance",
            AdmissionChoice::L2Kt => "IpI8Distance + squared-L2 reconstruction",
            AdmissionChoice::NativeIpU8 | AdmissionChoice::NativeCosineU8 => "IpU8Distance",
            #[cfg(feature = "benchmark")]
            AdmissionChoice::AdsF32 => "ADSampling f32 L2",
        };
        let scores = match self.admission.metric() {
            SearchMetric::SquaredL2 => "squared L2, lower is better",
            SearchMetric::InnerProduct => "inner product, higher is better",
            SearchMetric::Cosine => "cosine similarity, higher is better",
        };
        format!(
            "{} → {} → {}\nPreparation: {normalization}\nAdmission kernel: {kernel} (runtime SIMD dispatch)\nRequested exact result scores: {scores}",
            self.prefilter, self.admission, self.rerank
        )
    }
}

#[cfg(test)]
mod selector_tests {
    use super::*;

    #[test]
    #[cfg_attr(
        not(feature = "benchmark"),
        expect(
            clippy::single_element_loop,
            reason = "the benchmark feature adds exact f32 admission to this test matrix"
        )
    )]
    fn exact_storage_cannot_be_followed_by_quantized_reranking() {
        for admission in [
            #[cfg(feature = "benchmark")]
            AdmissionChoice::AdsF32,
            AdmissionChoice::NativeL2U8,
        ] {
            let cascade = Cascade {
                prefilter: PrefilterChoice::None,
                admission,
                rerank: RerankChoice::U16,
            };
            assert!(cascade
                .validate_cascade_options(SearchMetric::SquaredL2)
                .unwrap_err()
                .to_string()
                .contains("precision"));
        }
        let shared = Cascade {
            prefilter: PrefilterChoice::None,
            admission: AdmissionChoice::L2U16,
            rerank: RerankChoice::U16,
        };
        shared
            .validate_cascade_options(SearchMetric::SquaredL2)
            .unwrap();
        let requirements = shared.dataset_requirements();
        assert_eq!(requirements[0], requirements[1]);
    }

    #[test]
    fn dimension_and_metric_select_independent_defaults() {
        let low = Cascade::default_for_specified_dimension_and_metric(128, SearchMetric::SquaredL2);
        let high =
            Cascade::default_for_specified_dimension_and_metric(960, SearchMetric::SquaredL2);
        assert_eq!(low.admission, AdmissionChoice::L2U8);
        assert_eq!(high.admission, AdmissionChoice::L2Kt);
        assert_eq!(high.prefilter, PrefilterChoice::Jl);
        assert_eq!(
            Cascade::default_for_specified_dimension_and_metric(511, SearchMetric::SquaredL2)
                .prefilter,
            PrefilterChoice::None
        );
        assert_eq!(
            Cascade::default_for_specified_dimension_and_metric(512, SearchMetric::SquaredL2)
                .prefilter,
            PrefilterChoice::Jl
        );
        for metric in [SearchMetric::InnerProduct, SearchMetric::Cosine] {
            let c = Cascade::default_for_specified_dimension_and_metric(1536, metric);
            assert_eq!(c.prefilter, PrefilterChoice::None);
            assert_eq!(
                c.admission,
                if metric == SearchMetric::Cosine {
                    AdmissionChoice::CosineI8
                } else {
                    AdmissionChoice::IpI8
                }
            );
            assert_eq!(
                c.rerank,
                if metric == SearchMetric::Cosine {
                    RerankChoice::CosineF32
                } else {
                    RerankChoice::IpF32
                }
            );
            c.validate_cascade_options(metric).unwrap();
        }
        assert!(low
            .validate_cascade_options(SearchMetric::InnerProduct)
            .is_err());
        assert!(high
            .validate_cascade_options(SearchMetric::SquaredL2)
            .is_ok());
    }
}

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
    /// Reranking identifier; native u8 uses none.
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
    /// Inner-product and cosine search share the MIPS-oriented admission and
    /// rerank path. Dimension alone is not used to enable an additional
    /// prefilter because it does not establish sketch quality for MIPS.
    pub fn default_for_specified_dimension_and_metric(
        dimension: usize,
        metric: SearchMetric,
    ) -> Self {
        match metric {
            SearchMetric::L2 if dimension >= HIGH_DIMENSION_L2 => Self {
                prefilter: PrefilterChoice::Jl,
                admission: AdmissionChoice::L2Kt,
                rerank: RerankChoice::F32,
            },
            SearchMetric::L2 => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::L2U8,
                rerank: RerankChoice::F32,
            },
            // Dimension alone does not establish sketch quality for MIPS.
            SearchMetric::InnerProduct | SearchMetric::Cosine => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::MipsI8,
                rerank: RerankChoice::IpF32,
            },
        }
    }

    #[cfg(any(feature = "cli", feature = "benchmark"))]
    pub fn label(self) -> String {
        format!(
            "{:?} -> {:?} -> {:?}",
            self.prefilter, self.admission, self.rerank
        )
    }
}

#[cfg(test)]
mod selector_tests {
    use super::*;

    #[test]
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
                .validate_cascade_options(SearchMetric::L2)
                .unwrap_err()
                .to_string()
                .contains("precision"));
        }
        let shared = Cascade {
            prefilter: PrefilterChoice::None,
            admission: AdmissionChoice::L2U16,
            rerank: RerankChoice::U16,
        };
        shared.validate_cascade_options(SearchMetric::L2).unwrap();
        let requirements = shared.dataset_requirements();
        assert_eq!(requirements[0], requirements[1]);
    }

    #[test]
    fn dimension_and_metric_select_independent_defaults() {
        let low = Cascade::default_for_specified_dimension_and_metric(128, SearchMetric::L2);
        let high = Cascade::default_for_specified_dimension_and_metric(960, SearchMetric::L2);
        assert_eq!(low.admission, AdmissionChoice::L2U8);
        assert_eq!(high.admission, AdmissionChoice::L2Kt);
        assert_eq!(high.prefilter, PrefilterChoice::Jl);
        assert_eq!(
            Cascade::default_for_specified_dimension_and_metric(511, SearchMetric::L2).prefilter,
            PrefilterChoice::None
        );
        assert_eq!(
            Cascade::default_for_specified_dimension_and_metric(512, SearchMetric::L2).prefilter,
            PrefilterChoice::Jl
        );
        for metric in [SearchMetric::InnerProduct, SearchMetric::Cosine] {
            let c = Cascade::default_for_specified_dimension_and_metric(1536, metric);
            assert_eq!(c.prefilter, PrefilterChoice::None);
            assert_eq!(c.admission, AdmissionChoice::MipsI8);
            assert_eq!(c.rerank, RerankChoice::IpF32);
            c.validate_cascade_options(metric).unwrap();
        }
        assert!(low
            .validate_cascade_options(SearchMetric::InnerProduct)
            .is_err());
        assert!(high.validate_cascade_options(SearchMetric::L2).is_ok());
    }
}

impl Cascade {
    pub(crate) fn native_l2_u8() -> Self {
        Self {
            prefilter: PrefilterChoice::None,
            admission: AdmissionChoice::NativeL2U8,
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

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Validated choices and defaults for the search cascade.
use crate::sidecars::requirements::SidecarRecipe;
use crate::{Error, Result};
use std::str::FromStr;
/// Distance / similarity objective used by the search pipeline.
///
/// The selected metric determines:
///
/// - which admission and rerank implementations are valid,
/// - how calibration thresholds are interpreted,
/// - whether vector normalization is required by the data path.
///
/// `InnerProduct` also accepts the common CLI/config aliases `mips` and `ip`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum SearchMetric {
    /// Squared Euclidean-distance search.
    L2,

    /// Maximum inner-product search (MIPS).
    #[serde(alias = "mips", alias = "ip")]
    #[cfg_attr(feature = "cli", value(alias = "mips", alias = "ip"))]
    InnerProduct,

    /// Cosine-similarity search.
    ///
    /// The search stages use the inner-product-oriented pipeline after the
    /// corresponding vector normalization has been applied.
    Cosine,
}

impl SearchMetric {
    pub fn calibration_metric(self) -> orion_ann_core::CalibrationMetric {
        match self {
            Self::L2 => orion_ann_core::CalibrationMetric::L2,
            Self::InnerProduct => orion_ann_core::CalibrationMetric::InnerProduct,
            Self::Cosine => orion_ann_core::CalibrationMetric::Cosine,
        }
    }
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
    pub(crate) fn dataset_requirements(
        self,
    ) -> Vec<crate::sidecars::requirements::DatasetRequirement> {
        use crate::sidecars::requirements::DatasetRequirement as R;
        let mut result: Vec<_> = self
            .sidecar_requirements()
            .into_iter()
            .map(R::Sidecar)
            .collect();
        if matches!(
            self.admission,
            AdmissionChoice::NativeL2U8 | AdmissionChoice::AdsF32
        ) || matches!(self.rerank, RerankChoice::F32 | RerankChoice::IpF32)
        {
            result.push(R::Base);
        }
        result
    }

    /// Resolve storage dependencies independently of the search-stage execution.
    /// Duplicate consumers intentionally name the same requirement; the dataset
    /// budget/initializer deduplicates by representation and build options.
    pub(crate) fn sidecar_requirements(
        self,
    ) -> Vec<crate::sidecars::requirements::SidecarRequirement> {
        use crate::sidecars::requirements::SidecarKind as K;
        let mut kinds = Vec::new();
        let mips = matches!(
            self.admission,
            AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16 | AdmissionChoice::PqIp
        );
        match self.prefilter {
            PrefilterChoice::None => {}
            PrefilterChoice::Jl => kinds.push(if mips { K::JlSparseMips } else { K::JlSparse }),
            PrefilterChoice::JlHadamard => kinds.push(K::JlHadamard),
            PrefilterChoice::Rabitq => kinds.push(K::Rabitq),
        }
        match self.admission {
            AdmissionChoice::PqL2 | AdmissionChoice::PqIp => kinds.push(K::Pq),
            AdmissionChoice::L2U8 => kinds.push(K::L2U8),
            AdmissionChoice::L2U16 => kinds.push(K::L2U16),
            AdmissionChoice::MipsI8 => kinds.push(K::MipsI8),
            AdmissionChoice::MipsI16 => kinds.push(K::MipsI16),
            AdmissionChoice::L2Kt => kinds.push(K::L2Kt),
            AdmissionChoice::NativeL2U8 | AdmissionChoice::AdsF32 => {}
        }
        if self.rerank == RerankChoice::U16 {
            kinds.push(K::L2U16);
        }
        kinds.into_iter().map(K::requirement).collect()
    }

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
            AdmissionChoice::NativeL2U8 | AdmissionChoice::AdsF32 => P::Exact,
        };
        let rerank = match self.rerank {
            RerankChoice::None => None,
            RerankChoice::F32 | RerankChoice::IpF32 => Some(P::Exact),
            RerankChoice::U16 => Some(P::Quantized(K::L2U16)),
        };
        validate_precision(prefilter, admission, rerank)
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
        for admission in [AdmissionChoice::AdsF32, AdmissionChoice::NativeL2U8] {
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

/// Prefilter tier choice. Maps to `--prefilter <none|jl|jl-hadamard|rabitq>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum PrefilterChoice {
    /// No prefilter — graph neighbours go straight to admission.
    None,
    /// JL Sparse 1024-bit Hamming filter — one cache line / vertex.
    Jl,
    /// **JL Hadamard 1024-bit Hamming filter** — same byte width and
    /// same `JLHammingDistance` kernel as JL Sparse, but each output
    /// bit is `sign(HD₃HD₂HD₁ · x)[i]` so it sees **all D dims** via
    /// Hadamard mixing instead of NZ=9 sparse samples. Higher per-bit
    /// information content → tighter Hamming distance distribution →
    /// more discriminative cutoff at iso-recall.
    JlHadamard,
    /// RaBitQ rotation + sign-pack Hamming filter — dense Gaussian
    /// rotation; signature quality similar to JlHadamard but the
    /// O(N²) rotation cost is ~3× the FWHT cost on high-D.
    Rabitq,
}

/// Admission (PQ-ranking) tier choice. Native bytes and quantized bytes
/// have distinct variants so logs and experiments preserve their semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum AdmissionChoice {
    /// Independent product quantization with squared-L2 ADC.
    PqL2,
    /// Same codes with negative-inner-product ADC (also normalized cosine).
    PqIp,
    /// Affine-quantized sidecar derived from the f32 base.
    L2U8,
    /// Exact squared L2 over the original u8 base; no sidecar.
    NativeL2U8,
    L2U16,
    L2Kt,
    MipsI8,
    MipsI16,
    /// ADSampling f32 admission — scaled-partial-sum L2 with chunk-
    /// boundary early-abort. ε is fixed at `DEFAULT_ADS_EPSILON`
    /// (override via `ORION_ADS_EPSILON` env var). Requires the
    /// dataset and queries to be pre-rotated by the ADSampling
    /// rotator; L2 is rotation-invariant so the graph topology is
    /// unaffected.
    AdsF32,
}

impl AdmissionChoice {
    /// Full-precision metric corresponding to the admission ordering.
    #[cfg(any(feature = "cli", feature = "benchmark"))]
    pub fn calibration_metric(self) -> orion_ann_core::CalibrationMetric {
        match self {
            Self::MipsI8 | Self::MipsI16 | Self::PqIp => {
                orion_ann_core::CalibrationMetric::InnerProduct
            }
            _ => orion_ann_core::CalibrationMetric::L2,
        }
    }
}

/// ADSampling confidence ε constant — higher ⇒ tighter confidence
/// band ⇒ fewer early-aborts but safer admits. Default `2.1` matches
/// the `--algorithms ads-comparison` reference run in
/// `benchmark/src/main.rs::run_ads_comparison`.
pub const DEFAULT_ADS_EPSILON: f32 = 2.1;

/// Read `ORION_ADS_EPSILON` once per process (OnceLock-cached).
#[cfg(feature = "benchmark")]
pub fn ads_epsilon() -> f32 {
    use std::sync::OnceLock;
    static EPS: OnceLock<f32> = OnceLock::new();
    *EPS.get_or_init(|| {
        std::env::var("ORION_ADS_EPSILON")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(DEFAULT_ADS_EPSILON)
    })
}

#[cfg(not(feature = "benchmark"))]
pub fn ads_epsilon() -> f32 {
    DEFAULT_ADS_EPSILON
}

/// Final-precision rerank tier choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum RerankChoice {
    F32,
    IpF32,
    U16,
    /// Skip the rerank pass — emit the admission PQ's top-k directly.
    /// Useful as the rerank-disabled arm in a cascade-stage ablation
    /// (`cascade_ablation` bin), or when the admission tier already
    /// ranks at sufficient precision (f32 admission, RaBitQ B=4 with
    /// per-vertex correction).
    None,
}

impl FromStr for PrefilterChoice {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "none" | "no" | "off" => Ok(Self::None),
            "jl" | "jl-sparse" => Ok(Self::Jl),
            "jl-hadamard" | "jl-h" | "jlh" => Ok(Self::JlHadamard),
            "rabitq" | "rbq" => Ok(Self::Rabitq),
            other => Err(Error::from(format!(
                "unknown prefilter '{other}' (expected none|jl|jl-hadamard|rabitq)"
            ))),
        }
    }
}

impl FromStr for AdmissionChoice {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "pq-l2" | "pq" => Ok(Self::PqL2),
            "pq-ip" => Ok(Self::PqIp),
            "l2-u8" | "l2u8" => Ok(Self::L2U8),
            "native-l2-u8" => Ok(Self::NativeL2U8),
            "l2-u16" | "l2u16" => Ok(Self::L2U16),
            "l2-kt" | "l2kt" | "kt" => Ok(Self::L2Kt),
            "mips-i8" | "mipsi8" => Ok(Self::MipsI8),
            "mips-i16" | "mipsi16" => Ok(Self::MipsI16),
            "ads-f32" | "adsf32" | "ads" => Ok(Self::AdsF32),
            other => Err(Error::from(format!(
                "unknown admission '{other}' \
                 (expected pq-l2|pq-ip|native-l2-u8|l2-u8|l2-u16|l2-kt|mips-i8|mips-i16|ads-f32)"
            ))),
        }
    }
}

impl FromStr for RerankChoice {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "ip-f32" | "ipf32" => Ok(Self::IpF32),
            "u16" => Ok(Self::U16),
            "none" | "no" | "off" => Ok(Self::None),
            other => Err(Error::from(format!(
                "unknown rerank '{other}' (expected f32|ip-f32|u16|none)"
            ))),
        }
    }
}

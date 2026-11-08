/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Accepted stage names and aliases shared by every adapter.
use crate::{Error, Result};
use std::str::FromStr;

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
    #[cfg(feature = "benchmark")]
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
#[cfg(feature = "benchmark")]
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
            #[cfg(feature = "benchmark")]
            "ads-f32" | "adsf32" | "ads" => Ok(Self::AdsF32),
            #[cfg(not(feature = "benchmark"))]
            "ads-f32" | "adsf32" | "ads" => Err(Error::unsupported(
                "ADSampling is available only with the benchmark feature",
            )),
            other => Err(Error::from(format!(
                "unknown admission '{other}' \
                 (expected pq-l2|pq-ip|native-l2-u8|l2-u8|l2-u16|l2-kt|mips-i8|mips-i16)"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ads_aliases_follow_the_benchmark_feature() {
        for name in ["ads-f32", "adsf32", "ads"] {
            let parsed = name.parse::<AdmissionChoice>();
            #[cfg(feature = "benchmark")]
            assert_eq!(parsed.unwrap(), AdmissionChoice::AdsF32);
            #[cfg(not(feature = "benchmark"))]
            {
                let error = parsed.unwrap_err();
                assert_eq!(error.code(), crate::ErrorCode::UnsupportedConfiguration);
                assert!(error.to_string().contains("benchmark feature"));
            }
        }
    }
}

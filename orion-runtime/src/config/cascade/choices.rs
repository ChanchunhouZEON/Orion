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
    /// Product codes with negative-inner-product ADC on raw coordinates.
    PqIp,
    /// Negative-inner-product ADC after base/query L2 normalization.
    PqCosine,
    /// Affine-quantized sidecar derived from the f32 base.
    L2U8,
    /// Exact squared L2 over the original u8 base; no sidecar.
    NativeL2U8,
    /// Unsigned native-byte dot product; no quantization.
    NativeIpU8,
    /// Native-byte cosine with exact stored squared norms.
    NativeCosineU8,
    L2U16,
    L2Kt,
    /// Symmetric i8 quantization of raw inner-product coordinates.
    IpI8,
    /// Symmetric i16 quantization of raw inner-product coordinates.
    IpI16,
    /// Symmetric i8 quantization after base/query L2 normalization.
    /// No correction for norms changed by quantization is applied.
    CosineI8,
    /// Symmetric i16 quantization after base/query L2 normalization.
    CosineI16,
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
    /// Search objective accepted by this arm. Cosine normalization belongs to
    /// input preparation, while its kernels share IP arithmetic.
    pub fn metric(self) -> super::SearchMetric {
        use super::SearchMetric;
        match self {
            Self::IpI8 | Self::IpI16 | Self::PqIp | Self::NativeIpU8 => SearchMetric::InnerProduct,
            Self::CosineI8 | Self::CosineI16 | Self::PqCosine | Self::NativeCosineU8 => {
                SearchMetric::Cosine
            }
            _ => SearchMetric::SquaredL2,
        }
    }

    /// Whether the underlying arithmetic uses negative dot products.
    pub(crate) fn uses_inner_product(self) -> bool {
        self.metric() != super::SearchMetric::SquaredL2
    }

    /// Full-precision metric corresponding to the admission ordering.
    #[cfg(any(feature = "cli", feature = "benchmark"))]
    pub fn calibration_metric(self) -> orion_ann_core::CalibrationMetric {
        self.metric().calibration_metric()
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
    /// Exact inner product of raw f32 coordinates.
    IpF32,
    /// f32 IP rerank on normalized coordinates; output scores use cosine.
    CosineF32,
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
            "pq-cosine" => Ok(Self::PqCosine),
            "l2-u8" | "l2u8" => Ok(Self::L2U8),
            "native-l2-u8" => Ok(Self::NativeL2U8),
            "native-ip-u8" => Ok(Self::NativeIpU8),
            "native-cosine-u8" => Ok(Self::NativeCosineU8),
            "l2-u16" | "l2u16" => Ok(Self::L2U16),
            "l2-kt" | "l2kt" | "kt" => Ok(Self::L2Kt),
            "ip-i8" | "mips-i8" | "mipsi8" => Ok(Self::IpI8),
            "cosine-i8" => Ok(Self::CosineI8),
            "ip-i16" | "mips-i16" | "mipsi16" => Ok(Self::IpI16),
            "cosine-i16" => Ok(Self::CosineI16),
            #[cfg(feature = "benchmark")]
            "ads-f32" | "adsf32" | "ads" => Ok(Self::AdsF32),
            #[cfg(not(feature = "benchmark"))]
            "ads-f32" | "adsf32" | "ads" => Err(Error::unsupported(
                "ADSampling is available only with the benchmark feature",
            )),
            other => Err(Error::from(format!(
                "unknown admission '{other}' \
                 (expected pq-l2|pq-ip|pq-cosine|native-l2-u8|native-ip-u8|native-cosine-u8|l2-u8|l2-u16|l2-kt|ip-i8|ip-i16|cosine-i8|cosine-i16)"
            ))),
        }
    }
}

impl FromStr for RerankChoice {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "l2-f32" | "f32" => Ok(Self::F32),
            "ip-f32" | "ipf32" => Ok(Self::IpF32),
            "cosine-f32" => Ok(Self::CosineF32),
            "l2-u16" | "u16" => Ok(Self::U16),
            "none" | "no" | "off" => Ok(Self::None),
            other => Err(Error::from(format!(
                "unknown rerank '{other}' (expected l2-f32|ip-f32|cosine-f32|l2-u16|none)"
            ))),
        }
    }
}

impl std::fmt::Display for AdmissionChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PqL2 => "pq-l2",
            Self::PqIp => "pq-ip",
            Self::PqCosine => "pq-cosine",
            Self::L2U8 => "l2-u8",
            Self::L2U16 => "l2-u16",
            Self::L2Kt => "l2-kt",
            Self::IpI8 => "ip-i8",
            Self::IpI16 => "ip-i16",
            Self::CosineI8 => "cosine-i8",
            Self::CosineI16 => "cosine-i16",
            Self::NativeL2U8 => "native-l2-u8",
            Self::NativeIpU8 => "native-ip-u8",
            Self::NativeCosineU8 => "native-cosine-u8",
            #[cfg(feature = "benchmark")]
            Self::AdsF32 => "ads-f32",
        })
    }
}

impl std::fmt::Display for RerankChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F32 => "l2-f32",
            Self::IpF32 => "ip-f32",
            Self::CosineF32 => "cosine-f32",
            Self::U16 => "l2-u16",
            Self::None => "none",
        })
    }
}

impl std::fmt::Display for PrefilterChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::Jl => "jl",
            Self::JlHadamard => "jl-hadamard",
            Self::Rabitq => "rabitq",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_roundtrip_and_legacy_ip_aliases_do_not_mean_cosine() {
        for name in [
            "ip-i8",
            "ip-i16",
            "cosine-i8",
            "cosine-i16",
            "pq-cosine",
            "pq-ip",
            "pq-l2",
        ] {
            let choice: AdmissionChoice = name.parse().unwrap();
            assert_eq!(choice.to_string(), name);
        }
        for name in ["l2-f32", "ip-f32", "cosine-f32", "l2-u16", "none"] {
            let choice: RerankChoice = name.parse().unwrap();
            assert_eq!(choice.to_string(), name);
        }
        assert_eq!(
            "mips-i8".parse::<AdmissionChoice>().unwrap(),
            AdmissionChoice::IpI8
        );
        assert_eq!(
            "mips-i16".parse::<AdmissionChoice>().unwrap().metric(),
            super::super::SearchMetric::InnerProduct
        );
    }

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

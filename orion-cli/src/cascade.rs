/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Composable cascade dispatch (benchmark harness side)
//!
//! The three CLI axes (`--prefilter`, `--admission`, `--rerank`) map
//! to the enums below. The harness owns the build-and-pin handshake
//! so the search library can stay unaware of `mlock(2)` — each axis
//! pins its own sidecar(s) at `pin_cascade` time, and a single
//! dyn-dispatch call to `search_batch_unified` runs the actual search.
//!
//! Why this layer:
//!   * The lib's stage traits are object-safe; the harness builds each
//!     axis as a `Box<dyn Stage>` and pipes the three boxes into one
//!     non-generic search call.
//!   * Pinning is a benchmark concern; lib knows nothing of `mlock`.
//!     Per-axis pin helpers live next to the per-axis build helpers so
//!     the CLI mapping and pin policy stay close to each other.
//!
//! ### Two consumption shapes
//!
//! 1. `benchmark` crate's `main.rs` / `OrionRunner`: imports
//!    via the `runner` module tree (`runner::cascade::*`).
//! 2. `orion_sweep` bin: imports via `#[path = "../runner/cascade.rs"]`
//!    since it doesn't sit on top of a benchmark lib crate.
//!
//! Both paths satisfy `crate::utils::mlock_bytes` because every
//! consumer in the crate has a `utils` module reachable as
//! `crate::utils` (either the real one, or a `#[path]`-imported copy
//! inside the bin).
//!
//! The cascade helpers are consumed by the `orion` /
//! `orion_sweep` bins via `#[path]`, not from the `benchmark` main
//! binary — so to that compilation unit they look unused. Suppress
//! the dead-code warning at the module level rather than tagging
//! every helper individually.

use diskann::common::ANNResult;
use orion::algorithm::search::stage::admission::{
    AdsF32Admission, L2KTAdmission, L2U16Admission, L2U8Admission, MipsI16Admission,
    MipsI8Admission,
};
use orion::algorithm::search::stage::prefilter::{
    JlHadamardPrefilter, JlMipsPrefilter, JlPrefilter, RabitqPrefilter,
};
use orion::algorithm::search::stage::rerank::{F32Rerank, IpF32Rerank, NoRerank, U16Rerank};
use orion::Orion;
use std::str::FromStr;
use vector::FullPrecisionDistance;

/// Distance / similarity objective used by the search pipeline.
///
/// The selected metric determines:
///
/// - which admission and rerank implementations are valid,
/// - how calibration thresholds are interpreted,
/// - whether vector normalization is required by the data path.
///
/// `InnerProduct` also accepts the common CLI/config aliases `mips` and `ip`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, clap::ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
pub enum SearchMetric {
    /// Squared Euclidean-distance search.
    L2,

    /// Maximum inner-product search (MIPS).
    #[serde(alias = "mips", alias = "ip")]
    #[value(alias = "mips", alias = "ip")]
    InnerProduct,

    /// Cosine-similarity search.
    ///
    /// The search stages use the inner-product-oriented pipeline after the
    /// corresponding vector normalization has been applied.
    Cosine,
}

impl SearchMetric {
    pub fn calibration_metric(self) -> orion::CalibrationMetric {
        match self {
            Self::L2 => orion::CalibrationMetric::L2,
            Self::InnerProduct => orion::CalibrationMetric::InnerProduct,
            Self::Cosine => orion::CalibrationMetric::Cosine,
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
    pub fn validate_cascade_options(self, metric: SearchMetric) -> Result<(), String> {
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
            AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16
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

        Ok(())
    }

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
        assert!(low.validate_cascade_options(SearchMetric::InnerProduct).is_err());
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
    /// Affine-quantized sidecar derived from the f32 base.
    L2U8,
    /// Exact squared L2 over the original u8 base; no sidecar.
    NativeL2U8,
    L2U16,
    L2Kt,
    MipsI8,
    MipsI16,
    /// ADSampling f32 admission — scaled-partial-sum L2 with chunk-
    /// boundary early-abort. ε is fixed at [`DEFAULT_ADS_EPSILON`]
    /// (override via `ORION_ADS_EPSILON` env var). Requires the
    /// dataset and queries to be pre-rotated by the ADSampling
    /// rotator; L2 is rotation-invariant so the graph topology is
    /// unaffected.
    AdsF32,
}

impl AdmissionChoice {
    /// Full-precision metric corresponding to the admission ordering.
    pub fn calibration_metric(self) -> orion::CalibrationMetric {
        match self {
            Self::MipsI8 | Self::MipsI16 => orion::CalibrationMetric::InnerProduct,
            _ => orion::CalibrationMetric::L2,
        }
    }
}

/// ADSampling confidence ε constant — higher ⇒ tighter confidence
/// band ⇒ fewer early-aborts but safer admits. Default `2.1` matches
/// the `--algorithms ads-comparison` reference run in
/// `benchmark/src/main.rs::run_ads_comparison`.
pub const DEFAULT_ADS_EPSILON: f32 = 2.1;

/// Read `ORION_ADS_EPSILON` once per process (OnceLock-cached).
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
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "none" | "no" | "off" => Ok(Self::None),
            "jl" | "jl-sparse" => Ok(Self::Jl),
            "jl-hadamard" | "jl-h" | "jlh" => Ok(Self::JlHadamard),
            "rabitq" | "rbq" => Ok(Self::Rabitq),
            other => Err(format!(
                "unknown prefilter '{other}' (expected none|jl|jl-hadamard|rabitq)"
            )),
        }
    }
}

impl FromStr for AdmissionChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "l2-u8" | "l2u8" => Ok(Self::L2U8),
            "native-l2-u8" => Ok(Self::NativeL2U8),
            "l2-u16" | "l2u16" => Ok(Self::L2U16),
            "l2-kt" | "l2kt" | "kt" => Ok(Self::L2Kt),
            "mips-i8" | "mipsi8" => Ok(Self::MipsI8),
            "mips-i16" | "mipsi16" => Ok(Self::MipsI16),
            "ads-f32" | "adsf32" | "ads" => Ok(Self::AdsF32),
            other => Err(format!(
                "unknown admission '{other}' \
                 (expected native-l2-u8|l2-u8|l2-u16|l2-kt|mips-i8|mips-i16|ads-f32)"
            )),
        }
    }
}

impl FromStr for RerankChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "ip-f32" | "ipf32" => Ok(Self::IpF32),
            "u16" => Ok(Self::U16),
            "none" | "no" | "off" => Ok(Self::None),
            other => Err(format!(
                "unknown rerank '{other}' (expected f32|ip-f32|u16|none)"
            )),
        }
    }
}

/// Build the prefilter adapter. `None` choice → `None` Box (the search
/// loop's `Option<&dyn _>` short-circuits).
///
/// `admission` is consulted to decide whether the JL prefilter should
/// run in MIPS mode (per PA's `Mips_JL_Sparse_Point_Normalized`
/// design — `popcount × ‖v‖` instead of raw popcount). The L2 and
/// MIPS scoring rules diverge once the base vectors aren't unit-norm,
/// so the prefilter must agree with what the admission tier is
/// computing.
pub fn build_prefilter<'a, const N: usize>(
    orion: &'a Orion<N>,
    choice: PrefilterChoice,
    admission: AdmissionChoice,
) -> Option<Box<dyn orion::algorithm::search::stage::PrefilterStage<N> + 'a>>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let admission_is_mips = matches!(
        admission,
        AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16
    );
    match choice {
        PrefilterChoice::None => None,
        PrefilterChoice::Jl => {
            if admission_is_mips {
                // NZ=9, `popcount × ‖v‖` scoring — PA's MIPS-JL recipe
                // (`Mips_JL_Sparse_Point_Normalized`). The MIPS-only
                // adapter wraps `JLSparseDatasetMips`, which carries
                // the per-vertex `‖v‖` slab the L2 type doesn't.
                let ds = orion.ensure_quantized_dataset_jl_mips();
                Some(Box::new(JlMipsPrefilter::new(ds)))
            } else {
                // NZ=9, raw popcount — the L2-JL default.
                let ds = orion.ensure_quantized_dataset_jl();
                Some(Box::new(JlPrefilter::new(ds)))
            }
        }
        PrefilterChoice::JlHadamard => {
            let ds = orion.ensure_quantized_dataset_jl_hadamard();
            Some(Box::new(JlHadamardPrefilter::new(ds)))
        }
        PrefilterChoice::Rabitq => {
            let ds = orion.ensure_quantized_dataset_rabitq();
            Some(Box::new(RabitqPrefilter::new(ds)))
        }
    }
}

/// Build an admission adapter for an f32 index.
///
/// Returns a configuration error for native-byte admission, including callers
/// that do not use the standalone CLI validation.
pub fn build_admission<'a, const N: usize>(
    orion: &'a Orion<N>,
    choice: AdmissionChoice,
) -> ANNResult<Box<dyn orion::algorithm::search::stage::AdmissionStage<N> + 'a>>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    Ok(match choice {
        AdmissionChoice::NativeL2U8 => {
            return Err(diskann::common::ANNError::log_index_config_error(
                "admission".into(),
                "native-l2-u8 requires a native u8 index; the f32 dispatcher cannot execute it".into(),
            ));
        }
        AdmissionChoice::L2U8 => Box::new(L2U8Admission::new(orion.ensure_quantized_dataset())),
        AdmissionChoice::L2U16 => {
            Box::new(L2U16Admission::new(orion.ensure_quantized_dataset_l2_u16()))
        }
        AdmissionChoice::L2Kt => {
            Box::new(L2KTAdmission::new(orion.ensure_quantized_dataset_l2_kt()))
        }
        AdmissionChoice::MipsI8 => {
            Box::new(MipsI8Admission::new(orion.ensure_quantized_dataset_mips()))
        }
        AdmissionChoice::MipsI16 => Box::new(MipsI16Admission::new(
            orion.ensure_quantized_dataset_mips_i16(),
        )),
        AdmissionChoice::AdsF32 => Box::new(AdsF32Admission::new(&orion.dataset, ads_epsilon())),
    })
}

/// Build the rerank adapter.
pub fn build_rerank<'a, const N: usize>(
    orion: &'a Orion<N>,
    choice: RerankChoice,
) -> Box<dyn orion::algorithm::search::stage::RerankStage<N> + 'a>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        RerankChoice::F32 => Box::new(F32Rerank::new(&orion.dataset)),
        RerankChoice::IpF32 => Box::new(IpF32Rerank::new(&orion.dataset)),
        RerankChoice::U16 => Box::new(U16Rerank::new(orion.ensure_quantized_dataset_l2_u16())),
        RerankChoice::None => Box::new(NoRerank),
    }
}

/// Pin the JL sidecar when the prefilter is JL. Uses the
/// admission-family hint to pick the L2 (NZ=9) vs MIPS (NZ=11)
/// sidecar — matches `build_prefilter`'s dispatch.
pub fn pin_prefilter<const N: usize>(
    orion: &Orion<N>,
    choice: PrefilterChoice,
    admission: AdmissionChoice,
) where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let admission_is_mips = matches!(
        admission,
        AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16
    );
    match choice {
        PrefilterChoice::None => {}
        PrefilterChoice::Jl => {
            if admission_is_mips {
                let q = orion.ensure_quantized_dataset_jl_mips();
                crate::utils::mlock_bytes(
                    "prefilter (JL codes, mips)",
                    q.codes.as_ptr(),
                    q.codes.len(),
                );
                crate::utils::mlock_bytes(
                    "prefilter (JL indices, mips)",
                    q.indices.as_ptr() as *const u8,
                    q.indices.len() * std::mem::size_of::<u32>(),
                );
            } else {
                let q = orion.ensure_quantized_dataset_jl();
                crate::utils::mlock_bytes("prefilter (JL codes)", q.codes.as_ptr(), q.codes.len());
                crate::utils::mlock_bytes(
                    "prefilter (JL indices)",
                    q.indices.as_ptr() as *const u8,
                    q.indices.len() * std::mem::size_of::<u32>(),
                );
            }
        }
        PrefilterChoice::JlHadamard => {
            let q = orion.ensure_quantized_dataset_jl_hadamard();
            crate::utils::mlock_bytes(
                "prefilter (JL Hadamard codes)",
                q.codes.as_slice().as_ptr(),
                q.codes.as_slice().len(),
            );
            // signs1/2/3 are 3 × D_PAD bytes (= 3072 bytes at D_PAD=1024);
            // negligible but pin for symmetry.
            crate::utils::mlock_bytes(
                "prefilter (JL Hadamard signs1)",
                q.signs1.as_ptr() as *const u8,
                q.signs1.len(),
            );
            crate::utils::mlock_bytes(
                "prefilter (JL Hadamard signs2)",
                q.signs2.as_ptr() as *const u8,
                q.signs2.len(),
            );
            crate::utils::mlock_bytes(
                "prefilter (JL Hadamard signs3)",
                q.signs3.as_ptr() as *const u8,
                q.signs3.len(),
            );
        }
        PrefilterChoice::Rabitq => {
            let q = orion.ensure_quantized_dataset_rabitq();
            crate::utils::mlock_bytes(
                "prefilter (RaBitQ codes)",
                q.codes.as_slice().as_ptr(),
                q.codes.as_slice().len(),
            );
            // Rotation matrix is hot at every search setup (rotate_query
            // is O(N²)). Pin it so the f32 matrix stays L1-resident
            // across queries.
            crate::utils::mlock_bytes(
                "prefilter (RaBitQ rotation)",
                q.rotation.as_slice().as_ptr() as *const u8,
                q.rotation.as_slice().len() * std::mem::size_of::<f32>(),
            );
        }
    }
}

/// Pin the admission sidecar. L2-KT pins both i8 base + per-vertex
/// `‖x‖²` companion (sink touches both every hop).
pub fn pin_admission<const N: usize>(orion: &Orion<N>, choice: AdmissionChoice)
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        AdmissionChoice::NativeL2U8 => {} // Native storage is pinned with its base.
        AdmissionChoice::L2U8 => {
            let q = orion.ensure_quantized_dataset();
            crate::utils::mlock_bytes(
                "admission (u8 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<u8>(),
            );
        }
        AdmissionChoice::L2U16 => {
            let q = orion.ensure_quantized_dataset_l2_u16();
            crate::utils::mlock_bytes(
                "admission (u16 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<u16>(),
            );
        }
        AdmissionChoice::L2Kt => {
            let q = orion.ensure_quantized_dataset_l2_kt();
            crate::utils::mlock_bytes(
                "admission (L2-KT i8 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<i8>(),
            );
            crate::utils::mlock_bytes(
                "admission (L2-KT ‖x‖²)",
                q.norms_sq.as_ptr() as *const u8,
                q.norms_sq.len() * std::mem::size_of::<i32>(),
            );
        }
        AdmissionChoice::MipsI8 => {
            let q = orion.ensure_quantized_dataset_mips();
            crate::utils::mlock_bytes(
                "admission (MIPS i8 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<i8>(),
            );
        }
        AdmissionChoice::MipsI16 => {
            let q = orion.ensure_quantized_dataset_mips_i16();
            crate::utils::mlock_bytes(
                "admission (MIPS i16 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<i16>(),
            );
        }
        AdmissionChoice::AdsF32 => {
            // ADS reads straight from the f32 base — pinned via
            // `pin_rerank` for the f32/ip-f32 cases, so we skip
            // double-pinning here. If a future cascade pairs ADS with
            // `U16` rerank, the f32 base would go un-pinned; in that
            // (currently unused) combination, ADS still needs the f32
            // base resident, so pin it here as a safety net.
            if !matches!(orion.dataset.data.len(), 0) {
                let d = orion.dataset.data.as_slice();
                crate::utils::mlock_bytes(
                    "admission (ADS f32 base)",
                    d.as_ptr() as *const u8,
                    d.len() * std::mem::size_of::<f32>(),
                );
            }
        }
    }
}

/// Pin the rerank base. `F32`/`IpF32` share the f32 base; `U16` pins
/// the u16 sidecar (idempotent if `AdmissionChoice::L2U16` already did).
pub fn pin_rerank<const N: usize>(orion: &Orion<N>, choice: RerankChoice)
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        RerankChoice::F32 | RerankChoice::IpF32 => {
            let d = orion.dataset.data.as_slice();
            crate::utils::mlock_bytes(
                "rerank (f32 base)",
                d.as_ptr() as *const u8,
                d.len() * std::mem::size_of::<f32>(),
            );
        }
        RerankChoice::U16 => {
            let q = orion.ensure_quantized_dataset_l2_u16();
            crate::utils::mlock_bytes(
                "rerank (u16 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<u16>(),
            );
        }
        // NoRerank reads from `scratch.pq` only — no sidecar to pin.
        RerankChoice::None => {}
    }
}

/// Pin every sidecar referenced by the chosen cascade. `mlock(2)` is
/// idempotent so calling this in both warmup and timed-sweep is fine.
/// Does **not** pin the `PhasedGraph` slab — the caller pins that
/// once regardless of cascade choice.
pub fn pin_cascade<const N: usize>(
    orion: &Orion<N>,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    pin_prefilter(orion, prefilter, admission);
    pin_admission(orion, admission);
    pin_rerank(orion, rerank);
}

/// Single-query dispatch through the chosen cascade — same wiring
/// shape as [`search_batch_compose`], just routed to `search_unified`
/// instead of `search_batch_unified`. Used by per-query diagnostic
/// loops in `main.rs` (ablation study, search_profile, etc.).
#[allow(clippy::too_many_arguments)]
pub fn search_compose<const N: usize>(
    orion: &Orion<N>,
    query: &[f32; N],
    k: usize,
    search_list_size: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) -> ANNResult<Vec<u32>>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let ad = build_admission(orion, admission)?;
    let pf = build_prefilter(orion, prefilter, admission);
    let rr = build_rerank(orion, rerank);
    orion.search_unified(
        query,
        k,
        search_list_size,
        window_size,
        epsilon,
        early_exit_limit,
        pf.as_deref(),
        ad.as_ref(),
        rr.as_ref(),
    )
}

/// One-shot batched dispatch through the chosen cascade.
///
/// Builds all three adapters via the per-axis builders, then hands
/// the boxed dyn refs to `search_batch_unified` (dyn vtable on the
/// hot loop). Call `pin_cascade` once before the timed loop to keep
/// every sidecar resident.
#[allow(clippy::too_many_arguments)]
pub fn search_batch_compose<const N: usize>(
    orion: &Orion<N>,
    queries: &[[f32; N]],
    k: usize,
    search_list_size: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) -> ANNResult<Vec<Vec<u32>>>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let ad = build_admission(orion, admission)?;
    let pf = build_prefilter(orion, prefilter, admission);
    let rr = build_rerank(orion, rerank);
    orion.search_batch_unified(
        queries,
        k,
        search_list_size,
        window_size,
        epsilon,
        early_exit_limit,
        pf.as_deref(),
        ad.as_ref(),
        rr.as_ref(),
    )
}

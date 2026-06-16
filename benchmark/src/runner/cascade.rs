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
//! 1. `benchmark` crate's `main.rs` / `StagedDiskANNRunner`: imports
//!    via the `runner` module tree (`runner::cascade::*`).
//! 2. `staged_sweep` bin: imports via `#[path = "../runner/cascade.rs"]`
//!    since it doesn't sit on top of a benchmark lib crate.
//!
//! Both paths satisfy `crate::utils::mlock_bytes` because every
//! consumer in the crate has a `utils` module reachable as
//! `crate::utils` (either the real one, or a `#[path]`-imported copy
//! inside the bin).
//!
//! The cascade helpers are consumed by the `staged_diskann` /
//! `staged_sweep` bins via `#[path]`, not from the `benchmark` main
//! binary — so to that compilation unit they look unused. Suppress
//! the dead-code warning at the module level rather than tagging
//! every helper individually.

#![allow(dead_code)]

use diskann::common::ANNResult;
use staged_diskann::algorithm::search::stage::admission::{
    AdsF32Admission, L2KTAdmission, L2U16Admission, L2U8Admission, MipsI16Admission,
    MipsI8Admission,
};
use staged_diskann::algorithm::search::stage::prefilter::{
    JlHadamardPrefilter, JlMipsPrefilter, JlPrefilter, RabitqPrefilter,
};
use staged_diskann::algorithm::search::stage::rerank::{F32Rerank, IpF32Rerank, U16Rerank};
use staged_diskann::StagedDiskANN;
use std::str::FromStr;
use vector::FullPrecisionDistance;

/// Prefilter tier choice. Maps to `--prefilter <none|jl|jl-hadamard|rabitq>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Admission (PQ-ranking) tier choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionChoice {
    L2U8,
    L2U16,
    L2Kt,
    MipsI8,
    MipsI16,
    /// ADSampling f32 admission — scaled-partial-sum L2 with chunk-
    /// boundary early-abort. ε is fixed at [`DEFAULT_ADS_EPSILON`]
    /// (override via `STAGED_ADS_EPSILON` env var). Requires the
    /// dataset and queries to be pre-rotated by the ADSampling
    /// rotator; L2 is rotation-invariant so the graph topology is
    /// unaffected.
    AdsF32,
}

/// ADSampling confidence ε constant — higher ⇒ tighter confidence
/// band ⇒ fewer early-aborts but safer admits. Default `2.1` matches
/// the `--algorithms ads-comparison` reference run in
/// `benchmark/src/main.rs::run_ads_comparison`.
pub const DEFAULT_ADS_EPSILON: f32 = 2.1;

/// Read `STAGED_ADS_EPSILON` once per process (OnceLock-cached).
pub fn ads_epsilon() -> f32 {
    use std::sync::OnceLock;
    static EPS: OnceLock<f32> = OnceLock::new();
    *EPS.get_or_init(|| {
        std::env::var("STAGED_ADS_EPSILON")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(DEFAULT_ADS_EPSILON)
    })
}

/// Final-precision rerank tier choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankChoice {
    F32,
    IpF32,
    U16,
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
            "l2-u16" | "l2u16" => Ok(Self::L2U16),
            "l2-kt" | "l2kt" | "kt" => Ok(Self::L2Kt),
            "mips-i8" | "mipsi8" => Ok(Self::MipsI8),
            "mips-i16" | "mipsi16" => Ok(Self::MipsI16),
            "ads-f32" | "adsf32" | "ads" => Ok(Self::AdsF32),
            other => Err(format!(
                "unknown admission '{other}' \
                 (expected l2-u8|l2-u16|l2-kt|mips-i8|mips-i16|ads-f32)"
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
            other => Err(format!("unknown rerank '{other}' (expected f32|ip-f32|u16)")),
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
    staged: &'a StagedDiskANN<N>,
    choice: PrefilterChoice,
    admission: AdmissionChoice,
) -> Option<Box<dyn staged_diskann::algorithm::search::stage::PrefilterStage<N> + 'a>>
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
                let ds = staged.ensure_quantized_dataset_jl_mips();
                Some(Box::new(JlMipsPrefilter::new(ds)))
            } else {
                // NZ=9, raw popcount — the L2-JL default.
                let ds = staged.ensure_quantized_dataset_jl();
                Some(Box::new(JlPrefilter::new(ds)))
            }
        }
        PrefilterChoice::JlHadamard => {
            let ds = staged.ensure_quantized_dataset_jl_hadamard();
            Some(Box::new(JlHadamardPrefilter::new(ds)))
        }
        PrefilterChoice::Rabitq => {
            let ds = staged.ensure_quantized_dataset_rabitq();
            Some(Box::new(RabitqPrefilter::new(ds)))
        }
    }
}

/// Build the admission adapter.
pub fn build_admission<'a, const N: usize>(
    staged: &'a StagedDiskANN<N>,
    choice: AdmissionChoice,
) -> Box<dyn staged_diskann::algorithm::search::stage::AdmissionStage<N> + 'a>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        AdmissionChoice::L2U8 => Box::new(L2U8Admission::new(staged.ensure_quantized_dataset())),
        AdmissionChoice::L2U16 => {
            Box::new(L2U16Admission::new(staged.ensure_quantized_dataset_l2_u16()))
        }
        AdmissionChoice::L2Kt => {
            Box::new(L2KTAdmission::new(staged.ensure_quantized_dataset_l2_kt()))
        }
        AdmissionChoice::MipsI8 => {
            Box::new(MipsI8Admission::new(staged.ensure_quantized_dataset_mips()))
        }
        AdmissionChoice::MipsI16 => Box::new(MipsI16Admission::new(
            staged.ensure_quantized_dataset_mips_i16(),
        )),
        AdmissionChoice::AdsF32 => {
            Box::new(AdsF32Admission::new(&staged.dataset, ads_epsilon()))
        }
    }
}

/// Build the rerank adapter.
pub fn build_rerank<'a, const N: usize>(
    staged: &'a StagedDiskANN<N>,
    choice: RerankChoice,
) -> Box<dyn staged_diskann::algorithm::search::stage::RerankStage<N> + 'a>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        RerankChoice::F32 => Box::new(F32Rerank::new(&staged.dataset)),
        RerankChoice::IpF32 => Box::new(IpF32Rerank::new(&staged.dataset)),
        RerankChoice::U16 => Box::new(U16Rerank::new(staged.ensure_quantized_dataset_l2_u16())),
    }
}

/// Pin the JL sidecar when the prefilter is JL. Uses the
/// admission-family hint to pick the L2 (NZ=9) vs MIPS (NZ=11)
/// sidecar — matches `build_prefilter`'s dispatch.
pub fn pin_prefilter<const N: usize>(
    staged: &StagedDiskANN<N>,
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
                let q = staged.ensure_quantized_dataset_jl_mips();
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
                let q = staged.ensure_quantized_dataset_jl();
                crate::utils::mlock_bytes(
                    "prefilter (JL codes)",
                    q.codes.as_ptr(),
                    q.codes.len(),
                );
                crate::utils::mlock_bytes(
                    "prefilter (JL indices)",
                    q.indices.as_ptr() as *const u8,
                    q.indices.len() * std::mem::size_of::<u32>(),
                );
            }
        }
        PrefilterChoice::JlHadamard => {
            let q = staged.ensure_quantized_dataset_jl_hadamard();
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
            let q = staged.ensure_quantized_dataset_rabitq();
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
pub fn pin_admission<const N: usize>(staged: &StagedDiskANN<N>, choice: AdmissionChoice)
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        AdmissionChoice::L2U8 => {
            let q = staged.ensure_quantized_dataset();
            crate::utils::mlock_bytes(
                "admission (u8 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<u8>(),
            );
        }
        AdmissionChoice::L2U16 => {
            let q = staged.ensure_quantized_dataset_l2_u16();
            crate::utils::mlock_bytes(
                "admission (u16 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<u16>(),
            );
        }
        AdmissionChoice::L2Kt => {
            let q = staged.ensure_quantized_dataset_l2_kt();
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
            let q = staged.ensure_quantized_dataset_mips();
            crate::utils::mlock_bytes(
                "admission (MIPS i8 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<i8>(),
            );
        }
        AdmissionChoice::MipsI16 => {
            let q = staged.ensure_quantized_dataset_mips_i16();
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
            if !matches!(
                staged.dataset.data.len(),
                0
            ) {
                let d = staged.dataset.data.as_slice();
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
pub fn pin_rerank<const N: usize>(staged: &StagedDiskANN<N>, choice: RerankChoice)
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    match choice {
        RerankChoice::F32 | RerankChoice::IpF32 => {
            let d = staged.dataset.data.as_slice();
            crate::utils::mlock_bytes(
                "rerank (f32 base)",
                d.as_ptr() as *const u8,
                d.len() * std::mem::size_of::<f32>(),
            );
        }
        RerankChoice::U16 => {
            let q = staged.ensure_quantized_dataset_l2_u16();
            crate::utils::mlock_bytes(
                "rerank (u16 base)",
                q.data.as_ptr() as *const u8,
                q.data.len() * std::mem::size_of::<u16>(),
            );
        }
    }
}

/// Pin every sidecar referenced by the chosen cascade. `mlock(2)` is
/// idempotent so calling this in both warmup and timed-sweep is fine.
/// Does **not** pin the `PhasedGraph` slab — the caller pins that
/// once regardless of cascade choice.
pub fn pin_cascade<const N: usize>(
    staged: &StagedDiskANN<N>,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    pin_prefilter(staged, prefilter, admission);
    pin_admission(staged, admission);
    pin_rerank(staged, rerank);
}

/// Single-query dispatch through the chosen cascade — same wiring
/// shape as [`search_batch_compose`], just routed to `search_unified`
/// instead of `search_batch_unified`. Used by per-query diagnostic
/// loops in `main.rs` (ablation study, search_profile, etc.).
#[allow(clippy::too_many_arguments)]
pub fn search_compose<const N: usize>(
    staged: &StagedDiskANN<N>,
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
    let pf = build_prefilter(staged, prefilter, admission);
    let ad = build_admission(staged, admission);
    let rr = build_rerank(staged, rerank);
    staged.search_unified(
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
    staged: &StagedDiskANN<N>,
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
    let pf = build_prefilter(staged, prefilter, admission);
    let ad = build_admission(staged, admission);
    let rr = build_rerank(staged, rerank);
    staged.search_batch_unified(
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

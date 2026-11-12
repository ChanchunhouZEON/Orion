/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Metric-dependent cascade selection and shared search dispatch.
//!
//! Both frontends use these stage factories. Optional pinning remains an
//! explicit hook for benchmark callers; embedded searches do not lock memory
//! or flush the cache as part of normal API execution.

use crate::{Error, Result};
use orion_ann_core::index::CompressedIndex;
use orion_ann_core::model::dataset::rows::CascadeData;
use orion_ann_core::search::cascade::stage::admission::{
    L2KTAdmission, L2U16Admission, L2U8Admission, MipsI16Admission, MipsI8Admission,
};
use orion_ann_core::search::cascade::stage::prefilter::{
    JlHadamardPrefilter, JlMipsPrefilter, JlPrefilter, NoPrefilter, RabitqPrefilter,
};
use orion_ann_core::search::cascade::stage::rerank::{NoRerank, U16Rerank};
#[cfg(feature = "benchmark")]
use orion_ann_core::{DynamicCascade, DynamicSearchConfig};

pub use crate::config::cascade::*;
/// Build an official prefilter adapter, including `NoPrefilter`. The core
/// resolves the no-op stage to its no-filter traversal fast path.
///
/// `admission` is consulted to decide whether the JL prefilter should
/// run in MIPS mode (per PA's `Mips_JL_Sparse_Point_Normalized`
/// design — `popcount × ‖v‖` instead of raw popcount). The L2 and
/// MIPS scoring rules diverge once the base vectors aren't unit-norm,
/// so the prefilter must agree with what the admission tier is
/// computing.
pub fn build_prefilter<'a, D: CascadeData>(
    orion: &'a CompressedIndex<D>,
    choice: PrefilterChoice,
    admission: AdmissionChoice,
) -> Result<Box<orion_ann_core::search::cascade::stage::Prefilter<'a>>> {
    let admission_is_mips = admission.uses_inner_product();
    Ok(match choice {
        PrefilterChoice::None => Box::new(NoPrefilter),
        PrefilterChoice::Jl => {
            if admission_is_mips {
                // NZ=9, `popcount × ‖v‖` scoring — PA's MIPS-JL recipe
                // (`Mips_JL_Sparse_Point_Normalized`). The MIPS-only
                // adapter wraps `JLSparseDatasetMips`, which carries
                // the per-vertex `‖v‖` slab the L2 type doesn't.
                let ds = orion.try_ensure_quantized_dataset_jl_mips()?;
                Box::new(JlMipsPrefilter::new(ds))
            } else {
                // NZ=9, raw popcount — the L2-JL default.
                let ds = orion.try_ensure_quantized_dataset_jl()?;
                Box::new(JlPrefilter::new(ds))
            }
        }
        PrefilterChoice::JlHadamard => {
            let ds = orion.try_ensure_quantized_dataset_jl_hadamard()?;
            Box::new(JlHadamardPrefilter::new(ds))
        }
        PrefilterChoice::Rabitq => {
            let ds = orion.try_ensure_quantized_dataset_rabitq()?;
            Box::new(RabitqPrefilter::new(ds))
        }
    })
}

/// Build an admission adapter independently of the base element width.
///
/// Returns a configuration error for native-byte admission, including callers
/// with an incompatible base, including callers outside standalone CLI validation.
pub fn build_admission<'a, D: CascadeData>(
    orion: &'a CompressedIndex<D>,
    choice: AdmissionChoice,
) -> Result<Box<orion_ann_core::search::cascade::stage::Admission<'a>>> {
    Ok(match choice {
        AdmissionChoice::PqL2 | AdmissionChoice::PqIp | AdmissionChoice::PqCosine => Box::new(
            orion_ann_core::search::cascade::stage::admission::PqAdmission::new(
                orion.try_ensure_quantized_dataset_pq()?,
                if choice.uses_inner_product() {
                    orion_ann_core::model::dataset::quantized_dataset::pq::PqMetric::InnerProduct
                } else {
                    orion_ann_core::model::dataset::quantized_dataset::pq::PqMetric::L2
                },
            ),
        ),
        AdmissionChoice::NativeL2U8
        | AdmissionChoice::NativeIpU8
        | AdmissionChoice::NativeCosineU8 => {
            if orion.dataset().padded_u8().is_none() {
                return Err(Error::configuration(
                    "admission",
                    "native-byte admission requires u8 base storage",
                ));
            }
            Box::new(orion.exact_stage(match choice {
                AdmissionChoice::NativeL2U8 => orion_ann_core::ScoreMetric::SquaredL2,
                AdmissionChoice::NativeIpU8 => orion_ann_core::ScoreMetric::InnerProduct,
                _ => orion_ann_core::ScoreMetric::CosineSimilarity,
            })?)
        }
        AdmissionChoice::L2U8 => {
            Box::new(L2U8Admission::new(orion.try_ensure_quantized_dataset()?))
        }
        AdmissionChoice::L2U16 => Box::new(L2U16Admission::new(
            orion.try_ensure_quantized_dataset_l2_u16()?,
        )),
        AdmissionChoice::L2Kt => Box::new(L2KTAdmission::new(
            orion.try_ensure_quantized_dataset_l2_kt()?,
        )),
        AdmissionChoice::IpI8 | AdmissionChoice::CosineI8 => Box::new(MipsI8Admission::new(
            orion.try_ensure_quantized_dataset_mips()?,
        )),
        AdmissionChoice::IpI16 | AdmissionChoice::CosineI16 => Box::new(MipsI16Admission::new(
            orion.try_ensure_quantized_dataset_mips_i16()?,
        )),
        #[cfg(feature = "benchmark")]
        AdmissionChoice::AdsF32 => orion.dataset().ads(ads_epsilon())?,
    })
}

/// Build the rerank adapter.
pub fn build_rerank<'a, D: CascadeData>(
    orion: &'a CompressedIndex<D>,
    choice: RerankChoice,
) -> Result<Box<orion_ann_core::search::cascade::stage::Rerank<'a>>> {
    if orion.dataset().padded_u8().is_some() {
        let metric = match choice {
            RerankChoice::F32 => Some(orion_ann_core::ScoreMetric::SquaredL2),
            RerankChoice::IpF32 => Some(orion_ann_core::ScoreMetric::InnerProduct),
            RerankChoice::CosineF32 => Some(orion_ann_core::ScoreMetric::CosineSimilarity),
            _ => None,
        };
        if let Some(metric) = metric {
            return Ok(Box::new(orion.exact_stage(metric)?));
        }
    }
    Ok(match choice {
        RerankChoice::F32 => orion.dataset().truth(false),
        RerankChoice::IpF32 | RerankChoice::CosineF32 => orion.dataset().truth(true),
        RerankChoice::U16 => Box::new(U16Rerank::new(orion.try_ensure_quantized_dataset_l2_u16()?)),
        RerankChoice::None => Box::new(NoRerank),
    })
}

/// Single-query dispatch through the chosen cascade — same wiring
/// shape as [`search_batch_compose`], just routed to `search_observed`
/// instead of `search_batch`. Used by per-query diagnostic
/// loops in `main.rs` (ablation study, search_profile, etc.).
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "benchmark")]
pub fn search_compose<D: CascadeData>(
    orion: &CompressedIndex<D>,
    query: &[f32],
    k: usize,
    search_list_size: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) -> Result<Vec<u32>> {
    let recipe = Cascade {
        prefilter,
        admission,
        rerank,
    };
    let metric = admission.metric();
    recipe.validate_cascade_options(metric)?;
    crate::sidecars::requirements::prepare(orion, &recipe.sidecar_requirements())?;
    let ad = build_admission(orion, admission)?;
    let pf = build_prefilter(orion, prefilter, admission)?;
    let rr = build_rerank(orion, rerank)?;
    orion.search_observed(
        orion_ann_core::search::query::QueryInput::F32(query),
        DynamicSearchConfig {
            k,
            search_list_size,
            window_size,
            epsilon,
            early_exit_limit,
        },
        DynamicCascade {
            prefilter: pf.as_ref(),
            admission: ad.as_ref(),
            rerank: rr.as_ref(),
        },
        orion_ann_core::search::diagnostics::NeighborMode::default(),
        &mut orion_ann_core::search::diagnostics::NoopObserver,
    )
}

/// One-shot batched dispatch through the chosen cascade.
///
/// Builds all three adapters via the per-axis builders, then hands
/// the boxed dyn refs to `search_batch` (dyn vtable on the
/// hot loop). Prepare the requirements before timing; optional scoped pinning
/// consumes those allocations without fitting or changing the search recipe.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "benchmark")]
pub fn search_batch_compose<D: CascadeData>(
    orion: &CompressedIndex<D>,
    queries: &[f32],
    k: usize,
    search_list_size: usize,
    window_size: usize,
    epsilon: f32,
    early_exit_limit: usize,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) -> Result<orion_ann_core::SearchResult> {
    let recipe = Cascade {
        prefilter,
        admission,
        rerank,
    };
    let metric = admission.metric();
    recipe.validate_cascade_options(metric)?;
    crate::sidecars::requirements::prepare(orion, &recipe.sidecar_requirements())?;
    let ad = build_admission(orion, admission)?;
    let pf = build_prefilter(orion, prefilter, admission)?;
    let rr = build_rerank(orion, rerank)?;
    orion.search_batch(
        orion_ann_core::search::query::QueryInput::F32(queries),
        DynamicSearchConfig {
            k,
            search_list_size,
            window_size,
            epsilon,
            early_exit_limit,
        },
        DynamicCascade {
            prefilter: pf.as_ref(),
            admission: ad.as_ref(),
            rerank: rr.as_ref(),
        },
        None,
    )
}

#[cfg(all(test, feature = "benchmark"))]
#[path = "tests/cascade.rs"]
mod tests;

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resident sidecars selected for benchmark timing.
use crate::config::cascade::{AdmissionChoice, PrefilterChoice, RerankChoice};
use orion_ann_core::{index::compressed_index::CompressedIndex, model::dataset::rows::CascadeData};
/// Pin the JL sidecar when the prefilter is JL. Uses the
/// admission-family hint to pick the L2 (NZ=9) vs MIPS (NZ=11)
/// sidecar — matches `build_prefilter`'s dispatch.
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn pin_prefilter<D: CascadeData>(
    orion: &CompressedIndex<D>,
    choice: PrefilterChoice,
    admission: AdmissionChoice,
) {
    let admission_is_mips = matches!(
        admission,
        AdmissionChoice::MipsI8 | AdmissionChoice::MipsI16
    );
    match choice {
        PrefilterChoice::None => {}
        PrefilterChoice::Jl => {
            if admission_is_mips {
                let q = match orion.try_ensure_quantized_dataset_jl_mips() {
                    Ok(sidecar) => sidecar,
                    Err(error) => {
                        log::warn!("cannot pin sidecar: {error}");
                        return;
                    }
                };
                crate::search::utils::mlock_bytes(
                    "prefilter (JL codes, mips)",
                    q.codes().as_ptr(),
                    q.codes().len(),
                );
                crate::search::utils::mlock_bytes(
                    "prefilter (JL indices, mips)",
                    q.indices().as_ptr() as *const u8,
                    q.indices().len() * std::mem::size_of::<u32>(),
                );
            } else {
                let q = match orion.try_ensure_quantized_dataset_jl() {
                    Ok(sidecar) => sidecar,
                    Err(error) => {
                        log::warn!("cannot pin sidecar: {error}");
                        return;
                    }
                };
                crate::search::utils::mlock_bytes(
                    "prefilter (JL codes)",
                    q.codes().as_ptr(),
                    q.codes().len(),
                );
                crate::search::utils::mlock_bytes(
                    "prefilter (JL indices)",
                    q.indices().as_ptr() as *const u8,
                    q.indices().len() * std::mem::size_of::<u32>(),
                );
            }
        }
        PrefilterChoice::JlHadamard => {
            let q = match orion.try_ensure_quantized_dataset_jl_hadamard() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "prefilter (JL Hadamard codes)",
                q.codes().as_ptr(),
                q.codes().len(),
            );
            // signs1/2/3 are 3 × D_PAD bytes (= 3072 bytes at D_PAD=1024);
            // negligible but pin for symmetry.
            crate::search::utils::mlock_bytes(
                "prefilter (JL Hadamard signs1)",
                q.signs1().as_ptr() as *const u8,
                q.signs1().len(),
            );
            crate::search::utils::mlock_bytes(
                "prefilter (JL Hadamard signs2)",
                q.signs2().as_ptr() as *const u8,
                q.signs2().len(),
            );
            crate::search::utils::mlock_bytes(
                "prefilter (JL Hadamard signs3)",
                q.signs3().as_ptr() as *const u8,
                q.signs3().len(),
            );
        }
        PrefilterChoice::Rabitq => {
            let q = match orion.try_ensure_quantized_dataset_rabitq() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "prefilter (RaBitQ codes)",
                q.codes().as_ptr(),
                q.codes().len(),
            );
            // Rotation matrix is hot at every search setup (rotate_query
            // is O(N²)). Pin it so the f32 matrix stays L1-resident
            // across queries.
            crate::search::utils::mlock_bytes(
                "prefilter (RaBitQ rotation)",
                q.rotation().as_ptr() as *const u8,
                q.rotation().len() * std::mem::size_of::<f32>(),
            );
        }
    }
}

/// Pin the admission sidecar. L2-KT pins both i8 base + per-vertex
/// `‖x‖²` companion (sink touches both every hop).
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn pin_admission<D: CascadeData>(orion: &CompressedIndex<D>, choice: AdmissionChoice) {
    match choice {
        AdmissionChoice::NativeL2U8 => {} // Native storage is pinned with its base.
        AdmissionChoice::L2U8 => {
            let q = match orion.try_ensure_quantized_dataset() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "admission (u8 base)",
                q.data().as_ptr() as *const u8,
                q.data().len() * std::mem::size_of::<u8>(),
            );
        }
        AdmissionChoice::L2U16 => {
            let q = match orion.try_ensure_quantized_dataset_l2_u16() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "admission (u16 base)",
                q.data().as_ptr() as *const u8,
                q.data().len() * std::mem::size_of::<u16>(),
            );
        }
        AdmissionChoice::L2Kt => {
            let q = match orion.try_ensure_quantized_dataset_l2_kt() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "admission (L2-KT i8 base)",
                q.data().as_ptr() as *const u8,
                q.data().len() * std::mem::size_of::<i8>(),
            );
            crate::search::utils::mlock_bytes(
                "admission (L2-KT ‖x‖²)",
                q.norms_sq().as_ptr() as *const u8,
                q.norms_sq().len() * std::mem::size_of::<i32>(),
            );
        }
        AdmissionChoice::MipsI8 => {
            let q = match orion.try_ensure_quantized_dataset_mips() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "admission (MIPS i8 base)",
                q.data().as_ptr() as *const u8,
                q.data().len() * std::mem::size_of::<i8>(),
            );
        }
        AdmissionChoice::MipsI16 => {
            let q = match orion.try_ensure_quantized_dataset_mips_i16() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "admission (MIPS i16 base)",
                q.data().as_ptr() as *const u8,
                q.data().len() * std::mem::size_of::<i16>(),
            );
        }
        AdmissionChoice::AdsF32 => {
            // ADS reads straight from the f32 base — pinned via
            // `pin_rerank` for the f32/ip-f32 cases, so we skip
            // double-pinning here. If a future cascade pairs ADS with
            // `U16` rerank, the f32 base would go un-pinned; in that
            // (currently unused) combination, ADS still needs the f32
            // base resident, so pin it here as a safety net.
            if !matches!(orion.dataset().allocation().len(), 0) {
                let d = orion.dataset().allocation();
                crate::search::utils::mlock_bytes(
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
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn pin_rerank<D: CascadeData>(orion: &CompressedIndex<D>, choice: RerankChoice) {
    match choice {
        RerankChoice::F32 | RerankChoice::IpF32 => {
            let d = orion.dataset().allocation();
            crate::search::utils::mlock_bytes(
                "rerank (f32 base)",
                d.as_ptr() as *const u8,
                d.len() * std::mem::size_of::<f32>(),
            );
        }
        RerankChoice::U16 => {
            let q = match orion.try_ensure_quantized_dataset_l2_u16() {
                Ok(sidecar) => sidecar,
                Err(error) => {
                    log::warn!("cannot pin sidecar: {error}");
                    return;
                }
            };
            crate::search::utils::mlock_bytes(
                "rerank (u16 base)",
                q.data().as_ptr() as *const u8,
                q.data().len() * std::mem::size_of::<u16>(),
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
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn pin_cascade<D: CascadeData>(
    orion: &CompressedIndex<D>,
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
) {
    pin_prefilter(orion, prefilter, admission);
    pin_admission(orion, admission);
    pin_rerank(orion, rerank);
}

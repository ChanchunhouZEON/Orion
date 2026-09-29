/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve storage representation and search-stage compatibility before any
//! large dataset allocation.
//!
//! The resulting [`SearchPlan`] is the validated execution contract shared by
//! loading, preflight estimation, memory accounting, and search dispatch.

use super::config::{GraphSource, VectorStorageKind};
use super::data::VectorFormat;
use crate::cascade::{
    AdmissionChoice,
    Cascade,
    PrefilterChoice,
    RerankChoice,
    SearchMetric,
};

/// Fully resolved search execution plan.
///
/// Resolution collapses the user-facing combination of storage, metric,
/// vector formats, cascade stages, and graph source into one of the supported
/// execution paths:
///
/// - [`SearchPlan::F32Cascade`] keeps the base in `f32` and uses the configured
///   precision cascade.
/// - [`SearchPlan::NativeU8L2`] keeps the base in its original `u8` form and
///   performs native byte-space L2 admission without auxiliary reranking.
///
/// Once constructed through [`SearchPlan::resolve`], a plan is expected to be
/// internally compatible and may be consumed directly by loaders and search
/// dispatch without repeating option validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPlan {
    /// Conventional `f32` resident storage with a configurable search cascade.
    F32Cascade { cascade: Cascade },

    /// Native `u8` resident storage with exact L2 admission over the byte base.
    NativeU8L2,
}

/// Resident-storage properties derived from a validated [`SearchPlan`].
///
/// This is the shared source of truth for loaders, preflight checks, and
/// resident-memory accounting, avoiding independent interpretations of the
/// user-facing configuration.
pub struct StorageLayout {
    /// Resident base-vector representation.
    pub kind: VectorStorageKind,

    /// Bytes occupied by one resident coordinate.
    pub element_bytes: usize,

    /// Whether the plan additionally requires a quantized `u8` L2 admission
    /// sidecar alongside an `f32` resident base.
    pub has_l2_u8_sidecar: bool,
}

impl SearchPlan {
    /// Resolves user-facing search options into one validated execution plan.
    ///
    /// Validation happens before large allocations so unsupported combinations
    /// fail early. In particular, native `u8` execution is intentionally
    /// stricter than the generic `f32` cascade path because it preserves the
    /// original byte representation rather than constructing auxiliary
    /// quantized storage.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        vector_storage: VectorStorageKind,
        dimension: usize,
        metric: SearchMetric,
        base_format: VectorFormat,
        query_format: VectorFormat,
        cascade: Cascade,
        graph_source: GraphSource,
    ) -> Result<Self, String> {
        // The conventional f32 path accepts the configured cascade unchanged.
        //
        // `NativeL2U8`, however, means "score directly against a resident u8
        // base". It therefore cannot be selected when the base itself is loaded
        // as f32. `L2U8` is the corresponding choice when an auxiliary
        // quantized sidecar is desired on top of f32 storage.
        if vector_storage == VectorStorageKind::F32 {
            return if cascade.admission == AdmissionChoice::NativeL2U8 {
                Err(
                    "native-l2-u8 requires --vector-storage u8; \
                     use l2-u8 for a quantized f32 sidecar"
                        .into(),
                )
            } else {
                Ok(Self::F32Cascade { cascade })
            };
        }

        // Native u8 execution currently has a deliberately narrow contract:
        // SIFT-style 128-dimensional vectors under the L2 metric. Keeping this
        // restriction here ensures downstream code can rely on those invariants
        // without branching again.
        if dimension != 128 || metric != SearchMetric::L2 {
            return Err(
                "native u8 storage currently requires \
                 128-dimensional L2 vectors"
                    .into(),
            );
        }

        // Native byte storage must come from byte-encoded inputs directly.
        // Float encodings are rejected rather than silently quantized because
        // quantization requires an explicit policy and would defeat the purpose
        // of preserving the original resident representation.
        if !matches!(
            base_format,
            VectorFormat::Bvecs | VectorFormat::U8bin
        ) || !matches!(
            query_format,
            VectorFormat::Bvecs | VectorFormat::U8bin
        ) {
            return Err(
                "native u8 requires bvecs/u8bin base and queries; \
                 it does not quantize floats"
                    .into(),
            );
        }

        // The native-u8 path is not a configurable three-stage cascade:
        //
        //     no prefilter -> native byte L2 admission -> no rerank
        //
        // Requiring the explicit stage choices keeps the resolved plan and the
        // run log aligned with the actual execution path.
        if cascade.prefilter != PrefilterChoice::None
            || cascade.admission != AdmissionChoice::NativeL2U8
            || cascade.rerank != RerankChoice::None
        {
            return Err(
                "native u8 requires --prefilter none \
                 --admission native-l2-u8 --rerank none"
                    .into(),
            );
        }

        // Native u8 currently consumes a ParlayANN STAG graph export/cache.
        // The in-process Rust graph builder still operates on f32 data, so
        // permitting another graph source here would imply a construction path
        // that does not exist.
        if graph_source != GraphSource::Parlayann {
            return Err(
                "native u8 uses a ParlayANN STAG export/cache; \
                 the Rust builder remains f32"
                    .into(),
            );
        }

        Ok(Self::NativeU8L2)
    }

    /// Returns the effective cascade represented by this execution plan.
    ///
    /// Native u8 has no stored `Cascade` field because its stage composition is
    /// fixed by construction; this method reconstructs that canonical form when
    /// callers need a uniform view for logging or reporting.
    pub fn cascade(self) -> Cascade {
        match self {
            Self::F32Cascade { cascade } => cascade,

            Self::NativeU8L2 => Cascade {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::NativeL2U8,
                rerank: RerankChoice::None,
            },
        }
    }

    /// Returns the resident-storage layout implied by this plan.
    ///
    /// This keeps storage decisions derived from the validated execution plan
    /// rather than duplicating mutable configuration elsewhere.
    pub fn storage(self) -> StorageLayout {
        match self {
            Self::NativeU8L2 => StorageLayout {
                kind: VectorStorageKind::U8,
                element_bytes: 1,
                has_l2_u8_sidecar: false,
            },

            Self::F32Cascade { cascade } => StorageLayout {
                kind: VectorStorageKind::F32,
                element_bytes: 4,

                // `L2U8` denotes a quantized admission sidecar attached to an
                // otherwise f32-resident search path.
                has_l2_u8_sidecar:
                cascade.admission == AdmissionChoice::L2U8,
            },
        }
    }
}

// Preserve the existing public run-log shape without storing duplicate mutable
// copies of vector storage and cascade configuration inside `SearchPlan`.
//
// Both fields are instead derived from the validated plan at serialization
// time, keeping the plan itself as the single source of truth.
impl serde::Serialize for SearchPlan {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(2))?;

        map.serialize_entry(
            "vector_storage",
            &self.storage().kind,
        )?;

        map.serialize_entry(
            "cascade",
            &self.cascade(),
        )?;

        map.end()
    }
}
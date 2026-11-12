/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve base representation and search compatibility together, before I/O.
//! Only this module can assemble a plan; other layers borrow validated views.
use crate::config::cascade::{AdmissionChoice, Cascade, SearchMetric};
use crate::config::{GraphSource, VectorStorageKind};
use crate::data::VectorFormat;
use crate::{Error, Result};

/// Immutable compatibility boundary for base representation and search recipe.
/// The private fields prevent independently replacing one half of a valid plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexPlan {
    base: BaseRepresentation,
    search: Cascade,
}

/// Coordinate representation, independent of the source file and search stages.
/// Padding/stride remain properties of the dataset's PaddedLayout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BaseRepresentation(VectorStorageKind);
impl IndexPlan {
    /// Resolves user-facing search options into one validated execution plan.
    ///
    /// Validation happens before large allocations so unsupported combinations
    /// fail early. Storage selects row layout; every representation uses the
    /// same precision-checked cascade and optional quantized sidecars.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        vector_storage: VectorStorageKind,
        dimension: usize,
        metric: SearchMetric,
        base_format: VectorFormat,
        _query_format: VectorFormat,
        cascade: Cascade,
        graph_source: GraphSource,
    ) -> Result<Self> {
        cascade.validate_cascade_options(metric)?;
        match vector_storage {
            VectorStorageKind::F32 => {
                orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(
                    dimension,
                )?;
            }
            VectorStorageKind::U8 => {
                orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::IpU8Distance>(
                    dimension,
                )?;
            }
        }

        // Quantized stages also have accumulator bounds. Check them before
        // allocating the base instead of letting lazy sidecar fitting panic.
        {
            match cascade.admission {
                AdmissionChoice::L2U8 => {
                    orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(
                        dimension,
                    )?;
                }
                AdmissionChoice::L2U16 => {
                    orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2U16Distance>(
                        dimension,
                    )?;
                }
                AdmissionChoice::IpI8 | AdmissionChoice::CosineI8 | AdmissionChoice::L2Kt => {
                    orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::IpI8Distance>(
                        dimension,
                    )?;
                }
                AdmissionChoice::IpI16 | AdmissionChoice::CosineI16 => {
                    orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::IpI16Distance>(
                        dimension,
                    )?;
                }
                _ => {}
            }
        }

        // The conventional f32 path accepts the configured cascade unchanged.
        //
        // Native admissions mean "score directly against a resident u8
        // base". It therefore cannot be selected when the base itself is loaded
        // as f32. `L2U8` is the corresponding choice when an auxiliary
        // quantized sidecar is desired on top of f32 storage.
        if vector_storage == VectorStorageKind::F32 {
            return if matches!(
                cascade.admission,
                AdmissionChoice::NativeL2U8
                    | AdmissionChoice::NativeIpU8
                    | AdmissionChoice::NativeCosineU8
            ) {
                Err(Error::unsupported(
                    "native admission requires --vector-storage u8; use a quantized or exact f32 cascade for f32 storage",
                )
                .at("configure"))
            } else {
                Ok(Self {
                    base: BaseRepresentation(vector_storage),
                    search: cascade,
                })
            };
        }

        // Native cosine keeps byte coordinates; norms provide normalization.
        if metric == SearchMetric::SquaredL2 {
            orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(
                dimension,
            )?;
        }
        // Native byte storage must come from byte-encoded inputs directly.
        // Float encodings are rejected rather than silently quantized because
        // quantization requires an explicit policy and would defeat the purpose
        // of preserving the original resident representation.
        let base = BaseRepresentation(vector_storage);
        base.validate_format(base_format)?;
        // Query dtype is independent of resident base storage.

        #[cfg(feature = "benchmark")]
        if cascade.admission == AdmissionChoice::AdsF32 {
            return Err(Error::unsupported("ADSampling requires f32 base storage"));
        }

        // Native u8 uses the ParlayANN in-memory builder or a STAG export/cache.
        // The in-process Rust graph builder still operates on f32 data, so
        // permitting another graph source here would imply a construction path
        // that does not exist.
        if !matches!(graph_source, GraphSource::Parlayann | GraphSource::Supplied) {
            return Err(Error::unsupported(
                "native u8 requires ParlayANN or supplied partitions; the Rust builder remains f32",
            )
            .at("configure"));
        }

        Ok(Self {
            base: BaseRepresentation(vector_storage),
            search: cascade,
        })
    }

    pub(crate) fn base(self) -> BaseRepresentation {
        self.base
    }
    pub(crate) fn cascade(self) -> Cascade {
        self.search
    }

    /// Resolve auxiliary allocations for the run phase from the validated recipe.
    /// The same list drives preparation and preflight; storage arithmetic stays
    /// with the quantized dataset implementation.
    pub(crate) fn sidecar_requirements(
        self,
        prepare_only: bool,
    ) -> Vec<crate::sidecars::requirements::SidecarRequirement> {
        if prepare_only {
            return Vec::new();
        }
        self.search.sidecar_requirements()
    }
}

impl BaseRepresentation {
    pub(crate) fn validate_format(self, format: VectorFormat) -> Result<()> {
        format.validate_base()?;
        if self.0 == VectorStorageKind::U8
            && !matches!(format, VectorFormat::Bvecs | VectorFormat::U8bin)
        {
            return Err(Error::unsupported(
                "native u8 base requires bvecs/u8bin; base loading does not quantize floats",
            )
            .at("configure"));
        }
        Ok(())
    }

    pub(crate) fn kind(self) -> VectorStorageKind {
        self.0
    }
    pub(crate) fn element_bytes(self) -> usize {
        match self.0 {
            VectorStorageKind::F32 => size_of::<f32>(),
            VectorStorageKind::U8 => size_of::<u8>(),
        }
    }
}

// Keep run logs and snapshot execution signatures stable across this refactor.
impl serde::Serialize for IndexPlan {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("vector_storage", &self.base.kind())?;
        map.serialize_entry("cascade", &self.search)?;
        map.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::cascade::{PrefilterChoice, RerankChoice};

    fn native_cascade() -> Cascade {
        Cascade {
            prefilter: PrefilterChoice::None,
            admission: AdmissionChoice::NativeL2U8,
            rerank: RerankChoice::None,
        }
    }

    #[test]
    fn resolved_plans_preserve_the_existing_serialized_identity() {
        // This was the original SearchPlan wire representation. Snapshot
        // execution signatures hash these exact bytes, including field order.
        #[derive(serde::Serialize)]
        struct PreviousWire {
            vector_storage: VectorStorageKind,
            cascade: Cascade,
        }
        for (storage, format, source, cascade, element_bytes) in [
            (
                VectorStorageKind::F32,
                VectorFormat::Bvecs,
                GraphSource::Rust,
                Cascade::default_for_specified_dimension_and_metric(128, SearchMetric::SquaredL2),
                4,
            ),
            (
                VectorStorageKind::U8,
                VectorFormat::U8bin,
                GraphSource::Parlayann,
                native_cascade(),
                1,
            ),
        ] {
            let plan = IndexPlan::resolve(
                storage,
                128,
                SearchMetric::SquaredL2,
                format,
                format,
                cascade,
                source,
            )
            .unwrap();
            assert_eq!(plan.base().kind(), storage);
            assert_eq!(plan.base().element_bytes(), element_bytes);
            assert_eq!(
                serde_json::to_vec(&plan).unwrap(),
                serde_json::to_vec(&PreviousWire {
                    vector_storage: storage,
                    cascade
                })
                .unwrap()
            );
        }
    }

    #[test]
    fn resolver_rejects_incompatible_representation_and_search_recipes() {
        let native = native_cascade();
        let quantized =
            Cascade::default_for_specified_dimension_and_metric(128, SearchMetric::SquaredL2);
        for (storage, metric, format, source, cascade) in [
            (
                VectorStorageKind::F32,
                SearchMetric::SquaredL2,
                VectorFormat::Fbin,
                GraphSource::Rust,
                native,
            ),
            (
                VectorStorageKind::U8,
                SearchMetric::InnerProduct,
                VectorFormat::U8bin,
                GraphSource::Parlayann,
                native,
            ),
            (
                VectorStorageKind::U8,
                SearchMetric::SquaredL2,
                VectorFormat::Fbin,
                GraphSource::Parlayann,
                native,
            ),
            (
                VectorStorageKind::U8,
                SearchMetric::SquaredL2,
                VectorFormat::U8bin,
                GraphSource::Rust,
                native,
            ),
            (
                VectorStorageKind::F32,
                SearchMetric::InnerProduct,
                VectorFormat::Fbin,
                GraphSource::Rust,
                quantized,
            ),
        ] {
            assert!(
                IndexPlan::resolve(storage, 128, metric, format, format, cascade, source).is_err()
            );
        }
    }
}

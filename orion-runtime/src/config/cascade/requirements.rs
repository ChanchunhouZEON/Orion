/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Translate consumers to dataset requirements; allocation deduplicates them.
use super::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice};
use crate::sidecars::requirements::SidecarRecipe;

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
        let uses_base = match self.admission {
            AdmissionChoice::NativeL2U8
            | AdmissionChoice::NativeIpU8
            | AdmissionChoice::NativeCosineU8 => true,
            #[cfg(feature = "benchmark")]
            AdmissionChoice::AdsF32 => true,
            _ => false,
        };
        if uses_base
            || matches!(
                self.rerank,
                RerankChoice::F32 | RerankChoice::IpF32 | RerankChoice::CosineF32
            )
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
        let mips = self.admission.uses_inner_product();
        match self.prefilter {
            PrefilterChoice::None => {}
            PrefilterChoice::Jl => kinds.push(if mips { K::JlSparseMips } else { K::JlSparse }),
            PrefilterChoice::JlHadamard => kinds.push(K::JlHadamard),
            PrefilterChoice::Rabitq => kinds.push(K::Rabitq),
        }
        match self.admission {
            AdmissionChoice::PqL2 | AdmissionChoice::PqIp | AdmissionChoice::PqCosine => {
                kinds.push(K::Pq)
            }
            AdmissionChoice::L2U8 => kinds.push(K::L2U8),
            AdmissionChoice::L2U16 => kinds.push(K::L2U16),
            AdmissionChoice::IpI8 | AdmissionChoice::CosineI8 => kinds.push(K::MipsI8),
            AdmissionChoice::IpI16 | AdmissionChoice::CosineI16 => kinds.push(K::MipsI16),
            AdmissionChoice::L2Kt => kinds.push(K::L2Kt),
            AdmissionChoice::NativeL2U8
            | AdmissionChoice::NativeIpU8
            | AdmissionChoice::NativeCosineU8 => {}
            #[cfg(feature = "benchmark")]
            AdmissionChoice::AdsF32 => {}
        }
        if self.rerank == RerankChoice::U16 {
            kinds.push(K::L2U16);
        }
        kinds.into_iter().map(K::requirement).collect()
    }
}

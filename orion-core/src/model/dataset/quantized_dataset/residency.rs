/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Every owned, initialized search buffer. Pinning does not build datasets.
use super::{
    jl::{
        hadamard::JlHadamardDataset,
        sparse::{JLSparseDataset, JLSparseDatasetMips},
    },
    kernel_trick::L2KTDataset,
    rabitq::{binary::RabitQDataset, multibit::RabitQ4Dataset},
    scalar::{ScalarQuantizedDataset, spec::QuantSpec},
};

/// Enumerates initialized backing spans without asserting physical RAM residency.
///
/// Callers apply placement policy before locking pages; mapped spans may be included.
pub(crate) trait ResidentRegions {
    /// Borrows initialized payload spans for residency selection without copying them.
    ///
    /// This inventory may include mappings; the caller applies placement and pin
    /// policy. A borrowed slice does not imply that its pages currently occupy RAM.
    fn resident_regions(&self) -> Vec<&[u8]>;
}
impl<Q: QuantSpec> ResidentRegions for ScalarQuantizedDataset<Q> {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![bytemuck::cast_slice(self.data())]
    }
}
impl ResidentRegions for L2KTDataset {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.data()),
            bytemuck::cast_slice(self.norms_sq()),
        ]
    }
}
impl<const B: usize, const NZ: usize> ResidentRegions for JLSparseDataset<B, NZ> {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![bytemuck::cast_slice(self.indices()), self.codes()]
    }
}
impl<const B: usize, const NZ: usize> ResidentRegions for JLSparseDatasetMips<B, NZ> {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.indices()),
            bytemuck::cast_slice(self.norms()),
            self.codes(),
        ]
    }
}
impl ResidentRegions for JlHadamardDataset {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.signs1()),
            bytemuck::cast_slice(self.signs2()),
            bytemuck::cast_slice(self.signs3()),
            self.codes(),
        ]
    }
}
impl ResidentRegions for RabitQDataset {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.rotation()),
            bytemuck::cast_slice(self.norms()),
            bytemuck::cast_slice(self.s_values()),
            self.codes(),
        ]
    }
}
impl ResidentRegions for RabitQ4Dataset {
    /// Borrows payload spans eligible for page locking without transferring their ownership.
    ///
    /// See [`ResidentRegions::resident_regions`] for the shared contract.
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.rotation()),
            bytemuck::cast_slice(self.norms()),
            bytemuck::cast_slice(self.taus()),
            self.codes(),
        ]
    }
}

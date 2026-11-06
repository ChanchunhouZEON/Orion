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

pub(crate) trait ResidentRegions {
    fn resident_regions(&self) -> Vec<&[u8]>;
}
impl<Q: QuantSpec> ResidentRegions for ScalarQuantizedDataset<Q> {
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![bytemuck::cast_slice(self.data())]
    }
}
impl ResidentRegions for L2KTDataset {
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.data()),
            bytemuck::cast_slice(self.norms_sq()),
        ]
    }
}
impl<const B: usize, const NZ: usize> ResidentRegions for JLSparseDataset<B, NZ> {
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![bytemuck::cast_slice(self.indices()), self.codes()]
    }
}
impl<const B: usize, const NZ: usize> ResidentRegions for JLSparseDatasetMips<B, NZ> {
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.indices()),
            bytemuck::cast_slice(self.norms()),
            self.codes(),
        ]
    }
}
impl ResidentRegions for JlHadamardDataset {
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
    fn resident_regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(self.rotation()),
            bytemuck::cast_slice(self.norms()),
            bytemuck::cast_slice(self.taus()),
            self.codes(),
        ]
    }
}

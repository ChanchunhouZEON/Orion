/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Checked adapters around the existing computation entry points.
//! Search continues to use its specialized stream adapters directly.
use super::{PreparedQuery, QuantizedDatasetCompute, sealed};
use crate::model::dataset::{
    JLSparseDataset, JLSparseDatasetMips, JlHadamardDataset, L2KTDataset, QuantSpec,
    RabitQ4Dataset, ScalarQuantizedDataset, rabitq_dataset::RabitQDataset,
};
use crate::{Error, Result};

/// Encoded query bound to its encoder; fields are private to prevent rebinding.
struct EncodedQuery<'a, D: Compute> {
    dataset: &'a D,
    query: D::Query,
}
impl<'a, D: Compute> EncodedQuery<'a, D> {
    pub(super) fn new(dataset: &'a D, query: D::Query) -> Self {
        Self { dataset, query }
    }
}
impl<D: Compute> sealed::Sealed for EncodedQuery<'_, D> {}
impl<D: Compute> PreparedQuery for EncodedQuery<'_, D> {
    fn distance(&self, id: u32) -> Result<f32> {
        self.dataset.score_query(&self.query, id)
    }
}
impl<D: Compute> std::fmt::Debug for EncodedQuery<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedQuery").finish_non_exhaustive()
    }
}

// This trait is deliberately not a public supertrait. Otherwise its encoding
// methods would remain callable through generic public bounds.
trait Compute: sealed::Sealed {
    type Query;
    fn encode_query_checked(&self, query: &[f32]) -> Result<Self::Query>;
    fn score_query(&self, query: &Self::Query, id: u32) -> Result<f32>;
}
impl<D: Compute> QuantizedDatasetCompute for D {
    fn prepare_query(&self, query: &[f32]) -> Result<impl PreparedQuery + '_> {
        Ok(EncodedQuery::new(self, self.encode_query_checked(query)?))
    }
}
fn check_query(query: &[f32], dimension: usize) -> Result<()> {
    if query.len() != dimension {
        return Err(Error::dimension("sidecar query", dimension, query.len()));
    }
    super::super::validation::finite(query, false)
}
fn check_row(id: u32, rows: usize) -> Result<()> {
    if id as usize >= rows {
        return Err(Error::invalid_argument("sidecar row ID out of bounds").with_detail("id", id));
    }
    Ok(())
}
fn check_len(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::dimension("encoded query", expected, actual));
    }
    Ok(())
}
impl<Q: QuantSpec> Compute for ScalarQuantizedDataset<Q> {
    type Query = Vec<Q::Storage>;
    fn encode_query_checked(&self, query: &[f32]) -> Result<Self::Query> {
        self.try_quantize_query(query, true)
    }
    fn score_query(&self, query: &Self::Query, id: u32) -> Result<f32> {
        self.distance(id, query)
    }
}
impl Compute for L2KTDataset {
    type Query = Vec<i8>;
    fn encode_query_checked(&self, query: &[f32]) -> Result<Self::Query> {
        check_query(query, self.dimension())?;
        Ok(self.quantize_query_padded(query))
    }
    fn score_query(&self, query: &Self::Query, id: u32) -> Result<f32> {
        check_row(id, self.len())?;
        check_len(query.len(), self.stride())?;
        let mut neg_ip = 0.0;
        // SAFETY: immutable dataset and checked row/stride cover every stream load.
        unsafe {
            orion_ann_vector::DistanceStream::<orion_ann_vector::IpI8Distance>::new(
                self.data().as_ptr().cast(),
                self.stride(),
                self.stride(),
                query.as_ptr().cast(),
                &[id],
                0,
            )
            .run(|_, distance| neg_ip = distance);
        }
        Ok(self.query_norm_sq(query) as f32 + self.norms_sq()[id as usize] as f32 + 2.0 * neg_ip)
    }
}
macro_rules! jl_compute {
    ($type:ty, [$($generics:tt)*], $score:ident, $stride:expr) => {
        impl<$($generics)*> Compute for $type {
            type Query = Box<[u8]>;
            fn encode_query_checked(&self, query: &[f32]) -> Result<Self::Query> {
                check_query(query, self.dimension())?;
                Ok(self.encode_query(query))
            }
            fn score_query(&self, query: &Self::Query, id: u32) -> Result<f32> {
                check_row(id, self.len())?;
                check_len(query.len(), ($stride)(self))?;
                Ok(self.$score(query, id) as f32)
            }
        }
    };
}
jl_compute!(JLSparseDataset<BITS, NZ>, [const BITS: usize, const NZ: usize], hamming,
    |_: &JLSparseDataset<BITS, NZ>| JLSparseDataset::<BITS, NZ>::STRIDE);
jl_compute!(JLSparseDatasetMips<BITS, NZ>, [const BITS: usize, const NZ: usize], mips_distance,
    |_: &JLSparseDatasetMips<BITS, NZ>| JLSparseDatasetMips::<BITS, NZ>::STRIDE);
jl_compute!(JlHadamardDataset, [], hamming, |ds: &JlHadamardDataset| {
    super::super::jl_hadamard_stride(ds.d_pad())
});

/// Rotation-space query and its original squared norm.
#[derive(Debug)]
struct RotatedQuery {
    coordinates: Vec<f32>,
    norm_squared: f32,
}
macro_rules! rotated_compute {
    ($type:ty) => {
        impl Compute for $type {
            type Query = RotatedQuery;
            fn encode_query_checked(&self, query: &[f32]) -> Result<Self::Query> {
                check_query(query, self.dimension())?;
                let mut coordinates = vec![0.0; self.dimension()];
                self.rotate_query(query, &mut coordinates);
                Ok(RotatedQuery {
                    coordinates,
                    norm_squared: query.iter().map(|v| v * v).sum(),
                })
            }
            fn score_query(&self, query: &Self::Query, id: u32) -> Result<f32> {
                check_row(id, self.len())?;
                check_len(query.coordinates.len(), self.dimension())?;
                Ok(self.estimate_l2_sq(&query.coordinates, query.norm_squared, id))
            }
        }
    };
}
rotated_compute!(RabitQDataset);
rotated_compute!(RabitQ4Dataset);

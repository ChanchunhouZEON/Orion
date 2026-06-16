/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! IP-based f32 truth rerank for MIPS-family recipes.
//!
//! Reads the top-(k · rerank_factor) PQ entries against the f32 base
//! via `DistanceStream<IpF32Distance, N>`, returns `-IP` so the
//! "smaller == closer" convention works directly in the sort, and
//! **unit-normalises the query inside the rerank** before the stream
//! pass.
//!
//! Query normalisation is required because the f32 base lives on the
//! unit sphere for MIPS workloads (build-time normalisation); a
//! non-normalised query against a unit base via inner product
//! produces a magnitude-skewed ranking that silently loses 10+ pp
//! of recall at low L.

use super::super::super::NDC_F32;
use super::super::super::utils::dstream_la_truth;
use super::RerankStage;
use crate::model::Neighbor;
use crate::model::scratch::InMemSearchScratch;
use diskann::model::InmemDataset;

/// IP-based f32 truth rerank stage adapter.
///
/// Use for any recipe whose admission tier is MIPS-family — i.e.
/// `LowDimMips` (i8 sdot) and `MipsPrecise` (i16 IP). The L2-based
/// [`super::f32_truth::F32Rerank`] also produces a correct ranking
/// for unit-normalised vectors (since `‖q − x‖² = 2 − 2·IP` is
/// monotone in IP), but only when the query is also unit-normalised
/// — and `F32Rerank` doesn't normalise.
pub struct IpF32Rerank<'a, const N: usize> {
    pub dataset: &'a InmemDataset<f32, N>,
}

impl<'a, const N: usize> IpF32Rerank<'a, N> {
    #[inline]
    pub fn new(dataset: &'a InmemDataset<f32, N>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> RerankStage<N> for IpF32Rerank<'a, N> {
    fn rerank(
        &self,
        query: &[f32; N],
        pq_entries: &[Neighbor],
        k: usize,
        rerank_factor: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Vec<u32> {
        let _ = pq_entries;

        let beam_n = scratch.pq.size();
        let num_check = (k * rerank_factor).min(beam_n);
        if num_check == 0 {
            return Vec::new();
        }

        // Pad the query into the SIMD-aligned scratch buffer.
        let f32_stride_bytes = N * 4;
        let f32_compute_bytes = f32_stride_bytes.div_ceil(32) * 32;
        let f32_padded_lanes = f32_compute_bytes / 4;
        scratch.q_query_f32_padded.clear();
        scratch.q_query_f32_padded.resize(f32_padded_lanes, 0.0);
        scratch.q_query_f32_padded[..N].copy_from_slice(query);

        scratch.id_scratch.clear();
        for i in 0..num_check {
            scratch.id_scratch.push(scratch.pq[i].id);
        }

        let dataset_base_ptr = self.dataset.get_data().as_ptr() as *const u8;
        let mut rerank_buf: Vec<(u32, f32)> = Vec::with_capacity(num_check);
        unsafe {
            let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), num_check);
            vector::DistanceStream::<vector::IpF32Distance, N>::new(
                dataset_base_ptr,
                f32_stride_bytes,
                f32_compute_bytes,
                scratch.q_query_f32_padded.as_ptr() as *const u8,
                id_in,
                dstream_la_truth(),
            )
            .run(|id, dist| {
                rerank_buf.push((id, dist));
            });
        }
        NDC_F32.add(num_check as u64);
        rerank_buf.sort_unstable_by(|a, b| {
            a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
        });

        let n_out = k.min(rerank_buf.len());
        let mut out: Vec<u32> = Vec::with_capacity(n_out);
        unsafe {
            let dst = out.as_mut_ptr();
            let src = rerank_buf.as_ptr();
            let full_chunks = n_out / 10;
            for c in 0..full_chunks {
                let base = c * 10;
                *dst.add(base) = (*src.add(base)).0;
                *dst.add(base + 1) = (*src.add(base + 1)).0;
                *dst.add(base + 2) = (*src.add(base + 2)).0;
                *dst.add(base + 3) = (*src.add(base + 3)).0;
                *dst.add(base + 4) = (*src.add(base + 4)).0;
                *dst.add(base + 5) = (*src.add(base + 5)).0;
                *dst.add(base + 6) = (*src.add(base + 6)).0;
                *dst.add(base + 7) = (*src.add(base + 7)).0;
                *dst.add(base + 8) = (*src.add(base + 8)).0;
                *dst.add(base + 9) = (*src.add(base + 9)).0;
            }
            for i in (full_chunks * 10)..n_out {
                *dst.add(i) = (*src.add(i)).0;
            }
            out.set_len(n_out);
        }
        out
    }
}

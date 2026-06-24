/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! u16 truth rerank — reads from the u16 sidecar (`-quantize_bits 16`
//! in PA terms).
//!
//! Rationale: at end-of-beam we only re-score `k·rerank_factor`
//! candidates (= 20 for k=10, rf=2), but each f32 vertex on GIST is
//! 30 cache lines (3.84 KB) while a u16 vertex is 15 lines (1.92 KB)
//! — 2× DRAM bandwidth saving for the rerank pass. The cascade's
//! final ordering is u16-precise, which matches PA's recipe exactly.

use super::super::super::NDC_F32;
use super::super::super::utils::dstream_la_truth;
use super::RerankStage;
use crate::model::Neighbor;
use crate::model::scratch::InMemSearchScratch;
use crate::model::{L2U16, QuantizedDataset};

/// u16-based truth rerank stage adapter.
///
/// Pairs with [`super::super::admission::L2U16Admission`] for PA-
/// aligned recipes that want the u16 admission tier to also be the
/// rerank precision.
pub struct U16Rerank<'a, const N: usize> {
    pub dataset: &'a QuantizedDataset<L2U16, N>,
}

impl<'a, const N: usize> U16Rerank<'a, N> {
    #[inline]
    pub fn new(dataset: &'a QuantizedDataset<L2U16, N>) -> Self {
        Self { dataset }
    }
}

impl<'a, const N: usize> RerankStage<N> for U16Rerank<'a, N> {
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

        // u16 STRIDE is N rounded up to ALIGN_ELEMS=16 (32-byte
        // stride); compute_bytes = STRIDE × 2 B is therefore already
        // a multiple of the kernel's 32-byte chunk size — no extra
        // padding needed.
        let stride_elems = QuantizedDataset::<L2U16, N>::STRIDE;
        let stride_bytes = stride_elems * std::mem::size_of::<u16>();
        let q_query_padded = self.dataset.quantize_query_padded(query);

        scratch.id_scratch.clear();
        for i in 0..num_check {
            scratch.id_scratch.push(scratch.pq[i].id);
        }

        let mut rerank_buf: Vec<(u32, f32)> = Vec::with_capacity(num_check);
        unsafe {
            let id_in = std::slice::from_raw_parts(scratch.id_scratch.as_ptr(), num_check);
            vector::DistanceStream::<vector::L2U16Distance, N>::new(
                self.dataset.data.as_slice().as_ptr() as *const u8,
                stride_bytes,
                stride_bytes,
                q_query_padded.as_ptr() as *const u8,
                id_in,
                dstream_la_truth(),
            )
            .run(|id, dist| {
                rerank_buf.push((id, dist));
            });
        }
        // Reuse the f32-truth counter slot — the rerank stage is what
        // it measures, regardless of the underlying storage precision.
        NDC_F32.add(num_check as u64);
        rerank_buf.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

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

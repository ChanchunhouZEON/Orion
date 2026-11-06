/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Exact L2 admission over original byte coordinates, borrowing the base buffer.
//!
//! This is distinct from affine-quantized L2U8Admission: there is no quantizer,
//! scale or sidecar. The same admission scores are the final ranking scores.

use super::{AdmissionSession, AdmissionStage};
use crate::model::Neighbor;
use orion_ann_diskann::model::InmemDataset;
use orion_ann_vector::FullPrecisionDistance;

pub struct NativeU8Admission<'a, const N: usize> {
    dataset: &'a InmemDataset<u8, N>,
}

impl<'a, const N: usize> NativeU8Admission<'a, N> {
    pub fn new(dataset: &'a InmemDataset<u8, N>) -> Self {
        assert!(N > 0, "stage dimension must be positive");
        let logical_len = dataset
            .num_active_pts
            .checked_mul(N)
            .expect("dataset span overflow");
        assert!(
            logical_len <= dataset.data.len(),
            "dataset allocation shorter than active rows"
        );
        // Both supported SIMD backends can consume complete 64-byte blocks.
        // The limit also keeps signed SIMD sums from overflowing.
        assert!(N > 0 && N % 64 == 0 && N <= 32_768);
        Self { dataset }
    }
}

// SAFETY: shape describes the borrowed base; sessions own checked query storage
// and cannot outlive that base.
unsafe impl<const N: usize> AdmissionStage for NativeU8Admission<'_, N> {
    fn precision(
        &self,
    ) -> Option<crate::model::dataset::quantized_dataset::capabilities::Precision> {
        use crate::model::dataset::quantized_dataset::capabilities::Precision as P;
        Some(P::Exact)
    }

    fn shape(&self) -> Option<(usize, usize)> {
        Some((N, self.dataset.num_active_pts))
    }
    fn open<'a>(&'a self, query: &[f32]) -> Box<dyn AdmissionSession + 'a> {
        let query: &[f32; N] = query.try_into().expect("query dimension mismatch");
        // The stage interface uses f32 queries. Reject fractional/out-of-range
        // values instead of silently turning exact search into quantization.
        assert!(
            query
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0 && *x <= 255.0 && x.fract() == 0.0),
            "native u8 admission requires original byte-valued queries"
        );
        Box::new(NativeU8Session {
            dataset: self.dataset,
            query: query.map(|x| x as u8),
        })
    }
}

struct NativeU8Session<'a, const N: usize> {
    dataset: &'a InmemDataset<u8, N>,
    query: [u8; N],
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<const N: usize> AdmissionSession for NativeU8Session<'_, N> {
    fn entry_distance(&self, id: u32) -> f32 {
        let start = id as usize * N;
        let vertex: &[u8; N] = self.dataset.data[start..start + N].try_into().unwrap();
        <[u8; N]>::distance_compare(&self.query, vertex, orion_ann_vector::Metric::L2)
    }

    unsafe fn admit_stream(
        &self,
        id_scratch: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        lookahead_lines: usize,
    ) -> usize {
        let mut w = 0;
        unsafe {
            orion_ann_vector::DistanceStream::<orion_ann_vector::L2U8Distance>::new(
                self.dataset.data.as_ptr(),
                N,
                N,
                self.query.as_ptr(),
                id_scratch,
                lookahead_lines,
            )
            .run(|id, distance| {
                out.add(w).write(Neighbor::new(id, distance));
                w += (distance < cutoff) as usize;
            });
        }
        w
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<'a, const N: usize> std::fmt::Debug for NativeU8Admission<'a, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeU8Admission")
            .field("dimension", &(N))
            .field("points", &self.dataset.num_active_pts)
            .finish_non_exhaustive()
    }
}

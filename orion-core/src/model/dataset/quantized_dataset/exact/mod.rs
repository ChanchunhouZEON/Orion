/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! base:  borrowed PaddedDataset<T>, N rows of [D coordinates | zero padding]
//! query: borrowed PaddedDataset<T>, matching physical row layout
//! ```
//!
//! Exact sessions borrow existing rows and do not own a quantized sidecar or
//! file codec. The kernel determines arithmetic; the generic dataset determines
//! storage. Native cosine additionally borrows the squared-norm auxiliary array.

//! Identity representation: borrow the base and padded queries without a sidecar.
//!
//! Traversal owns validation and candidate queues; this session only computes
//! exact distances using the same SIMD stream as the native-storage path.
use crate::{model::Neighbor, search::cascade::stage::AdmissionSession};
use bytemuck::Pod;
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::DistanceKernelBackend;

pub(crate) struct ExactSession<'a, T: Pod + Send + Sync, K: DistanceKernelBackend<Storage = T>> {
    base: &'a PaddedDataset<T>,
    queries: &'a PaddedDataset<T>,
    query_id: usize,
    // K selects stateless kernel functions; the session does not own a K value.
    kernel: std::marker::PhantomData<fn() -> K>,
}

impl<'a, T: Pod + Send + Sync, K: DistanceKernelBackend<Storage = T>> ExactSession<'a, T, K> {
    pub(crate) fn new(
        base: &'a PaddedDataset<T>,
        queries: &'a PaddedDataset<T>,
        query_id: usize,
    ) -> Self {
        Self {
            base,
            queries,
            query_id,
            kernel: std::marker::PhantomData,
        }
    }
}

// SAFETY: streams retain input IDs, write only within the input-sized output
// region, and return the initialized compacted prefix length.
unsafe impl<T: Pod + Send + Sync, K: DistanceKernelBackend<Storage = T>> AdmissionSession
    for ExactSession<'_, T, K>
{
    fn entry_distance(&self, id: u32) -> f32 {
        self.base
            .distance::<K>(id, self.queries, self.query_id)
            .expect("validated dynamic entry and layout")
    }

    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        near: usize,
    ) -> usize {
        let mut count = 0;
        // Search preparation checks the kernel/layout once; the graph loop
        // supplies valid row IDs under AdmissionSession's safety contract.
        unsafe {
            self.base
                .stream_unchecked::<K>(self.queries, self.query_id, ids, near)
        }
        .run(|id, distance| {
            unsafe {
                out.add(count).write(Neighbor::new(id, distance));
            }
            count += (distance < cutoff) as usize;
        });
        count
    }
}

#[cfg(feature = "benchmark")]
pub(crate) mod reference;

pub(crate) mod native;

pub mod stage;

mod mixed;

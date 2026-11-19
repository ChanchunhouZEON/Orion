/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Shared cascade beam search
//!
//! Quantized and identity entry points prepare query sessions and reuse the
//! traversal in `search_sessions`. Legacy typed entry wrappers are available
//! only with the `benchmark` feature. Tuning constants live in `tuning`;
//! instrumentation and legacy diagnostics live in `super::diagnostics`.

#[cfg(feature = "benchmark")]
use super::diagnostics::NoopObserver;
use super::diagnostics::{NeighborMode, SearchObserver};
use crate::Result;
#[cfg(feature = "benchmark")]
use crate::StaticOrion;
use crate::index::CompressedIndex;
use crate::model::Neighbor as DNeighbor;
use crate::model::search::scratch::{InMemScratchPool, InMemSearchScratch};

#[cfg(feature = "benchmark")]
use orion_ann_vector::FullPrecisionDistance;
#[cfg(feature = "benchmark")]
use rayon::prelude::*;

use super::{
    NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT,
    RAW_VISIT_COUNT, VISIT_COUNT,
};
#[cfg(feature = "benchmark")]
use crate::search::cascade::stage::prefilter::NoPrefilter;
use crate::search::cascade::stage::{AdmissionSession, PrefilterSession};
#[cfg(feature = "benchmark")]
use crate::search::cascade::stage::{AdmissionStage, PrefilterStage, RerankStage};
#[cfg(feature = "benchmark")]
use tuning::AlignedQuery;
use tuning::{FLUSH_INTERVAL, dstream_la_q, insert_route_mul, linear_merge_mul};

/// Parses the benchmark rerank multiplier, defaulting to two and rejecting zero or malformed overrides.
#[cfg(feature = "benchmark")]
fn parse_rerank_factor(value: Option<&str>) -> Result<usize, &'static str> {
    match value {
        None => Ok(2),
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|&factor| factor > 0)
            .ok_or("ORION_RERANK_FACTOR must be a positive integer"),
    }
}

/// Process-wide experiment override; the production default remains two.
#[cfg(feature = "benchmark")]
fn rerank_factor() -> usize {
    use std::sync::OnceLock;
    static FACTOR: OnceLock<usize> = OnceLock::new();
    *FACTOR.get_or_init(|| {
        let value = std::env::var("ORION_RERANK_FACTOR");
        let factor = match value {
            Ok(ref value) => parse_rerank_factor(Some(value)),
            Err(std::env::VarError::NotPresent) => parse_rerank_factor(None),
            Err(_) => Err("ORION_RERANK_FACTOR must be valid Unicode"),
        }
        .expect("invalid rerank factor");
        log::info!("Rerank factor={factor} (budget=min(k*factor, beam size))");
        factor
    })
}

/// Returns the production multiplier limiting the beam prefix submitted to reranking.
#[cfg(not(feature = "benchmark"))]
fn rerank_factor() -> usize {
    2
}

/// Peel hops at search start — fixed warm-up window that bypasses
/// DCC, early-exit, and the prefilter. Default `3` matches the
/// legacy `expand_peeled_hop_l2_q` count. Set via
/// `ORION_PEEL_HOPS=0` to disable (useful on low-D recipes where
/// the search converges in tens of hops and the peel is overhead
/// relative to the main loop).
#[inline]
#[cfg(feature = "benchmark")]
fn peel_hops() -> usize {
    use std::sync::OnceLock;
    static PEEL: OnceLock<usize> = OnceLock::new();
    *PEEL.get_or_init(|| {
        std::env::var("ORION_PEEL_HOPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3)
    })
}

/// Returns the production count of initial unfiltered expansions before adaptive traversal.
#[cfg(not(feature = "benchmark"))]
fn peel_hops() -> usize {
    3
}

// Compatibility signatures for existing research executables. Production and
// integration tests use the validated cascade/native-L2 entry points instead.
#[cfg(feature = "benchmark")]
impl<const N: usize, T: Send + Sync> StaticOrion<N, T>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Unified cascade beam search.
    ///
    /// Drives a single beam loop through the three stage traits.
    /// `prefilter` is optional (`None` = no rejection tier, e.g. on
    /// SIFT-class workloads where the admission kernel is already
    /// cheap enough). `admission` is mandatory (it defines the PQ-
    /// ranked distance metric). `rerank` is mandatory but can be
    /// [`NoRerank`](crate::search::cascade::stage::rerank::NoRerank)
    /// if the admission tier already ranks at sufficient precision.
    ///
    /// # Stage interaction contract
    ///
    /// * **Peel** (PEEL_HOPS = 3) runs at search start to fill the
    ///   PQ before any threshold-state machinery activates.
    ///
    /// * **Prefilter threshold** is a multiplicative slack over the
    ///   running mean of prefilter distances on PQ entries. The mean
    ///   is incrementally re-computed only when `pq_back_id` changes
    ///   (mirrors PA `beamSearch.h:142`).
    ///
    /// * **Admission cutoff** is the PQ tail distance (`pq_worst`)
    ///   when the PQ is full, otherwise `f32::MAX`. The admission
    ///   stream cmov-compacts admits into the per-hop staging buffer.
    ///
    /// * **Rerank** runs once at end of beam over the top
    ///   `k * rerank_factor` PQ entries (default factor two).
    ///
    /// # Generic monomorphisation
    ///
    /// The function body inlines every trait method call. With ~5-7
    /// distinct recipes in the dispatcher we get ~5-7 specialised
    /// hot loops — bloat budget ~20 KiB compiled text.
    #[allow(clippy::too_many_arguments)]
    pub fn search_unified<P, A, R>(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: Option<&P>,
        admission: &A,
        rerank: &R,
    ) -> Result<Vec<u32>>
    where
        P: ?Sized + PrefilterStage,
        A: ?Sized + AdmissionStage,
        R: ?Sized + RerankStage,
    {
        let mode = NeighborMode::default();
        self.search_unified_observed(
            query,
            k,
            search_list_size,
            window_size,
            epsilon,
            early_exit_limit,
            prefilter,
            admission,
            rerank,
            mode,
            &mut NoopObserver,
        )
    }

    /// Same production loop, with explicit neighbor policy and optional instrumentation.
    #[allow(clippy::too_many_arguments)]
    pub fn search_unified_observed<P, A, R, O>(
        &self,
        query: &[f32; N],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: Option<&P>,
        admission: &A,
        rerank: &R,
        mode: NeighborMode,
        observer: &mut O,
    ) -> Result<Vec<u32>>
    where
        P: ?Sized + PrefilterStage,
        A: ?Sized + AdmissionStage,
        R: ?Sized + RerankStage,
        O: SearchObserver,
    {
        let t_setup = crate::search::diagnostics::metrics::SetupTimer::now();
        let aligned = AlignedQuery(*query);

        // The benchmark adapter still accepts an optional factory. Resolve
        // it here; all session execution uses an explicit no-op or active stage.
        let p_session = match prefilter {
            Some(stage) => stage.open(&aligned.0),
            None => NoPrefilter.open(&aligned.0),
        };
        let a_session: Box<dyn AdmissionSession + '_> = admission.open(&aligned.0);

        self.search_sessions(
            k,
            search_list_size,
            window_size,
            epsilon,
            early_exit_limit,
            p_session.as_ref(),
            a_session.as_ref(),
            mode,
            observer,
            t_setup,
            |scratch, factor| rerank.rerank(query, &[], k, factor, scratch),
        )
    }

    /// Parallel batch wrapper. Mirrors the `search_batch_*` shape on
    /// every other search variant — rayon `par_iter_mut().zip(...)`
    /// with no manual chunking (empirically tied with PA's parlay
    /// scheduler at our work granularity).
    #[allow(clippy::too_many_arguments)]
    pub fn search_batch_unified<P, A, R>(
        &self,
        queries: &[[f32; N]],
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: Option<&P>,
        admission: &A,
        rerank: &R,
    ) -> Result<Vec<Vec<u32>>>
    where
        P: ?Sized + PrefilterStage + Sync,
        A: ?Sized + AdmissionStage + Sync,
        R: ?Sized + RerankStage + Sync,
    {
        self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::for_workers(rayon::current_num_threads(), search_list_size)
        });

        let n = queries.len();
        let mut results: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        results
            .par_iter_mut()
            .zip(queries.par_iter())
            .try_for_each(|(out, query)| -> Result<()> {
                *out = self.search_unified(
                    query,
                    k,
                    search_list_size,
                    window_size,
                    epsilon,
                    early_exit_limit,
                    prefilter,
                    admission,
                    rerank,
                )?;
                Ok(())
            })?;
        Ok(results)
    }
}

// Resident acquisition is always ready. Poll once on the caller, without an
// executor, task allocation or blocking bridge. The shared generic future is
// specialized for PhasedGraph; explicit-storage callers await the same body.
impl<D, S> CompressedIndex<D, S> {
    /// Borrows reusable scratch and drives the shared traversal to completion for resident graph access.
    ///
    /// The graph-read future must be immediately ready; an unexpected suspension is an
    /// internal invariant violation, not a request to block the caller on an executor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_sessions<O: SearchObserver, R>(
        &self,
        k: usize,
        search_list_size: usize,
        window_size: usize,
        epsilon: f32,
        early_exit_limit: usize,
        prefilter: &dyn PrefilterSession,
        a_session: &dyn AdmissionSession,
        mode: NeighborMode,
        observer: &mut O,
        t_setup: crate::search::diagnostics::metrics::SetupTimer,
        finish: impl FnOnce(&mut InMemSearchScratch, usize) -> R,
    ) -> Result<R> {
        use std::{
            future::Future,
            pin::pin,
            task::{Context, Poll, Waker},
        };
        let pool = self.inmem_scratch_pool.get_or_init(|| {
            InMemScratchPool::for_workers(rayon::current_num_threads(), search_list_size)
        });
        let mut guard = pool.acquire();
        let mut run = pin!(shared::traverse(
            &self.graph,
            self.entry,
            guard.scratch(),
            k,
            search_list_size,
            window_size,
            epsilon,
            early_exit_limit,
            prefilter,
            a_session,
            mode,
            observer,
            t_setup,
            finish,
        ));
        match run.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(result) => result,
            Poll::Pending => unreachable!("resident graph access cannot suspend"),
        }
    }
}

mod shared;
#[cfg(feature = "experimental-storage")]
pub(crate) use shared::traverse;
#[cfg(all(test, feature = "experimental-storage"))]
mod reference;

#[cfg(all(test, feature = "benchmark"))]
mod rerank_factor_tests {
    use super::parse_rerank_factor;

    #[test]
    fn default_and_explicit_factors() {
        assert_eq!(parse_rerank_factor(None), Ok(2));
        assert_eq!(parse_rerank_factor(Some("2")), Ok(2));
        assert_eq!(parse_rerank_factor(Some("6")), Ok(6));
    }

    #[test]
    fn invalid_factors_are_not_silently_defaulted() {
        for value in ["0", "-1", "", "1.5", "six"] {
            assert!(parse_rerank_factor(Some(value)).is_err());
        }
    }
}

/// Traversal constants and prefetch policy shared by official stages.
pub mod tuning;

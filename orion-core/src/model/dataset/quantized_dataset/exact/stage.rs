/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! Exact stages borrow padded base rows and optional squared norms. A query
//! owns one matching padded row; sessions borrow it for the traversal. No base
//! copy, quantized payload or additional file format is introduced here.
use super::{ExactSession, native::NativeSession};
use crate::model::{
    Neighbor,
    dataset::{native_u8::NativeU8Norms, quantized_dataset::capabilities::Precision},
    search::scratch::InMemSearchScratch,
};
use crate::search::{
    cascade::stage::{AdmissionSession, AdmissionStage, RerankStage},
    query::{self, QueryInput},
};
use crate::{Error, Result, ScoreMetric};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::{
    DistanceKernelBackend, IpF32Distance, IpU8Distance, L2F32Distance, L2U8Distance, PaddedLayout,
};

/// Original float or byte base view with optional byte cosine norms.
#[derive(Debug, Clone, Copy)]
enum Base<'a> {
    F32(&'a PaddedDataset<f32>),
    U8(&'a PaddedDataset<u8>, Option<&'a NativeU8Norms>),
}
/// Exact representation of the immutable base, usable at admission or rerank.
#[derive(Debug, Clone, Copy)]
pub struct ExactStage<'a> {
    base: Base<'a>,
    metric: ScoreMetric,
}

impl<'a> ExactStage<'a> {
    /// Returns the exact metric interpreted by this stage, independently of base coordinate type.
    pub(crate) fn metric(self) -> ScoreMetric {
        self.metric
    }

    /// Validates an f32 base layout for the exact kernel and rejects zero vectors for cosine.
    pub(crate) fn f32(base: &'a PaddedDataset<f32>, metric: ScoreMetric) -> Result<Self> {
        match metric {
            ScoreMetric::SquaredL2 => validate::<L2F32Distance>(base.layout())?,
            _ => validate::<IpF32Distance>(base.layout())?,
        }
        if metric == ScoreMetric::CosineSimilarity
            && (0..base.len()).any(|id| base.row(id).unwrap().iter().all(|&v| v == 0.))
        {
            return Err(Error::invalid_argument(
                "cosine requires nonzero base vectors",
            ));
        }
        Ok(Self {
            base: Base::F32(base),
            metric,
        })
    }

    /// Validates a byte base layout and requires precomputed norms for exact cosine.
    pub(crate) fn u8(
        base: &'a PaddedDataset<u8>,
        norms: Option<&'a NativeU8Norms>,
        metric: ScoreMetric,
    ) -> Result<Self> {
        match metric {
            ScoreMetric::SquaredL2 => validate::<L2U8Distance>(base.layout())?,
            _ => validate::<IpU8Distance>(base.layout())?,
        }
        if metric == ScoreMetric::CosineSimilarity && norms.is_none() {
            return Err(Error::invalid_state(
                "exact cosine requires prepared base norms",
            ));
        }
        Ok(Self {
            base: Base::U8(base, norms),
            metric,
        })
    }

    /// Returns logical dimension and row count for either base representation.
    fn shape_value(self) -> (usize, usize) {
        match self.base {
            Base::F32(base) => (base.layout().logical_dim(), base.len()),
            Base::U8(base, _) => (base.layout().logical_dim(), base.len()),
        }
    }

    /// Validates and prepares one query for this exact stage without a shared prepared-query cache.
    pub(crate) fn prepare(self, q: QueryInput<'_>) -> Result<PreparedExact> {
        self.prepare_shared(q, None)
    }

    /// Reuse the common floating transport when its transform matches this
    /// exact stage. Padded storage is still allocated directly in its final layout.
    pub(crate) fn prepare_shared(
        self,
        q: QueryInput<'_>,
        floating: Option<&[f32]>,
    ) -> Result<PreparedExact> {
        q.validate()?;
        if q.len() != self.shape_value().0 {
            return Err(Error::dimension(
                "queries.dimension",
                self.shape_value().0,
                q.len(),
            ));
        }
        if self.metric == ScoreMetric::CosineSimilarity && q.is_zero() {
            return Err(Error::invalid_argument(
                "cosine requires nonzero query vectors",
            ));
        }
        match self.base {
            Base::F32(base) if q.lossless_f32() => {
                if let Some(values) = floating {
                    return Ok(PreparedExact::F32(query::pad(values, base.layout())?));
                }
                let mut values = q.to_f32();
                if q.needs_unit_norm() {
                    query::normalize(values.to_mut())?;
                }
                Ok(PreparedExact::F32(query::pad(&values, base.layout())?))
            }
            Base::U8(base, _) => {
                if let Some(bytes) = q.to_u8() {
                    let padded = query::pad(&bytes, base.layout())?;
                    let norm = if self.metric == ScoreMetric::CosineSimilarity {
                        (bytes
                            .iter()
                            .map(|&x| u64::from(x) * u64::from(x))
                            .sum::<u64>() as f64)
                            .sqrt()
                    } else {
                        0.0
                    };
                    Ok(PreparedExact::U8(padded, norm))
                } else {
                    Ok(PreparedExact::Wide(q.prepared_wide()?))
                }
            }
            _ => Ok(PreparedExact::Wide(q.prepared_wide()?)),
        }
    }

    /// Representation reuse requires the same base, metric and physical layout,
    /// not merely equal coordinate widths or dimensions.
    pub(crate) fn same_representation(self, other: Self) -> bool {
        if self.metric != other.metric {
            return false;
        }
        match (self.base, other.base) {
            (Base::F32(a), Base::F32(b)) => std::ptr::eq(a, b),
            (Base::U8(a, _), Base::U8(b, _)) => std::ptr::eq(a, b),
            _ => false,
        }
    }

    /// Binds prepared query storage to the exact kernel selected by base type and metric.
    pub(crate) fn session<'q>(self, query: &'q PreparedExact) -> ExactHandle<'q>
    where
        'a: 'q,
    {
        match (self.base, query, self.metric) {
            (Base::F32(base), PreparedExact::Wide(q), metric) => {
                ExactHandle::MixedF32(super::mixed::MixedSession::new(base, q, metric))
            }
            (Base::U8(base, _), PreparedExact::Wide(q), metric) => {
                ExactHandle::MixedU8(super::mixed::MixedSession::new(base, q, metric))
            }
            (Base::F32(base), PreparedExact::F32(q), ScoreMetric::SquaredL2) => {
                ExactHandle::F32L2(ExactSession::new(base, q, 0))
            }
            (Base::F32(base), PreparedExact::F32(q), ScoreMetric::CosineSimilarity) => {
                ExactHandle::F32Cos(FloatCosineSession {
                    base,
                    query: q,
                    query_norm: float_norm(q.row(0).unwrap()),
                })
            }
            (Base::F32(base), PreparedExact::F32(q), _) => {
                ExactHandle::F32Ip(ExactSession::new(base, q, 0))
            }
            (Base::U8(base, _), PreparedExact::U8(q, _), ScoreMetric::SquaredL2) => {
                ExactHandle::U8L2(ExactSession::new(base, q, 0))
            }
            (Base::U8(base, norms), PreparedExact::U8(q, norm), ScoreMetric::InnerProduct) => {
                ExactHandle::U8Ip(NativeSession {
                    base,
                    query: q,
                    norms,
                    query_norm: *norm,
                })
            }
            (Base::U8(base, norms), PreparedExact::U8(q, norm), ScoreMetric::CosineSimilarity) => {
                ExactHandle::U8Cos(NativeSession {
                    base,
                    query: q,
                    norms,
                    query_norm: *norm,
                })
            }
            _ => unreachable!("prepared query belongs to its exact stage"),
        }
    }

    /// Computes the original metric score for a validated result ID, independent of quantized admission.
    pub(crate) fn score(self, query: &PreparedExact, id: u32, metric: ScoreMetric) -> f32 {
        match (self.base, query) {
            (Base::F32(base), PreparedExact::Wide(q)) => crate::search::query::numeric::score(
                q,
                base.row(id as usize).expect("validated ID"),
                metric,
            ),
            (Base::U8(base, _), PreparedExact::Wide(q)) => crate::search::query::numeric::score(
                q,
                base.row(id as usize).expect("validated ID"),
                metric,
            ),
            (Base::F32(base), PreparedExact::F32(q)) => crate::search::output::exact_score(
                q.row(0).unwrap(),
                base.row(id as usize).unwrap(),
                metric,
            ),
            (Base::U8(base, norms), PreparedExact::U8(q, qnorm)) => {
                let q = q.row(0).unwrap();
                let row = base.row(id as usize).unwrap();
                if metric == ScoreMetric::SquaredL2 {
                    return q
                        .iter()
                        .zip(row)
                        .map(|(&a, &b)| {
                            let d = i64::from(a) - i64::from(b);
                            (d * d) as u64
                        })
                        .sum::<u64>() as f32;
                }
                let dot = q
                    .iter()
                    .zip(row)
                    .map(|(&a, &b)| u64::from(a) * u64::from(b))
                    .sum::<u64>();
                if metric == ScoreMetric::InnerProduct {
                    return dot as f32;
                }
                let qnorm = if self.metric == ScoreMetric::CosineSimilarity {
                    *qnorm
                } else {
                    (q.iter().map(|&a| u64::from(a) * u64::from(a)).sum::<u64>() as f64).sqrt()
                };
                let norm = norms.map_or_else(
                    || {
                        row.iter()
                            .map(|&a| u64::from(a) * u64::from(a))
                            .sum::<u64>()
                    },
                    |n| n.get(id as usize),
                );
                (dot as f64 / (qnorm * (norm as f64).sqrt())).clamp(0., 1.) as f32
            }
            _ => unreachable!("prepared query belongs to its exact stage"),
        }
    }

    /// Rescores the selected beam prefix exactly and returns the best k IDs with deterministic ID tie-breaking.
    pub(crate) fn ranked(
        self,
        query: &PreparedExact,
        k: usize,
        factor: usize,
        scratch: &mut InMemSearchScratch,
    ) -> Vec<u32> {
        let n = k.saturating_mul(factor).min(scratch.pq.size());
        scratch.id_scratch.clear();
        scratch
            .id_scratch
            .extend(scratch.pq.neighbors().iter().take(n).map(|v| {
                assert!(
                    (v.id as usize) < self.shape_value().1,
                    "rerank row ID out of bounds"
                );
                v.id
            }));
        // A session streams into initialized capacity under the same validated-ID
        // contract as traversal. There is no per-candidate element dispatch.
        let mut scored = vec![Neighbor::new(0, 0.); n];
        let handle = self.session(query);
        let count = unsafe {
            handle.session().admit_stream(
                &scratch.id_scratch,
                scored.as_mut_ptr(),
                f32::INFINITY,
                crate::search::traversal::tuning::dstream_la_truth(),
            )
        };
        scored.truncate(count);
        crate::search::NDC_F32.add(n as u64);
        scored.sort_unstable_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        scored.into_iter().take(k).map(|v| v.id).collect()
    }
}

/// Rejects a padded layout that cannot be consumed safely by the selected exact stream kernel.
fn validate<K: DistanceKernelBackend>(layout: PaddedLayout) -> Result<()> {
    let block = K::stream_block_bytes(layout.compute_bytes());
    if layout.logical_dim() > K::MAX_COORDINATES
        || block == 0
        || !layout.compute_bytes().is_multiple_of(block)
    {
        return Err(Error::invalid_argument(
            "dataset layout is incompatible with exact kernel",
        ));
    }
    Ok(())
}

/// Prepared exact query representation selected without narrowing mixed coordinates.
pub(crate) enum PreparedExact {
    Wide(Vec<f64>),
    F32(PaddedDataset<f32>),
    U8(PaddedDataset<u8>, f64),
}

/// Dispatch handle owning or borrowing the prepared exact query state.
pub(crate) enum ExactHandle<'a> {
    MixedF32(super::mixed::MixedSession<'a, f32>),
    MixedU8(super::mixed::MixedSession<'a, u8>),
    F32Cos(FloatCosineSession<'a>),
    F32L2(ExactSession<'a, f32, L2F32Distance>),
    F32Ip(ExactSession<'a, f32, IpF32Distance>),
    U8L2(ExactSession<'a, u8, L2U8Distance>),
    U8Ip(NativeSession<'a, false>),
    U8Cos(NativeSession<'a, true>),
}
impl ExactHandle<'_> {
    /// Borrows the common admission interface for the selected exact computation session.
    pub(crate) fn session(&self) -> &dyn AdmissionSession {
        match self {
            Self::MixedF32(s) => s,
            Self::MixedU8(s) => s,
            Self::F32Cos(s) => s,
            Self::F32L2(s) => s,
            Self::F32Ip(s) => s,
            Self::U8L2(s) => s,
            Self::U8Ip(s) => s,
            Self::U8Cos(s) => s,
        }
    }
}

/// Exact-stage session retaining owned query preparation for the search lifetime.
struct OwnedSession<'a> {
    stage: ExactStage<'a>,
    query: PreparedExact,
}
// SAFETY: preparation validates layouts; borrowed typed sessions implement the
// compacted-output contract. The owned query outlives every borrowed session.
unsafe impl AdmissionSession for OwnedSession<'_> {
    /// Scores the entry node in the admission distance domain.
    ///
    /// See [`AdmissionSession::entry_distance`] for the shared contract.
    fn entry_distance(&self, id: u32) -> f32 {
        self.stage.session(&self.query).session().entry_distance(id)
    }

    /// Scores candidates and compacts qualifying neighbors into caller-owned output.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`AdmissionSession::admit_stream`]'s pointer, shape and lifetime contract.
    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        near: usize,
    ) -> usize {
        unsafe {
            self.stage
                .session(&self.query)
                .session()
                .admit_stream(ids, out, cutoff, near)
        }
    }
}
// SAFETY: stage shape is the borrowed immutable base shape; construction validates
// the chosen kernel, and query preparation validates logical coordinates/layout.
unsafe impl AdmissionStage for ExactStage<'_> {
    /// Identifies the precision used to validate cascade ordering.
    ///
    /// See [`AdmissionStage::precision`] for the shared contract.
    fn precision(&self) -> Option<Precision> {
        Some(Precision::Exact)
    }

    /// Describes the base shape required by this admission representation.
    ///
    /// See [`AdmissionStage::shape`] for the shared contract.
    fn shape(&self) -> Option<(usize, usize)> {
        Some(self.shape_value())
    }

    /// Exposes an exact-stage adapter when this admission representation supports one.
    ///
    /// See [`AdmissionStage::exact`] for the shared contract.
    fn exact(&self) -> Option<ExactStage<'_>> {
        Some(*self)
    }

    /// Prepares the query state consumed by this admission stage.
    ///
    /// See [`AdmissionStage::open`] for the shared contract.
    fn open<'q>(&'q self, q: &[f32]) -> Box<dyn AdmissionSession + 'q> {
        Box::new(OwnedSession {
            stage: *self,
            query: self
                .prepare(QueryInput::F32(q))
                .expect("validated exact query"),
        })
    }
}
impl RerankStage for ExactStage<'_> {
    /// Identifies final-stage precision for cascade compatibility checks.
    ///
    /// See [`RerankStage::precision`] for the shared contract.
    fn precision(&self) -> Option<Precision> {
        Some(Precision::Exact)
    }

    /// Describes the base shape required by this reranking representation.
    ///
    /// See [`RerankStage::shape`] for the shared contract.
    fn shape(&self) -> Option<(usize, usize)> {
        Some(self.shape_value())
    }

    /// Exposes original-metric scoring when this reranker has an exact-stage adapter.
    ///
    /// See [`RerankStage::exact`] for the shared contract.
    fn exact(&self) -> Option<ExactStage<'_>> {
        Some(*self)
    }

    /// Selects final IDs from the search scratch using this stage's ranking policy.
    ///
    /// See [`RerankStage::rerank`] for the shared contract.
    fn rerank(
        &self,
        q: &[f32],
        _: &[Neighbor],
        k: usize,
        factor: usize,
        s: &mut InMemSearchScratch,
    ) -> Vec<u32> {
        self.ranked(
            &self
                .prepare(QueryInput::F32(q))
                .expect("validated exact query"),
            k,
            factor,
            s,
        )
    }
}

/// Computes an L2 norm with f64 accumulation for exact cosine conversion.
fn float_norm(row: &[f32]) -> f64 {
    row.iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}
/// Exact cosine also accepts non-normalized f32 rows at the low-level boundary.
/// Normalized production f32 recipes retain their existing IP stage/kernel.
pub(crate) struct FloatCosineSession<'a> {
    base: &'a PaddedDataset<f32>,
    query: &'a PaddedDataset<f32>,
    query_norm: f64,
}
impl FloatCosineSession<'_> {
    /// Converts a negative dot product to bounded cosine distance using query and base norms.
    fn distance(&self, id: u32, dot: f32) -> f32 {
        let norm = float_norm(self.base.row(id as usize).expect("valid exact cosine ID"));
        (1. + f64::from(dot) / (self.query_norm * norm)).clamp(0., 2.) as f32
    }
}
// SAFETY: the typed stream preserves IDs and supplies at most one initialized
// neighbor per candidate. Norm calculation borrows the same checked base rows.
unsafe impl AdmissionSession for FloatCosineSession<'_> {
    /// Scores the entry node in the admission distance domain.
    ///
    /// See [`AdmissionSession::entry_distance`] for the shared contract.
    fn entry_distance(&self, id: u32) -> f32 {
        self.distance(
            id,
            self.base
                .distance::<IpF32Distance>(id, self.query, 0)
                .expect("validated cosine layout"),
        )
    }

    /// Scores candidates and compacts qualifying neighbors into caller-owned output.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`AdmissionSession::admit_stream`]'s pointer, shape and lifetime contract.
    unsafe fn admit_stream(
        &self,
        ids: &[u32],
        out: *mut Neighbor,
        cutoff: f32,
        near: usize,
    ) -> usize {
        let mut count = 0;
        unsafe {
            self.base
                .stream_unchecked::<IpF32Distance>(self.query, 0, ids, near)
        }
        .run(|id, dot| {
            let distance = self.distance(id, dot);
            unsafe {
                out.add(count).write(Neighbor::new(id, distance));
            }
            count += (distance < cutoff) as usize;
        });
        count
    }
}

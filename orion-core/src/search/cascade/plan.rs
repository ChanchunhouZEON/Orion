/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Bind logical query coordinates and official stages before traversal.
use super::{
    DynamicCascade,
    execute::{self, QueryPlan, SessionHandle},
    stage,
};
use crate::model::{
    dataset::{
        quantized_dataset::exact::stage::ExactHandle,
        rows::{CascadeData, RowTransform},
    },
    scratch::InMemSearchScratch,
};
use crate::search::query::QueryInput;
use crate::search::query::prepared::{PreparedQuery, QueryRequirements};
use crate::{Result, ScoreMetric};

pub(super) struct CascadePlan<'a, D> {
    base: &'a D,
    recipe: DynamicCascade<'a>,
    requirements: QueryRequirements<'a>,
}
impl<'a, D> CascadePlan<'a, D> {
    pub(super) fn new(base: &'a D, recipe: DynamicCascade<'a>, transform: RowTransform) -> Self {
        Self {
            base,
            recipe,
            requirements: QueryRequirements {
                admission: recipe.admission.exact(),
                rerank: recipe.rerank.exact(),
                filter: recipe.prefilter.precision().is_some(),
                float_rerank: recipe.rerank.exact().is_none()
                    && recipe.rerank.precision().is_some(),
                transform,
            },
        }
    }
}
pub(super) enum OpenedAdmission<'a> {
    Quantized(Box<dyn stage::AdmissionSession + 'a>),
    Exact(ExactHandle<'a>),
}
impl SessionHandle for OpenedAdmission<'_> {
    fn session(&self) -> &dyn stage::AdmissionSession {
        match self {
            Self::Quantized(s) => s.as_ref(),
            Self::Exact(s) => s.session(),
        }
    }
}
impl<D: CascadeData> QueryPlan for CascadePlan<'_, D> {
    type Prepared<'a>
        = PreparedQuery<'a>
    where
        Self: 'a;
    type Admission<'a>
        = OpenedAdmission<'a>
    where
        Self: 'a;
    fn validate(&self) -> Result<()> {
        self.recipe.validate(
            crate::model::dataset::rows::Rows::dimension(self.base),
            self.base.len(),
        )
    }
    fn prepare<'a>(&'a self, raw: QueryInput<'a>) -> Result<Self::Prepared<'a>> {
        PreparedQuery::new(raw, self.requirements)
    }

    fn admission<'a, 'q: 'a>(&'a self, q: &'a Self::Prepared<'q>) -> Self::Admission<'a>
    where
        Self: 'q,
    {
        match (self.requirements.admission, q.admission.as_ref()) {
            (Some(stage), Some(query)) => OpenedAdmission::Exact(stage.session(query)),
            _ => OpenedAdmission::Quantized(
                self.recipe
                    .admission
                    .open(q.floating.as_deref().expect("quantized query prepared")),
            ),
        }
    }
    fn prefilter<'a, 'q: 'a>(
        &'a self,
        q: &'a Self::Prepared<'q>,
    ) -> Box<dyn stage::PrefilterSession + 'a>
    where
        Self: 'q,
    {
        // NoPrefilter ignores coordinates; avoid materializing an f32 view
        // solely to call its factory for exact-only byte queries.
        self.recipe
            .prefilter
            .open(q.floating.as_deref().unwrap_or(&[]))
    }
    fn finish<'a>(
        &'a self,
        q: &Self::Prepared<'a>,
        scratch: &mut InMemSearchScratch,
        k: usize,
        factor: usize,
        scores: Option<ScoreMetric>,
    ) -> execute::SearchRow {
        let (ids, reused) = if let (Some(stage), Some(query)) = (
            self.requirements.rerank,
            if q.reuse_admission {
                q.admission.as_ref()
            } else {
                q.rerank.as_ref()
            },
        ) {
            let ids = stage.ranked(query, k, factor, scratch);
            let values = scores.map(|metric| {
                ids.iter()
                    .map(|&id| stage.score(query, id, metric))
                    .collect()
            });
            (ids, values)
        } else if let Some(metric) = scores {
            self.recipe.rerank.rerank_scored(
                q.floating.as_deref().unwrap_or(&[]),
                k,
                factor,
                scratch,
                metric,
            )
        } else {
            (
                self.recipe.rerank.rerank(
                    q.floating.as_deref().unwrap_or(&[]),
                    &[],
                    k,
                    factor,
                    scratch,
                ),
                None,
            )
        };
        let values = scores.map(|metric| {
            reused.unwrap_or_else(|| {
                if let (Some(stage), Some(query)) =
                    (self.requirements.admission, q.admission.as_ref())
                {
                    if self.recipe.rerank.precision().is_none()
                        && metric == ScoreMetric::SquaredL2
                        && stage.metric() == ScoreMetric::SquaredL2
                    {
                        // Preserve exact L2's original queue values and reduction order.
                        return scratch
                            .pq
                            .neighbors()
                            .iter()
                            .take(k)
                            .map(|n| n.distance)
                            .collect();
                    }
                    return ids
                        .iter()
                        .map(|&id| stage.score(query, id, metric))
                        .collect();
                }
                // Score original coordinates, never normalized/quantized admission
                // transport. A single conversion buffer is reused for all result IDs.
                let mut row = Vec::new();
                ids.iter()
                    .map(|&id| {
                        q.raw
                            .score(self.base.row_f32(id as usize, &mut row), metric)
                    })
                    .collect()
            })
        });
        (ids, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::cascade::stage::{NoPrefilter, rerank::NoRerank};
    use crate::{Orion, PaddedDataset, PaddedLayout};
    use orion_ann_vector::L2U8Distance;

    #[test]
    fn no_prefilter_session_needs_no_float_transport_for_exact_bytes() {
        let base = PaddedDataset::from_packed(
            &[1u8, 2, 3, 4, 5, 6],
            PaddedLayout::for_kernel::<L2U8Distance>(3).unwrap(),
        )
        .unwrap();
        let index = Orion::from_graph_partitions(
            base,
            &[(vec![1], vec![], vec![]), (vec![0], vec![], vec![])],
            0,
            1,
            0,
        )
        .unwrap();
        let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
        let recipe = DynamicCascade {
            prefilter: &NoPrefilter,
            admission: &exact,
            rerank: &NoRerank,
        };
        let plan = CascadePlan::new(index.dataset(), recipe, RowTransform::Identity);
        plan.validate().unwrap();
        let query = plan.prepare(QueryInput::U8(&[1u8, 2, 3])).unwrap();
        assert!(query.floating.is_none());
        let session = plan.prefilter(&query);
        let mut scratch = InMemSearchScratch::new(1);
        let outcome = unsafe { session.apply(&mut scratch, 1, 32) };
        assert_eq!(outcome, stage::prefilter::PrefilterOutcome::default());
        assert!(query.admission.is_some());
        assert!(query.rerank.is_none());
    }
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Per-row preparation owns conversion and padding. Requirements are collected
//! once per cascade; sessions borrow the completed representations read-only.
//! Original coordinates remain borrowed for scoring, and no whole query batch
//! or base is converted. Encoders retain their own model-specific query state.
use super::QueryInput;
use crate::{
    Result,
    model::dataset::{
        quantized_dataset::exact::stage::{ExactStage, PreparedExact},
        rows::RowTransform,
    },
};
use std::borrow::Cow;

/// Representations and transforms needed to prepare one query for a selected cascade.
#[derive(Clone, Copy)]
pub(crate) struct QueryRequirements<'a> {
    /// Exact admission adapter, or None when admission needs floating transport for quantization.
    pub(crate) admission: Option<ExactStage<'a>>,

    /// Exact rerank adapter that may share admission query state.
    pub(crate) rerank: Option<ExactStage<'a>>,

    /// Whether a filtering representation requires floating query coordinates.
    pub(crate) filter: bool,

    /// Whether a non-exact reranker needs floating query transport.
    pub(crate) float_rerank: bool,

    /// Sidecar input transform, kept separate from original coordinates used for exact scores.
    pub(crate) transform: RowTransform,
}

/// Query-scoped storage shared by sessions and released after final result selection.
pub(crate) struct PreparedQuery<'a> {
    /// Original coordinates retained for exact business-facing result scores.
    pub(crate) raw: QueryInput<'a>,
    // None for exact-only byte search: no f32 transport allocation is needed.
    /// Optional floating transport; absent for exact-only byte queries.
    pub(crate) floating: Option<Cow<'a, [f32]>>,

    /// Prepared exact admission state when the selected representation needs it.
    pub(crate) admission: Option<PreparedExact>,

    /// Separate exact rerank state only when it cannot reuse admission preparation.
    pub(crate) rerank: Option<PreparedExact>,

    /// Whether rerank and admission use the same exact query representation.
    pub(crate) reuse_admission: bool,
}

impl<'a> PreparedQuery<'a> {
    /// Validates and prepares one row, reusing matching exact-stage state and avoiding unused conversions.
    pub(crate) fn new(raw: QueryInput<'a>, requirements: QueryRequirements<'_>) -> Result<Self> {
        raw.validate()?;
        let needs_float =
            requirements.admission.is_none() || requirements.filter || requirements.float_rerank;
        let floating = if needs_float {
            let mut values = raw.to_f32();
            if requirements.transform == RowTransform::UnitL2 || raw.needs_unit_norm() {
                crate::search::query::normalize(values.to_mut())?;
            }
            Some(values)
        } else {
            None
        };
        // A sidecar-only unit transform must not change raw exact coordinates.
        let shared_float =
            if requirements.transform == RowTransform::Identity || raw.needs_unit_norm() {
                floating.as_deref()
            } else {
                None
            };
        let admission = requirements
            .admission
            .map(|stage| stage.prepare_shared(raw, shared_float))
            .transpose()?;
        let reuse_admission = requirements
            .admission
            .zip(requirements.rerank)
            .is_some_and(|(a, b)| a.same_representation(b));
        let rerank = requirements
            .rerank
            .filter(|_| !reuse_admission)
            .map(|stage| stage.prepare_shared(raw, shared_float))
            .transpose()?;
        Ok(PreparedQuery {
            raw,
            floating,
            admission,
            rerank,
            reuse_admission,
        })
    }
}

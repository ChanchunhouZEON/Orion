/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::Result;
use serde::{Deserialize, Serialize};

pub const TRACE_VERSION: u32 = 1;
pub const MAX_TRACE_JSON_BYTES: usize = 8 * 1024 * 1024;

/// Limits cover retained events and variable-size ID/score payloads, including
/// reserved final-result slots. Exhaustion stops recording, never search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceLimits {
    /// Maximum retained event count; reaching it truncates recording, not search.
    pub max_events: u32,

    /// Budget for retained ID/score items, including reserved final-result slots.
    pub max_items: u32,
}
impl Default for TraceLimits {
    /// Uses bounded recording defaults suitable for interactive trace replay.
    fn default() -> Self {
        Self {
            max_events: 2048,
            max_items: 32768,
        }
    }
}
impl TraceLimits {
    /// Checks recording caps and reserves room for the final `k` result IDs.
    ///
    /// # Errors
    ///
    /// Rejects oversized limits or an item budget smaller than `k`.
    pub fn validate(self, k: u32) -> Result<()> {
        if self.max_events > 8192 || self.max_items > 65536 || self.max_items < k {
            return Err(crate::argument(
                "trace requires max_events <= 8192 and k <= max_items <= 65536",
            ));
        }
        Ok(())
    }
}

/// Scores belong to the admission/prefilter domain, not necessarily exact L2.
/// Non-finite values have explicit JSON spellings rather than silently becoming null.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Score {
    Finite(f32),
    Special(SpecialScore),
}

/// JSON-safe representations of non-finite distance values.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialScore {
    PositiveInfinity,
    NegativeInfinity,
    Nan,
}
impl From<f32> for Score {
    /// Preserves non-finite values explicitly instead of serializing them as JSON null.
    fn from(v: f32) -> Self {
        if v.is_finite() {
            Self::Finite(v)
        } else if v.is_nan() {
            Self::Special(SpecialScore::Nan)
        } else if v.is_sign_positive() {
            Self::Special(SpecialScore::PositiveInfinity)
        } else {
            Self::Special(SpecialScore::NegativeInfinity)
        }
    }
}

/// A scored internal node ID in the distance domain of the emitting stage.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    /// Internal graph node ID, not an application-provided external ID.
    pub id: u32,

    /// Stage-local score; it must not be interpreted as an exact business similarity.
    pub distance: Score,
}

/// Traversal phase associated with a recorded node expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Peel,
    Exploration,
    Converged,
}

/// Reason the search stopped, independent of whether recording was truncated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    FrontierExhausted,
    EarlyExit,
}

/// Search settings needed to interpret a trace without loading the index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceMetadata {
    /// Logical vector dimension, excluding any SIMD padding.
    pub dimension: u32,

    /// Number of addressable graph nodes; bounds all recorded internal IDs.
    pub points: u32,

    /// Requested result count and reserved final-result recording capacity.
    pub k: u32,

    /// Search candidate capacity, which must be at least `k`.
    pub beam: u32,

    /// Whether calibrated adaptive traversal was enabled for this search.
    pub adaptive: bool,

    /// Metric label describing the search configuration.
    pub metric: String,

    /// Base representation label describing the indexed vector storage.
    pub storage: String,

    /// Selected prefilter representation, including its disabled form.
    pub prefilter: String,

    /// Selected representation used to score admission candidates.
    pub admission: String,

    /// Selected final reranking representation, including its disabled form.
    pub rerank: String,
}
impl TraceMetadata {
    /// Rejects impossible search shapes and unbounded or empty configuration labels.
    fn validate(&self) -> Result<()> {
        if self.dimension == 0
            || self.points == 0
            || self.k == 0
            || self.k > self.points
            || self.beam < self.k
        {
            return Err(crate::invalid(
                "invalid trace dimension, point count, k or beam",
            ));
        }
        if [
            &self.metric,
            &self.storage,
            &self.prefilter,
            &self.admission,
            &self.rerank,
        ]
        .iter()
        .any(|s| s.is_empty() || s.len() > 64)
        {
            return Err(crate::invalid("trace labels must contain 1..=64 bytes"));
        }
        Ok(())
    }
}

/// An ordered prefix of the real production loop. Candidates are already
/// visited-set-deduplicated; admitted candidates may later leave the beam.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TraceEvent {
    Entry {
        candidate: Candidate,
    },
    Expansion {
        node: u32,
        step: u32,
        phase: Phase,
        candidates: Vec<u32>,
    },
    Prefilter {
        cutoff: Option<Score>,
        admission_cutoff: Option<Score>,
        tested: u32,
        rejected: u32,
    },
    Admission {
        evaluated: u32,
        admitted: Vec<Candidate>,
    },
    Beam {
        candidates: Vec<Candidate>,
    },
    Rerank {
        candidates: u32,
    },
}
impl TraceEvent {
    /// Counts variable-size ID/score payload items charged to the recorder budget.
    pub fn items(&self) -> usize {
        match self {
            Self::Entry { .. } => 1,
            Self::Expansion { candidates, .. } => candidates.len(),
            Self::Admission { admitted, .. } => admitted.len(),
            Self::Beam { candidates } => candidates.len(),
            _ => 0,
        }
    }

    /// Checks node bounds, score encodings and count relationships for one event.
    fn validate(&self, points: u32) -> Result<()> {
        let score = |s: &Score| !matches!(s, Score::Finite(v) if !v.is_finite());
        let candidate = |c: &Candidate| c.id < points && score(&c.distance);
        let valid = match self {
            Self::Entry { candidate: c } => candidate(c),
            Self::Expansion {
                node, candidates, ..
            } => *node < points && candidates.iter().all(|&id| id < points),
            Self::Prefilter {
                cutoff,
                admission_cutoff,
                tested,
                rejected,
            } => rejected <= tested && cutoff.iter().chain(admission_cutoff).all(score),
            Self::Admission {
                evaluated,
                admitted,
            } => admitted.len() <= *evaluated as usize && admitted.iter().all(candidate),
            Self::Beam { candidates } => candidates.iter().all(candidate),
            Self::Rerank { candidates } => *candidates <= points,
        };
        if !valid {
            return Err(crate::invalid("invalid trace event ID, score or count"));
        }
        Ok(())
    }
}

/// Portable search results plus a bounded, ordered prefix of traversal events.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrace {
    /// Wire format version; readers reject unsupported values.
    pub schema_version: u32,

    /// Search settings used to interpret event IDs, scores and phases.
    pub metadata: TraceMetadata,

    /// Recording limits against which the retained document is validated.
    pub limits: TraceLimits,

    /// Retained traversal prefix in production emission order.
    pub events: Vec<TraceEvent>,

    /// Whether recording stopped before the complete search event sequence.
    pub truncated: bool,

    /// Search termination reason; truncation alone does not change it.
    pub stop_reason: StopReason,
    /// Always the real completed search result, even when the event prefix is truncated.
    pub result_ids: Vec<u32>,
}
impl SearchTrace {
    /// Checks schema compatibility, recording budgets and event/result invariants.
    ///
    /// # Errors
    ///
    /// Rejects malformed documents, out-of-range IDs and incomplete event boundaries
    /// unless the trace is explicitly marked truncated.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TRACE_VERSION {
            return Err(crate::invalid("unsupported trace schema version"));
        }
        self.metadata.validate()?;
        self.limits.validate(self.metadata.k)?;
        let items = self
            .events
            .iter()
            .try_fold(self.metadata.k as usize, |n, e| n.checked_add(e.items()))
            .ok_or_else(|| crate::invalid("trace payload size overflow"))?;
        if self.events.len() > self.limits.max_events as usize
            || items > self.limits.max_items as usize
        {
            return Err(crate::invalid(
                "trace exceeds its declared recording limits",
            ));
        }
        if self.result_ids.len() > self.metadata.k as usize
            || self.result_ids.iter().any(|&id| id >= self.metadata.points)
        {
            return Err(crate::invalid("invalid final result IDs"));
        }
        for event in &self.events {
            event.validate(self.metadata.points)?;
        }
        if !self.truncated
            && (!matches!(self.events.first(), Some(TraceEvent::Entry { .. }))
                || !matches!(self.events.last(), Some(TraceEvent::Rerank { .. })))
        {
            return Err(crate::invalid(
                "complete traces must begin with entry and end with rerank",
            ));
        }
        Ok(())
    }

    /// Validates and serializes the trace within the JSON byte-size limit.
    ///
    /// # Errors
    ///
    /// Returns validation, serialization or output-size errors without changing the trace.
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        let json = serde_json::to_string(self).map_err(|e| crate::invalid(e.to_string()))?;
        if json.len() > MAX_TRACE_JSON_BYTES {
            return Err(crate::invalid("trace JSON exceeds 8 MiB"));
        }
        Ok(json)
    }
    /// Reject oversized input before parsing. The byte cap also bounds allocation
    /// while validating hostile documents; declared limits are checked afterwards.
    pub fn from_json(json: &str) -> Result<Self> {
        if json.len() > MAX_TRACE_JSON_BYTES {
            return Err(crate::invalid("trace JSON exceeds 8 MiB"));
        }
        let trace: Self = serde_json::from_str(json).map_err(|e| crate::invalid(e.to_string()))?;
        trace.validate()?;
        Ok(trace)
    }
}

/// Record lazily: payload copying never occurs once the prefix limit is reached.
pub struct TraceRecorder {
    trace: SearchTrace,
    items: usize,
}
impl TraceRecorder {
    /// Starts bounded recording and reserves final-result space before the first event.
    ///
    /// # Errors
    ///
    /// Rejects invalid metadata or limits using the same checks as a restored trace.
    pub fn new(metadata: TraceMetadata, limits: TraceLimits) -> Result<Self> {
        metadata.validate()?;
        limits.validate(metadata.k)?;
        let items = metadata.k as usize;
        Ok(Self {
            trace: SearchTrace {
                schema_version: TRACE_VERSION,
                metadata,
                limits,
                events: Vec::new(),
                truncated: false,
                stop_reason: StopReason::FrontierExhausted,
                result_ids: Vec::new(),
            },
            items,
        })
    }

    /// Appends an event lazily when both recording budgets permit it.
    ///
    /// The closure is not called after truncation or when the declared cost will not fit.
    /// A mismatch between declared and actual item counts also stops recording.
    pub fn record(&mut self, items: usize, make: impl FnOnce() -> TraceEvent) {
        if self.trace.truncated {
            return;
        }
        if self.trace.events.len() >= self.trace.limits.max_events as usize
            || items > self.trace.limits.max_items as usize - self.items
        {
            self.trace.truncated = true;
            return;
        }
        let event = make();
        // Defend the public recorder's budget even if a producer miscounts.
        if event.items() != items {
            self.trace.truncated = true;
            return;
        }
        self.items += items;
        self.trace.events.push(event);
    }

    /// Marks search termination as early exit without allocating another event.
    pub fn early_exit(&mut self) {
        self.trace.stop_reason = StopReason::EarlyExit;
    }

    /// Attaches the actual search result and validates the completed document.
    ///
    /// # Errors
    ///
    /// Rejects invalid IDs or malformed events supplied by the producer.
    pub fn finish(mut self, result_ids: Vec<u32>) -> Result<SearchTrace> {
        self.trace.result_ids = result_ids;
        self.trace.validate()?;
        Ok(self.trace)
    }
}

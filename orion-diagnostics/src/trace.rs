use crate::Result;
use serde::{Deserialize, Serialize};

pub const TRACE_VERSION: u32 = 1;
pub const MAX_TRACE_JSON_BYTES: usize = 8 * 1024 * 1024;

/// Limits cover retained events and variable-size ID/score payloads, including
/// reserved final-result slots. Exhaustion stops recording, never search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceLimits {
    pub max_events: u32,
    pub max_items: u32,
}
impl Default for TraceLimits {
    fn default() -> Self {
        Self {
            max_events: 2048,
            max_items: 32768,
        }
    }
}
impl TraceLimits {
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialScore {
    PositiveInfinity,
    NegativeInfinity,
    Nan,
}
impl From<f32> for Score {
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: u32,
    pub distance: Score,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Peel,
    Exploration,
    Converged,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    FrontierExhausted,
    EarlyExit,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceMetadata {
    pub dimension: u32,
    pub points: u32,
    pub k: u32,
    pub beam: u32,
    pub adaptive: bool,
    pub metric: String,
    pub storage: String,
    pub prefilter: String,
    pub admission: String,
    pub rerank: String,
}
impl TraceMetadata {
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
    pub fn items(&self) -> usize {
        match self {
            Self::Entry { .. } => 1,
            Self::Expansion { candidates, .. } => candidates.len(),
            Self::Admission { admitted, .. } => admitted.len(),
            Self::Beam { candidates } => candidates.len(),
            _ => 0,
        }
    }
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrace {
    pub schema_version: u32,
    pub metadata: TraceMetadata,
    pub limits: TraceLimits,
    pub events: Vec<TraceEvent>,
    pub truncated: bool,
    pub stop_reason: StopReason,
    /// Always the real completed search result, even when the event prefix is truncated.
    pub result_ids: Vec<u32>,
}
impl SearchTrace {
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
    pub fn early_exit(&mut self) {
        self.trace.stop_reason = StopReason::EarlyExit;
    }
    pub fn finish(mut self, result_ids: Vec<u32>) -> Result<SearchTrace> {
        self.trace.result_ids = result_ids;
        self.trace.validate()?;
        Ok(self.trace)
    }
}

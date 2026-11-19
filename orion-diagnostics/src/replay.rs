/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::{Result, SearchTrace, TraceEvent};

/// Replay owns a validated trace. Cursor positions refer to events, not graph hops.
/// Seeking does not execute search or reconstruct a graph/beam state.
pub struct TraceReplay {
    trace: SearchTrace,
    position: usize,
}
impl TraceReplay {
    /// Validates an owned trace and starts replay before its first event.
    ///
    /// # Errors
    ///
    /// Returns the trace validation error before creating a cursor.
    pub fn new(trace: SearchTrace) -> Result<Self> {
        trace.validate()?;
        Ok(Self { trace, position: 0 })
    }

    /// Parses a size-bounded trace document and starts replay at event zero.
    ///
    /// See [`SearchTrace::from_json`] for the document validation contract.
    pub fn from_json(json: &str) -> Result<Self> {
        Self::new(SearchTrace::from_json(json)?)
    }

    /// Borrows the validated document without advancing the replay cursor.
    pub fn trace(&self) -> &SearchTrace {
        &self.trace
    }

    /// Returns the index of the next event to be replayed; the end position is valid.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Moves to an event boundary without executing graph search.
    ///
    /// # Errors
    ///
    /// Rejects positions beyond the event count and leaves the cursor unchanged.
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.trace.events.len() {
            return Err(crate::argument("trace cursor exceeds event count"));
        }
        self.position = position;
        Ok(())
    }
    /// A page is at most 1024 events; repeated calls do not clone retained payloads.
    pub fn next_events(&mut self, count: usize) -> Result<&[TraceEvent]> {
        if count == 0 || count > 1024 {
            return Err(crate::argument("trace page size must be in 1..=1024"));
        }
        let start = self.position;
        self.position += count.min(self.trace.events.len() - start);
        Ok(&self.trace.events[start..self.position])
    }
}

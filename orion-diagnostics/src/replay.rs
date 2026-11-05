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
    pub fn new(trace: SearchTrace) -> Result<Self> {
        trace.validate()?;
        Ok(Self { trace, position: 0 })
    }
    pub fn from_json(json: &str) -> Result<Self> {
        Self::new(SearchTrace::from_json(json)?)
    }
    pub fn trace(&self) -> &SearchTrace {
        &self.trace
    }
    pub fn position(&self) -> usize {
        self.position
    }
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

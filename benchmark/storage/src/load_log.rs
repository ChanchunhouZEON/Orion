/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Preparation-only observations; no logging enters the timed search path.
use std::{
    io::{self, Read},
    path::Path,
    sync::OnceLock,
    time::Instant,
};

/// Snapshot process and verified job-cgroup memory at a preparation boundary.
/// Missing platform counters are omitted; diagnostics never change load success.
pub fn stage(object: &str, phase: &str) {
    static START: OnceLock<Instant> = OnceLock::new();
    let process = std::fs::read_to_string("/proc/self/status").ok().map(|s| {
        s.lines()
            .filter(|l| {
                ["VmRSS:", "VmHWM:", "VmSwap:", "RssAnon:", "RssFile:"]
                    .iter()
                    .any(|p| l.starts_with(p))
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    let mut counters = serde_json::Map::new();
    let mut ancestors = serde_json::Map::new();
    // The Python runner supplies the enforcing ancestor discovered by cgroup
    // validation, rather than the leaf task whose limit can be unlimited.
    if let Ok(path) = std::env::var("ORION_STORAGE_CGROUP") {
        for name in [
            "memory.current",
            "memory.peak",
            "memory.max",
            "memory.events",
            "memory.stat",
            "memory.pressure",
            "io.pressure",
            "memory.swap.current",
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
            "memory.failcnt",
        ] {
            if let Ok(value) = std::fs::read_to_string(Path::new(&path).join(name)) {
                counters.insert(name.into(), value.trim().into());
            }
        }
        // Equal limits can exist at several levels. The job parent can OOM
        // while the chosen task ancestor reports no local limit failures.
        for parent in Path::new(&path).ancestors().skip(1) {
            if !parent.starts_with("/sys/fs/cgroup") {
                break;
            }
            let mut values = serde_json::Map::new();
            for name in [
                "memory.max",
                "memory.current",
                "memory.events",
                "memory.pressure",
                "io.pressure",
            ] {
                if let Ok(value) = std::fs::read_to_string(parent.join(name)) {
                    values.insert(name.into(), value.trim().into());
                }
            }
            if !values.is_empty() {
                ancestors.insert(parent.display().to_string(), values.into());
            }
        }
    }
    let process_io = std::fs::read_to_string("/proc/self/io").ok();
    eprintln!(
        "{}",
        serde_json::json!({"kind":"storage_stage", "object":object, "phase":phase,
        "elapsed_seconds":START.get_or_init(Instant::now).elapsed().as_secs_f64(),
        "process":process, "process_io":process_io, "cgroup":counters, "cgroup_ancestors":ancestors})
    );
}

/// Observe the existing decoder through Read, without adding a public core hook.
/// The fixed native header precedes allocation; payload reads precede row validation.
pub struct NativeGraphReader<R> {
    reader: R,
    expected: u64,
    consumed: u64,
    payload_started: bool,
}
impl<R> NativeGraphReader<R> {
    /// Wrap a graph stream whose complete byte length was obtained from metadata.
    pub fn new(reader: R, expected: u64) -> Self {
        Self {
            reader,
            expected,
            consumed: 0,
            payload_started: false,
        }
    }
}
impl<R: Read> Read for NativeGraphReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.consumed >= 16 && !self.payload_started {
            stage("graph", "allocation.complete");
            stage("graph", "read.begin");
            self.payload_started = true;
        }
        let n = self.reader.read(out)?;
        self.consumed += n as u64;
        if self.consumed == 16 && n != 0 {
            stage("graph", "header.complete");
            stage("graph", "allocation.begin");
        }
        if self.consumed == self.expected && n != 0 {
            stage("graph", "read.complete");
            // Native decode performs structural validation before returning.
            stage("graph", "final_reclaim_and_validation.begin");
        }
        Ok(n)
    }
}

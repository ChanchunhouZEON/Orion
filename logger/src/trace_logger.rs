/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use log;

/// Trace logger that delegates to the standard log crate.
/// Replaces the Windows ETW-based TraceLogger from the reference implementation.
pub struct TraceLogger;

impl TraceLogger {
    pub fn new() -> Self {
        TraceLogger
    }
}

impl Default for TraceLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Log a trace message at the specified level.
pub fn trace_log(level: log::Level, message: &str) {
    match level {
        log::Level::Error => log::error!("{}", message),
        log::Level::Warn => log::warn!("{}", message),
        log::Level::Info => log::info!("{}", message),
        log::Level::Debug => log::debug!("{}", message),
        log::Level::Trace => log::trace!("{}", message),
    }
}

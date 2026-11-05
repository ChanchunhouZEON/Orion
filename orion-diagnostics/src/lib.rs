/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Bounded, versioned diagnostics. The default native adapter consumes Orion's
//! observer hooks. Disable default features for the OS-independent replay layer.
#[cfg(feature = "native")]
mod native;
mod replay;
mod trace;
#[cfg(feature = "native")]
pub use native::RecordingObserver;

pub use orion_ann_error::{Error, ErrorCode, ErrorReport, Result};
pub use replay::TraceReplay;
pub use trace::*;

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid_data(message).at("trace")
}
fn argument(message: impl Into<String>) -> Error {
    Error::invalid_argument(message).at("trace")
}

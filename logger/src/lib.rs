/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

pub mod error_logger;
pub mod trace_logger;

use once_cell::sync::OnceCell;

static LOGGER_INITIALIZED: OnceCell<()> = OnceCell::new();

/// Initialize the logger. Safe to call multiple times; only the first call takes effect.
pub fn init_logger() {
    LOGGER_INITIALIZED.get_or_init(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format_timestamp_millis()
            .init();
    });
}

#[derive(thiserror::Error, Debug)]
pub enum LogError {
    #[error("Logging error: {0}")]
    LoggingError(String),
}

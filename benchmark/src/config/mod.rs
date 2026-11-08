/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Dataset shortcuts and YAML are benchmark-only inputs.
mod cli;
pub mod presets;
pub use cli::Args;
pub use presets::{
    load_dataset_config, load_dataset_config_by_dim, load_sweep_config, ResolvedSweep,
    SweepOverrides,
};

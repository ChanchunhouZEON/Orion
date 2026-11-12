/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Complete engine auxiliaries and restore readiness for saved calibrations.
//! Fresh indexes still require explicit calibration. A calibrated snapshot must
//! have its recipe sidecars ready before the handle becomes visible to callers.
use super::super::view::ReadView;
use crate::{config::ResolvedRunConfig, Result};

pub(super) fn prepare(
    view: &mut ReadView,
    config: &ResolvedRunConfig,
    pool: &rayon::ThreadPool,
    restore_calibrations: bool,
) -> Result<()> {
    pool.install(|| view.engine.prepare_auxiliary(config, &view.object_store))?;
    if restore_calibrations {
        // A snapshot may have been saved before its first search. Complete
        // any missing recipe sidecars before sharing the restored handle.
        pool.install(|| view.engine.prepare_search(config))?;
    }
    Ok(())
}

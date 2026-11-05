/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve the automatic graph source from available caches, exports and builders.
use crate::{
    config::{GraphSource, ResolvedRunConfig, VectorStorageKind},
    Error, Result,
};
use std::path::PathBuf;
impl ResolvedRunConfig {
    /// Only absence allows fallback. Invalid PA caches/exports and permission
    /// errors are surfaced instead of quietly changing the experiment's graph.
    pub fn select_automatic_graph(&mut self, custom_cache_dir: bool) -> Result<()> {
        use crate::graph::cache::CachePlan;

        let count = match self.num_points {
            Some(n) => Some(n),
            None => match std::fs::metadata(&self.base) {
                Ok(_) => {
                    Some(crate::data::inspect_vector_file(&self.base, self.base_format)?.count)
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => return Err(Error::io("metadata", &self.base, e)),
            },
        }
        .map(|n| n.min(self.max_points));

        if let Some(n) = count {
            let cache = CachePlan::inspect(self, n).map_err(|error| {
                let message = format!(
                    "automatic PA cache lookup failed in {}: {}; no Rust graph was substituted",
                    self.cache_dir.display(),
                    error.message()
                );
                error.with_message(message).at("resolve_graph")
            })?;
            if cache.is_hit() {
                self.graph_selection_reason = "PA cache available";
                return Ok(());
            }
        }
        if let Some(path) = crate::graph::staged::available_staged_export(self)? {
            if let Some(n) = count {
                crate::graph::staged::resolve_staged_export(self, n)?;
            }
            self.staged_file = Some(path);
            self.graph_selection_reason = "PA staged export available";
            return Ok(());
        }
        if cfg!(feature = "parlay") {
            self.graph_selection_reason = "PA in-memory build required";
            return Ok(());
        }
        if self.storage() == VectorStorageKind::U8 {
            // Offline preflight can still estimate this configuration. Execution
            // will report the missing PA input before allocating the dataset.
            self.graph_selection_reason =
                "native u8 requires a PA cache or staged export; no Rust fallback";
            return Ok(());
        }

        self.graph_source = GraphSource::Rust;
        if !custom_cache_dir {
            self.cache_dir = PathBuf::from("cache/orion");
        }
        self.cache_namespace = self.compute_cache_namespace();
        self.graph_selection_reason = if count
            .map(|n| CachePlan::inspect(self, n).map(|cache| cache.is_hit()))
            .transpose()?
            .unwrap_or(false)
        {
            "PA unavailable; Rust cache available"
        } else {
            "PA unavailable; Rust build required"
        };
        Ok(())
    }
}

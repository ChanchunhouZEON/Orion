/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Build or restore one complete index session.
//!
//! `load_request` is the ordered lifecycle: validate the request, hold a working
//! cache lease, load original rows, verify the graph source and budget, then
//! normalize and prepare the engine. Search state and application identities are
//! completed before publishing `Index`. Helpers stay private to this directory;
//! storage codecs, builders and search kernels remain in their existing modules.
mod graph;
mod identity;
mod input;
mod search;
mod snapshot;
mod source;
mod working;

use super::{snapshot::Snapshot, BaseInput, Index};
use crate::{Error, IndexOptions, Result};

impl Index {
    pub(super) fn load(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
    ) -> Result<Self> {
        let ids = identity::restore_ids(snapshot.as_ref(), options.memory_budget_gib)?;
        Self::load_with_ids(input, options, snapshot, ids)
    }

    pub(super) fn load_with_ids(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
        external_ids: Option<super::ids::IdMap>,
    ) -> Result<Self> {
        Self::load_request(
            input,
            options,
            snapshot,
            external_ids,
            #[cfg(feature = "experimental-graph")]
            None,
        )
    }

    pub(super) fn load_request(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
        external_ids: Option<super::ids::IdMap>,
        #[cfg(feature = "experimental-graph")] supplied: Option<super::supplier::SuppliedGraph>,
    ) -> Result<Self> {
        source::validate_request(
            &options,
            snapshot.as_ref(),
            #[cfg(feature = "experimental-graph")]
            supplied.as_ref(),
        )?;
        let working_directory = snapshot.is_none().then(|| options.cache_dir.clone());
        // Keep this guard in the orchestration scope, including identity/label
        // publication. Returning an error at any step also releases the lease.
        let _lease = working_directory
            .as_deref()
            .map(crate::storage::WriterLease::acquire)
            .transpose()?;
        let input::LoadedInput {
            mut base,
            mut config,
            snapshot_graph,
        } = input::load(
            input,
            &options,
            snapshot.as_ref(),
            external_ids.as_ref(),
            #[cfg(feature = "experimental-graph")]
            supplied.as_ref(),
        )?;
        let points = base.rows.len();
        let (source, calibrations) = source::resolve(
            &config,
            points,
            &base.sha256,
            snapshot,
            snapshot_graph.as_deref(),
            #[cfg(feature = "experimental-graph")]
            supplied,
        )?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()
            .map_err(Error::thread_pool)?;
        // Snapshot/supplier verification must see the original coordinates.
        // Normalize our array copy only after those identity checks succeed.
        base.validate_and_normalize_array(&config)?;
        let mut view = pool.install(|| graph::prepare(&config, base.rows, source, &base.sha256))?;
        source::bind_supplier_identity(&mut config, &view.payloads)?;
        search::prepare(&mut view, &config, &pool, !calibrations.is_empty())?;
        let labels = identity::attach(&mut view, external_ids.as_ref(), points)?;
        Ok(Self {
            labels,
            view,
            pool,
            config,
            options,
            base_file: base.file,
            calibrations,
            external_ids,
            working_directory,
        })
    }
}

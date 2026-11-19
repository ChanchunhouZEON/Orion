/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Build or restore one complete index session.
//!
//! `Index::load_request` owns the ordering and lifetime constraints spanning
//! these helpers. Storage codecs, builders and search kernels remain in their
//! existing modules; source-specific details belong to the corresponding callee.
mod graph;
mod identity;
mod input;
mod search;
mod snapshot;
mod source;
pub(super) mod storage;
mod working;

use super::{snapshot::Snapshot, BaseInput, Index};
use crate::{Error, IndexOptions, Result};

impl Index {
    /// Restores optional business IDs before assembling the index generation.
    pub(super) fn load(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
    ) -> Result<Self> {
        let ids = identity::restore_ids(snapshot.as_ref(), options.memory_budget_gib)?;
        Self::load_with_ids(input, options, snapshot, ids)
    }

    /// Assembles a normal build/restore request with an already validated ID mapping.
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

    /// Assemble a handle before making any of its state visible to the caller.
    ///
    /// Errors drop the local view, storage reservations and writer lease; no
    /// partially initialized Index escapes. This is not a filesystem transaction:
    /// valid cache objects published by earlier steps may remain for later reuse.
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
        // Working-cache coordination spans the whole request, not just graph
        // construction: identity/label attachment can still publish objects.
        // Snapshot restore uses immutable objects, so it has no working-cache
        // directory or lease; this does not mean every later step is read-only.
        let working_directory = snapshot.is_none().then(|| options.cache_dir.clone());
        let _lease = working_directory
            .as_deref()
            .map(crate::storage::WriterLease::acquire)
            .transpose()?;

        // Input identity must describe the original coordinates. The returned
        // config also reflects the actual shape and the budget available after
        // external IDs; keep using it rather than re-resolving from options.
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

        // Fix the source and check its identity/budget before handing rows to a
        // builder. An invalid snapshot is an error, not permission to rebuild;
        // otherwise saved calibrations could be paired with a different graph.
        let (source, calibrations) = source::resolve(
            &config,
            points,
            &base.sha256,
            snapshot,
            snapshot_graph.as_deref(),
            #[cfg(feature = "experimental-graph")]
            supplied,
        )?;
        // Own the Rayon pool with the resulting Index. Both graph preparation
        // below and search preparation use this pool instead of the global one.
        // Pool creation failure still leaves all loaded state local to this call.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()
            .map_err(Error::thread_pool)?;

        // Only now may array-backed coordinates be transformed: snapshot/supplier
        // checks above require the original digest. This also applies the chosen
        // base placement to our array copy; passing rows onward does not imply
        // resident storage. File-backed preparation has already handled its rows.
        base.validate_and_normalize_array(&config)?;

        // Transfer row ownership into the view. A fresh supplier's final cache
        // identity depends on the imported partitions and is available only now;
        // bind it before the following search preparation and identity attachment.
        // Graph preparation also calls storage::configure_object_storage before
        // fitting/restoring sidecars and norms. Calibration consumes that fixed
        // policy; it does not decide whether representations should be resident.
        let mut view = pool.install(|| graph::prepare(&config, base.rows, source, &base.sha256))?;
        source::bind_supplier_identity(&mut config, &view.payloads)?;

        // Restored calibration requires its search dependencies to be usable
        // before publication. A fresh build keeps calibration explicit: this
        // step must not silently run expensive calibration on the user's behalf.
        search::prepare(&mut view, &config, &pool, !calibrations.is_empty())?;

        // Labels must bind to this view's base and external-ID identity. Delay
        // exposing the handle until that final check succeeds; a usable graph
        // alone is insufficient to publish an application-consistent Index.
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

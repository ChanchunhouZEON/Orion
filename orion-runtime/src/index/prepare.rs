/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve inputs and prepare a resident generation on the session pool.
//! `Index::load` is the lifecycle entry point; `prepare` below it handles the
//! storage-specific engine after input, provenance, and budgets are resolved.
use super::{
    base::Rows,
    snapshot::{CheckedManifest, Payloads, Snapshot},
    view::{DynamicEngine, ReadView},
    BaseInput, Index,
};
use crate::{
    config::{GraphSourcePolicy, ResolvedRunConfig},
    data::LoadedBase,
    graph::cache::CachePlan,
    preflight as resources,
    search::{F32CascadeBackend, NativeU8L2Backend, SearchBackend},
    Error, IndexOptions, Result,
};
use std::collections::BTreeMap;
enum GraphInput {
    Snapshot(CheckedManifest),
    Working(CachePlan),
}

impl Index {
    pub(super) fn load(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
    ) -> Result<Self> {
        let ids = snapshot
            .as_ref()
            .and_then(|s| s.payloads.external_ids.as_ref().map(|id| (s, id)))
            .map(|(s, id)| {
                super::ids::IdMap::load(
                    &s.object_store,
                    &id.sha256,
                    id.points,
                    options.memory_budget_gib,
                )
            })
            .transpose()?;
        Self::load_with_ids(input, options, snapshot, ids)
    }

    pub(super) fn load_with_ids(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
        external_ids: Option<super::ids::IdMap>,
    ) -> Result<Self> {
        let working_directory = snapshot.is_none().then(|| options.cache_dir.clone());
        let _lease = working_directory
            .as_deref()
            .map(crate::storage::WriterLease::acquire)
            .transpose()?;
        let mut allocation_options = options.clone();
        // Reserve application identity alongside graph/base rather than silently
        // spending the entire user budget on the core representation.
        if let (Some(ids), Some(budget)) = (&external_ids, options.memory_budget_gib) {
            allocation_options.memory_budget_gib =
                Some(budget - ids.0.len() as f64 * 8.0 / 1073741824.0);
        }
        let snapshot_graph = snapshot
            .as_ref()
            .map(|s| crate::storage::object(&s.object_store, &s.payloads.graph_sha256, "pgraph"))
            .transpose()?;
        let (mut base, mut config) =
            input.validate_and_load(&allocation_options, snapshot_graph.as_deref())?;
        if config.graph_source_policy == GraphSourcePolicy::Auto {
            config.select_automatic_graph(true)?;
        }

        let points = base.rows.len();
        if external_ids
            .as_ref()
            .is_some_and(|ids| ids.0.len() != points)
        {
            return Err(Error::invalid_argument(
                "external ID count must equal indexed point count",
            ));
        }
        let saved = snapshot
            .map(|manifest| manifest.verify(&config, points, &base.sha256))
            .transpose()?;
        let (source, calibrations) = match saved {
            Some(mut manifest) => {
                resources::check_index_budget(&config, points, snapshot_graph.as_deref())?;
                let calibrations = std::mem::take(&mut manifest.calibrations);
                (GraphInput::Snapshot(manifest), calibrations)
            }
            None => {
                let plan = CachePlan::resolve(&config, points)?;
                resources::check_selected_budget(&config, points, &plan)?;
                (GraphInput::Working(plan), BTreeMap::new())
            }
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()
            .map_err(Error::thread_pool)?;
        // Identity above refers to the original input. Only after verifying it
        // may we normalize our array copy for cosine's unit-vector search space.
        // L2/IP retain their coordinates; file rows were handled by the decoder.
        base.validate_and_normalize_array(&config)?;
        let mut view = pool.install(|| match base.rows {
            Rows::F32(rows) => prepare::<F32CascadeBackend>(
                &config,
                rows,
                source,
                &base.sha256,
                crate::graph::prepare_f32,
            ),
            Rows::U8(rows) => prepare::<NativeU8L2Backend>(
                &config,
                rows,
                source,
                &base.sha256,
                crate::graph::prepare_u8,
            ),
        })?;
        if !calibrations.is_empty() {
            // A snapshot may have been saved before its first search. Complete
            // any missing recipe sidecars before sharing the restored handle.
            pool.install(|| view.engine.prepare_search(&config))?;
        }
        // Publish application identities once. Later calibration-only saves
        // reference this immutable object without re-encoding the mapping.
        if view.payloads.external_ids.is_none() {
            if let Some(ids) = &external_ids {
                view.payloads.external_ids = Some(super::snapshot::ExternalIdObject {
                    sha256: ids.save(&view.object_store)?,
                    points: ids.0.len(),
                });
            }
        }
        Ok(Self {
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

fn prepare<B: SearchBackend + 'static>(
    config: &ResolvedRunConfig,
    mut data: LoadedBase<B::Element>,
    source: GraphInput,
    base_sha256: &str,
    load: impl FnOnce(
        &ResolvedRunConfig,
        &mut LoadedBase<B::Element>,
        CachePlan,
    ) -> Result<crate::graph::LoadedGraph<B::Element>>,
) -> Result<ReadView> {
    match source {
        GraphInput::Snapshot(saved) => {
            let files = saved.payloads;
            let path = crate::storage::object(&saved.store, &files.graph_sha256, "pgraph")?;
            let dataset = data.take_index_dataset()?;
            let storage = crate::placement::graph_options(
                config,
                &dataset,
                std::fs::metadata(&path)?.len().saturating_sub(16),
            )?;
            let graph = crate::graph::persistence::load_verified_with_storage(
                &path,
                dataset,
                &saved.metadata,
                &files.graph_sha256,
                &storage,
            )?;
            let mut index = graph.value;
            crate::placement::configure(&index, config, files.sidecars.keys().cloned())?;
            crate::sidecars::restore_snapshot(&mut index, &saved.store, &files.sidecars)?;
            let receipts = crate::sidecars::track(
                &mut index,
                saved.store.clone(),
                None,
                files.sidecars.clone(),
                None,
            );
            Ok(ReadView {
                engine: Box::new(DynamicEngine::<B> {
                    index,
                    receipts,
                    auto_pin: Default::default(),
                    pin_report: Default::default(),
                }),
                object_store: saved.store,
                payloads: files,
                graph_bytes: graph.bytes,
                metadata_bytes: saved.metadata.len() as u64,
            })
        }
        GraphInput::Working(plan) => {
            let points = data.num_points;
            let mut graph = load(config, &mut data, plan)?;
            let record = graph.generation;
            let (bytes, digest) = crate::storage::put(&graph.store, "cache.json", |w| {
                std::io::Write::write_all(w, &record.encoded)?;
                Ok(())
            })?;
            let files = Payloads {
                graph_sha256: record.manifest.graph_sha256,
                metadata_sha256: record.manifest.metadata_sha256,
                provenance_sha256: digest,
                provenance_bytes: bytes,
                base_sha256: base_sha256.to_owned(),
                sidecars: BTreeMap::new(),
                external_ids: None,
            };
            let receipts = crate::sidecars::track(
                &mut graph.index,
                graph.store.clone(),
                Some(graph.working_key),
                BTreeMap::new(),
                crate::sidecars::derived_key(config, points)?,
            );
            Ok(ReadView {
                engine: Box::new(DynamicEngine::<B> {
                    auto_pin: Default::default(),
                    pin_report: Default::default(),
                    index: graph.index,
                    receipts,
                }),
                object_store: graph.store.canonicalize()?,
                payloads: files,
                graph_bytes: graph.graph_bytes,
                metadata_bytes: graph.metadata_bytes,
            })
        }
    }
}

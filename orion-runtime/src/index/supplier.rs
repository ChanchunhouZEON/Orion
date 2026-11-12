/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Experimental construction boundary; shared lifecycle remains in prepare/mod.rs.
use super::{
    snapshot::Payloads,
    view::{DynamicEngine, ReadView},
    BaseInput, Index,
};
use crate::{
    config::{GraphSourcePolicy, ResolvedRunConfig},
    data::BaseElement,
    data::LoadedBase,
    Error, IndexOptions, Result, SearchMetric,
};
use orion_ann_core::{
    supplier::{PartitionElement, PartitionMetric, PartitionSource, PartitionSpec, StagedReader},
    CompressedIndex,
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Write},
    path::Path,
};

pub(crate) struct SuppliedGraph {
    spec: PartitionSpec,
    source: Box<dyn PartitionSource + Send>,
}

impl SuppliedGraph {
    pub(super) fn validate_base(
        &self,
        config: &ResolvedRunConfig,
        points: usize,
        digest: &str,
    ) -> Result<()> {
        let spec = &self.spec;
        if points != spec.points as usize
            || config.dimension != spec.dimension as usize
            || matches!(
                (config.storage(), spec.element),
                (crate::VectorStorageKind::F32, PartitionElement::U8)
                    | (crate::VectorStorageKind::U8, PartitionElement::F32)
            )
        {
            return Err(Error::invalid_argument(
                "supplied graph shape/type differs from the indexed base",
            ));
        }
        if let Some(expected) = &spec.base_sha256 {
            crate::integrity::check("supplier base", digest, expected)?;
        }
        Ok(())
    }
}

impl Index {
    /// Import a third-party graph with no builder fallback or repartitioning.
    ///
    /// `options.graph` must retain its default: capacities and entry come solely
    /// from `spec`. All other options (metric, cascade, workers, budget, placement)
    /// apply normally. Base row order must be exactly the supplier's node order.
    /// The source is consumed during construction, never retained for search.
    ///
    /// # Errors
    /// Rejects incompatible metric/type/shape/base digest, invalid partitions,
    /// source failures and insufficient budgets. No partial index is returned.
    /// Calibration remains explicit; valid edges do not guarantee graph quality.
    pub fn from_partitions(
        input: BaseInput,
        spec: PartitionSpec,
        source: impl PartitionSource + Send + 'static,
        mut options: IndexOptions,
    ) -> Result<Self> {
        spec.validate()?;
        configure_options(&spec, &mut options)?;
        Self::load_request(
            input,
            options,
            None,
            None,
            Some(SuppliedGraph {
                spec,
                source: Box::new(source),
            }),
        )
    }

    /// Import a portable STAG v4 file. Header metadata is bounded and validated
    /// before base allocation. The open file is held through the whole import.
    pub fn from_staged(
        input: BaseInput,
        path: impl AsRef<Path>,
        options: IndexOptions,
    ) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::io("open supplier graph", path.as_ref(), e))?;
        let bytes = file.metadata()?.len();
        let reader = StagedReader::new(BufReader::with_capacity(8 * 1024 * 1024, file), bytes)?;
        Self::from_partitions(input, reader.spec().clone(), reader, options)
    }

    /// Adapt legacy STAG v3 with explicit supplier metadata. The original entry
    /// is preserved and all rows pass the same validation as native suppliers.
    pub fn from_legacy_staged(
        input: BaseInput,
        path: impl AsRef<Path>,
        spec: PartitionSpec,
        options: IndexOptions,
    ) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::io("open legacy supplier graph", path.as_ref(), e))?;
        let bytes = file.metadata()?.len();
        let reader =
            StagedReader::legacy_v3(BufReader::with_capacity(8 * 1024 * 1024, file), bytes, spec)?;
        Self::from_partitions(input, reader.spec().clone(), reader, options)
    }
}

pub(crate) fn configure_options(spec: &PartitionSpec, options: &mut IndexOptions) -> Result<()> {
    if options.graph != crate::GraphConfig::default() {
        return Err(Error::invalid_argument(
            "supplier import takes its graph recipe from PartitionSpec; keep options.graph at default",
        ));
    }
    let metric = match spec.metric {
        PartitionMetric::SquaredL2 => SearchMetric::SquaredL2,
        PartitionMetric::InnerProduct => SearchMetric::InnerProduct,
        PartitionMetric::Cosine => SearchMetric::Cosine,
    };
    if options.metric != metric {
        return Err(Error::invalid_argument(
            "supplier metric/preprocessing differs from IndexOptions.metric",
        ));
    }
    options.graph.source = GraphSourcePolicy::Supplied;
    options.graph.degree = spec.max_degree;
    options.graph.max_extra = spec.max_extra as usize;
    Ok(())
}

/// Encode a validated staged file with 8 MiB buffering, then atomically publish
/// without replacing an existing destination. Failed writes clean up the temp.
/// Parent directory must already exist. Returns the complete file's SHA-256.
pub fn write_staged_file(
    path: impl AsRef<Path>,
    spec: &PartitionSpec,
    source: &mut (impl PartitionSource + ?Sized),
) -> Result<String> {
    let path = path.as_ref();
    let encoded = crate::storage::encode_file(path, |w| {
        orion_ann_core::build::supplier::write_staged(w, spec, source)
    })?;
    std::fs::hard_link(&encoded.temporary.0, path)
        .map_err(|e| Error::io("publish supplier graph", path, e))?;
    crate::storage::sync_dir(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    Ok(encoded.digest)
}

pub(super) fn prepare<T: BaseElement + 'static>(
    config: &ResolvedRunConfig,
    mut data: LoadedBase<T>,
    mut supplied: SuppliedGraph,
    base_sha256: &str,
) -> Result<ReadView>
where
    orion_ann_diskann::model::PaddedDataset<T>: orion_ann_core::model::dataset::rows::CascadeData,
{
    let dataset = data.take_index_dataset()?;
    let storage =
        crate::placement::graph_options(config, &dataset, supplied.spec.slab_bytes()? as u64)?;
    let mut index: CompressedIndex<_> = CompressedIndex::from_partition_source(
        dataset,
        &supplied.spec,
        &mut *supplied.source,
        &storage,
    )?;
    let store = crate::storage::object_dir(&config.cache_dir)?;
    let mut metadata = Vec::new();
    let (graph_bytes, graph_sha256) = crate::storage::put(&store, "pgraph", |w| {
        index.encode_native_graph(&mut metadata, w)
    })?;
    let (metadata_bytes, metadata_sha256) = crate::storage::put(&store, "bin", |w| {
        w.write_all(&metadata)?;
        Ok(())
    })?;
    let (provenance_bytes, provenance_sha256) = crate::storage::put(&store, "cache.json", |w| {
        serde_json::to_writer(
            w,
            &serde_json::json!({"supplier":supplied.spec, "base_sha256":base_sha256, "graph_sha256":graph_sha256, "metadata_sha256":metadata_sha256}),
        )?;
        Ok(())
    })?;
    crate::search::bind_rows(&index, config)?;
    crate::placement::configure(&index, config, std::iter::empty())?;
    let receipts = crate::sidecars::track(&mut index, store.clone(), None, BTreeMap::new(), None);
    Ok(ReadView {
        engine: Box::new(DynamicEngine::<T> {
            index,
            receipts,
            auto_pin: Default::default(),
            pin_report: Default::default(),
        }),
        object_store: store.canonicalize()?,
        graph_bytes,
        metadata_bytes,
        payloads: Payloads {
            graph_sha256,
            metadata_sha256,
            provenance_sha256,
            provenance_bytes,
            base_sha256: base_sha256.to_owned(),
            sidecars: BTreeMap::new(),
            external_ids: None,
            labels: None,
            supplier: Some(super::snapshot::manifest::SupplierProvenance {
                name: supplied.spec.origin.name,
                revision: supplied.spec.origin.revision,
            }),
        },
    })
}

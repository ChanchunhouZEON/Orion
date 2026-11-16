/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Translate CLI input into the same options used by Rust and Python.
use crate::args::Args;
use orion_ann::execution::ResolvedRunConfig;
use orion_ann::{execution, Error, GraphSourcePolicy, IndexOptions, Result, VectorFormat};

impl Args {
    pub(crate) fn index_options(&self) -> Result<IndexOptions> {
        let mut options = if let Some(path) = &self.config {
            let yaml = std::fs::read_to_string(path)
                .map_err(|error| Error::io("read config", path, error))?;
            IndexOptions::from_yaml(&yaml)
                .map_err(|error| error.with_detail("path", path.display().to_string()))?
        } else {
            IndexOptions::default()
        };

        // Overlay only explicit flags. Replacing whole nested structs would
        // silently discard YAML siblings, especially per-object placements.
        macro_rules! override_field {
            ($source:ident, $($target:ident).+) => {
                if let Some(value) = &self.$source {
                    options.$($target).+ = Clone::clone(value);
                }
            };
            (optional $source:ident, $($target:ident).+) => {
                if self.$source.is_some() {
                    options.$($target).+ = Clone::clone(&self.$source);
                }
            };
        }
        override_field!(optional dataset_name, dataset_name);
        override_field!(metric, metric);
        override_field!(optional vector_storage, storage);
        override_field!(optional base_format, base_format);
        override_field!(optional prefilter, cascade.prefilter);
        override_field!(optional admission, cascade.admission);
        override_field!(optional rerank, cascade.rerank);
        override_field!(threads, threads);
        override_field!(window_size, window_size);
        override_field!(cache_dir, cache_dir);
        override_field!(optional memory_budget_gib, memory_budget_gib);
        override_field!(optional max_points, max_points);
        override_field!(storage_options, storage_options.mode);
        override_field!(optional mmap_dir, storage_options.directory);
        override_field!(graph_source, graph.source);
        override_field!(optional staged_file, graph.staged_file);
        override_field!(graph_degree, graph.degree);
        override_field!(build_l, graph.build_l);
        override_field!(alpha, graph.alpha);
        override_field!(max_extra, graph.max_extra);
        if self.no_auto_pin {
            options.auto_pin = false;
        }
        options
            .storage_options
            .objects
            .extend(self.object_overrides()?);

        Ok(options)
    }

    pub(crate) fn object_overrides(
        &self,
    ) -> Result<std::collections::BTreeMap<orion_ann::StorageObject, orion_ann::StorageMode>> {
        let mut overrides = std::collections::BTreeMap::new();
        for &(object, mode) in &self.object_storage {
            if overrides.insert(object, mode).is_some() {
                return Err(Error::invalid_argument(format!(
                    "duplicate object-storage override: {object}"
                )));
            }
        }
        Ok(overrides)
    }

    #[cfg(test)]
    pub(crate) fn resolve_search(&self, beam: Option<usize>) -> Result<ResolvedRunConfig> {
        self.resolve_with_options(beam, true, self.index_options()?)
    }

    /// Actual execution hashes the base while Index loads it. Resolve only its
    /// shape here; configuration-only/preflight commands still need an identity.
    #[cfg(test)]
    pub(crate) fn resolve_execution(&self, beam: Option<usize>) -> Result<ResolvedRunConfig> {
        self.resolve_with_options(beam, false, self.index_options()?)
    }

    pub(crate) fn resolve_with_options(
        &self,
        beam: Option<usize>,
        fingerprint: bool,
        #[allow(unused_mut)] mut options: IndexOptions,
    ) -> Result<ResolvedRunConfig> {
        #[cfg(feature = "experimental-graph")]
        if let Some(path) = &self.supplier_staged {
            let file =
                std::fs::File::open(path).map_err(|e| Error::io("open supplier graph", path, e))?;
            let reader = orion_ann::supplier::StagedReader::new(
                std::io::BufReader::new(&file),
                file.metadata()?.len(),
            )?;
            let spec = reader.spec();
            execution::configure_supplier(spec, &mut options)?;
        }
        let base = self.base.as_ref().ok_or("--base is required")?;
        let format = options
            .base_format
            .map(Ok)
            .unwrap_or_else(|| VectorFormat::infer(base))?;
        let header = match std::fs::metadata(base) {
            Ok(_) => Some(execution::inspect_vector_file(base, format)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && self.preflight => None,
            Err(e) => return Err(Error::io("open", base, e)),
        };
        let dimension = match (self.dimension, header.as_ref()) {
            (Some(d), Some(h)) if d != h.dimension => {
                return Err(Error::invalid_argument(
                    "--dimension differs from the base header",
                ))
            }
            (Some(d), _) => d,
            (None, Some(h)) => h.dimension,
            _ => {
                return Err(Error::invalid_argument(
                    "offline preflight requires --dimension and --max-points",
                ))
            }
        };
        let count = header
            .as_ref()
            .map(|h| h.count)
            .or(options.max_points)
            .ok_or("offline preflight requires --dimension and --max-points")?;
        // Configuration-only commands need the service identity for cache lookup.
        // Actual execution leaves fingerprinting to Index's single decoding pass;
        // this preliminary config is used only for CLI query/evaluation settings.
        let content_id = if fingerprint && header.is_some() {
            use sha2::Digest;
            let mut input = std::fs::File::open(base).map_err(|e| Error::io("open", base, e))?;
            let mut hash = sha2::Sha256::new();
            std::io::copy(&mut input, &mut hash)?;
            Some(format!("file-{:x}", hash.finalize()))
        } else {
            None
        };
        let mut config = execution::resolve_options(
            &options,
            base.clone(),
            format,
            dimension,
            count,
            content_id,
        )?;
        config.query = self.query.clone();
        config.query_format = self.query_format.map(Ok).unwrap_or_else(|| {
            self.query
                .as_deref()
                .filter(|p| *p != std::path::Path::new("-"))
                .map(VectorFormat::infer)
                .unwrap_or(Ok(format))
        })?;
        config.groundtruth = if self.no_groundtruth {
            None
        } else {
            self.groundtruth.clone()
        };
        config.prepare_only = self.prepare_only;
        config.query_batch_size = self.query_batch_size;
        if config.query_batch_size == 0 {
            return Err(Error::invalid_argument(
                "--query-batch-size must be positive",
            ));
        }
        if config.query.is_none() && !self.prepare_only && !self.preflight {
            return Err(Error::invalid_argument(
                "--query is required for search (use --prepare-only to build without queries)",
            ));
        }
        config.sweep.k = self.k.unwrap_or(10);
        let beam = beam.unwrap_or(config.sweep.k.max(64));
        if config.sweep.k == 0 || beam < config.sweep.k {
            return Err(Error::invalid_argument("require 1 <= k <= search L"));
        }
        config.sweep.search_list_sizes = vec![beam];
        config.sweep.calibration_l =
            orion_ann_core::calibration_search_list_size!(48, config.sweep.k);
        if fingerprint && config.graph_source_policy == GraphSourcePolicy::Auto {
            config.select_automatic_graph(true)?;
        }
        // Query format is a CLI concern but storage compatibility is a shared rule.
        execution::validate_query_format(&config)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use orion_ann::{CascadeConfig, GraphConfig, SearchMetric, VectorStorageKind};
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    struct Input(PathBuf);
    impl Input {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "orion-config-parity-{}-{}.u8bin",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut bytes = [16u32.to_le_bytes(), 32u32.to_le_bytes()].concat();
            bytes.extend([3u8; 16 * 32]);
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
        fn args(&self, extra: &[&str]) -> Args {
            let mut values = vec![
                "orion",
                "--base",
                self.0.to_str().unwrap(),
                "--prepare-only",
                "--graph-source",
                "parlayann",
            ];
            values.extend_from_slice(extra);
            Args::try_parse_from(values).unwrap()
        }
        fn service(&self, options: &IndexOptions) -> Result<ResolvedRunConfig> {
            use sha2::Digest;
            let id = format!(
                "file-{:x}",
                sha2::Sha256::digest(std::fs::read(&self.0).unwrap())
            );
            execution::resolve_options(
                options,
                self.0.clone(),
                VectorFormat::U8bin,
                32,
                16,
                Some(id),
            )
        }
    }
    impl Drop for Input {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn cli_and_service_share_defaults_aliases_and_cache_identity() {
        let file = Input::new();
        let cases = [
            (vec![], IndexOptions::default()),
            (
                vec![
                    "--vector-storage",
                    "f32",
                    "--admission",
                    " L2U16 ",
                    "--rerank",
                    "f32",
                    "--window-size",
                    "64",
                    "--memory-budget-gib",
                    "2",
                    "--no-auto-pin",
                ],
                IndexOptions {
                    storage: Some(VectorStorageKind::F32),
                    cascade: CascadeConfig {
                        admission: Some("l2-u16".into()),
                        rerank: Some("f32".into()),
                        ..Default::default()
                    },
                    window_size: 64,
                    memory_budget_gib: Some(2.),
                    auto_pin: false,
                    ..Default::default()
                },
            ),
            (
                vec![
                    "--vector-storage",
                    "f32",
                    "--metric",
                    "ip",
                    "--admission",
                    "mipsi8",
                    "--rerank",
                    "ipf32",
                ],
                IndexOptions {
                    metric: SearchMetric::InnerProduct,
                    storage: Some(VectorStorageKind::F32),
                    ..Default::default()
                },
            ),
        ];
        for (flags, mut options) in cases {
            options.graph.source = GraphSourcePolicy::Parlayann;
            let cli = file.args(&flags).resolve_search(None).unwrap();
            let mut service = file.service(&options).unwrap();
            // The CLI's single-request k/beam is separate from Index construction.
            service.sweep.k = 10;
            assert_eq!(
                serde_json::to_value(&cli).unwrap(),
                serde_json::to_value(&service).unwrap()
            );
            assert_eq!(cli.cache_namespace, service.cache_namespace);
        }
    }

    #[test]
    fn execution_defers_identity_to_the_verified_base_load() {
        let file = Input::new();
        let args = file.args(&[]);
        let execution = args.resolve_execution(None).unwrap();
        let inspection = args.resolve_search(None).unwrap();
        assert!(execution.base_content_id.is_none());
        assert!(inspection.base_content_id.is_some());
        assert_eq!(execution.dimension, inspection.dimension);
        assert_eq!(execution.storage(), inspection.storage());
        assert_eq!(execution.metric, inspection.metric);
        assert_eq!(execution.sweep.k, inspection.sweep.k);
    }

    #[test]
    fn configuration_errors_are_identical_across_adapters() {
        let file = Input::new();
        for (flags, options) in [
            (
                vec!["--threads", "0"],
                IndexOptions {
                    threads: 0,
                    ..Default::default()
                },
            ),
            (
                vec!["--threads", "1000000"],
                IndexOptions {
                    threads: 1_000_000,
                    ..Default::default()
                },
            ),
            (
                vec!["--window-size", "65"],
                IndexOptions {
                    window_size: 65,
                    ..Default::default()
                },
            ),
            (
                vec!["--memory-budget-gib", "0"],
                IndexOptions {
                    memory_budget_gib: Some(0.),
                    ..Default::default()
                },
            ),
            (
                vec!["--alpha", "0.98"],
                IndexOptions {
                    graph: GraphConfig {
                        alpha: 0.98,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ),
            (
                vec!["--admission", "unknown"],
                IndexOptions {
                    cascade: CascadeConfig {
                        admission: Some("unknown".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ),
            (
                vec!["--rerank", "u16"],
                IndexOptions {
                    cascade: CascadeConfig {
                        rerank: Some("u16".into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ),
        ] {
            let actual = file.args(&flags).resolve_search(None).unwrap_err();
            let expected = file.service(&options).unwrap_err();
            assert_eq!(actual.report(), expected.report(), "{flags:?}");
        }
    }
}

#[cfg(test)]
mod yaml_tests {
    use super::*;
    use clap::Parser;
    use orion_ann::{StorageMode, StorageObject};

    #[test]
    fn flags_override_individual_yaml_fields_and_keep_siblings() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("options.yaml");
        std::fs::write(
            &path,
            r#"
threads: 2
auto_pin: false
base_format: u8bin
max_points: 1000000
graph: {degree: 32, build_l: 96}
cascade: {prefilter: none, admission: native-l2-u8, rerank: none}
storage_options:
  mode: auto
  objects: {graph: resident, base: resident}
"#,
        )
        .unwrap();
        let args = Args::parse_from([
            "orion",
            "--config",
            path.to_str().unwrap(),
            "--base",
            "missing.payload",
            "--preflight",
            "--dimension",
            "128",
            "--threads",
            "8",
            "--graph-degree",
            "64",
            "--object-storage",
            "base=mapped",
        ]);
        let options = args.index_options().unwrap();
        assert_eq!(options.threads, 8);
        assert!(!options.auto_pin);
        assert_eq!(options.graph.degree, 64);
        assert_eq!(options.graph.build_l, 96);
        assert_eq!(options.cascade.rerank.as_deref(), Some("none"));
        assert_eq!(
            options.storage_options.objects[&StorageObject::Base],
            StorageMode::Mapped
        );
        assert_eq!(
            options.storage_options.objects[&StorageObject::Graph],
            StorageMode::Resident
        );
        // Prove resolution uses the captured configuration, not a second file read.
        std::fs::remove_file(path).unwrap();
        let resolved = args.resolve_with_options(None, true, options).unwrap();
        assert_eq!(resolved.max_points, 1000000);
        assert_eq!(resolved.base_format, VectorFormat::U8bin);
    }
}

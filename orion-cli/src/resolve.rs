/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Translate CLI input into the same options used by Rust and Python.
use crate::args::Args;
use orion_ann::execution::ResolvedRunConfig;
use orion_ann::{
    execution, CascadeConfig, Error, GraphSourcePolicy, IndexOptions, Result, VectorFormat,
};

impl Args {
    pub(crate) fn resolve_search(&self, beam: Option<usize>) -> Result<ResolvedRunConfig> {
        let defaults = IndexOptions::default();
        let mut options = IndexOptions {
            metric: self.metric.unwrap_or(defaults.metric),
            storage: self.vector_storage,
            base_format: self.base_format,
            cascade: CascadeConfig {
                prefilter: self.prefilter.clone(),
                admission: self.admission.clone(),
                rerank: self.rerank.clone(),
            },
            threads: self.threads.unwrap_or(defaults.threads),
            window_size: self.window_size.unwrap_or(defaults.window_size),
            cache_dir: self.cache_dir.clone().unwrap_or(defaults.cache_dir),
            memory_budget_gib: self.memory_budget_gib,
            auto_pin: !self.no_auto_pin,
            max_points: self.max_points,
            ..defaults
        };
        options.graph.source = self.graph_source.unwrap_or(options.graph.source);
        options.graph.staged_file = self.staged_file.clone();
        options.graph.degree = self.graph_degree.unwrap_or(options.graph.degree);
        options.graph.build_l = self.build_l.unwrap_or(options.graph.build_l);
        options.graph.alpha = self.alpha.unwrap_or(options.graph.alpha);
        options.graph.max_extra = self.max_extra.unwrap_or(options.graph.max_extra);

        let base = self.base.as_ref().ok_or("--base is required")?;
        let format = self
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
            .or(self.max_points)
            .ok_or("offline preflight requires --dimension and --max-points")?;
        // Match service file identities; changing a label/path must not reuse
        // an index belonging to different bytes. Offline estimates have no ID.
        let content_id = if header.is_some() {
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
        if config.graph_source_policy == GraphSourcePolicy::Auto {
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
    use orion_ann::{GraphConfig, SearchMetric, VectorStorageKind};
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

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use super::*;

#[test]
fn bundled_metadata_resolves_all_named_datasets() {
    let root: RawRoot = serde_yaml::from_str(DEFAULT_CONFIG).unwrap();
    for (name, raw) in &root.datasets {
        let dataset = resolve_dataset(&root, name);
        assert_eq!(dataset.name, *name);
        assert_eq!(dataset.dimension, raw.dimension);
        let paths = raw.paths.as_ref().unwrap();
        assert_eq!(dataset.base_path, PathBuf::from(&paths.base));
        assert_eq!(dataset.query_path, PathBuf::from(&paths.query));
        assert_eq!(
            dataset.groundtruth_path,
            PathBuf::from(paths.groundtruth.as_ref().unwrap())
        );
        assert!(dataset.orion.graph_degree > 0);
        assert!(dataset.diskann.graph_degree > 0);
    }
}

#[test]
fn metadata_overrides_do_not_leak_between_engines() {
    let mut root: RawRoot = serde_yaml::from_str(DEFAULT_CONFIG).unwrap();
    root.datasets.insert(
        "fixture".into(),
        serde_yaml::from_str(
            r#"
dimension: 33
paths:
  base: base.fbin
  query: query.fbin
  groundtruth: gt.ivecs
orion:
  alpha: 1.3
  graph_degree: 23
  build_search_list_size: 47
  max_extra: 7
  window_size: 5
  prefilter: none
  admission: l2-u16
  rerank: f32
base_graph:
  source: rust
  staged_file: graph-${max_extra}.staged
"#,
        )
        .unwrap(),
    );
    let ds = resolve_dataset(&root, "fixture");
    assert_eq!(
        (
            ds.orion.alpha,
            ds.orion.graph_degree,
            ds.orion.build_search_list_size,
            ds.orion.max_extra,
            ds.orion.window_size
        ),
        (1.3, 23, 47, 7, 5)
    );
    assert_eq!(ds.base_graph.source, BaseGraphSource::Rust);
    assert_eq!(
        ds.base_graph.staged_file.unwrap(),
        PathBuf::from("graph-7.staged")
    );
    assert_eq!(ds.diskann.alpha, root.defaults.diskann.alpha);
    assert_eq!(ds.diskann.graph_degree, root.defaults.diskann.graph_degree);
    for source in ["auto", "Rust", "parlayann", "Parlayann", "PA", "pa"] {
        root.datasets
            .get_mut("fixture")
            .unwrap()
            .base_graph
            .as_mut()
            .unwrap()
            .source = Some(source.into());
        let actual = resolve_dataset(&root, "fixture").base_graph.source;
        assert_eq!(
            actual,
            match source {
                "auto" => BaseGraphSource::Auto,
                "Rust" => BaseGraphSource::Rust,
                _ => BaseGraphSource::Parlayann,
            }
        );
    }
}

#[test]
fn placeholder_paths_preserve_unicode_and_reject_incomplete_substitutions() {
    assert_eq!(
        try_resolve_placeholders("数据/图.staged"),
        Some("数据/图.staged".into())
    );
    assert_eq!(
        try_resolve_placeholders("a${ORION_TEST_VARIABLE_THAT_DOES_NOT_EXIST_20261105}"),
        None
    );
    assert_eq!(try_resolve_placeholders("a${unterminated"), None);
    assert_eq!(
        try_resolve_placeholders("$literal"),
        Some("$literal".into())
    );
}

use orion_ann::{Error, ErrorCode};
use std::error::Error as _;

#[test]
fn errors_are_small_send_sync_values_and_reports_preserve_context() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Error>();
    assert_eq!(std::mem::size_of::<Error>(), std::mem::size_of::<usize>());
    let error = Error::dimension("queries.dimension", 128, 127)
        .at("read_queries")
        .at("search");
    assert_eq!(error.code(), ErrorCode::DimensionMismatch);
    assert_eq!(
        error.to_string(),
        "[ORION_DIMENSION_MISMATCH] read_queries: queries.dimension must be 128, got 127"
    );
    let report = serde_json::to_value(error.report()).unwrap();
    assert_eq!(report["operation"], "read_queries");
    assert_eq!(report["details"]["expected"], 128);
    assert_eq!(report["details"]["actual"], 127);
}

#[test]
fn io_sources_survive_runtime_and_diskann_adapters() {
    let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "fixture denied");
    let backend = orion_ann_diskann::common::ANNError::from(io);
    assert!(matches!(
        backend,
        orion_ann_diskann::common::ANNError::IOError { .. }
    ));
    let error = Error::from(backend).at("load_graph");
    assert_eq!(error.code(), ErrorCode::Io);
    let source = error.source().unwrap();
    assert!(
        source
            .downcast_ref::<orion_ann_diskann::common::ANNError>()
            .is_some()
    );
    assert_eq!(
        source
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert_eq!(error.report().causes.last().unwrap(), "fixture denied");
}

#[test]
fn truncated_graph_headers_return_an_error_before_allocation() {
    let path =
        std::env::temp_dir().join(format!("orion-invalid-graph-{}.pgraph", std::process::id()));
    let mut header = Vec::new();
    for value in [u32::MAX, 64, u32::MAX, 0] {
        header.extend_from_slice(&value.to_ne_bytes());
    }
    std::fs::write(&path, header).unwrap();
    let error = orion_ann::PhasedGraph::load(&path).unwrap_err();
    std::fs::remove_file(path).unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidData);
}

#[test]
fn invalid_build_controls_return_errors_without_entering_backend() {
    let invalid_shape =
        orion_ann::build_diskann_index(&[1.0, 2.0], 1, 3, 1.2, 4, 8, false, None, None, true, 2);
    assert!(matches!(invalid_shape, Err(e) if e.code() == ErrorCode::InvalidArgument));
    let invalid_pq = orion_ann::build_diskann_index(
        &[1.0, 2.0, 3.0],
        1,
        3,
        1.2,
        1,
        4,
        true,
        Some(0),
        Some(8),
        false,
        0,
    );
    assert!(matches!(invalid_pq, Err(e) if e.code() == ErrorCode::UnsupportedConfiguration));
}

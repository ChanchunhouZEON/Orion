/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use orion_ann_diskann::common::ANNError;
use std::error::Error as _;

#[test]
fn backend_errors_preserve_sources_and_context() {
    let errors: Vec<(ANNError, &str)> = vec![
        (
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "fixture denied").into(),
            "fixture denied",
        ),
        (u8::try_from(300u32).unwrap_err().into(), "TryFromIntError"),
        (
            std::alloc::Layout::from_size_align(1, 3)
                .unwrap_err()
                .into(),
            "MemoryAllocLayoutError",
        ),
        (
            <[u8; 4]>::try_from(&[0u8][..]).unwrap_err().into(),
            "array from slice",
        ),
        (
            bincode::error::EncodeError::Other("fixture encode").into(),
            "fixture encode",
        ),
        (
            bincode::error::DecodeError::Other("fixture decode").into(),
            "fixture decode",
        ),
    ];
    for (error, expected) in errors {
        assert!(error.to_string().contains(expected));
        assert!(error.source().is_some());
        assert!(!format!("{error:?}").is_empty());
    }
    assert!(
        ANNError::log_disk_io_request_alignment_error("offset=3".into())
            .to_string()
            .contains("offset=3")
    );
    assert!(
        ANNError::log_pq_error("bad codebook".into())
            .to_string()
            .contains("bad codebook")
    );
}

#[cfg(feature = "orion")]
#[test]
fn public_errors_keep_backend_classification() {
    use orion_ann_error::{Error, ErrorCode};
    for (backend, expected) in [
        (
            ANNError::log_index_config_error("degree".into(), "bad degree".into()),
            ErrorCode::InvalidArgument,
        ),
        (
            ANNError::from(std::io::Error::from(std::io::ErrorKind::UnexpectedEof)),
            ErrorCode::InvalidData,
        ),
        (
            ANNError::from(std::io::Error::from(std::io::ErrorKind::NotFound)),
            ErrorCode::Io,
        ),
        (
            ANNError::log_lock_poison_error("poisoned fixture".into()),
            ErrorCode::InvalidState,
        ),
        (
            ANNError::from(std::alloc::Layout::from_size_align(1, 3).unwrap_err()),
            ErrorCode::ResourceExhausted,
        ),
    ] {
        let original = backend.to_string();
        let public: Error = backend.into();
        assert_eq!(public.code(), expected);
        assert_eq!(public.source().unwrap().to_string(), original);
    }
}

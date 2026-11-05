/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

mod support;
use orion_ann_platform::{AccessMode, UnixAlignedFileReader, UnixFileHandle};
use std::{
    io::{ErrorKind, Read, Write},
    sync::Arc,
};
use support::TestDir;

#[test]
fn file_modes_preserve_or_truncate_as_documented() {
    let dir = TestDir::new();
    let path = dir.file("file");
    assert!(UnixFileHandle::open(&path, AccessMode::Read).is_err());
    let handle = UnixFileHandle::open(&path, AccessMode::Write).unwrap();
    handle.file().write_all(b"payload").unwrap();
    assert_eq!(handle.file_size().unwrap(), 7);
    assert!(handle.raw_fd() >= 0);
    drop(handle);
    let handle = UnixFileHandle::open(&path, AccessMode::ReadWrite).unwrap();
    handle.file().write_all(b"new").unwrap();
    drop(handle);
    let handle = UnixFileHandle::open(&path, AccessMode::Read).unwrap();
    let mut data = String::new();
    handle.file().read_to_string(&mut data).unwrap();
    assert_eq!(data, "newload");
    drop(handle);
    assert_eq!(
        UnixFileHandle::open(&path, AccessMode::Write)
            .unwrap()
            .file_size()
            .unwrap(),
        0
    );
    assert!(UnixFileHandle::default().raw_fd() >= 0);
}

#[test]
fn mapped_reads_check_eof_and_overflow_without_panicking() {
    let dir = TestDir::new();
    let path = dir.file("mapped");
    std::fs::write(&path, b"0123456789").unwrap();
    let reader = UnixAlignedFileReader::open(&path).unwrap();
    assert!(reader.raw_fd() >= 0);
    assert_eq!(reader.read_slice(9, 1).unwrap(), b"9");
    assert_eq!(reader.read_slice(10, 0).unwrap(), b"");
    for (offset, len) in [(9, 2), (11, 0), (u64::MAX, 1), (1, usize::MAX)] {
        assert_eq!(
            reader.read_slice(offset, len).unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );
    }
    for offset in [10, u64::MAX] {
        let mut target = [99];
        assert_eq!(
            reader.read(offset, &mut target).unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );
        assert_eq!(target, [99], "failed reads must not change the output");
    }
}

#[test]
fn contexts_are_reused_within_threads_and_separate_between_threads() {
    let dir = TestDir::new();
    let path = dir.file("contexts");
    std::fs::write(&path, b"0123456789").unwrap();
    let reader = UnixAlignedFileReader::open(&path).unwrap();
    let parent = reader.get_ctx();
    assert!(Arc::ptr_eq(&parent, &reader.get_ctx()));
    std::thread::scope(|scope| {
        let children: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let context = reader.get_ctx();
                    assert!(Arc::ptr_eq(&context, &reader.get_ctx()));
                    assert!(!Arc::ptr_eq(&context, &parent));
                    assert_eq!(reader.read_slice(2, 3).unwrap(), b"234");
                    context
                })
            })
            .collect();
        let contexts: Vec<_> = children.into_iter().map(|c| c.join().unwrap()).collect();
        for i in 0..contexts.len() {
            for j in 0..i {
                assert!(!Arc::ptr_eq(&contexts[i], &contexts[j]));
            }
        }
    });
}

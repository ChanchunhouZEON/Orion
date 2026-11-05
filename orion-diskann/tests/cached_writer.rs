/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

mod support;
use orion_ann_diskann::utils::CachedWriter;
use support::TestDir;

#[test]
fn cached_writer_preserves_bytes_across_chunk_boundaries() {
    let dir = TestDir::new();
    // Includes exact cache fills, overflow, direct large writes and drop flush.
    for cache in [1, 8, 31, 128] {
        let path = dir.file(&format!("cache-{cache}"));
        let mut expected = Vec::new();
        {
            let mut writer = CachedWriter::new(&path, cache).unwrap();
            for len in [0, 1, 7, 8, 9, 31, 129, 3] {
                let bytes: Vec<_> = (0..len).map(|i| (i * 37 + len) as u8).collect();
                writer.write(&bytes).unwrap();
                expected.extend(bytes);
            }
            writer.flush().unwrap();
            assert_eq!(writer.get_file_size(), expected.len() as u64);
            writer.flush().unwrap();
            assert_eq!(writer.get_file_size(), expected.len() as u64);
            writer.write(b"tail").unwrap();
            expected.extend(b"tail");
        }
        assert_eq!(std::fs::read(path).unwrap(), expected);
    }
}

#[test]
fn reset_flushes_payload_before_overwriting_the_header() {
    let dir = TestDir::new();
    let path = dir.file("reset");
    let mut writer = CachedWriter::new(&path, 16).unwrap();
    writer.write(b"0000payload").unwrap();
    writer.reset().unwrap();
    writer.write(b"0007").unwrap();
    drop(writer);
    assert_eq!(std::fs::read(path).unwrap(), b"0007payload");
    assert!(CachedWriter::new(&dir.file("zero"), 0).is_err());
    assert!(CachedWriter::new(&dir.file("missing/file"), 8).is_err());
}

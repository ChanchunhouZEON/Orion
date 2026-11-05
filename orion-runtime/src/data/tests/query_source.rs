/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::{
    data::{
        query::{QuerySource, VectorReader},
        VectorFormat,
    },
    SearchMetric,
};
use std::io::{self, Cursor, Read};

// Deliberately not Seek: partial reads model pipes and network readers.
struct Chunks(Cursor<Vec<u8>>);
impl Read for Chunks {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = out.len().min(3);
        self.0.read(&mut out[..n])
    }
}

#[test]
fn all_formats_stream_in_partial_reads_and_preserve_batch_boundaries() {
    for format in [
        VectorFormat::Bvecs,
        VectorFormat::Fvecs,
        VectorFormat::U8bin,
        VectorFormat::Fbin,
    ] {
        let binary = matches!(format, VectorFormat::Fbin | VectorFormat::U8bin);
        let bytes = matches!(format, VectorFormat::Bvecs | VectorFormat::U8bin);
        let mut encoded = Vec::new();
        if binary {
            encoded.extend(3u32.to_le_bytes());
            encoded.extend(2u32.to_le_bytes());
        }
        for row in [[1u8, 2], [3, 4], [5, 6]] {
            if !binary {
                encoded.extend(2u32.to_le_bytes());
            }
            for v in row {
                if bytes {
                    encoded.push(v);
                } else {
                    encoded.extend(f32::from(v).to_le_bytes());
                }
            }
        }
        let mut source =
            VectorReader::new(Chunks(Cursor::new(encoded)), format, 2, SearchMetric::L2).unwrap();
        assert_eq!(source.dimension(), 2);
        assert_eq!(source.count_hint(), binary.then_some(3));
        let mut batch = vec![];
        assert_eq!(source.read_batch(&mut batch, 2).unwrap(), 2);
        assert_eq!(batch, [1., 2., 3., 4.]);
        assert_eq!(source.read_batch(&mut batch, 2).unwrap(), 1);
        assert_eq!(batch, [5., 6.]);
        assert_eq!(source.read_batch(&mut batch, 2).unwrap(), 0);
    }
}

#[test]
fn malformed_streams_fail_instead_of_silently_dropping_records() {
    let good = [2u32.to_le_bytes().as_slice(), &[1, 2]].concat();
    for suffix in [
        vec![2],
        [2u32.to_le_bytes().as_slice(), &[1]].concat(),
        [3u32.to_le_bytes().as_slice(), &[1, 2, 3]].concat(),
    ] {
        let mut source = VectorReader::new(
            Cursor::new([good.clone(), suffix].concat()),
            VectorFormat::Bvecs,
            2,
            SearchMetric::L2,
        )
        .unwrap();
        assert!(source.read_batch(&mut vec![], 8).is_err());
    }
    let encoded = [1u32.to_le_bytes(), 2u32.to_le_bytes()].concat();
    let mut source = VectorReader::new(
        Cursor::new([encoded, vec![1, 2, 3]].concat()),
        VectorFormat::U8bin,
        2,
        SearchMetric::L2,
    )
    .unwrap();
    assert!(source.read_batch(&mut vec![], 8).is_err());
}

#[test]
fn query_headers_and_batch_sizes_fail_with_context() {
    for (bytes, format, dimension) in [
        (vec![], VectorFormat::Bvecs, 2),
        (vec![2], VectorFormat::Bvecs, 2),
        (2u32.to_le_bytes().to_vec(), VectorFormat::Bvecs, 0),
        (3u32.to_le_bytes().to_vec(), VectorFormat::Fvecs, 2),
        (0u32.to_le_bytes().to_vec(), VectorFormat::U8bin, 2),
        (1u32.to_le_bytes().to_vec(), VectorFormat::Fbin, 2),
        (
            [1u32.to_le_bytes(), 3u32.to_le_bytes()].concat(),
            VectorFormat::Fbin,
            2,
        ),
    ] {
        assert!(
            VectorReader::new(Cursor::new(bytes), format, dimension, SearchMetric::L2).is_err()
        );
    }
    let mut reader = VectorReader::new(
        Cursor::new([2u32.to_le_bytes().as_slice(), &[3, 4]].concat()),
        VectorFormat::Bvecs,
        2,
        SearchMetric::Cosine,
    )
    .unwrap();
    let mut output = vec![9.];
    assert!(reader.read_batch(&mut output, 0).is_err());
    assert_eq!(reader.read_batch(&mut output, 1).unwrap(), 1);
    assert!((output[0] - 0.6).abs() < 1e-6);
    assert!((output[1] - 0.8).abs() < 1e-6);
    let debug = format!("{reader:?}");
    assert!(debug.contains("dimension: 2"));
    assert!(!debug.contains("record_buf"));
}

#[test]
fn query_reader_retries_interruptions_but_propagates_io_errors() {
    struct InterruptOnce {
        input: Cursor<Vec<u8>>,
        interrupted: bool,
    }
    impl Read for InterruptOnce {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if !self.interrupted {
                self.interrupted = true;
                return Err(io::ErrorKind::Interrupted.into());
            }
            self.input.read(out)
        }
    }
    let input = [1u32.to_le_bytes().as_slice(), &[42]].concat();
    let mut reader = VectorReader::new(
        InterruptOnce {
            input: Cursor::new(input),
            interrupted: false,
        },
        VectorFormat::Bvecs,
        1,
        SearchMetric::L2,
    )
    .unwrap();
    let mut result = vec![];
    assert_eq!(reader.read_batch(&mut result, 8).unwrap(), 1);
    assert_eq!(result, [42.]);
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "denied test input",
            ))
        }
    }
    assert!(
        VectorReader::new(Broken, VectorFormat::Bvecs, 1, SearchMetric::L2)
            .unwrap_err()
            .to_string()
            .contains("denied test input")
    );
}

#[test]
fn query_reader_rejects_non_finite_values_and_trailing_records() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = [1u32.to_le_bytes(), value.to_le_bytes()].concat();
        let mut reader =
            VectorReader::new(Cursor::new(input), VectorFormat::Fvecs, 1, SearchMetric::L2)
                .unwrap();
        assert!(reader.read_batch(&mut vec![], 1).is_err());
    }
    let input = [
        1u32.to_le_bytes().as_slice(),
        1u32.to_le_bytes().as_slice(),
        &[7, 0, 0, 0, 0],
    ]
    .concat();
    let mut reader =
        VectorReader::new(Cursor::new(input), VectorFormat::U8bin, 1, SearchMetric::L2).unwrap();
    assert!(reader.read_batch(&mut vec![], 8).is_err());
}

#[cfg(any(feature = "cli", feature = "benchmark"))]
#[test]
fn ground_truth_truncates_to_k_and_validates_every_id_and_width() {
    use crate::data::query::GroundTruthReader;
    let encode = |words: &[u32]| {
        words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut reader = GroundTruthReader::new(Cursor::new(encode(&[3, 0, 2, 1, 3, 2, 1, 0])), 2, 3);
    assert_eq!(reader.next().unwrap(), Some(vec![0, 2]));
    assert_eq!(reader.next().unwrap(), Some(vec![2, 1]));
    assert_eq!(reader.next().unwrap(), None);
    assert!(format!("{reader:?}").contains("rows_read: 2"));
    for words in [&[1, 0][..], &[4, 0, 1, 2, 0], &[3, 0, 1, 3], &[3, 0, 1]] {
        let mut reader = GroundTruthReader::new(Cursor::new(encode(words)), 2, 3);
        assert!(reader.next().is_err(), "invalid GT: {words:?}");
    }
    let mut reader = GroundTruthReader::new(Cursor::new(encode(&[3, 0, 1, 2, 2, 0, 1])), 2, 3);
    assert!(reader.next().unwrap().is_some());
    assert!(reader.next().is_err());
}

#[cfg(feature = "file-queries")]
#[test]
fn file_queries_cross_decode_batches_and_validate_format_policy() {
    use crate::data::query::read_query_file;
    use crate::VectorStorageKind;
    let path = std::env::temp_dir().join(format!("orion-query-file-{}.bvecs", std::process::id()));
    let mut bytes = vec![];
    let mut expected = vec![];
    for row in 0..2053 {
        bytes.extend(2u32.to_le_bytes());
        let values = [(row % 251) as u8, ((row + 1) % 251) as u8];
        bytes.extend(values);
        expected.extend(values.map(f32::from));
    }
    std::fs::write(&path, bytes).unwrap();
    assert_eq!(
        read_query_file(&path, None, 2, VectorStorageKind::U8).unwrap(),
        expected
    );
    assert_eq!(
        read_query_file(&path, Some(VectorFormat::Bvecs), 2, VectorStorageKind::F32).unwrap(),
        expected
    );
    assert!(read_query_file(&path, Some(VectorFormat::Fvecs), 2, VectorStorageKind::U8).is_err());
    assert!(read_query_file(&path, None, 3, VectorStorageKind::F32).is_err());
    std::fs::remove_file(&path).unwrap();
    assert!(read_query_file(&path, None, 2, VectorStorageKind::U8).is_err());
    assert!(read_query_file(
        &path.with_extension("unknown"),
        None,
        2,
        VectorStorageKind::F32
    )
    .is_err());
}

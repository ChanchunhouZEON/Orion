use orion_cli::{
    cascade::SearchMetric,
    cli::{
        data::VectorFormat,
        query::{QuerySource, VectorReader},
    },
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

use orion_runtime::data::{padded, VectorFormat};
use orion_runtime::search::cascade::SearchMetric;
use vector::{L2F32Distance, L2U8Distance, PaddedLayout};

struct Temp(std::path::PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn file(bytes: &[u8]) -> Temp {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!(
        "orion-padded-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&p, bytes).unwrap();
    Temp(p)
}

#[test]
fn formats_and_arrays_share_padded_layout() {
    for dim in [3, 100, 128, 129] {
        let rows = 5;
        let layout = PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap();
        let values: Vec<f32> = (0..rows * dim).map(|i| (i % 251) as f32).collect();
        let array = padded::from_packed(&values, layout, SearchMetric::L2).unwrap();
        for format in [
            VectorFormat::Fbin,
            VectorFormat::U8bin,
            VectorFormat::Fvecs,
            VectorFormat::Bvecs,
        ] {
            let binary = matches!(format, VectorFormat::Fbin | VectorFormat::U8bin);
            let float = matches!(format, VectorFormat::Fbin | VectorFormat::Fvecs);
            let mut bytes = vec![];
            if binary {
                bytes.extend((rows as u32).to_le_bytes());
                bytes.extend((dim as u32).to_le_bytes());
            }
            for row in values.chunks_exact(dim) {
                if !binary {
                    bytes.extend((dim as u32).to_le_bytes());
                }
                for &x in row {
                    if float {
                        bytes.extend(x.to_le_bytes());
                    } else {
                        bytes.push(x as u8);
                    }
                }
            }
            let f = file(&bytes);
            let loaded =
                padded::load::<f32>(&f.0, format, layout, rows, SearchMetric::L2, None).unwrap();
            assert_eq!(loaded.as_slice(), array.as_slice());
            let prefix =
                padded::load::<f32>(&f.0, format, layout, 2, SearchMetric::L2, None).unwrap();
            assert_eq!(
                prefix.as_slice(),
                &array.as_slice()[..2 * layout.stride_elements()]
            );
            assert!(
                padded::load::<f32>(&f.0, format, layout, rows, SearchMetric::L2, Some(1))
                    .err()
                    .unwrap()
                    .contains("padding")
            );
            if !float {
                let u8layout = PaddedLayout::for_kernel::<L2U8Distance>(dim).unwrap();
                let loaded =
                    padded::load::<u8>(&f.0, format, u8layout, rows, SearchMetric::L2, None)
                        .unwrap();
                for i in 0..rows {
                    assert_eq!(
                        loaded.row(i).unwrap(),
                        values[i * dim..(i + 1) * dim]
                            .iter()
                            .map(|x| *x as u8)
                            .collect::<Vec<_>>()
                    );
                }
            }
            if !binary {
                let header_offset = 4 + dim * if float { 4 } else { 1 };
                bytes[header_offset..header_offset + 4].copy_from_slice(&7u32.to_le_bytes());
                let bad = file(&bytes);
                assert!(
                    padded::load::<f32>(&bad.0, format, layout, rows, SearchMetric::L2, None)
                        .is_err()
                );
            }
            bytes.pop();
            let truncated = file(&bytes);
            assert!(padded::load::<f32>(
                &truncated.0,
                format,
                layout,
                rows,
                SearchMetric::L2,
                None
            )
            .is_err());
        }
    }
}

#[test]
fn cosine_and_query_reuse_exclude_padding() {
    let layout = PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap();
    let mut data =
        padded::from_packed(&[3., 4., 0., 0., 0., 2.], layout, SearchMetric::Cosine).unwrap();
    assert_eq!(data.row(0).unwrap(), &[0.6, 0.8, 0.]);
    assert_eq!(data.row(1).unwrap(), &[0., 0., 1.]);
    data.set_row(0, &[1., 2., 3.]).unwrap();
    for row in data.as_slice().chunks_exact(layout.stride_elements()) {
        assert!(row[3..].iter().all(|&x| x == 0.));
    }
    assert!(padded::from_packed(&[0.; 3], layout, SearchMetric::Cosine).is_err());
    assert!(padded::from_packed(&[f32::NAN, 0., 1.], layout, SearchMetric::L2).is_err());
}

#[test]
fn native_reads_cross_io_blocks_with_and_without_padding() {
    for dim in [31, 64] {
        let rows = (8 * 1024 * 1024) / dim + 3;
        let values: Vec<u8> = (0..rows * dim).map(|i| (i % 251) as u8).collect();
        let mut bytes = Vec::with_capacity(8 + values.len());
        bytes.extend((rows as u32).to_le_bytes());
        bytes.extend((dim as u32).to_le_bytes());
        bytes.extend(&values);
        let input = file(&bytes);
        let layout = PaddedLayout::for_kernel::<L2U8Distance>(dim).unwrap();
        let loaded = padded::load::<u8>(
            &input.0,
            VectorFormat::U8bin,
            layout,
            rows,
            SearchMetric::L2,
            None,
        )
        .unwrap();
        for (id, row) in loaded
            .as_slice()
            .chunks_exact(layout.stride_elements())
            .enumerate()
        {
            assert_eq!(&row[..dim], &values[id * dim..(id + 1) * dim]);
            assert!(row[dim..].iter().all(|&v| v == 0));
        }
    }
}

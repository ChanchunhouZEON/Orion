use orion_ann_runtime::{BaseInput, ErrorCode, GraphConfig, Index, IndexOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::current_dir()?.join("index-fixtures");
    let bytes: Vec<u8> = (0..96 * 13)
        .map(|i| ((i * 73 + i / 13 * 11) % 251) as u8)
        .collect();
    let floats: Vec<f32> = bytes.iter().map(|&x| x as f32).collect();
    for native in [false, true] {
        let dir = root.join(if native { "u8" } else { "f32" });
        let input = || {
            if native {
                BaseInput::from_u8(&bytes, 13)
            } else {
                BaseInput::from_f32(&floats, 13)
            }
        };
        let mut index = Index::new(
            input()?,
            IndexOptions {
                threads: 2,
                cache_dir: dir.join("build"),
                graph: GraphConfig {
                    degree: 8,
                    build_l: 24,
                    max_extra: 4,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        assert_eq!(index.info()["graph_source"], "parlayann");
        let queries = floats[..8 * 13].to_vec();
        assert_eq!(
            index
                .search(queries.clone(), 3, 24, true)
                .err()
                .unwrap()
                .code(),
            ErrorCode::InvalidState
        );
        index.calibrate(queries.clone(), 3, Some(24))?;
        let expected = index.search(queries.clone(), 3, 24, true)?;
        assert_eq!(expected.counts, vec![3; 8]);
        index.save(&dir.join("snapshot"))?;
        let mut restored = Index::open(&dir.join("snapshot"), input()?, Some(1), None)?;
        assert_eq!(restored.search(queries, 3, 24, true)?.ids, expected.ids);
    }
    println!("Packaged Rust consumer: default ParlayANN f32/u8 + calibration/search/cache passed");
    Ok(())
}

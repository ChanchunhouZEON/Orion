/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(feature = "parlay")]
use orion_ann_parlay::{BuildOptions, Builder};
#[test]
fn default_runtime_build_calibration_cache_and_import_are_equivalent() {
    use orion_ann::GraphSourcePolicy;
    use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions};
    let root = std::env::temp_dir().join(format!("orion-ffi-runtime-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let n = 128;
    let dim = 13;
    let bytes: Vec<u8> = (0..n * dim)
        .map(|i| ((i * 17 + i / 13 * 23) % 251) as u8)
        .collect();
    let floats: Vec<f32> = bytes.iter().map(|&x| f32::from(x)).collect();
    let recipe = BuildOptions {
        degree: 12,
        build_l: 24,
        max_extra: 8,
        threads: 2,
        alpha: 1.25,
        ..BuildOptions::default()
    };
    for native in [false, true] {
        let dir = root.join(if native { "u8" } else { "f32" });
        std::fs::create_dir_all(&dir).unwrap();
        let staged = dir.join("reference.staged");
        let mut builder = if native {
            Builder::u8(&bytes, n, dim, dim, recipe).unwrap()
        } else {
            Builder::f32(&floats, n, dim, dim, recipe).unwrap()
        };
        builder.save_reference(&staged).unwrap();
        drop(builder);
        let input = || {
            if native {
                BaseInput::from_u8(&bytes, dim).unwrap()
            } else {
                BaseInput::from_f32(&floats, dim).unwrap()
            }
        };
        let options = IndexOptions {
            cache_dir: dir.join("memory"),
            threads: 2,
            graph: GraphConfig {
                degree: 12,
                build_l: 24,
                max_extra: 8,
                alpha: 1.25,
                ..GraphConfig::default()
            },
            ..IndexOptions::default()
        };
        // Standalone core and service callers must produce the same slab as the
        // reference STAG route, with exactly the same partition/extras recipe.
        let core_config = orion_ann_core::BuildConfig {
            degree: recipe.degree,
            build_l: recipe.build_l as usize,
            max_extra: recipe.max_extra as usize,
            threads: recipe.threads as usize,
            alpha: recipe.alpha as f32,
            ..Default::default()
        };
        let core_bytes = if native {
            orion_ann_core::Orion::<u8>::build_from_slice(&bytes, dim, core_config)
                .unwrap()
                .graph()
                .buffer_bytes()
                .to_vec()
        } else {
            orion_ann_core::Orion::<f32>::build_from_slice(&floats, dim, core_config)
                .unwrap()
                .graph()
                .buffer_bytes()
                .to_vec()
        };
        assert_eq!(
            core_bytes,
            if native {
                orion_ann_core::Orion::from_staged_graph(
                    orion_ann_core::PaddedDataset::from_packed(
                        &bytes,
                        orion_ann_core::PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(
                            dim,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                    std::io::BufReader::new(std::fs::File::open(&staged).unwrap()),
                    std::fs::metadata(&staged).unwrap().len(),
                    recipe.degree,
                    0,
                )
                .unwrap()
                .graph()
                .buffer_bytes()
                .to_vec()
            } else {
                orion_ann_core::Orion::from_staged_graph(
                    orion_ann_core::PaddedDataset::from_packed(
                        &floats,
                        orion_ann_core::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(dim)
                            .unwrap(),
                    )
                    .unwrap(),
                    std::io::BufReader::new(std::fs::File::open(&staged).unwrap()),
                    std::fs::metadata(&staged).unwrap().len(),
                    recipe.degree,
                    0,
                )
                .unwrap()
                .graph()
                .buffer_bytes()
                .to_vec()
            }
        );
        let mut memory = Index::new(input(), options.clone()).unwrap();
        let graph_path = std::fs::read_dir(dir.join("memory.objects"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.extension().is_some_and(|ext| ext == "pgraph"))
            .unwrap();
        assert_eq!(core_bytes, std::fs::read(graph_path).unwrap()[16..]);

        let mut import_options = options.clone();
        import_options.cache_dir = dir.join("import");
        import_options.graph.source = GraphSourcePolicy::Parlayann;
        import_options.graph.staged_file = Some(staged.clone());
        let mut imported = Index::new(input(), import_options).unwrap();
        let queries = floats[..dim * 16].to_vec();
        assert_eq!(
            memory
                .search(
                    orion_ann::QueryBatch::from_f32(&queries, dim).unwrap(),
                    &orion_ann::SearchOptions::new(3, 24)
                )
                .err()
                .unwrap()
                .code(),
            orion_ann_core::ErrorCode::InvalidState
        );
        for index in [&mut memory, &mut imported] {
            index.calibrate(queries.clone(), 3, Some(24)).unwrap();
        }
        let a = memory
            .search(
                orion_ann::QueryBatch::from_f32(&queries, dim).unwrap(),
                &orion_ann::SearchOptions::new(3, 24),
            )
            .unwrap();
        let b = imported
            .search(
                orion_ann::QueryBatch::from_f32(&queries, dim).unwrap(),
                &orion_ann::SearchOptions::new(3, 24),
            )
            .unwrap();
        assert_eq!(a.ids, b.ids);
        assert_eq!(a.counts, b.counts);
        let snapshot = memory.save(&dir.join("saved")).unwrap();
        drop(memory);
        let restored = Index::open(&snapshot, input()).unwrap();
        assert_eq!(
            a.ids,
            restored
                .search(
                    orion_ann::QueryBatch::from_f32(&queries, dim).unwrap(),
                    &orion_ann::SearchOptions::new(3, 24)
                )
                .unwrap()
                .ids
        );
        let manifests: Vec<_> = std::fs::read_dir(dir.join("memory"))
            .unwrap()
            .filter_map(|e| {
                let p = e.unwrap().path();
                p.to_string_lossy().ends_with("cache.json").then_some(p)
            })
            .collect();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifests[0]).unwrap()).unwrap();
        assert_eq!(
            manifest["builder_recipe"],
            orion_ann_core::build::PARLAY_RECIPE
        );
        assert!(manifest["imported_from"].is_null());
        // Explicit missing STAG is never replaced by an implicit build.
        let mut strict = options;
        strict.cache_dir = dir.join("strict");
        strict.graph.staged_file = Some(dir.join("missing.staged"));
        assert!(Index::new(input(), strict).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

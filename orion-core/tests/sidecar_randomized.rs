/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Opt-in deterministic randomized stress; no datasets or benchmark artifacts needed.
use orion_ann_core::model::{L2U8, ScalarQuantizedDataset};
use orion_ann_core::{PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;
use rand::{RngExt, SeedableRng, rngs::StdRng};

#[test]
#[ignore = "standalone randomized sidecar stress"]
fn random_shapes_preserve_rows_padding_and_distances() {
    let cases = std::env::var("ORION_SIDECAR_STRESS_CASES")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(10_000);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut rng = StdRng::seed_from_u64(0x5341_4645_5349_4445);
        for _ in 0..cases {
            let dim = rng.random_range(1..=257);
            let n = rng.random_range(1..=9);
            let packed: Vec<f32> = (0..n * dim)
                .map(|_| rng.random_range(-100.0..100.0))
                .collect();
            let data = PaddedDataset::from_packed(
                &packed,
                PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap(),
            )
            .unwrap();
            let sidecar = ScalarQuantizedDataset::<L2U8>::try_from_f32_dataset(&data).unwrap();
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-200.0..200.0)).collect();
            let q = sidecar.try_quantize_query(&query, true).unwrap();
            assert!(q[dim..].iter().all(|&v| v == 0));
            for id in 0..n {
                let row = sidecar.row(id as u32).unwrap();
                assert_eq!(row.len(), dim);
                let expected: u32 = row
                    .iter()
                    .zip(&q)
                    .map(|(&a, &b)| (a as i32 - b as i32).pow(2) as u32)
                    .sum();
                assert_eq!(sidecar.distance(id as u32, &q).unwrap(), expected as f32);
                assert!(
                    sidecar.data()[id * sidecar.stride() + dim..(id + 1) * sidecar.stride()]
                        .iter()
                        .all(|&v| v == 0)
                );
            }
        }
    });
    eprintln!("verified {cases} random sidecar shapes");
}

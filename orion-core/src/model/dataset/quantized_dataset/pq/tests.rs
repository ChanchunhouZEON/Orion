/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use super::super::{
    PreparedQuery, QuantizedDataset, QuantizedDatasetCompute, QuantizedDatasetShape,
};
use super::*;
use crate::{PaddedDataset, PaddedLayout};
fn base(rows: usize, dim: usize) -> PaddedDataset<f32> {
    let v: Vec<_> = (0..rows * dim)
        .map(|i| ((i * 17 + i / dim * 3) % 251) as f32 / 32.0 - 3.0)
        .collect();
    PaddedDataset::from_packed(
        &v,
        PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(dim).unwrap(),
    )
    .unwrap()
}
#[test]
fn adc_matches_reconstructed_vectors_for_unequal_chunks_and_both_metrics() {
    for (dim, m) in [(1, 1), (17, 3), (63, 7), (128, 16), (260, 65)] {
        let b = base(40, dim);
        let options = PqBuildOptions {
            chunks: Some(m),
            ..Default::default()
        };
        let pq = PqDataset::build(&b, &options).unwrap();
        assert_eq!(
            pq.resident_memory(),
            PqDataset::estimate_resident_memory(
                QuantizedDatasetShape {
                    rows: 40,
                    dimension: dim
                },
                &options
            )
            .unwrap()
        );
        let q = b.row(3).unwrap();
        for metric in [PqMetric::L2, PqMetric::InnerProduct] {
            let query = pq.query(q, metric).unwrap();
            let ids = [39, 0, 3, 5, 3, 18];
            let mut scored = Vec::new();
            query.stream(&ids, 32, |i, d| scored.push((i, d))).unwrap();
            assert_eq!(scored.iter().map(|x| x.0).collect::<Vec<_>>(), ids);
            for (id, d) in scored {
                let mut exact = 0.0;
                for (chunk, r) in pq.table.offsets.windows(2).enumerate() {
                    let width = r[1] - r[0];
                    let code = pq.codes()[id as usize * m + chunk] as usize;
                    for j in 0..width {
                        let x = pq.table.centers[r[0] * 256 + code * width + j]
                            + pq.table.mean[r[0] + j];
                        exact += if metric == PqMetric::L2 {
                            (q[r[0] + j] - x).powi(2)
                        } else {
                            -q[r[0] + j] * x
                        };
                    }
                }
                assert!(
                    (d - exact).abs() <= 2e-5 * exact.abs().max(1.0),
                    "dim={dim}, m={m}, {d} vs {exact}"
                );
            }
        }
        assert_eq!(pq.prepare_query(q).unwrap().distance(3).unwrap(), 0.0);
    }
}
#[test]
fn training_is_deterministic_across_thread_counts_and_handles_degenerate_data() {
    let b = base(300, 17);
    let opt = PqBuildOptions {
        chunks: Some(3),
        iterations: 2,
        ..Default::default()
    };
    let build = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| PqDataset::build(&b, &opt).unwrap())
    };
    let a = build(1);
    let b = build(4);
    assert_eq!(a.codes(), b.codes());
    assert_eq!(a.table.centers.as_slice(), b.table.centers.as_slice());
    let flat = vec![1.0; 300 * 4];
    let table = FixedChunkPQTable::train(&flat, 300, 4, 2).unwrap();
    assert!(table.encode(&flat, 300).unwrap().iter().all(|&c| c == 0));
}
#[test]
fn on_demand_codes_roundtrip_and_corruption_are_checked() {
    let b = base(31, 17);
    let mut pq = PqDataset::build(&b, &Default::default()).unwrap();
    let mut bytes = vec![];
    pq.encode(&mut bytes).unwrap();
    let baseline = pq
        .query(b.row(2).unwrap(), PqMetric::L2)
        .unwrap()
        .distance(7)
        .unwrap();
    let paged = PqDataset::decode_with_storage(
        &mut bytes.as_slice(),
        bytes.len() as u64,
        PqCodeStorage::OnDemand,
    )
    .unwrap();
    assert_eq!(paged.codes(), pq.codes());
    assert_eq!(paged.code_storage(), PqCodeStorage::OnDemand);
    assert_eq!(
        baseline,
        paged
            .query(b.row(2).unwrap(), PqMetric::L2)
            .unwrap()
            .distance(7)
            .unwrap()
    );
    pq.set_code_storage(PqCodeStorage::OnDemand).unwrap();
    pq.set_code_storage(PqCodeStorage::Resident).unwrap();
    assert_eq!(paged.codes(), pq.codes());
    for cut in [0, 8, 31, bytes.len() - 1] {
        assert!(PqDataset::decode(&mut &bytes[..cut], cut as u64).is_err());
    }
    let mut corrupt = bytes.clone();
    corrupt[32..40].copy_from_slice(&1u64.to_le_bytes());
    assert!(PqDataset::decode(&mut corrupt.as_slice(), corrupt.len() as u64).is_err());
    assert!(pq.pin_codes(0).is_err());
    assert!(pq.codebook_pinned_bytes() > 0 || pq.codebook_pin_error().is_some());
}
#[test]
fn malformed_input_is_rejected_without_output() {
    assert!(FixedChunkPQTable::train(&[], 0, 3, 1).is_err());
    assert!(FixedChunkPQTable::train(&[f32::NAN], 1, 1, 1).is_err());
    assert!(FixedChunkPQTable::train(&[0.0], 1, 1, 2).is_err());
    let b = base(3, 7);
    let pq = PqDataset::build(&b, &Default::default()).unwrap();
    assert!(pq.query(&[0.0], PqMetric::L2).is_err());
    assert!(pq.query(&[f32::INFINITY; 7], PqMetric::L2).is_err());
    let query = pq.prepare_query(&[0.0; 7]).unwrap();
    assert!(query.distance(3).is_err());
    let query = pq.query(&[0.0; 7], PqMetric::L2).unwrap();
    assert!(
        query
            .stream(&[0, 3], 32, |_, _| panic!("must reject before emitting"))
            .is_err()
    );
}

#[test]
fn imported_centroid_layout_preserves_mean_translation_for_l2_and_ip() {
    let dimension = 5;
    let centers: Vec<f32> = (0..256)
        .flat_map(|c| (0..dimension).map(move |d| c as f32 * 0.25 + d as f32))
        .collect();
    let mean = vec![3.0, -2.0, 0.5, 1.0, -0.75];
    let offsets = vec![0, 1, 3, 5];
    let table =
        FixedChunkPQTable::new(dimension, 3, centers.clone(), mean.clone(), offsets.clone())
            .unwrap();
    let code = vec![0u8, 127, 255];
    let pq = PqDataset::from_parts(
        table,
        super::storage::Codes::from_reader(
            &mut code.as_slice(),
            code.len(),
            super::storage::PqCodeStorage::Resident,
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let query = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut reconstructed = vec![0.0; dimension];
    for (chunk, r) in offsets.windows(2).enumerate() {
        for d in r[0]..r[1] {
            reconstructed[d] = centers[code[chunk] as usize * dimension + d] + mean[d];
        }
    }
    for metric in [PqMetric::L2, PqMetric::InnerProduct] {
        let expected: f32 = query
            .iter()
            .zip(&reconstructed)
            .map(|(q, x)| {
                if metric == PqMetric::L2 {
                    (q - x) * (q - x)
                } else {
                    -q * x
                }
            })
            .sum();
        assert!((pq.query(&query, metric).unwrap().distance(0).unwrap() - expected).abs() < 1e-3);
    }
    let mut bytes = vec![];
    pq.encode(&mut bytes).unwrap();
    // Header + 4 offsets: first mean word must reject a quiet NaN.
    bytes[64..68].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(PqDataset::decode(&mut bytes.as_slice(), bytes.len() as u64).is_err());
}

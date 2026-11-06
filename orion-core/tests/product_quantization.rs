/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! PQ is available in ordinary builds, with training selected at runtime.
#[test]
fn attaching_pq_keeps_codes_and_rejects_other_base_shapes() {
    use orion_ann_core::{FixedChunkPQTable, Orion, PaddedDataset, PaddedLayout};
    use std::sync::Arc;
    fn index() -> Orion {
        let base = PaddedDataset::from_packed(
            &[1.; 4 * 8],
            PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(8).unwrap(),
        )
        .unwrap();
        Orion::from_graph_partitions(
            base,
            &(0..4)
                .map(|i| (vec![(i + 1) % 4], vec![], vec![]))
                .collect::<Vec<_>>(),
            0,
            1,
            0,
        )
        .unwrap()
    }
    fn table(dimension: usize) -> Arc<FixedChunkPQTable> {
        Arc::new(
            FixedChunkPQTable::new(
                dimension,
                1,
                vec![0.; 256 * dimension],
                vec![0.; dimension],
                vec![0, dimension],
            )
            .unwrap(),
        )
    }
    let pq = table(8);
    let encoded = vec![0, 3, 127, 255];
    let attached = index()
        .with_product_quantization(pq.clone(), encoded.clone())
        .unwrap();
    assert_eq!(attached.pq_codes(), Some(encoded.as_slice()));
    assert_eq!(attached.num_pq_chunks(), Some(1));
    assert!(std::ptr::eq(attached.pq_table().unwrap(), pq.as_ref()));
    for (pq, codes) in [(table(7), encoded.clone()), (table(8), vec![0; 3])] {
        assert_eq!(
            index()
                .with_product_quantization(pq, codes)
                .unwrap_err()
                .code(),
            orion_ann_core::ErrorCode::InvalidArgument
        );
    }
}

#[test]
fn graph_builder_trains_pq_on_request_in_ordinary_builds() {
    use orion_ann_core::model::dataset::quantized_dataset::pq::FixedChunkPQTable;
    let rows = 32;
    let dimension = 32;
    let values: Vec<f32> = (0..rows * dimension)
        .map(|i| ((i * 17 + i / dimension * 7) % 251) as f32 / 32.)
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for enabled in [false, true] {
        let built = pool
            .install(|| {
                orion_ann_core::build_diskann_index(
                    &values,
                    rows,
                    dimension,
                    1.2,
                    4,
                    8,
                    enabled,
                    Some(4),
                    Some(8),
                    true,
                    2,
                )
            })
            .unwrap();
        assert_eq!(built.partitions.len(), rows);
        if !enabled {
            assert!(built.pq.is_none());
            assert!(built.pq_codes.is_none());
            continue;
        }
        // The family module and the graph builder share one table type/codec.
        let table: &FixedChunkPQTable = built.pq.as_deref().unwrap();
        let codes = built.pq_codes.as_ref().unwrap();
        assert_eq!(codes.len(), rows * 4);
        assert_eq!(table.encode(&values, rows).unwrap(), *codes);
        let mut query = values[..dimension].to_vec();
        table.preprocess_query(&mut query);
        let distances = table.populate_chunk_distances(&query);
        for code in codes.chunks_exact(4) {
            let adc = table.adc_distance(code, &distances);
            let direct = table.l2_distance(&query, code);
            assert!((adc - direct).abs() <= 1e-4 * direct.max(1.));
        }
        assert!(table.adc_distance(&codes[..4], &distances) < 1e-4);
    }
}

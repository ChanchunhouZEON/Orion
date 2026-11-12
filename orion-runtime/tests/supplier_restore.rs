/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Run twice with the same ORION_SUPPLIER_RESTORE_DIR: feature on, then off.
use orion_ann::{BaseInput, Index, QueryBatch, SearchOptions};

#[test]
#[ignore = "two-build contract; CI provides an isolated artifact directory"]
fn snapshot_needs_no_supplier_feature() {
    let path = std::path::PathBuf::from(std::env::var_os("ORION_SUPPLIER_RESTORE_DIR").unwrap());
    for (native, metric, name) in [
        (false, orion_ann::SearchMetric::SquaredL2, "f32"),
        (true, orion_ann::SearchMetric::SquaredL2, "u8-l2"),
        (true, orion_ann::SearchMetric::InnerProduct, "u8-ip"),
        (true, orion_ann::SearchMetric::Cosine, "u8-cosine"),
    ] {
        let path = path.join(name);
        #[cfg(not(feature = "experimental-graph"))]
        let _ = metric;
        let values: Vec<f32> = (0..48).map(|i| ((i * 19) % 37) as f32).collect();
        let base = || {
            if native {
                BaseInput::from_u8(&values.iter().map(|v| *v as u8).collect::<Vec<_>>(), 3).unwrap()
            } else {
                BaseInput::from_f32(&values, 3).unwrap()
            }
        };
        #[cfg(feature = "experimental-graph")]
        {
            use orion_ann::{IndexOptions, supplier::*};
            std::fs::create_dir_all(&path).unwrap();
            let spec = PartitionSpec {
                origin: PartitionOrigin {
                    name: "feature-off-fixture".into(),
                    revision: None,
                },
                points: 16,
                dimension: 3,
                element: if native {
                    PartitionElement::U8
                } else {
                    PartitionElement::F32
                },
                metric: match metric {
                    orion_ann::SearchMetric::SquaredL2 => PartitionMetric::SquaredL2,
                    orion_ann::SearchMetric::InnerProduct => PartitionMetric::InnerProduct,
                    orion_ann::SearchMetric::Cosine => PartitionMetric::Cosine,
                },
                entry: 7,
                max_degree: 12,
                max_extra: 3,
                base_sha256: None,
            };
            let rows = (0..16)
                .map(|i| {
                    let ids: Vec<u32> = (1..16).map(|n| (i + n) % 16).collect();
                    (ids[..6].to_vec(), ids[6..12].to_vec(), ids[12..].to_vec())
                })
                .collect();
            let mut index = Index::from_partitions(
                base(),
                spec,
                OwnedPartitions::new(rows),
                IndexOptions {
                    metric,
                    cache_dir: path.join("cache"),
                    threads: 2,
                    auto_pin: false,
                    ..Default::default()
                },
            )
            .unwrap();
            index.calibrate(values.clone(), 3, Some(16)).unwrap();
            index.save(&path.join("snapshot")).unwrap();
            let expected = index
                .search(
                    QueryBatch::from_f32(&values, 3).unwrap(),
                    &SearchOptions::new(3, 16),
                )
                .unwrap();
            std::fs::write(
                path.join("expected.json"),
                serde_json::to_vec(&expected.ids).unwrap(),
            )
            .unwrap();
        }
        let restored = Index::open(path.join("snapshot"), base()).unwrap();
        let result = restored
            .search(
                QueryBatch::from_f32(&values, 3).unwrap(),
                &SearchOptions::new(3, 16),
            )
            .unwrap();
        let expected: Vec<u32> =
            serde_json::from_slice(&std::fs::read(path.join("expected.json")).unwrap()).unwrap();
        assert_eq!(result.ids, expected);
        assert_eq!(restored.info()["supplier"]["name"], "feature-off-fixture");
    }
}

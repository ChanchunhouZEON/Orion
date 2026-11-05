/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_core::{Orion, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;

fn from_rows(values: &[f32]) -> Orion {
    Orion::from_graph_partitions(
        PaddedDataset::from_packed(
            values,
            PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap(),
        )
        .unwrap(),
        &[(vec![1], vec![], vec![]), (vec![0], vec![], vec![])],
        0,
        1,
        0,
    )
    .unwrap()
}
#[test]
fn equal_shape_different_base_cannot_reuse_sidecar() {
    let temp = std::env::temp_dir().join(format!("orion-working-sidecars-{}", std::process::id()));
    std::fs::create_dir(&temp).unwrap();
    let work = temp.join("work");
    std::fs::create_dir(&work).unwrap();
    let path = work.join("index.bin");
    let mut first = from_rows(&[0., 1., 2., 3., 4., 5.]);
    super::attach(&mut first, &path).unwrap();
    let before = first
        .try_ensure_quantized_dataset()
        .unwrap()
        .data()
        .to_vec();
    let mut second = from_rows(&[3., 4., 5., 0., 1., 2.]);
    super::attach(&mut second, &path).unwrap();
    let after = second
        .try_ensure_quantized_dataset()
        .unwrap()
        .data()
        .to_vec();
    assert_ne!(before, after);
    let expected =
        orion_ann_core::model::ScalarQuantizedDataset::<orion_ann_core::model::L2U8>::from_f32_dataset(
            second.dataset(),
        );
    assert_eq!(after, expected.data());
    // A hit must reuse the payload. A rebuild publishes a fresh file whose
    // modification time would differ from this known timestamp.
    let identity: super::working::Identity =
        serde_json::from_slice(&std::fs::read(path.with_extension("qds.identity.json")).unwrap())
            .unwrap();
    let payload = temp
        .join("work.objects")
        .join(format!("{}.qds", identity.payload_sha256));
    let marker = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
    std::fs::File::options()
        .write(true)
        .open(&payload)
        .unwrap()
        .set_modified(marker)
        .unwrap();
    let mut third = from_rows(&[3., 4., 5., 0., 1., 2.]);
    super::attach(&mut third, &path).unwrap();
    assert_eq!(third.try_ensure_quantized_dataset().unwrap().data(), after);
    assert_eq!(
        std::fs::metadata(&payload).unwrap().modified().unwrap(),
        marker
    );
    // Tampering with a well-shaped payload must also invalidate its identity.
    let qp = payload;
    let mut bytes = std::fs::read(&qp).unwrap();
    bytes[24] ^= 255;
    std::fs::write(&qp, bytes).unwrap();
    let mut fourth = from_rows(&[3., 4., 5., 0., 1., 2.]);
    super::attach(&mut fourth, &path).unwrap();
    assert_eq!(fourth.try_ensure_quantized_dataset().unwrap().data(), after);
    std::fs::remove_dir_all(temp).unwrap();
}

#[test]
fn failed_snapshot_verification_does_not_install_a_decoded_value() {
    let source = from_rows(&[0., 1., 2., 3., 4., 5.]);
    let mut bytes = Vec::new();
    source
        .ensure_quantized_dataset()
        .encode(&mut bytes)
        .unwrap();
    let mut target = from_rows(&[0., 1., 2., 3., 4., 5.]);
    let failure = target.restore_sidecar("qds", |decode| {
        decode(&mut bytes.as_slice(), bytes.len() as u64)?;
        Err(crate::Error::cache("digest mismatch"))
    });
    assert!(failure.is_err());
    target
        .visit_sidecars(|_, _| panic!("unverified value was published"))
        .unwrap();
    target
        .restore_sidecar("qds", |decode| {
            decode(&mut bytes.as_slice(), bytes.len() as u64)
        })
        .unwrap();
    assert!(target
        .restore_sidecar("../qds", |_| panic!("unknown codec opened a file"))
        .is_err());
}

#[test]
fn a_matching_digest_cannot_override_the_base_shape() {
    use orion_ann_core::model::{ScalarQuantizedDataset, L2U8};
    use sha2::{Digest, Sha256};

    let temp = std::env::temp_dir().join(format!("orion-sidecar-shape-{}", std::process::id()));
    std::fs::create_dir(&temp).unwrap();
    let work = temp.join("work");
    std::fs::create_dir(&work).unwrap();
    let path = work.join("index.bin");
    let data = [0., 1., 2., 3., 4., 5.];
    let mut index = from_rows(&data);
    super::attach(&mut index, &path).unwrap();

    // Encode the same six coordinates as three 2D rows. Even a record naming
    // the current base and the correct payload digest cannot make it a 3D cache.
    let other =
        PaddedDataset::from_packed(&data, PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap())
            .unwrap();
    let mut bytes = Vec::new();
    ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&other)
        .encode(&mut bytes)
        .unwrap();
    let payload = path.with_extension("qds");
    std::fs::create_dir(temp.join("work.objects")).unwrap();
    std::fs::write(
        temp.join("work.objects")
            .join(format!("{:x}.qds", Sha256::digest(&bytes))),
        &bytes,
    )
    .unwrap();
    let identity = super::working::Identity {
        dataset_sha256: super::working::fingerprint(index.dataset()).unwrap(),
        payload_sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    std::fs::write(
        super::working::metadata_path(&payload),
        serde_json::to_vec(&identity).unwrap(),
    )
    .unwrap();

    assert_eq!(index.try_ensure_quantized_dataset().unwrap().dimension(), 3);
    let repaired: super::working::Identity =
        serde_json::from_slice(&std::fs::read(super::working::metadata_path(&payload)).unwrap())
            .unwrap();
    assert_ne!(repaired.payload_sha256, identity.payload_sha256);
    std::fs::remove_dir_all(temp).unwrap();
}

#[test]
fn working_key_binds_raw_input_and_row_transformation() {
    use crate::{IndexOptions, SearchMetric, VectorFormat};
    let options = IndexOptions::default();
    let c = options
        .resolve(
            "base".into(),
            VectorFormat::Fbin,
            32,
            128,
            Some("file-abc".into()),
        )
        .unwrap();
    let key = |c: &crate::config::ResolvedRunConfig, points| {
        super::derived_key(c, points).unwrap().unwrap()
    };
    let original = key(&c, 128);
    assert_ne!(original, key(&c, 96));
    let mut cfg = c.clone();
    cfg.metric = SearchMetric::Cosine;
    assert_ne!(original, key(&cfg, 128));
    cfg = c.clone();
    cfg.dimension = 16;
    assert_ne!(original, key(&cfg, 128));
    cfg = c.clone();
    cfg.base_format = VectorFormat::Fvecs;
    assert_ne!(original, key(&cfg, 128));
    cfg = c.clone();
    cfg.base_content_id = Some("abc".into());
    assert_ne!(original, key(&cfg, 128));
    cfg.base_content_id = None;
    assert_eq!(super::derived_key(&cfg, 128).unwrap(), None);
}

#[test]
fn authenticated_input_replaces_legacy_record_once() {
    use super::{track, working::Identity};
    let root = std::env::temp_dir().join(format!("orion-sidecar-migration-{}", std::process::id()));
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    let key = work.join("index.bin");
    let identity = key.with_extension("qds.identity.json");
    let rows = [0., 1., 2., 3., 4., 5.];
    let mut legacy = from_rows(&rows);
    super::attach(&mut legacy, &key).unwrap();
    legacy.try_ensure_quantized_dataset().unwrap();
    let read =
        || -> Identity { serde_json::from_slice(&std::fs::read(&identity).unwrap()).unwrap() };
    let old = read();
    let reopen = || {
        let mut index = from_rows(&rows);
        track(
            &mut index,
            root.join("work.objects"),
            Some(key.clone()),
            Default::default(),
            Some("verified input".into()),
        );
        index.try_ensure_quantized_dataset().unwrap();
    };
    reopen();
    let new = read();
    assert_ne!(old.dataset_sha256, new.dataset_sha256);
    assert_eq!(old.payload_sha256, new.payload_sha256);
    let marker = std::time::UNIX_EPOCH + std::time::Duration::from_secs(123456789);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&identity)
        .unwrap()
        .set_modified(marker)
        .unwrap();
    reopen();
    assert_eq!(
        std::fs::metadata(&identity).unwrap().modified().unwrap(),
        marker
    );
    std::fs::remove_dir_all(root).unwrap();
}

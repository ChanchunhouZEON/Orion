/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(feature = "parlay")]
use orion_ann::{BaseInput, CascadeConfig, GraphConfig, Index, IndexOptions};
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::Path,
};

fn flip_byte(path: &Path, offset: u64) {
    use std::io::Read;
    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let modified = f.metadata().unwrap().modified().unwrap();
    let mut byte = [0];
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.read_exact(&mut byte).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&[byte[0] ^ 1]).unwrap();
    f.set_modified(modified).unwrap();
}

#[test]
fn snapshots_bind_every_consumed_payload() {
    let dir = std::env::temp_dir().join(format!("orion-auth-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let base = dir.join("base.fbin");
    let values: Vec<f32> = (0..128 * 32)
        .map(|i| ((i * 73 + i / 32) % 251) as f32)
        .collect();
    let mut raw = Vec::from(128u32.to_le_bytes());
    raw.extend(32u32.to_le_bytes());
    for v in &values {
        raw.extend(v.to_le_bytes());
    }
    fs::write(&base, raw).unwrap();
    let input = || BaseInput::from_file(base.clone());
    let mut index = Index::new(
        input(),
        IndexOptions {
            threads: 1,
            max_points: Some(96),
            cache_dir: dir.join("build"),
            graph: GraphConfig {
                degree: 8,
                build_l: 24,
                max_extra: 4,
                ..Default::default()
            },
            cascade: CascadeConfig {
                admission: Some("l2-u8".into()),
                prefilter: Some("none".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let q = values[..32 * 8].to_vec();
    index.calibrate(q.clone(), 3, Some(24)).unwrap();
    let expected = index
        .search(
            orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
            &orion_ann::SearchOptions::new(3, 24),
        )
        .unwrap();
    let snapshot = dir.join("snapshot");
    let objects = dir.join("build.objects");
    fn inventory(
        dir: &Path,
    ) -> std::collections::BTreeMap<std::ffi::OsString, (u64, std::time::SystemTime)> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                let stat = e.metadata().unwrap();
                (e.file_name(), (stat.len(), stat.modified().unwrap()))
            })
            .collect()
    }
    // Stamp every payload before recalibration so an accidental rewrite is observable.
    let marker = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
    for e in fs::read_dir(&objects).unwrap() {
        fs::OpenOptions::new()
            .write(true)
            .open(e.unwrap().path())
            .unwrap()
            .set_modified(marker)
            .unwrap();
    }
    let before = inventory(&objects);
    let reference = index.save(&snapshot).unwrap();
    assert_eq!(before, inventory(&objects));
    index.calibrate(q.clone(), 2, Some(24)).unwrap();
    let second = index.save(&dir.join("second")).unwrap();
    assert_eq!(before, inventory(&objects));
    for path in [reference.path(), second.path()] {
        assert_eq!(
            inventory(path).keys().collect::<Vec<_>>(),
            [&std::ffi::OsString::from("index.json")]
        );
    }
    fs::remove_dir_all(dir.join("build")).unwrap();
    assert_eq!(
        Index::open(&second, input())
            .unwrap()
            .search(
                orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
                &orion_ann::SearchOptions::new(3, 24)
            )
            .unwrap(),
        expected
    );
    assert!(Index::open(&second, input())
        .unwrap()
        .search(
            orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
            &orion_ann::SearchOptions::new(2, 24)
        )
        .is_ok());
    assert!(Index::open(&reference, input())
        .unwrap()
        .search(
            orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
            &orion_ann::SearchOptions::new(2, 24)
        )
        .is_err());
    let persisted = serde_json::to_vec(&reference).unwrap();
    let reference: orion_ann::SnapshotRef = serde_json::from_slice(&persisted).unwrap();
    let open = || Index::open(&reference, input());
    assert_eq!(
        open()
            .unwrap()
            .search(
                orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
                &orion_ann::SearchOptions::new(3, 24)
            )
            .unwrap(),
        expected
    );
    let files = || {
        fs::read_dir(dir.join("build.objects"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect::<Vec<_>>()
    };
    for path in files()
        .into_iter()
        .filter(|p| p.file_name().unwrap() != "index.json")
    {
        use sha2::{Digest, Sha256};
        let expected = format!("{:x}", Sha256::digest(fs::read(&path).unwrap()));
        assert_eq!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .split('.')
                .next()
                .unwrap(),
            expected
        );
    }
    let file_count = files().len();
    for ext in ["bin", "pgraph", "qds", "json"] {
        let path = files()
            .into_iter()
            .find(|p| p.extension().unwrap() == ext && p.file_name().unwrap() != "index.json")
            .unwrap();
        let offset = fs::metadata(&path).unwrap().len() / 2;
        flip_byte(&path, offset);
        assert!(open().is_err(), "accepted modified {ext}");
        assert!(
            Index::open(&snapshot, input()).is_err(),
            "directory accepted modified {ext}"
        );
        flip_byte(&path, offset);
    }
    // Unversioned snapshots are rejected. A versioned standalone directory
    // can explicitly use a relative object store without weakening verification.
    let legacy = dir.join("standalone");
    fs::create_dir(&legacy).unwrap();
    let mut old: serde_json::Value =
        serde_json::from_slice(&fs::read(snapshot.join("index.json")).unwrap()).unwrap();
    old.as_object_mut().unwrap().remove("format_version");
    fs::write(legacy.join("index.json"), serde_json::to_vec(&old).unwrap()).unwrap();
    for file in files() {
        fs::hard_link(&file, legacy.join(file.file_name().unwrap())).unwrap();
    }
    assert!(Index::open(&legacy, input()).is_err());
    old["format_version"] = 2.into();
    old["object_store"] = ".".into();
    fs::write(legacy.join("index.json"), serde_json::to_vec(&old).unwrap()).unwrap();
    assert_eq!(
        Index::open(&legacy, input())
            .unwrap()
            .search(
                orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
                &orion_ann::SearchOptions::new(3, 24)
            )
            .unwrap(),
        expected
    );
    // The raw input fingerprint covers even rows outside the indexed prefix.
    flip_byte(&base, 8 + 110 * 32 * 4);
    assert!(open().unwrap_err().to_string().contains("SHA-256"));
    let stale = dir.join("stale");
    assert!(index.save(&stale).is_err());
    assert!(!stale.exists());
    flip_byte(&base, 8 + 110 * 32 * 4);
    assert_eq!(
        open()
            .unwrap()
            .search(
                orion_ann::QueryBatch::from_f32(&q, 32).unwrap(),
                &orion_ann::SearchOptions::new(3, 24)
            )
            .unwrap(),
        expected
    );
    assert_eq!(files().len(), file_count);
    let moved = dir.join("same-content.fbin");
    fs::copy(&base, &moved).unwrap();
    assert!(Index::open(&reference, BaseInput::from_file(moved)).is_ok());
    // Replace the entire directory with another valid snapshot of the same base.
    // All files match its own manifest, but the original reference must reject it.
    index
        .calibrate(values[..32 * 8].to_vec(), 4, Some(24))
        .unwrap();
    let replacement = index.save(&dir.join("replacement")).unwrap();
    assert!(Index::open(&replacement, input()).is_ok());
    fs::rename(&snapshot, dir.join("original")).unwrap();
    fs::rename(replacement.path(), &snapshot).unwrap();
    assert!(Index::open(&snapshot, input()).is_ok());
    assert!(open()
        .unwrap_err()
        .to_string()
        .contains("manifest SHA-256 mismatch"));
    fs::remove_dir_all(dir).unwrap();
}

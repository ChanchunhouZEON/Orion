/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::{load_verified, save_graph, save_working};
use orion_ann_core::{DynamicSearchConfig, Orion, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

fn index() -> Orion {
    let data = PaddedDataset::from_packed(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap(),
    )
    .unwrap();
    Orion::from_graph_partitions(
        data,
        &[
            (vec![1, 2], vec![], vec![]),
            (vec![0, 2], vec![], vec![]),
            (vec![0, 1], vec![], vec![]),
        ],
        0,
        2,
        0,
    )
    .unwrap()
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "orion-state-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn saving_and_associating_preserve_search_and_loaded_sidecars() {
    let temp = Temp::new();
    let mut index = index();
    let search = DynamicSearchConfig {
        k: 2,
        search_list_size: 3,
        ..Default::default()
    };
    let expected = index.search_l2(&[0., 1.], search).unwrap();
    let base_ptr = index.dataset().as_slice().as_ptr();
    let sidecar_ptr = index.ensure_quantized_dataset() as *const _;
    let attached = temp.0.join("attached.bin");
    let digest = save_working(&mut index, &attached).unwrap();
    // Check the on-disk format independently of the new stream decoder.
    let graph = index.graph();
    let header = [
        graph.num_nodes() as u32,
        graph.max_degree(),
        graph.stride() as u32,
        graph.base_local_count(),
    ];
    let expected_graph = [bytemuck::cast_slice(&header), graph.buffer_bytes()].concat();
    assert_eq!(
        std::fs::read(attached.with_extension("pgraph")).unwrap(),
        expected_graph
    );
    use sha2::Digest;
    assert_eq!(
        digest,
        format!("{:x}", sha2::Sha256::digest(&expected_graph))
    );
    // Exporting a snapshot does not redirect subsequent lazy cache accesses.
    save_graph(&index, &temp.0.join("export.bin")).unwrap();
    assert_eq!(index.dataset().as_slice().as_ptr(), base_ptr);
    assert_eq!(index.ensure_quantized_dataset() as *const _, sidecar_ptr);
    assert_eq!(index.search_l2(&[0., 1.], search).unwrap(), expected);
    let restored: crate::integrity::Verified<Orion> = load_verified(
        &attached.with_extension("pgraph"),
        index.into_dataset(),
        &std::fs::read(&attached).unwrap(),
        &digest,
    )
    .unwrap();
    assert_eq!(
        restored.value.search_l2(&[0., 1.], search).unwrap(),
        expected
    );
}

#[test]
fn failed_save_does_not_redirect_the_existing_cache_association() {
    let temp = Temp::new();
    let mut index = index();
    let attached = temp.0.join("attached.bin");
    save_working(&mut index, &attached).unwrap();
    let blocker = temp.0.join("not-a-directory");
    std::fs::write(&blocker, b"block").unwrap();
    assert!(save_working(&mut index, &blocker.join("index.bin")).is_err());
    assert!(save_working(&mut index, std::path::Path::new("")).is_err());
    index.try_ensure_quantized_dataset().unwrap();
    assert!(attached.with_extension("qds.identity.json").exists());
    assert!(attached.with_extension("pgraph").exists());
}

#[cfg(feature = "benchmark")]
#[test]
fn benchmark_v5_cache_preserves_search_but_rejects_corrupt_layout() {
    let temp = Temp::new();
    let path = temp.0.join("old.bin");
    let source = index();
    save_graph(&source, &path).unwrap();
    // Historical standalone metadata is just version and entry point.
    std::fs::write(&path, [5u32.to_le_bytes(), 0u32.to_le_bytes()].concat()).unwrap();
    let search = DynamicSearchConfig {
        k: 2,
        search_list_size: 3,
        ..Default::default()
    };
    let expected = source.search_l2(&[0., 1.], search).unwrap();
    let loaded = super::load_working(&path, source.into_dataset()).unwrap();
    assert_eq!(loaded.search_l2(&[0., 1.], search).unwrap(), expected);

    let graph = path.with_extension("pgraph");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&graph)
        .unwrap();
    file.set_len(file.metadata().unwrap().len() - 1).unwrap();
    assert!(super::load_working(&path, index().into_dataset()).is_err());
    std::fs::write(&path, [4u32.to_le_bytes(), 0u32.to_le_bytes()].concat()).unwrap();
    assert!(super::load_working(&path, index().into_dataset()).is_err());
}

#[cfg(feature = "benchmark")]
#[test]
fn benchmark_load_does_not_disable_verified_reader() {
    let temp = Temp::new();
    let path = temp.0.join("current.bin");
    save_graph(&index(), &path).unwrap();
    // A well-formed graph loads for a trusted benchmark even without receipts.
    assert!(super::load_working(&path, index().into_dataset()).is_ok());
    let verified: crate::Result<crate::integrity::Verified<Orion>> = load_verified(
        &path.with_extension("pgraph"),
        index().into_dataset(),
        &std::fs::read(&path).unwrap(),
        &"0".repeat(64),
    );
    assert!(
        verified.is_err(),
        "benchmark feature must not disable SHA verification"
    );
    // New metadata includes shape; trusting contents must not skip that check.
    let mut metadata = std::fs::read(&path).unwrap();
    metadata[8..16].copy_from_slice(&3u64.to_le_bytes());
    std::fs::write(&path, metadata).unwrap();
    assert!(super::load_working(&path, index().into_dataset()).is_err());
}

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

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![allow(dead_code)]
use crate::model::PhasedGraph;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

pub type Partition = (Vec<u32>, Vec<u32>, Vec<u32>);

/// Each test owns only its own temporary directory, including on panic.
pub struct TempDir(pub PathBuf);
impl TempDir {
    pub fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "orion-graph-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// SplitMix64 fixes the generated workload independently of the rand version.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    pub fn below(&mut self, limit: usize) -> usize {
        self.next() as usize % limit
    }
    pub fn partition(&mut self, nodes: usize, degree: usize, extras: usize) -> Partition {
        let count = self.below(degree + 1);
        let local = self.below(count + 1);
        let extra = self.below(extras + 1);
        (
            (0..local).map(|_| self.below(nodes) as u32).collect(),
            (local..count).map(|_| self.below(nodes) as u32).collect(),
            (0..extra).map(|_| self.below(nodes) as u32).collect(),
        )
    }
}

pub fn check_node(view: &crate::PhasedGraph, node: usize, model: &Partition) {
    assert_eq!(view.local_neighbors(node), model.0);
    assert_eq!(view.remote_neighbors(node), model.1);
    assert_eq!(view.extra_candidates(node), model.2);
    assert_eq!(view.degree(node), model.0.len() + model.1.len());
}

impl PhasedGraph {
    pub(super) fn save(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
        self.encode(&mut writer)?;
        Ok(std::io::Write::flush(&mut writer)?)
    }

    pub(super) fn load(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::decode(bytes.as_slice(), bytes.len() as u64)
    }

    pub(super) fn load_staged(
        path: impl AsRef<std::path::Path>,
        nodes: usize,
        degree: u32,
    ) -> crate::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::decode_staged(bytes.as_slice(), bytes.len() as u64, nodes, degree)
    }
}

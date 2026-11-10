/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Service-owned output directories and operator-provided local input files.
use orion_ann::{Error, Result};
use std::path::{Component, Path, PathBuf};

/// Root namespace is operator-owned. Ownership survives process restart through
/// subject-scoped directories, independently of the in-memory job registry.
#[derive(Debug)]
pub(crate) struct StorageRoot {
    root: PathBuf,
}
impl StorageRoot {
    pub fn new(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)?;
        Ok(Self {
            root: root.canonicalize()?,
        })
    }
    pub fn for_subject(&self, principal: &crate::Principal) -> Result<Paths> {
        let tenants = self.root.join("tenants");
        let tenant = tenants.join(principal.subject());
        for path in [&tenants, &tenant] {
            std::fs::create_dir_all(path)?;
            // Do not accept aliases into another subject's namespace, even if
            // they stay inside the service root. Only operators write here.
            if path.canonicalize()? != *path {
                return Err(Error::invalid_argument(
                    "tenant directory cannot be a symlink",
                ));
            }
        }
        Paths::new(&tenant)
    }
}

#[derive(Debug)]
pub(crate) struct Paths {
    pub root: PathBuf,
}
impl Paths {
    pub fn new(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).map_err(|e| Error::io("http_root", root, e))?;
        let root = root
            .canonicalize()
            .map_err(|e| Error::io("http_root", root, e))?;
        for name in ["jobs", "snapshots"] {
            let path = root.join(name);
            std::fs::create_dir_all(&path).map_err(|e| Error::io("http_root", &path, e))?;
            if !path.canonicalize()?.starts_with(&root) {
                return Err(Error::invalid_argument(
                    "service directory escapes data root",
                ));
            }
        }
        Ok(Self { root })
    }
    pub fn input(&self, path: &Path) -> Result<PathBuf> {
        if path.is_absolute()
            || path
                .components()
                .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
        {
            return Err(Error::invalid_argument(
                "input paths must be relative to data root without '..'",
            )
            .at("http_path"));
        }
        let full = self.root.join(path);
        let canonical = full
            .canonicalize()
            .map_err(|e| Error::io("http_input", &full, e))?;
        if !canonical.starts_with(&self.root) {
            return Err(Error::invalid_argument("input symlink escapes data root").at("http_path"));
        }
        Ok(canonical)
    }
    pub fn snapshot(&self, id: &str) -> Result<PathBuf> {
        if id.is_empty()
            || id.len() > 100
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err(Error::invalid_argument("invalid snapshot ID").at("http_path"));
        }
        self.input(&PathBuf::from("snapshots").join(id))
    }
    pub fn snapshots(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(self.root.join("snapshots"))? {
            let entry = entry?;
            if entry.file_type()?.is_dir() && entry.path().join("index.json").is_file() {
                names.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        names.sort();
        Ok(names)
    }
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Linux memory counters and visible cgroup hierarchy.
use super::MemoryResources;

#[cfg(target_os = "linux")]
pub(super) fn observe(result: &mut MemoryResources) {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    result.total_bytes = kib_field(&meminfo, "MemTotal:");
    result.available_bytes = kib_field(&meminfo, "MemAvailable:");
    result.locked_bytes = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| kib_field(&s, "VmLck:"));
    // Respect both the leaf and every visible ancestor; a parent's quota can
    // be smaller than the leaf's. Resolve actual mounts, including namespaces.
    if let (Ok(groups), Ok(mounts)) = (
        std::fs::read_to_string("/proc/self/cgroup"),
        std::fs::read_to_string("/proc/self/mountinfo"),
    ) {
        constrain_cgroups(result, &groups, &mounts, |p| {
            std::fs::read_to_string(p).ok()
        });
    } else {
        result.available_bytes = None;
    }
}

fn kib_field(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        if words.next()? != key {
            return None;
        }
        words.next()?.parse::<u64>().ok()?.checked_mul(1024)
    })
}

fn constrain_cgroups(
    result: &mut MemoryResources,
    groups: &str,
    mounts: &str,
    mut read: impl FnMut(&std::path::Path) -> Option<String>,
) {
    use std::path::{Component, Path};
    for group in groups.lines() {
        let parts: Vec<_> = group.splitn(3, ':').collect();
        if parts.len() != 3 {
            continue;
        }
        let v2 = parts[0] == "0" && parts[1].is_empty();
        if !v2 && !parts[1].split(',').any(|c| c == "memory") {
            continue;
        }
        let mut resolved = false;
        for mount in mounts.lines() {
            let Some((left, right)) = mount.split_once(" - ") else {
                continue;
            };
            let a: Vec<_> = left.split_whitespace().collect();
            let b: Vec<_> = right.split_whitespace().collect();
            if a.len() < 5 || b.len() < 3 {
                continue;
            }
            if if v2 {
                b[0] != "cgroup2"
            } else {
                b[0] != "cgroup" || !b[2].split(',').any(|s| s == "memory")
            } {
                continue;
            }
            let decode = |s: &str| {
                s.replace("\\040", " ")
                    .replace("\\011", "\t")
                    .replace("\\134", "\\")
            };
            let root = decode(a[3]);
            let mount = decode(a[4]);
            let path = Path::new(parts[2]);
            if path.components().any(|c| matches!(c, Component::ParentDir)) {
                continue;
            }
            let Ok(relative) = path.strip_prefix(&root) else {
                continue;
            };
            let mut directory = Path::new(&mount).join(relative);
            resolved = true;
            loop {
                let (limit_file, used_file) = if v2 {
                    ("memory.max", "memory.current")
                } else {
                    ("memory.limit_in_bytes", "memory.usage_in_bytes")
                };
                if let Some(limit) = read(&directory.join(limit_file)) {
                    if limit.trim() != "max" {
                        match (
                            limit.trim().parse::<u64>(),
                            read(&directory.join(used_file))
                                .and_then(|s| s.trim().parse::<u64>().ok()),
                        ) {
                            (Ok(limit), Some(used)) => {
                                result.total_bytes = result.total_bytes.map(|n| n.min(limit));
                                result.available_bytes = result
                                    .available_bytes
                                    .map(|n| n.min(limit.saturating_sub(used)));
                            }
                            _ => result.available_bytes = None,
                        }
                    }
                } else {
                    // A v2 root without the memory controller has no memory.max.
                    // Missing controls below the root cannot safely be assumed unlimited.
                    if directory != Path::new(&mount) {
                        result.available_bytes = None;
                    }
                }
                if directory == Path::new(&mount) || !directory.pop() {
                    break;
                }
            }
        }
        if !resolved {
            result.available_bytes = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cgroup_ancestors_and_namespaced_mounts_bound_headroom() {
        let mut r = MemoryResources {
            total_bytes: Some(10000),
            available_bytes: Some(8000),
            ..Default::default()
        };
        constrain_cgroups(
            &mut r,
            "0::/tenant/job",
            "1 0 0:1 /tenant /cg rw - cgroup2 cgroup rw",
            |p| {
                Some(
                    match p.to_str()? {
                        "/cg/job/memory.max" => "max",
                        "/cg/memory.max" => "6000",
                        "/cg/memory.current" => "5000",
                        _ => return None,
                    }
                    .into(),
                )
            },
        );
        assert_eq!(r.total_bytes, Some(6000));
        assert_eq!(r.available_bytes, Some(1000));
    }
    #[test]
    fn missing_cgroup_observations_never_mean_unlimited() {
        let mut r = MemoryResources {
            available_bytes: Some(10000),
            ..Default::default()
        };
        constrain_cgroups(&mut r, "0::/job", "", |_| None);
        assert_eq!(r.available_bytes, None);
        assert_eq!(
            kib_field("MemAvailable: 12 kB", "MemAvailable:"),
            Some(12288)
        );
        assert_eq!(kib_field("MemAvailable: invalid kB", "MemAvailable:"), None);
    }
    #[test]
    fn legacy_controller_and_overcommitted_usage() {
        let mut r = MemoryResources {
            total_bytes: Some(1000),
            available_bytes: Some(900),
            ..Default::default()
        };
        constrain_cgroups(
            &mut r,
            "2:cpu,memory:/job",
            "1 0 0:1 / /cg rw - cgroup cgroup rw,memory",
            |p| {
                Some(
                    if p.ends_with("memory.limit_in_bytes") {
                        "100"
                    } else {
                        "200"
                    }
                    .into(),
                )
            },
        );
        assert_eq!(r.available_bytes, Some(0));
    }
}

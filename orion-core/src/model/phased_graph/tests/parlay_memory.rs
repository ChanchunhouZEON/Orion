/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::PhasedGraph;
use crate::{Error, ErrorCode, Result};
use orion_ann_parlay::{BuildOptions, Builder, Metric};
pub fn convert_error(error: orion_ann_parlay::Error) -> Error {
    let code = match error.code {
        1 => ErrorCode::InvalidArgument,
        2 => ErrorCode::ResourceExhausted,
        _ => ErrorCode::Backend,
    };
    Error::new(code, error.message.clone())
        .at("build_graph")
        .with_detail("backend", "parlayann")
        .with_source(error)
}

pub fn collect_graph(
    builder: &mut Builder<'_>,
    n: usize,
    degree: u32,
    extra: usize,
) -> Result<PhasedGraph> {
    let mut graph = PhasedGraph::try_empty(n, degree, extra)?;
    for node in 0..n {
        let part = builder
            .next_partition()
            .map_err(convert_error)?
            .ok_or_else(|| Error::invalid_data("ParlayANN returned too few partitions"))?;
        if part.extra.len() > extra {
            return Err(Error::invalid_data(
                "ParlayANN extras exceed configured capacity",
            ));
        }
        graph.set_partition(node, part.local, part.remote, part.extra)?;
    }
    if builder.next_partition().map_err(convert_error)?.is_some() {
        return Err(Error::invalid_data(
            "ParlayANN returned too many partitions",
        ));
    }
    Ok(graph)
}

use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);

fn compare(mut builder: Builder<'_>, n: usize, options: BuildOptions) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!(
        "orion-ffi-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let staged = dir.join("graph.staged");
    let reference = dir.join("reference.staged");
    builder.save_staged(&staged).unwrap();
    builder.save_reference(&reference).unwrap();
    assert_eq!(
        std::fs::read(&staged).unwrap(),
        std::fs::read(&reference).unwrap()
    );
    let memory =
        collect_graph(&mut builder, n, options.degree, options.max_extra as usize).unwrap();
    let imported = PhasedGraph::load_staged(&reference, n, options.degree).unwrap();
    assert_eq!(memory.buffer_bytes(), imported.buffer_bytes());
    let bytes = memory.buffer_bytes().to_vec();
    std::fs::remove_dir_all(dir).unwrap();
    bytes
}

#[test]
fn f32_u8_padded_rows_match_original_staged_and_thread_counts() {
    let n = 160;
    let dim = 13;
    let stride = 32;
    let bytes: Vec<u8> = (0..n * stride)
        .map(|i| {
            if i % stride < dim {
                ((i * 73 + i / stride * 11) % 251) as u8
            } else {
                255
            }
        })
        .collect();
    let floats: Vec<f32> = bytes
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if i % stride < dim {
                x as f32 / 255.
            } else {
                f32::NAN
            }
        })
        .collect();
    for metric in [Metric::L2, Metric::InnerProduct] {
        let mut expected = None;
        for threads in [1, 4] {
            let options = BuildOptions {
                degree: 12,
                build_l: 24,
                max_extra: 8,
                threads,
                metric,
                ..BuildOptions::default()
            };
            let result = compare(
                Builder::f32(&floats, n, dim, stride, options).unwrap(),
                n,
                options,
            );
            let direct = PhasedGraph::try_build_with(
                n,
                options.degree,
                options.max_extra as usize,
                |slab, ss| {
                    Builder::f32_into(&floats, n, dim, stride, options, slab, ss)
                        .map_err(convert_error)
                },
            )
            .unwrap();
            assert_eq!(direct.buffer_bytes(), result);
            let packed: Vec<f32> = floats
                .chunks_exact(stride)
                .flat_map(|row| row[..dim].iter().copied())
                .collect();
            let core = crate::Orion::<f32>::build_from_slice(
                &packed,
                dim,
                crate::BuildConfig {
                    degree: options.degree,
                    build_l: options.build_l as usize,
                    max_extra: options.max_extra as usize,
                    threads: threads as usize,
                    metric: match metric {
                        Metric::L2 => crate::BuildMetric::SquaredL2,
                        Metric::InnerProduct => crate::BuildMetric::InnerProduct,
                        Metric::Cosine => crate::BuildMetric::Cosine,
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(core.graph().buffer_bytes(), result);

            if let Some(expected) = &expected {
                assert_eq!(&result, expected);
            } else {
                expected = Some(result);
            }
        }
    }
    for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
        let mut expected = None;
        for threads in [1, 4] {
            let options = BuildOptions {
                degree: 12,
                build_l: 24,
                max_extra: 8,
                threads,
                metric,
                ..BuildOptions::default()
            };
            let result = compare(
                Builder::u8(&bytes, n, dim, stride, options).unwrap(),
                n,
                options,
            );
            let direct = PhasedGraph::try_build_with(
                n,
                options.degree,
                options.max_extra as usize,
                |slab, ss| {
                    Builder::u8_into(&bytes, n, dim, stride, options, slab, ss)
                        .map_err(convert_error)
                },
            )
            .unwrap();
            assert_eq!(direct.buffer_bytes(), result);
            if let Some(expected) = &expected {
                assert_eq!(&result, expected);
            } else {
                expected = Some(result);
            }
        }
    }
}

#[test]
fn single_node_and_no_extras() {
    let opts = BuildOptions {
        degree: 1,
        build_l: 2,
        max_extra: 0,
        threads: 1,
        ..BuildOptions::default()
    };
    compare(Builder::u8(&[7], 1, 1, 1, opts).unwrap(), 1, opts);
}

#[test]
fn slab_boundary_shapes_ties_and_repeated_builds() {
    for (n, degree, extras) in [(1, 1, 0), (17, 16, 8), (37, 16, 0), (37, 3, 17)] {
        // Deliberate equal distances exercise the unchanged distance-only sort.
        let data: Vec<u8> = (0..n * 7).map(|i| (i % 3) as u8).collect();
        for (threads, local_pct) in [(1, 1), (4, 60), (4, 100)] {
            let opts = BuildOptions {
                degree,
                max_extra: extras,
                build_l: 32,
                threads,
                local_pct,
                ..BuildOptions::default()
            };
            let expected = compare(Builder::u8(&data, n, 7, 7, opts).unwrap(), n, opts);
            for _ in 0..2 {
                let direct = PhasedGraph::try_build_with(n, degree, extras as usize, |slab, ss| {
                    Builder::u8_into(&data, n, 7, 7, opts, slab, ss).map_err(convert_error)
                })
                .unwrap();
                assert_eq!(direct.buffer_bytes(), expected);
            }
        }
    }
}

#[test]
fn unpublished_slab_rejects_backend_errors_and_invalid_rows() {
    use crate::{Error, ErrorCode};
    let err = PhasedGraph::try_build_with(2, 4, 2, |_, _| Err(Error::invalid_argument("probe")))
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidArgument);
    for (word, value) in [(0, 5), (1, 3), (2, 1), (3, 1), (4, 2)] {
        let result = PhasedGraph::try_build_with(2, 4, 2, |slab, _| {
            slab[word] = value;
            if word == 4 {
                slab[0] = 1;
            }
            Ok(())
        });
        assert_eq!(result.unwrap_err().code(), ErrorCode::InvalidData);
    }
    let options = BuildOptions {
        degree: 4,
        build_l: 8,
        max_extra: 2,
        ..BuildOptions::default()
    };
    let mut slab = [99; 32];
    assert!(Builder::u8_into(&[0; 8], 2, 4, 4, options, &mut slab, 9).is_err());
    assert_eq!(slab, [99; 32]); // Rejected before any worker can write.
    assert!(Builder::u8_into(&[0; 8], 2, 4, 4, options, &mut slab[..19], 10).is_err());
}

/// Run in separate processes under an OS RSS monitor; not part of routine CI.
#[test]
#[ignore]
fn slab_memory_probe() {
    let n: usize = std::env::var("ORION_PROBE_N")
        .unwrap_or("100000".into())
        .parse()
        .unwrap();
    let mode = std::env::var("ORION_PROBE_MODE").unwrap_or("slab".into());
    let options = BuildOptions {
        degree: 64,
        build_l: 96,
        max_extra: 16,
        threads: 4,
        ..BuildOptions::default()
    };
    let mut seed = 17u64;
    let data: Vec<u8> = (0..n * 16)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        })
        .collect();
    let graph = if mode == "owning" {
        let mut builder = Builder::u8(&data, n, 16, 16, options).unwrap();
        collect_graph(&mut builder, n, 64, 16).unwrap()
    } else {
        assert_eq!(mode, "slab");
        PhasedGraph::try_build_with(n, 64, 16, |slab, ss| {
            Builder::u8_into(&data, n, 16, 16, options, slab, ss).map_err(convert_error)
        })
        .unwrap()
    };
    // Stable checksum without retaining another copy of the final graph.
    let hash = graph
        .buffer_bytes()
        .iter()
        .fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
    println!(
        "mode={mode} nodes={n} slab_bytes={} checksum={hash:016x} independent_graph_bytes={}",
        graph.buffer_bytes().len(),
        if mode == "owning" { n * 328 } else { 0 }
    );
}

/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Deterministic-prefix replay and local geometric audits; no timing claims.
use orion_cli::cascade;
use orion_cli::cli;
use orion_cli::with_supported_dimension;

use clap::{Parser, ValueEnum};
use diskann::model::InmemDataset;
use orion::algorithm::search::{
    diagnostics::{Checkpoint, MatchedStateTrace, NeighborMode, NoopObserver},
    stage::{rerank::NoRerank, AdmissionSession, AdmissionStage, PrefilterStage, RerankStage},
    utils::AlignedQuery,
};
use orion::model::Neighbor;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs::OpenOptions, io::Write, path::PathBuf};
use vector::FullPrecisionDistance;

#[derive(Clone, Copy, ValueEnum)]
enum Track {
    Exact,
    Cascade,
    Both,
}

#[derive(Parser)]
#[command(
    name = "matched_state",
    about = "Matched-state L2 diagnostics (EE disabled; no QPS timing)"
)]
struct Args {
    #[command(flatten)]
    run: cli::config::Args,
    /// Evenly spaced queries; 0 means all. This binary does not measure QPS.
    #[arg(long, default_value_t = 200)]
    diagnostic_queries: usize,
    #[arg(long, value_enum, default_value = "both")]
    track: Track,
    /// Diagnostic L values, independently of the performance sweep.
    #[arg(long, value_delimiter = ',', default_value = "64,128,256")]
    diagnostic_ls: Vec<usize>,
    /// Euclidean alpha for existential witness and sampled local-coverage checks.
    /// Not inferred from builders that may use squared-distance alpha conventions.
    #[arg(long, default_value_t = 1.1)]
    geometry_alpha: f64,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn l2<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let a = AlignedQuery(*a);
    let b = AlignedQuery(*b);
    <[f32; N] as FullPrecisionDistance<f32, N>>::distance_compare(&a.0, &b.0, vector::Metric::L2)
}

struct ExactAdmission<'a, const N: usize>(&'a InmemDataset<f32, N>);
struct ExactSession<'a, const N: usize> {
    data: &'a InmemDataset<f32, N>,
    query: [f32; N],
}
impl<const N: usize> AdmissionStage<N> for ExactAdmission<'_, N> {
    fn open<'a>(&'a self, q: &[f32; N]) -> Box<dyn AdmissionSession + 'a> {
        Box::new(ExactSession {
            data: self.0,
            query: *q,
        })
    }
}
impl<const N: usize> AdmissionSession for ExactSession<'_, N> {
    fn entry_distance(&self, id: u32) -> f32 {
        l2(&self.query, unsafe { self.data.get_vertex_unchecked(id) })
    }
    unsafe fn admit_stream(&self, ids: &[u32], out: *mut Neighbor, cutoff: f32, _: usize) -> usize {
        let mut n = 0;
        for &id in ids {
            let d = self.entry_distance(id);
            if d < cutoff {
                unsafe {
                    out.add(n).write(Neighbor::new(id, d));
                }
                n += 1;
            }
        }
        n
    }
}

fn recall(ids: &[u32], truth: &[u32]) -> f64 {
    ids.iter().filter(|id| truth.contains(id)).count() as f64 / truth.len() as f64
}

fn summarize(rows: &[Value]) -> Value {
    let groups: Vec<_> = [
        "all",
        "no_switch",
        "coverage_holds",
        "coverage_violated",
        "coverage_not_applicable",
    ]
        .into_iter()
        .map(|group| {
            let selected: Vec<_> = rows
                .iter()
                .filter(|r| match group {
                    "all" => true,
                    "no_switch" => r["checkpoint"].is_null(),
                    "coverage_holds" => r["sampled_local_coverage_holds"] == true,
                    "coverage_violated" => r["sampled_local_coverage_holds"] == false,
                    _ => !r["checkpoint"].is_null() && r["sampled_local_coverage_holds"].is_null(),
                })
                .collect();
            let branches: Vec<_> = (0..3)
                .map(|i| {
                    let mut means = serde_json::Map::new();
                    for key in [
                        "recall",
                        "before_ndc",
                        "after_ndc",
                        "total_stage_ndc",
                        "after_expansions",
                    ] {
                        means.insert(
                            key.into(),
                            if selected.is_empty() {
                                Value::Null
                            } else {
                                json!(
                                selected
                                    .iter()
                                    .map(|r| r["branches"][i][key].as_f64().unwrap())
                                    .sum::<f64>()
                                    / selected.len() as f64
                            )
                            },
                        );
                    }
                    // Corresponding neighbor mode for `i`
                    json!({"mode": NeighborMode::ALL[i].name(), "means": means})
                })
                .collect();
            json!({"group": group, "query_count": selected.len(), "branches": branches})
        })
        .collect();
    json!(groups)
}

fn audit<const N: usize>(
    idx: &orion::Orion<N>,
    q: &[f32; N],
    truth: &[u32],
    cp: &Checkpoint,
    session: &dyn AdmissionSession,
    alpha: f64,
) -> Value {
    let graph = &idx.graph;
    let u = cp.node;
    let local = graph.local_neighbors(u as usize);
    let remote = graph.remote_neighbors(u as usize);
    let extra = graph.extra_candidates(u as usize);
    let mut evaluations = 0usize;

    // If the second parameter is `None`, use the query as the second vector.
    let mut distance = |a: u32, b: Option<u32>| {
        evaluations += 1;
        let v = unsafe { idx.dataset.get_vertex_unchecked(a) };
        let w = b
            .map(|id| unsafe { idx.dataset.get_vertex_unchecked(id) })
            .unwrap_or(q);
        (l2(v, w) as f64).max(0.0).sqrt()
    };

    let beam_ids: BTreeSet<_> = cp.beam.iter().map(|n| n.0).collect();

    let radius = cp
        .beam
        .iter()
        .map(|n| distance(n.0, None))
        .fold(0.0, f64::max);
    let ranked_tail_radius = distance(cp.beam.last().unwrap().0, None);

    let cutoff = if cp.beam.len() >= cp.capacity {
        f32::from_bits(cp.beam.last().unwrap().1)
    } else {
        f32::MAX
    };

    let mut zones = serde_json::Map::new();

    let mut scoring_evaluations = 0;
    for (name, ids) in [("local", local), ("remote", remote), ("extra", extra)] {
        let rows: Vec<_> = ids
            .iter()
            .map(|&v| {
                let seen = cp.seen.binary_search(&v).is_ok();
                let score = if seen {
                    None
                } else {
                    scoring_evaluations += 1;
                    Some(session.entry_distance(v))
                };
                let uv = distance(u, Some(v));
                let witnesses: Vec<_> = if name == "extra" || name == "remote" {
                    local
                        .iter()
                        .copied()
                        .filter(|&w| w != v)
                        .filter(|&w| alpha * distance(w, Some(v)) < uv)
                        .collect()
                } else {
                    Vec::new()
                };
                let available: Vec<_> = witnesses
                    .iter()
                    .copied()
                    .filter(|w| {
                        cp.beam.iter().any(|n| n.0 == *w && !n.2)
                            || cp.pending.iter().any(|n| n.0 == *w)
                    })
                    .collect();
                json!({"id": v, "already_seen": seen, "query_distance": distance(v, None),
                "distance_to_expanded": uv, "admission_score": score,
                "passes_cutoff_ignoring_prefilter": score.map(|d| d < cutoff),
                "ground_truth_topk": truth.contains(&v), "local_existential_witnesses": witnesses,
                "unexpanded_beam_or_pending_witnesses": available})
            })
            .collect();
        zones.insert(name.into(), json!(rows));
    }
    let coverage: Vec<_> = truth
        .iter()
        .filter(|id| !beam_ids.contains(id))
        .map(|&target| {
            let ut = distance(u, Some(target));
            let covered = |neighbors: &[u32], distance: &mut dyn FnMut(u32, Option<u32>) -> f64| {
                neighbors
                    .iter()
                    .any(|&w| w == target || alpha * distance(w, Some(target)) < ut)
            };
            json!({"target": target, "already_seen": cp.seen.binary_search(&target).is_ok(),
            "inside_beam_enclosing_ball": distance(target, None) <= radius,
            "local_alpha_progress": covered(local, &mut distance),
            "full_alpha_progress": covered(graph.neighbors(u as usize), &mut distance)})
        })
        .collect();
    let unseen = |ids: Vec<u32>| {
        ids.into_iter()
            .filter(|id| cp.seen.binary_search(id).is_err())
            .collect::<BTreeSet<_>>()
            .len()
    };
    json!({"node": u, "step": cp.step, "beam_full": cp.beam.len() == cp.capacity,
        "beam": cp.beam.iter().map(|n| json!({"id": n.0, "score_bits": n.1, "visited": n.2})).collect::<Vec<_>>(),
        "pending": cp.pending.iter().map(|n| json!({"id": n.0, "score_bits": n.1, "visited": n.2})).collect::<Vec<_>>(),
        "scc_state": cp.convergence, "prefilter_state_bits": cp.prefilter,
        "hops_since_flush": cp.hops_since_flush,
        "beam_enclosing_radius_l2": radius, "ranked_tail_radius_l2": ranked_tail_radius,
        "admission_cutoff": cutoff, "pending_count": cp.pending.len(), "seen_count": cp.seen.len(),
        "zones": zones, "sampled_topk_coverage": coverage,
        "unseen_full_budget": unseen(graph.neighbors(u as usize).to_vec()),
        "unseen_local_extra_budget": unseen(local.iter().chain(extra).copied().collect()),
        "audit_exact_distance_evaluations": evaluations,
        "audit_admission_score_evaluations": scoring_evaluations})
}

fn trace_json<const N: usize>(
    idx: &orion::Orion<N>,
    q: &[f32; N],
    truth: &[u32],
    trace: &MatchedStateTrace,
    ids: &[u32],
    mode: NeighborMode,
    rerank: bool,
    radius: Option<f64>,
) -> Value {
    let w = &trace.work;
    let rerank_ndc = if rerank { w.rerank_candidates } else { 0 };
    let outside: Vec<_> = trace
        .tail_nodes
        .iter()
        .filter_map(|&(id, _)| {
            radius.and_then(|r| {
                let d = (l2(q, unsafe { idx.dataset.get_vertex_unchecked(id) }) as f64)
                    .max(0.0)
                    .sqrt();
                (d > r).then_some(id)
            })
        })
        .collect();
    json!({"mode": mode.name(), "returned_ids": ids, "recall": recall(ids, truth),
        "before_ndc": w.before.ndc(), "after_ndc": w.after.ndc(), "rerank_ndc": rerank_ndc,
        "total_stage_ndc": w.before.ndc() + w.after.ndc() + rerank_ndc,
        "before_expansions": w.before.expansions, "after_expansions": w.after.expansions,
        "tail_nodes": trace.tail_nodes, "tail_nodes_outside_checkpoint_ball": outside,
        "ball_audit_distance_evaluations": if radius.is_some() { trace.tail_nodes.len() } else { 0 },
        "retained_at_any_tail_flush": trace.retained_after_switch,
        "admitted_in_tail": trace.admitted_after_switch,
        "first_discovery": trace.discovery.first_discovery,
        "first_admission": trace.discovery.first_admission})
}

fn run<const N: usize>(
    args: &Args,
    config: &cli::config::ResolvedRunConfig,
    mut data: cli::data::LoadedDataset,
) -> Result<(), String> {
    if config.prepare_only {
        cli::index::load_index::<N>(config, &mut data)?;
        return Ok(());
    }
    let ground_truth = data.ground_truth.take().ok_or("this diagnostic requires --groundtruth; ordinary orion search does not")?;
    let settings = &config.sweep;
    let queries: Vec<[f32; N]> = data
        .queries
        .iter()
        .map(|q| q.as_slice().try_into().expect("dimension validated"))
        .collect();
    if queries.is_empty() || settings.k == 0 || settings.calibration_samples == 0 {
        return Err("Queries, k and calibration samples must be nonzero".into());
    }
    let out = args.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "visualizations/matched_state_{}_k{}.json",
            config.dataset.as_str(),
            settings.k
        ))
    });
    if out.exists() {
        return Err(format!("Refusing to overwrite {}", out.display()));
    }
    let idx = cli::index::load_index::<N>(config, &mut data)?;


    let calib = idx
        .calibrate(
            &queries[..settings.calibration_samples.min(queries.len())],
            settings.calibration_l,
            config.window_size,
            orion::CalibrationConfig {
                k: settings.k,
                metric: orion::CalibrationMetric::L2,
            },
        )
        .map_err(|e| e.to_string())?;
    let count = if args.diagnostic_queries == 0 {
        queries.len()
    } else {
        args.diagnostic_queries.min(queries.len())
    };
    let sample: Vec<_> = (0..count).map(|i| i * queries.len() / count).collect();
    let tracks: &[bool] = match args.track {
        Track::Exact => &[true],
        Track::Cascade => &[false],
        Track::Both => &[true, false],
    };
    let mut results = Vec::new();
    for &exact in tracks {
        let track = if exact { "exact_l2" } else { "cascade" };
        let pf: Option<Box<dyn PrefilterStage<N> + '_>> = if exact {
            None
        } else {
            cascade::build_prefilter(&idx, config.search_plan().cascade().prefilter, config.search_plan().cascade().admission)
        };
        let ad: Box<dyn AdmissionStage<N> + '_> = if exact {
            Box::new(ExactAdmission(&idx.dataset))
        } else {
            cascade::build_admission(&idx, config.search_plan().cascade().admission).map_err(|e| e.to_string())?
        };
        let rr: Box<dyn RerankStage<N> + '_> = if exact {
            Box::new(NoRerank)
        } else {
            cascade::build_rerank(&idx, config.search_plan().cascade().rerank)
        };
        for &l in &args.diagnostic_ls {
            let mut rows = Vec::new();
            for &qi in &sample {
                let q = &queries[qi];
                let truth = &ground_truth[qi][..settings.k];
                let mut traces = Vec::new();
                let mut returned = Vec::new();
                for mode in NeighborMode::ALL {
                    let mut trace = MatchedStateTrace::new(truth);
                    let ids = idx
                        .search_unified_observed(
                            q,
                            settings.k,
                            l,
                            config.window_size,
                            calib.threshold,
                            usize::MAX,
                            pf.as_deref(),
                            ad.as_ref(),
                            rr.as_ref(),
                            mode,
                            &mut trace,
                        )
                        .map_err(|e| e.to_string())?;
                    let plain = idx
                        .search_unified_observed(
                            q,
                            settings.k,
                            l,
                            config.window_size,
                            calib.threshold,
                            usize::MAX,
                            pf.as_deref(),
                            ad.as_ref(),
                            rr.as_ref(),
                            mode,
                            &mut NoopObserver,
                        )
                        .map_err(|e| e.to_string())?;
                    if ids != plain {
                        return Err(format!(
                            "Observer changed results: {track}, L={l}, query={qi}"
                        ));
                    }
                    traces.push(trace);
                    returned.push(ids);
                }
                if traces[1..].iter().any(|t| {
                    t.checkpoint != traces[0].checkpoint
                        || t.work.before.ndc() != traces[0].work.before.ndc()
                }) {
                    return Err(format!("Checkpoint mismatch: {track}, L={l}, query={qi}"));
                }
                let checkpoint = traces[0]
                    .checkpoint
                    .as_ref()
                    .map(|cp| audit(&idx, q, truth, cp, ad.open(q).as_ref(), args.geometry_alpha));
                let radius = checkpoint
                    .as_ref()
                    .and_then(|v| v["beam_enclosing_radius_l2"].as_f64());
                let branches: Vec<_> = NeighborMode::ALL
                    .iter()
                    .enumerate()
                    .map(|(i, &mode)| {
                        trace_json(
                            &idx,
                            q,
                            truth,
                            &traces[i],
                            &returned[i],
                            mode,
                            !exact && config.search_plan().cascade().rerank != cascade::RerankChoice::None,
                            radius,
                        )
                    })
                    .collect();
                let remote_followup: Vec<_> = checkpoint.as_ref().map(|cp| cp["zones"]["remote"].as_array().unwrap()
                    .iter().filter(|v| v["already_seen"] == false).map(|v| {
                    let id = v["id"].as_u64().unwrap() as u32;
                    json!({"id": id, "admitted_later_full": traces[0].admitted_after_switch.contains(&id),
                            "retained_later_full": traces[0].retained_after_switch.contains(&id),
                            "expanded_later_full": traces[0].tail_nodes.iter().any(|n| n.0 == id),
                            "returned_full": returned[0].contains(&id)})
                }).collect()).unwrap_or_default();
                let lost: Vec<_> = truth
                    .iter()
                    .copied()
                    .filter(|id| returned[0].contains(id) && !returned[2].contains(id))
                    .collect();
                let gained: Vec<_> = truth
                    .iter()
                    .copied()
                    .filter(|id| !returned[0].contains(id) && returned[2].contains(id))
                    .collect();
                let local_coverage = checkpoint.as_ref().and_then(|cp| {
                    let targets: Vec<_> = cp["sampled_topk_coverage"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|t| t["inside_beam_enclosing_ball"] == true)
                        .collect();
                    (!targets.is_empty())
                        .then(|| targets.iter().all(|t| t["local_alpha_progress"] == true))
                });
                rows.push(json!({"query": qi, "target_ids": truth, "checkpoint_verified": checkpoint.is_some(),
                    "sampled_local_coverage_holds": local_coverage,
                    "checkpoint": checkpoint, "branches": branches, "remote_followup_full": remote_followup,
                    "full_only_topk": lost, "local_extra_only_topk": gained}));
            }
            let switched = rows
                .iter()
                .filter(|r| r["checkpoint_verified"] == true)
                .count();
            log::info!("{track} L={l}: {switched}/{count} matched checkpoints verified");
            results.push(json!({"track": track, "l": l, "query_count": count,
                "summary": summarize(&rows),
                "switched_count": switched, "no_switch_count": count - switched, "queries": rows}));
        }
    }
    let result = json!({"schema_version": 1, "experiment": "matched_state", "config": config,
        "diagnostic_ls": args.diagnostic_ls, "diagnostic_query_ids": sample,
        "threshold": calib.threshold, "geometry_alpha": args.geometry_alpha, "early_exit": false,
        "semantics": {
            "replay": "Three deterministic full-neighbor prefixes checked for exact logical-state equality at first convergence; reversible post-switch policies",
            "state": "Chosen vertex already marked expanded; checkpoint precedes neighbor collection; includes beam bits/visited, next cursor ID, pending buffer, seen IDs, DCC ring, EE state, prefilter cache and flush state",
            "exact": "Full-precision f32 L2 admission, no prefilter/rerank; same calibrated threshold as cascade, not identical checkpoints across tracks",
            "ball": "Exact Euclidean enclosing radius of checkpoint beam; in cascade this need not equal its admission-ranked tail distance",
            "coverage": "One-hop alpha progress toward top-k targets missing from checkpoint beam only; NOT global shortcut coverage",
            "witness": "Existential alpha*d(w,v)<d(u,v) in Euclidean distance; w != v; not historical pruning provenance; remote is retained, not a discarded candidate",
            "eligibility": "Counterfactual admission score vs checkpoint cutoff ignores prefilter; retained_later is actual full-branch PQ membership, not immediate insertion or causal contribution",
            "causality": "Full-only final targets show branch disagreement; do not attribute them to a specific remote edge without intervention",
            "work": "Search stage NDC excludes all audits; no QPS timing; exact comparisons are sampled empirical checks, not proofs",
            "budget": "Immediate unseen-neighbor counts and realized continuation NDC; no universal remaining-candidate bound asserted"
        }, "results": results});
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out)
        .map_err(|e| e.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    log::info!("Saved {}", out.display());
    Ok(())
}

fn execute(args: Args) -> Result<(), String> {
    let config = args.run.resolve_for_sweep()?;
    if config.metric != cascade::SearchMetric::L2 {
        return Err("Geometric audit currently requires L2".into());
    }
    if !args.geometry_alpha.is_finite() || args.geometry_alpha <= 1.0 {
        return Err("geometry-alpha must be finite and > 1".into());
    }
    if args.diagnostic_ls.is_empty()
        || args.diagnostic_ls.iter().any(|&l| l < config.sweep.k)
        || args.diagnostic_ls.iter().collect::<BTreeSet<_>>().len() != args.diagnostic_ls.len()
    {
        return Err("diagnostic-ls must be distinct values >= k".into());
    }
    if args.run.print_config {
        println!("{}", serde_json::to_string_pretty(&config).unwrap());
        return Ok(());
    }
    if args.run.preflight {
        println!("{}", serde_json::to_string_pretty(&cli::resources::preflight_report(&config)?).map_err(|e| e.to_string())?);
        return Ok(());
    }
    let data = cli::data::LoadedDataset::load(&config)?;
    // Sequential queries avoid interpreting instrumented runtime as throughput.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .map_err(|e| e.to_string())?;
    pool.install(|| with_supported_dimension!(config.dimension, |D| run::<D>(&args, &config, data)))
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    if let Err(e) = execute(Args::parse()) {
        log::error!("{e}");
        std::process::exit(2);
    }
}

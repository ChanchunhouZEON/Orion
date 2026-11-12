/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Closed-loop clients share one Index and its existing Rayon pool. Client
//! threads warm up before a common start barrier. Only Index::search is timed;
//! recording, result destruction, scheduling the next request and joins are not.
use super::{
    config::Protocol,
    statistics::{Bucket, Distribution, Percentiles},
};
use anyhow::{anyhow, ensure, Result};
use orion_ann::{Index, QueryBatch, SearchOptions};
use serde::Serialize;
use std::{
    hint::black_box,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Barrier, OnceLock,
    },
    time::{Duration, Instant},
};

#[derive(Debug, Serialize)]
pub struct Trial {
    pub requests: u64,
    pub wall_seconds: f64,
    /// Wall throughput includes harness recording/loop overhead. It is not the
    /// existing batch-sweep QPS measurement and not the inverse of mean latency.
    pub instrumented_qps: f64,
    pub minimum_reached: bool,
    pub latency: Option<Percentiles>,
    pub histogram: Vec<Bucket>,
    #[serde(skip)]
    pub distribution: Distribution,
}

struct Worker {
    requests: usize,
    finished: Instant,
    reached: bool,
    distribution: Distribution,
}

/// One query per request, with borrowed rows prepared outside the measured call.
/// Failed requests abort the trial; they are never omitted from a successful report.
pub fn measure(
    index: &Index,
    rows: &[QueryBatch<'_>],
    options: &SearchOptions,
    order: &[usize],
    concurrency: usize,
    protocol: &Protocol,
    timed: bool,
) -> Result<Trial> {
    ensure!(
        !rows.is_empty() && rows.iter().all(|q| q.len() == 1),
        "request latency requires nonempty single-row queries"
    );
    ensure!(order.len() == rows.len(), "order must cover every query");
    let mut sorted = order.to_vec();
    sorted.sort_unstable();
    ensure!(
        sorted.iter().copied().eq(0..rows.len()),
        "order must be a query permutation"
    );
    let search = |row| Ok(index.search(rows[row], options)?);
    if timed {
        closed_loop::<true, _, _>(order, concurrency, protocol, search)
    } else {
        closed_loop::<false, _, _>(order, concurrency, protocol, search)
    }
}

fn closed_loop<const TIMED: bool, T, F: Fn(usize) -> Result<T> + Sync>(
    order: &[usize],
    concurrency: usize,
    protocol: &Protocol,
    search: F,
) -> Result<Trial> {
    protocol.validate()?;
    ensure!(
        !order.is_empty() && (1..=64).contains(&concurrency),
        "invalid query count or concurrency"
    );
    let ready = Barrier::new(concurrency + 1);
    let go = Barrier::new(concurrency + 1);
    let start = OnceLock::new();
    let cancelled = AtomicBool::new(false);
    // Allocate requests from one shuffled cycle. Faster clients must not
    // over-sample their own (potentially easier) query partition. This ticket
    // is acquired outside the request timer; histograms remain thread-local.
    let next_query = AtomicUsize::new(0);
    let quota = protocol.min_requests.div_ceil(concurrency);
    let warmup_quota = protocol.warmup_requests.div_ceil(concurrency);
    let minimum = protocol.minimum();
    let maximum = protocol.maximum();
    let warmup_time = Duration::from_secs_f64(protocol.warmup_seconds);

    let workers = std::thread::scope(|scope| -> Result<Vec<Worker>> {
        let mut handles = Vec::with_capacity(concurrency);
        for worker in 0..concurrency {
            let (ready, go, start, cancelled, search) = (&ready, &go, &start, &cancelled, &search);
            let next_query = &next_query;
            handles.push(scope.spawn(move || {
                let run = || -> Result<Worker> {
                    let distribution = Distribution::new();
                    let warmup =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                            let began = Instant::now();
                            let mut n = 0;
                            let mut offset = worker % order.len();
                            while n < warmup_quota || began.elapsed() < warmup_time {
                                drop(black_box(search(order[offset])?));
                                n += 1;
                                offset = (offset + concurrency) % order.len();
                            }
                            Ok(())
                        }))
                        .map_err(|_| anyhow!("warmup client panicked"))
                        .and_then(|r| r);
                    // Even a warmup error must rendezvous so peers cannot hang.
                    ready.wait();
                    go.wait();
                    warmup?;
                    let mut distribution = distribution?;
                    let began: Instant = *start.get().expect("coordinator published start");
                    let mut n = 0;
                    let mut finished = began;
                    loop {
                        let elapsed = began.elapsed();
                        if cancelled.load(Ordering::Relaxed)
                            || elapsed >= maximum
                            || (n >= quota && elapsed >= minimum)
                        {
                            break;
                        }
                        let row = order[next_query.fetch_add(1, Ordering::Relaxed) % order.len()];
                        let clock = TIMED.then(Instant::now);
                        let result = search(row)?;
                        // Read the endpoint before black_box, drop and histogram work.
                        let elapsed_ns = clock.map(|t| t.elapsed().as_nanos());
                        drop(black_box(result));
                        if let Some(ns) = elapsed_ns {
                            distribution.record(ns.try_into()?)?;
                        }
                        n += 1;
                        finished = Instant::now();
                    }
                    Ok(Worker {
                        requests: n,
                        finished,
                        reached: n >= quota && began.elapsed() >= minimum,
                        distribution,
                    })
                };
                let result = run();
                if result.is_err() {
                    cancelled.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
        ready.wait();
        start.set(Instant::now()).expect("start set once");
        go.wait();
        let mut workers = Vec::with_capacity(concurrency);
        let mut failure = None;
        // Always join all clients, even if one failed. No background work leaks
        // into the next point and no partial distribution is reported as success.
        for handle in handles {
            match handle
                .join()
                .map_err(|_| anyhow!("latency client panicked"))
                .and_then(|r| r)
            {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(workers),
        }
    })?;
    let mut distribution = Distribution::new()?;
    let mut requests = 0;
    let mut finished = *start.get().unwrap();
    let mut minimum_reached = true;
    for worker in workers {
        requests += worker.requests as u64;
        finished = finished.max(worker.finished);
        minimum_reached &= worker.reached;
        distribution.merge(&worker.distribution)?;
    }
    let wall_seconds = finished.duration_since(*start.get().unwrap()).as_secs_f64();
    ensure!(
        requests > 0 && wall_seconds > 0.0,
        "no requests completed before deadline"
    );
    Ok(Trial {
        requests,
        wall_seconds,
        instrumented_qps: requests as f64 / wall_seconds,
        minimum_reached,
        latency: distribution.summary(),
        histogram: distribution.buckets(),
        distribution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol() -> Protocol {
        Protocol {
            min_requests: 800,
            min_seconds: 0.0,
            max_seconds: 10.0,
            warmup_requests: 80,
            warmup_seconds: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn warmup_is_excluded_and_shared_cycles_are_balanced() {
        let hits: Vec<_> = (0..10).map(|_| AtomicUsize::new(0)).collect();
        let result =
            closed_loop::<true, _, _>(&[9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 8, &protocol(), |row| {
                hits[row].fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
            .unwrap();
        assert_eq!(result.requests, 800);
        assert_eq!(result.latency.unwrap().samples, 800);
        assert_eq!(result.histogram.iter().map(|b| b.count).sum::<u64>(), 800);
        assert_eq!(
            hits.iter()
                .map(|h| h.load(Ordering::Relaxed))
                .sum::<usize>(),
            880
        );
        // Ignore warmup's different partitioning when checking timed cycles.
        let p = Protocol {
            warmup_requests: 0,
            ..protocol()
        };
        for h in &hits {
            h.store(0, Ordering::Relaxed);
        }
        closed_loop::<false, _, _>(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 8, &p, |row| {
            hits[row].fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .unwrap();
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 80));
    }

    #[test]
    fn warmup_panics_do_not_strand_clients_at_start_barrier() {
        let error =
            closed_loop::<true, (), _>(&[0], 8, &protocol(), |_| panic!("injected warmup panic"))
                .unwrap_err();
        assert!(error.to_string().contains("warmup client panicked"));
    }

    #[test]
    fn deadline_keeps_slow_samples_and_marks_incomplete() {
        let p = Protocol {
            min_requests: 1_000_000,
            max_seconds: 0.05,
            warmup_requests: 0,
            ..protocol()
        };
        let trial = closed_loop::<true, _, _>(&[0], 1, &p, |_| {
            std::thread::sleep(Duration::from_millis(10));
            Ok(())
        })
        .unwrap();
        assert!(!trial.minimum_reached);
        assert!(trial.requests < p.min_requests as u64);
        assert_eq!(trial.latency.unwrap().samples, trial.requests);
    }
}

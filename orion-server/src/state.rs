/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Bounded job admission, index publication and shared session lifetimes.
use crate::{
    error::exhausted, paths::StorageRoot, protocol::*, request_context::RequestContext, Principal,
    ServerConfig,
};
use orion_ann::{Error, ErrorReport, Index, OpenOptions, Result};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Job {
    pub id: String,
    pub owner: String,
    pub request_id: String,
    pub status: &'static str,
    pub stage: &'static str,
    /// Actual stage transitions, not an invented completion percentage.
    pub stages: Vec<&'static str>,
    pub index_id: Option<String>,
    pub snapshot_id: Option<String>,
    pub error: Option<ErrorReport>,
}
#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pub owner: String,
    pub index: Arc<RwLock<Index>>,
    pub info: Value,
}
#[derive(Debug, Default)]
pub(crate) struct Registry {
    pub jobs: BTreeMap<String, Job>,
    pub indexes: BTreeMap<String, Entry>,
    pub reserved: usize,
    serial: u64,
}
#[derive(Debug)]
pub(crate) struct Service {
    pub config: ServerConfig,
    pub paths: StorageRoot,
    pub registry: Mutex<Registry>,
    pub searches: Arc<Semaphore>,
    workers: Arc<Semaphore>,
    pending: Arc<Semaphore>,
}

pub(crate) enum Task {
    Build(BuildRequest),
    Load(LoadRequest),
}

impl Registry {
    pub fn entry(&self, principal: &Principal, id: &str) -> crate::error::ApiResult<Entry> {
        self.indexes
            .get(id)
            .filter(|e| e.owner == principal.subject())
            .cloned()
            .ok_or_else(|| crate::error::missing("index", id))
    }
    pub fn job(&self, principal: &Principal, id: &str) -> crate::error::ApiResult<&Job> {
        self.jobs
            .get(id)
            .filter(|j| j.owner == principal.subject())
            .ok_or_else(|| crate::error::missing("job", id))
    }
}

impl Service {
    pub fn new(config: ServerConfig) -> Result<Arc<Self>> {
        if [
            config.build_workers,
            config.max_pending_jobs,
            config.max_indexes,
            config.max_job_history,
            config.max_searches,
            config.max_threads,
            config.max_body_bytes,
            config.max_search_list_size,
            config.max_result_ids,
        ]
        .contains(&0)
            || [
                config.build_workers,
                config.max_pending_jobs,
                config.max_searches,
            ]
            .iter()
            .any(|&n| n > Semaphore::MAX_PERMITS)
        {
            return Err(Error::invalid_argument(
                "server limits must be positive and within semaphore capacity",
            ));
        }
        Ok(Arc::new(Self {
            paths: StorageRoot::new(&config.data_root)?,
            registry: Mutex::default(),
            workers: Arc::new(Semaphore::new(config.build_workers)),
            pending: Arc::new(Semaphore::new(config.max_pending_jobs)),
            searches: Arc::new(Semaphore::new(config.max_searches)),
            config,
        }))
    }
    pub fn registry(&self) -> Result<std::sync::MutexGuard<'_, Registry>> {
        self.registry
            .lock()
            .map_err(|_| Error::internal("service registry poisoned"))
    }
    pub fn search_budget(&self, rows: usize, k: usize, beam: usize) -> Result<()> {
        if beam > self.config.max_search_list_size
            || rows
                .checked_mul(k)
                .is_none_or(|n| n > self.config.max_result_ids)
        {
            return Err(exhausted(
                "search request exceeds candidate or result limit",
            ));
        }
        Ok(())
    }
    fn stage(&self, id: &str, stage: &'static str) -> Result<()> {
        let mut state = self.registry()?;
        let job = state
            .jobs
            .get_mut(id)
            .ok_or_else(|| Error::internal("active job missing"))?;
        job.status = "running";
        job.stage = stage;
        job.stages.push(stage);
        Ok(())
    }
    pub fn submit(
        self: &Arc<Self>,
        principal: Principal,
        context: RequestContext,
        task: Task,
    ) -> Result<Job> {
        let pending = self
            .pending
            .clone()
            .try_acquire_owned()
            .map_err(|_| exhausted("build/load queue is full"))?;
        let job = {
            let mut state = self.registry()?;
            if state.jobs.len() >= self.config.max_job_history {
                return Err(exhausted(
                    "job history is full; delete completed job records",
                ));
            }
            if state.indexes.len() + state.reserved >= self.config.max_indexes {
                return Err(exhausted("index capacity reached; unload an index first"));
            }
            state.serial += 1;
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let id = format!("job-{}-{stamp}-{}", std::process::id(), state.serial);
            let job = Job {
                id: id.clone(),
                owner: principal.subject().to_owned(),
                request_id: context.id.clone(),
                status: "queued",
                stage: "queued",
                stages: vec!["queued"],
                index_id: None,
                snapshot_id: None,
                error: None,
            };
            state.jobs.insert(id, job.clone());
            state.reserved += 1;
            job
        };
        let service = self.clone();
        let id = job.id.clone();
        let owner = job.owner.clone();
        let request_id = job.request_id.clone();
        tokio::spawn(async move {
            let outcome = match service.workers.clone().acquire_owned().await {
                Ok(worker) => {
                    let engine = service.clone();
                    let task_id = id.clone();
                    tokio::task::spawn_blocking(move || {
                        // Permits live inside blocking work: disconnecting a client
                        // cannot admit more CPU work while its previous task runs.
                        let (_worker, _pending) = (worker, pending);
                        engine.execute(&task_id, &principal, task)
                    })
                    .await
                    .unwrap_or_else(|_| Err(Error::internal("index worker panicked")))
                }
                Err(_) => Err(Error::invalid_state("build worker queue closed")),
            };
            if let Ok(mut state) = service.registry() {
                state.reserved -= 1;
                let (status, error, snapshot_id) = match outcome {
                    Ok((index, snapshot_id)) => {
                        let info = index.info();
                        state.indexes.insert(
                            id.clone(),
                            Entry {
                                owner: owner.clone(),
                                index: Arc::new(RwLock::new(index)),
                                info,
                            },
                        );
                        ("complete", None, Some(snapshot_id))
                    }
                    Err(error) => ("failed", Some(error.report()), None),
                };
                let job = state
                    .jobs
                    .get_mut(&id)
                    .expect("active jobs cannot be removed");
                job.status = status;
                job.stage = status;
                job.stages.push(status);
                job.error = error;
                job.snapshot_id = snapshot_id;
                job.index_id = (status == "complete").then_some(id.clone());
                eprintln!(
                    "{}",
                    serde_json::json!({"event":"job_finished","request_id":request_id,"job_id":id,"owner":owner,"status":status})
                );
            }
        });
        Ok(job)
    }

    fn execute(&self, id: &str, principal: &Principal, task: Task) -> Result<(Index, String)> {
        let paths = self.paths.for_subject(principal)?;
        let work = paths.root.join("jobs").join(id);
        std::fs::create_dir(&work).map_err(|e| Error::io("http_job_directory", &work, e))?;
        let mmap = work.join("mmap");
        std::fs::create_dir(&mmap)?;
        match task {
            Task::Build(mut request) => {
                self.stage(id, "building")?;
                if request.options.threads > self.config.max_threads {
                    return Err(exhausted("requested threads exceed server limit"));
                }
                if request.options.cache_dir != orion_ann::IndexOptions::default().cache_dir
                    || request.options.storage_options.directory.is_some()
                {
                    return Err(Error::invalid_argument(
                        "service owns cache_dir and mmap directory",
                    ));
                }
                request.options.cache_dir = work.join("cache");
                request.options.storage_options.directory = Some(mmap);
                if let Some(path) = request.options.graph.staged_file.as_ref() {
                    request.options.graph.staged_file = Some(paths.input(path)?);
                }
                if request.calibrations.len() > 16 {
                    return Err(Error::invalid_argument("at most 16 calibrations per build"));
                }
                let mut index = Index::new(request.base.prepare(&paths)?, request.options)?;
                if !request.calibrations.is_empty() {
                    self.stage(id, "calibrating")?;
                }
                for calibration in request.calibrations {
                    index.calibrate(
                        calibration.queries,
                        calibration.k,
                        calibration.search_list_size,
                    )?;
                }
                self.stage(id, "saving")?;
                let snapshot = paths.root.join("snapshots").join(id);
                index.save(&snapshot)?;
                Ok((index, id.to_owned()))
            }
            Task::Load(request) => {
                self.stage(id, "loading")?;
                let threads = request
                    .options
                    .threads
                    .unwrap_or(self.config.max_threads.min(8));
                if threads > self.config.max_threads {
                    return Err(exhausted("requested threads exceed server limit"));
                }
                let mut storage = request.options.storage_options.unwrap_or_default();
                if storage.directory.is_some() {
                    return Err(Error::invalid_argument("service owns mmap directory"));
                }
                storage.directory = Some(mmap);
                let index = OpenOptions {
                    // HTTP clients cannot redirect snapshot objects outside the
                    // service-owned store. Use the snapshot's recorded location.
                    object_store: None,
                    threads: Some(threads),
                    memory_budget_gib: request.options.memory_budget_gib,
                    auto_pin: request.options.auto_pin,
                    storage_options: Some(storage),
                }
                .open(
                    paths.snapshot(&request.snapshot_id)?,
                    request.base.prepare(&paths)?,
                )?;
                Ok((index, request.snapshot_id))
            }
        }
    }
}

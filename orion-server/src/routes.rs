/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Thin transport handlers. No CPU work or index locks on async executor threads.
use crate::{
    error::{exhausted, missing, ApiError, ApiResult},
    protocol::*,
    request_context::RequestContext,
    state::{Job, Service, Task},
    Principal,
};
use axum::{
    extract::{rejection::JsonRejection, Path, State},
    http::StatusCode,
    Extension, Json,
};
use orion_ann::{Error, SearchOptions};
use serde_json::{json, Value};
use std::sync::Arc;

type Shared = State<Arc<Service>>;
type Body<T> = Result<Json<T>, JsonRejection>;

pub(crate) async fn health() -> Json<Value> {
    Json(json!({"status":"ok", "api_version":1}))
}

// Transport-only Easter egg: it deliberately has no native Orion ErrorCode.
pub(crate) async fn teapot() -> (StatusCode, Json<Value>) {
    (
        StatusCode::IM_A_TEAPOT,
        Json(json!({"message": "I'm a teapot"})),
    )
}

pub(crate) async fn build(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
    Extension(context): Extension<RequestContext>,
    body: Body<BuildRequest>,
) -> ApiResult<(StatusCode, Json<Job>)> {
    Ok((
        StatusCode::ACCEPTED,
        Json(service.submit(principal, context, Task::Build(body?.0))?),
    ))
}
pub(crate) async fn load(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
    Extension(context): Extension<RequestContext>,
    body: Body<LoadRequest>,
) -> ApiResult<(StatusCode, Json<Job>)> {
    Ok((
        StatusCode::ACCEPTED,
        Json(service.submit(principal, context, Task::Load(body?.0))?),
    ))
}
pub(crate) async fn job(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
    Path(id): Path<String>,
) -> ApiResult<Json<Job>> {
    Ok(Json(service.registry()?.job(&principal, &id)?.clone()))
}
pub(crate) async fn remove_job(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let mut state = service.registry()?;
    let job = state.job(&principal, &id)?;
    if !matches!(job.status, "complete" | "failed") {
        return Err(Error::invalid_state("an active job cannot be deleted or cancelled").into());
    }
    state.jobs.remove(&id);
    Ok(StatusCode::NO_CONTENT)
}
pub(crate) async fn catalog(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
) -> ApiResult<Json<Value>> {
    let (indexes, jobs) = {
        let state = service.registry()?;
        (
            state
                .indexes
                .iter()
                .filter(|(_, e)| e.owner == principal.subject())
                .map(|(id, e)| json!({"id":id,"info":e.info}))
                .collect::<Vec<_>>(),
            state
                .jobs
                .values()
                .filter(|j| j.owner == principal.subject())
                .cloned()
                .collect::<Vec<_>>(),
        )
    };
    let snapshots =
        tokio::task::spawn_blocking(move || service.paths.for_subject(&principal)?.snapshots())
            .await
            .map_err(|_| Error::internal("catalog worker failed"))??;
    Ok(Json(
        json!({"indexes":indexes,"jobs":jobs,"snapshots":snapshots}),
    ))
}
pub(crate) async fn index(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    Ok(Json(service.registry()?.entry(&principal, &id)?.info))
}
pub(crate) async fn unload(
    State(service): Shared,
    Extension(principal): Extension<Principal>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let entry = {
        let mut state = service.registry()?;
        state.entry(&principal, &id)?;
        state.indexes.remove(&id)
    };
    // Last-drop may unmap large payloads; existing requests retain their Arc.
    tokio::task::spawn_blocking(move || drop(entry))
        .await
        .map_err(|_| Error::internal("unload worker failed"))?;
    Ok(StatusCode::NO_CONTENT)
}
pub(crate) async fn search(
    State(service): Shared,
    Path(id): Path<String>,
    Extension(principal): Extension<Principal>,
    body: Body<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let request = body?.0;
    service.search_budget(
        request.queries.batch()?.len(),
        request.k,
        request.search_list_size,
    )?;
    let permit = service
        .searches
        .clone()
        .try_acquire_owned()
        .map_err(|_| exhausted("search capacity reached"))?;
    let entry = service.registry()?.entry(&principal, &id)?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let index = entry
            .index
            .read()
            .map_err(|_| Error::invalid_state("index poisoned; unload it"))?;
        index.search(
            request.queries.batch()?,
            &SearchOptions::new(request.k, request.search_list_size),
        )
    })
    .await
    .map_err(|_| Error::internal("search worker panicked"))??;
    Ok(Json(SearchResponse {
        ids: result.ids,
        counts: result.counts,
        k: result.k,
    }))
}
pub(crate) async fn trace(
    State(service): Shared,
    Path(id): Path<String>,
    Extension(principal): Extension<Principal>,
    body: Body<TraceRequest>,
) -> ApiResult<Json<orion_ann::diagnostics::SearchTrace>> {
    let request = body?.0;
    service.search_budget(1, request.k, request.search_list_size)?;
    let permit = service
        .searches
        .clone()
        .try_acquire_owned()
        .map_err(|_| exhausted("search capacity reached"))?;
    let entry = service.registry()?.entry(&principal, &id)?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        // Native trace_search currently requires &mut self. Ordinary search
        // still uses shared read access and can run concurrently.
        let mut index = entry
            .index
            .write()
            .map_err(|_| Error::invalid_state("index poisoned; unload it"))?;
        index.trace_search(
            request.query,
            request.k,
            request.search_list_size,
            true,
            Default::default(),
        )
    })
    .await
    .map_err(|_| Error::internal("trace worker panicked"))??;
    Ok(Json(result))
}
pub(crate) async fn not_found() -> ApiError {
    missing("route", "")
}
pub(crate) async fn method_not_allowed() -> ApiError {
    ApiError(
        StatusCode::METHOD_NOT_ALLOWED,
        Error::invalid_argument("HTTP method not supported").at("http_route"),
    )
}

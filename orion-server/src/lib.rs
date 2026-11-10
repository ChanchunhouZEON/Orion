/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Local HTTP prototype over the public native Index API; no search kernels here.
mod auth;
mod error;
mod request_context;
pub use auth::{Authenticator, Principal, TokenAuthenticator};
mod paths;
mod protocol;
mod routes;
mod state;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use std::{path::PathBuf, sync::Arc};

/// Resource limits are per service, not an RSS guarantee. All file inputs must
/// be relative to the authenticated subject’s storage namespace. The executable
/// binds to loopback; remote clients use a TLS-terminating reverse proxy.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub data_root: PathBuf,
    /// Required: no anonymous mode, even behind a proxy.
    pub authenticator: Option<Arc<dyn Authenticator>>,
    pub build_workers: usize,
    pub max_pending_jobs: usize,
    pub max_indexes: usize,
    pub max_job_history: usize,
    pub max_searches: usize,
    pub max_threads: usize,
    pub max_body_bytes: usize,
    pub max_search_list_size: usize,
    pub max_result_ids: usize,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            data_root: "orion-service-data".into(),
            authenticator: None,
            build_workers: 1,
            max_pending_jobs: 8,
            max_indexes: 8,
            max_job_history: 128,
            max_searches: 8,
            max_threads: 64,
            max_body_bytes: 16 * 1024 * 1024,
            max_search_list_size: 65_536,
            max_result_ids: 1_000_000,
        }
    }
}

/// Construct a service router. The host owns the Tokio runtime and listener.
/// Returns an Orion error for invalid limits or inaccessible service storage.
pub fn router(config: ServerConfig) -> orion_ann::Result<Router> {
    let auth = config
        .authenticator
        .clone()
        .ok_or_else(|| orion_ann::Error::invalid_argument("an authenticator is required"))?;
    let limit = config.max_body_bytes;
    let service = state::Service::new(config)?;
    Ok(Router::new()
        .route("/health", get(routes::health))
        .route("/v1/teapot", get(routes::teapot))
        .route("/v1/catalog", get(routes::catalog))
        .route("/v1/indexes", post(routes::build))
        .route("/v1/indexes/load", post(routes::load))
        .route("/v1/jobs/{id}", get(routes::job).delete(routes::remove_job))
        .route(
            "/v1/indexes/{id}",
            get(routes::index).delete(routes::unload),
        )
        .route("/v1/indexes/{id}/search", post(routes::search))
        .route("/v1/indexes/{id}/trace", post(routes::trace))
        .fallback(routes::not_found)
        .method_not_allowed_fallback(routes::method_not_allowed)
        .layer(DefaultBodyLimit::max(limit))
        .layer(axum::middleware::map_response(error::normalize_rejection))
        .layer(axum::middleware::from_fn_with_state(
            auth,
            auth::authenticate,
        ))
        .layer(axum::middleware::from_fn(request_context::correlate))
        .with_state(service))
}

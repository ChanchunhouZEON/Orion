/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Server-issued correlation IDs cover authentication, decoding and native work.
use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug)]
pub(crate) struct RequestContext {
    pub id: String,
}
static SERIAL: AtomicU64 = AtomicU64::new(0);
pub(crate) async fn correlate(mut request: Request, next: Next) -> Response {
    let id = format!(
        "req-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    );
    let method = request.method().to_string();
    // Log route templates, never query strings, credentials or supplied paths.
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| "unmatched".into());
    request
        .extensions_mut()
        .insert(RequestContext { id: id.clone() });
    let start = Instant::now();
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        "x-request-id",
        id.parse().expect("generated ASCII request ID"),
    );
    response.headers_mut().insert(
        "cache-control",
        axum::http::HeaderValue::from_static("no-store"),
    );
    let principal = response.extensions_mut().remove::<crate::Principal>();
    eprintln!(
        "{}",
        serde_json::json!({"event":"http_request","request_id":id,"owner":principal.as_ref().map(|p| p.subject()),"method":method,"route":route,"status":response.status().as_u16(),"elapsed_ms":start.elapsed().as_millis()})
    );
    response
}

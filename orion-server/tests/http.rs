/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use axum::{
    body::{to_bytes, Body},
    http::Request,
    Router,
};
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};
use orion_ann_server::{router, ServerConfig, TokenAuthenticator};
use sha2::{Digest, Sha256};
use std::sync::Arc;
const ALICE: &str = "test-only-alice-random-key-not-for-deployment";
const BOB: &str = "test-only-bob-random-key-not-for-deployment";
fn auth() -> Arc<TokenAuthenticator> {
    let file = tempfile::NamedTempFile::new().unwrap();
    let records = json!([
        {"subject":"alice","token_sha256":format!("{:x}",Sha256::digest(ALICE))},
        {"subject":"bob","token_sha256":format!("{:x}",Sha256::digest(BOB))}
    ]);
    std::fs::write(file.path(), records.to_string()).unwrap();
    Arc::new(TokenAuthenticator::from_file(file.path()).unwrap())
}

use serde_json::{json, Value};
use tower::ServiceExt;

async fn call(app: &Router, method: &str, uri: &str, value: Option<Value>) -> (u16, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("authorization", format!("Bearer {ALICE}"))
        .header("content-type", "application/json")
        .body(Body::from(value.map(|v| v.to_string()).unwrap_or_default()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert!(response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .starts_with("req-"));
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    (
        status,
        if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        },
    )
}
async fn finish(app: &Router, job: &Value) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            let (status, report) = call(
                app,
                "GET",
                &format!("/v1/jobs/{}", job["id"].as_str().unwrap()),
                None,
            )
            .await;
            assert_eq!(status, 200);
            if matches!(report["status"].as_str(), Some("complete" | "failed")) {
                return report;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
fn values() -> Vec<u8> {
    (0..128 * 13)
        .map(|i| ((i * 17 + i / 13 * 23) % 251) as u8)
        .collect()
}
fn options() -> Value {
    json!({"threads":2,"auto_pin":false,"graph":{"degree":12,"build_l":24,"max_extra":8,"alpha":1.25}})
}
fn request(native: bool, calibrated: bool) -> Value {
    let bytes = values();
    json!({"base":{"kind":if native {"u8"} else {"f32"},"values":bytes,"dimension":13},
        "options":options(), "calibrations":if calibrated {
            json!([{"queries": &bytes[..13*8],"k":3,"search_list_size":24}])
        } else { json!([]) }})
}

#[tokio::test]
async fn teapot_preserves_authentication_json_and_request_context() {
    let root = tempfile::tempdir().unwrap();
    let app = router(ServerConfig {
        authenticator: Some(auth()),
        data_root: root.path().into(),
        ..Default::default()
    })
    .unwrap();
    let mut request_ids = std::collections::HashSet::new();
    for (token, expected) in [
        (None, 401),
        (Some("invalid"), 401),
        (Some(ALICE), 418),
        (Some(BOB), 418),
    ] {
        let mut request = Request::builder()
            .uri("/v1/teapot")
            .header("x-request-id", "client-forged-id");
        if let Some(token) = token {
            request = request.header("authorization", format!("Bearer {token}"));
        }
        let response = app
            .clone()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), expected);
        let id = response.headers()["x-request-id"].to_str().unwrap();
        assert!(id.starts_with("req-"));
        assert!(request_ids.insert(id.to_owned()));
        assert_eq!(response.headers()["cache-control"], "no-store");
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        if expected == 401 {
            assert_eq!(
                response.headers()["www-authenticate"],
                "Bearer realm=\"orion\""
            );
        }
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        if expected == 418 {
            assert_eq!(body, json!({"message": "I'm a teapot"}));
        } else {
            assert_eq!(body["code"], "ORION_UNAUTHENTICATED");
        }
    }
    assert_eq!(call(&app, "POST", "/v1/teapot", None).await.0, 405);
    assert_eq!(
        call(&app, "GET", "/v1/catalog", None).await.1,
        json!({"indexes": [], "jobs": [], "snapshots": []})
    );
}

#[tokio::test]
async fn http_build_search_trace_and_snapshot_match_rust_for_f32_and_u8() {
    for native in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let app = router(ServerConfig {
            authenticator: Some(auth()),
            data_root: root.path().into(),
            ..Default::default()
        })
        .unwrap();
        let (status, queued) = call(&app, "POST", "/v1/indexes", Some(request(native, true))).await;
        assert_eq!(status, 202);
        let job = finish(&app, &queued).await;
        assert_eq!(job["status"], "complete", "{job}");
        assert_eq!(
            job["stages"],
            json!(["queued", "building", "calibrating", "saving", "complete"])
        );
        let id = job["index_id"].as_str().unwrap();
        let bytes = values();
        let floats: Vec<f32> = bytes.iter().map(|&v| f32::from(v)).collect();
        let base = || {
            if native {
                BaseInput::from_u8(&bytes, 13).unwrap()
            } else {
                BaseInput::from_f32(&floats, 13).unwrap()
            }
        };
        let mut opts: IndexOptions = serde_json::from_value(options()).unwrap();
        opts.cache_dir = root.path().join("rust-cache");
        let mut direct = Index::new(base(), opts).unwrap();
        direct
            .calibrate(floats[..13 * 8].to_vec(), 3, Some(24))
            .unwrap();
        let queries = json!({"queries":{"kind":if native {"u8"} else {"f32"},"dimension":13,"values":&bytes[..26]},"k":3,"search_list_size":24});
        let batch = if native {
            QueryBatch::from_u8(&bytes[..26], 13)
        } else {
            QueryBatch::from_f32(&floats[..26], 13)
        }
        .unwrap();
        let expected = direct.search(batch, &SearchOptions::new(3, 24)).unwrap();
        let path = format!("/v1/indexes/{id}/search");
        let (a, b) = tokio::join!(
            call(&app, "POST", &path, Some(queries.clone())),
            call(&app, "POST", &path, Some(queries.clone()))
        );
        assert_eq!(a, b);
        assert_eq!(
            a,
            (
                200,
                json!({"ids":expected.ids,"counts":expected.counts,"k":3})
            )
        );
        let (_, trace) = call(
            &app,
            "POST",
            &format!("/v1/indexes/{id}/trace"),
            Some(json!({"query":&bytes[..13],"k":3,"search_list_size":24})),
        )
        .await;
        let expected_trace = direct
            .trace_search(floats[..13].to_vec(), 3, 24, true, Default::default())
            .unwrap();
        assert_eq!(trace, serde_json::to_value(expected_trace).unwrap());
        let snapshot = root.path().join("tenants/alice/snapshots").join(id);
        let restored = Index::open(&snapshot, base()).unwrap();
        assert_eq!(
            restored.search(batch, &SearchOptions::new(3, 24)).unwrap(),
            expected
        );
        assert_eq!(
            call(&app, "DELETE", &format!("/v1/indexes/{id}"), None)
                .await
                .0,
            204
        );
        let (status, queued) = call(&app,"POST","/v1/indexes/load",Some(json!({"snapshot_id":id,"base":request(native,false)["base"],"options":{"threads":2,"auto_pin":false}}))).await;
        assert_eq!(status, 202);
        let loaded = finish(&app, &queued).await;
        assert_eq!(loaded["status"], "complete", "{loaded}");
        let (_, result) = call(
            &app,
            "POST",
            &format!(
                "/v1/indexes/{}/search",
                loaded["index_id"].as_str().unwrap()
            ),
            Some(queries),
        )
        .await;
        assert_eq!(result, a.1);
        let (_, catalog) = call(&app, "GET", "/v1/catalog", None).await;
        assert_eq!(catalog["indexes"].as_array().unwrap().len(), 1);
        assert!(catalog["snapshots"]
            .as_array()
            .unwrap()
            .contains(&json!(id)));
    }
}

#[tokio::test]
async fn native_errors_are_preserved_and_failed_jobs_do_not_publish() {
    let root = tempfile::tempdir().unwrap();
    let app = router(ServerConfig {
        authenticator: Some(auth()),
        data_root: root.path().into(),
        ..Default::default()
    })
    .unwrap();
    let (_, queued) = call(&app, "POST", "/v1/indexes", Some(request(true, false))).await;
    let job = finish(&app, &queued).await;
    assert_eq!(job["status"], "complete");
    let bytes = values();
    let mut opts: IndexOptions = serde_json::from_value(options()).unwrap();
    opts.cache_dir = root.path().join("direct");
    let direct = Index::new(BaseInput::from_u8(&bytes, 13).unwrap(), opts).unwrap();
    let expected = direct
        .search(
            QueryBatch::from_u8(&bytes[..13], 13).unwrap(),
            &SearchOptions::new(3, 24),
        )
        .unwrap_err();
    let path = format!("/v1/indexes/{}/search", job["index_id"].as_str().unwrap());
    let (status,error)=call(&app,"POST",&path,Some(json!({"queries":{"kind":"u8","values":&bytes[..13],"dimension":13},"k":3,"search_list_size":24}))).await;
    assert_eq!(status, 409);
    assert_eq!(error, serde_json::to_value(expected.report()).unwrap());
    let malformed = QueryBatch::from_u8(&bytes[..12], 13).unwrap_err();
    let (status,error)=call(&app,"POST",&path,Some(json!({"queries":{"kind":"u8","values":&bytes[..12],"dimension":13},"k":3,"search_list_size":24}))).await;
    assert_eq!(status, 400);
    assert_eq!(error, serde_json::to_value(malformed.report()).unwrap());
    let (_, queued) = call(
        &app,
        "POST",
        "/v1/indexes",
        Some(json!({"base":{"kind":"file","path":"../outside.fbin"}})),
    )
    .await;
    let failed = finish(&app, &queued).await;
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["code"], "ORION_INVALID_ARGUMENT");
    assert!(failed["index_id"].is_null());
    assert_eq!(
        call(&app, "GET", "/v1/catalog", None).await.1["indexes"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn json_limits_routing_and_capacity_use_structured_errors() {
    let root = tempfile::tempdir().unwrap();
    let app = router(ServerConfig {
        authenticator: Some(auth()),
        data_root: root.path().into(),
        max_indexes: 1,
        ..Default::default()
    })
    .unwrap();
    for (method, path, body, status) in [
        ("POST", "/v1/indexes", Some(json!({"unknown":true})), 422),
        ("GET", "/v1/jobs/absent", None, 404),
        ("GET", "/absent", None, 404),
        ("PUT", "/health", None, 405),
    ] {
        let (actual, error) = call(&app, method, path, body).await;
        assert_eq!(actual, status, "{error}");
        assert!(error["code"].as_str().unwrap().starts_with("ORION_"));
    }
    let (_, queued) = call(&app, "POST", "/v1/indexes", Some(request(true, true))).await;
    assert_eq!(finish(&app, &queued).await["status"], "complete");
    let (status, error) = call(&app, "POST", "/v1/indexes", Some(request(true, true))).await;
    assert_eq!(status, 429);
    assert_eq!(error["code"], "ORION_RESOURCE_EXHAUSTED");
    let small = router(ServerConfig {
        authenticator: Some(auth()),
        data_root: root.path().join("small"),
        max_body_bytes: 16,
        ..Default::default()
    })
    .unwrap();
    let (status, error) = call(&small, "POST", "/v1/indexes", Some(request(true, true))).await;
    assert_eq!(status, 413);
    assert_eq!(error["operation"], "http_decode");

    let (status, error) = call(&app, "GET", "/v1/jobs/%FF", None).await;
    assert_eq!(status, 400);
    assert_eq!(error["operation"], "http_decode");
    let (status, error) = call(
        &app,
        "POST",
        "/v1/indexes/absent/search",
        Some(json!({
            "queries":{"kind":"f32","dimension":1,"values":[1]},"k":1,"search_list_size":100000
        })),
    )
    .await;
    assert_eq!(status, 429);
    assert_eq!(error["code"], "ORION_RESOURCE_EXHAUSTED");
    let id = queued["id"].as_str().unwrap();
    assert_eq!(
        call(&app, "DELETE", &format!("/v1/jobs/{id}"), None)
            .await
            .0,
        204
    );
    assert_eq!(
        call(&app, "GET", &format!("/v1/jobs/{id}"), None).await.0,
        404
    );
    assert_eq!(
        call(&app, "GET", &format!("/v1/indexes/{id}"), None)
            .await
            .0,
        200
    );
}

#[tokio::test]
async fn file_build_and_restart_load_preserve_verified_snapshot() {
    let root = tempfile::tempdir().unwrap();
    let bytes = values();
    let mut file = Vec::new();
    file.extend(128u32.to_le_bytes());
    file.extend(13u32.to_le_bytes());
    file.extend(&bytes);
    std::fs::create_dir_all(root.path().join("tenants/alice")).unwrap();
    std::fs::write(root.path().join("tenants/alice/base.u8bin"), file).unwrap();
    let config = ServerConfig {
        authenticator: Some(auth()),
        data_root: root.path().into(),
        ..Default::default()
    };
    let app = router(config.clone()).unwrap();
    let mut req = request(true, true);
    req["base"] = json!({"kind":"file","path":"base.u8bin"});
    let (_, queued) = call(&app, "POST", "/v1/indexes", Some(req)).await;
    let job = finish(&app, &queued).await;
    assert_eq!(job["status"], "complete", "{job}");
    drop(app);
    let restarted = router(config).unwrap();
    let (_, queued)=call(&restarted,"POST","/v1/indexes/load",Some(json!({"snapshot_id":job["snapshot_id"],"base":{"kind":"file","path":"base.u8bin"},"options":{"threads":2}}))).await;
    assert_eq!(finish(&restarted, &queued).await["status"], "complete");
    let (_, queued) = call(
        &restarted,
        "POST",
        "/v1/indexes/load",
        Some(json!({"snapshot_id":"../outside","base":{"kind":"file","path":"base.u8bin"}})),
    )
    .await;
    assert_eq!(finish(&restarted, &queued).await["status"], "failed");
}

// Explicit adversarial transport calls preserve headers for correlation checks.
async fn authorized(
    app: &Router,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<Value>,
    spoof: bool,
) -> (u16, Value, String) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if let Some(token) = token {
        request = request.header("authorization", format!("Bearer {token}"));
    }
    if spoof {
        request = request
            .header("x-orion-subject", "alice")
            .header("x-forwarded-user", "alice")
            .header("x-forwarded-for", "127.0.0.1")
            .header("x-request-id", "forged");
    }
    let response = app
        .clone()
        .oneshot(
            request
                .body(Body::from(body.map(|b| b.to_string()).unwrap_or_default()))
                .unwrap(),
        )
        .await
        .unwrap();
    let id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert_ne!(id, "forged");
    assert_eq!(response.headers()["cache-control"], "no-store");
    let status = response.status().as_u16();
    if status == 401 {
        assert_eq!(
            response.headers()["www-authenticate"],
            "Bearer realm=\"orion\""
        );
    }
    let bytes = to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    (
        status,
        if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        },
        id,
    )
}

#[tokio::test]
async fn authorization_covers_every_resource_and_persists_across_restart() {
    let root = tempfile::tempdir().unwrap();
    let config = ServerConfig {
        data_root: root.path().into(),
        authenticator: Some(auth()),
        ..Default::default()
    };
    let app = router(config.clone()).unwrap();
    for token in [None, Some("invalid")] {
        let (status, error, _) = authorized(
            &app,
            "POST",
            "/v1/indexes",
            token,
            Some(request(true, true)),
            true,
        )
        .await;
        assert_eq!(status, 401);
        assert_eq!(error["code"], "ORION_UNAUTHENTICATED");
    }
    assert_eq!(
        authorized(&app, "GET", "/health", None, None, false)
            .await
            .0,
        200
    );
    let (status, queued, request_id) = authorized(
        &app,
        "POST",
        "/v1/indexes",
        Some(ALICE),
        Some(request(true, true)),
        true,
    )
    .await;
    assert_eq!(status, 202);
    assert_eq!(queued["request_id"], request_id);
    assert_eq!(queued["owner"], "alice");
    let job = finish(&app, &queued).await;
    assert_eq!(job["request_id"], request_id);
    let id = job["index_id"].as_str().unwrap();
    let queries = json!({"queries":{"kind":"u8","dimension":13,"values":&values()[..13]},"k":3,"search_list_size":24});
    for (method, path, body) in [
        ("GET", format!("/v1/jobs/{id}"), None),
        ("DELETE", format!("/v1/jobs/{id}"), None),
        ("GET", format!("/v1/indexes/{id}"), None),
        ("DELETE", format!("/v1/indexes/{id}"), None),
        (
            "POST",
            format!("/v1/indexes/{id}/search"),
            Some(queries.clone()),
        ),
        (
            "POST",
            format!("/v1/indexes/{id}/trace"),
            Some(json!({"query":&values()[..13],"k":3,"search_list_size":24})),
        ),
    ] {
        assert_eq!(
            authorized(&app, method, &path, Some(BOB), body, true)
                .await
                .0,
            404,
            "{path}"
        );
    }
    let (_, catalog, _) = authorized(&app, "GET", "/v1/catalog", Some(BOB), None, true).await;
    assert_eq!(catalog, json!({"indexes":[],"jobs":[],"snapshots":[]}));
    // Unauthorized unload did not affect the owner, and each call has a new ID.
    let (status, _, search_id) = authorized(
        &app,
        "POST",
        &format!("/v1/indexes/{id}/search"),
        Some(ALICE),
        Some(queries),
        false,
    )
    .await;
    assert_eq!(status, 200);
    assert_ne!(search_id, request_id);
    drop(app);
    let restarted = router(config).unwrap();
    let load = json!({"snapshot_id":id,"base":request(true,false)["base"],"options":{"threads":2,"auto_pin":false}});
    let (_, queued, _) = authorized(
        &restarted,
        "POST",
        "/v1/indexes/load",
        Some(BOB),
        Some(load.clone()),
        false,
    )
    .await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        assert!(std::time::Instant::now() < deadline, "load job timed out");
        let (_, state, _) = authorized(
            &restarted,
            "GET",
            &format!("/v1/jobs/{}", queued["id"].as_str().unwrap()),
            Some(BOB),
            None,
            false,
        )
        .await;
        if state["status"] == "failed" {
            assert!(state["index_id"].is_null());
            break;
        }
        assert_ne!(state["status"], "complete");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let (_, queued, _) = authorized(
        &restarted,
        "POST",
        "/v1/indexes/load",
        Some(ALICE),
        Some(load),
        false,
    )
    .await;
    assert_eq!(finish(&restarted, &queued).await["status"], "complete");
}

#[tokio::test]
async fn authentication_fails_closed_and_rejects_ambiguous_credentials() {
    use orion_ann_server::{Authenticator, Principal};
    assert!(router(ServerConfig::default()).is_err());
    let root = tempfile::tempdir().unwrap();
    let app = router(ServerConfig {
        data_root: root.path().into(),
        authenticator: Some(auth()),
        ..Default::default()
    })
    .unwrap();
    let request = Request::builder()
        .uri("/v1/catalog")
        .header("authorization", format!("Bearer {ALICE}"))
        .header("authorization", format!("Bearer {BOB}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.oneshot(request).await.unwrap().status(), 401);
    #[derive(Debug)]
    struct Unavailable;
    impl Authenticator for Unavailable {
        fn authenticate<'a>(
            &'a self,
            _: &'a str,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = orion_ann::Result<Option<Principal>>> + Send + 'a>,
        > {
            Box::pin(async {
                Err(orion_ann::Error::internal(
                    "secret database connection detail",
                ))
            })
        }
    }
    let app = router(ServerConfig {
        data_root: root.path().into(),
        authenticator: Some(Arc::new(Unavailable)),
        ..Default::default()
    })
    .unwrap();
    let (status, error, _) = authorized(&app, "GET", "/v1/catalog", Some(ALICE), None, false).await;
    assert_eq!(status, 503);
    assert!(!error.to_string().contains("secret"));
}

#[tokio::test]
async fn file_namespaces_reject_cross_owner_paths_and_symlinks() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("tenants/bob")).unwrap();
    std::fs::write(root.path().join("tenants/bob/base.u8bin"), b"other owner").unwrap();
    let app = router(ServerConfig {
        data_root: root.path().into(),
        authenticator: Some(auth()),
        ..Default::default()
    })
    .unwrap();
    for path in ["../bob/base.u8bin", "/tmp/base.u8bin"] {
        let (_, queued) = call(
            &app,
            "POST",
            "/v1/indexes",
            Some(json!({"base":{"kind":"file","path":path}})),
        )
        .await;
        assert_eq!(finish(&app, &queued).await["status"], "failed");
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(
            root.path().join("tenants/bob/base.u8bin"),
            root.path().join("tenants/alice/alias.u8bin"),
        )
        .unwrap();
        let (_, queued) = call(
            &app,
            "POST",
            "/v1/indexes",
            Some(json!({"base":{"kind":"file","path":"alias.u8bin"}})),
        )
        .await;
        assert_eq!(
            finish(&app, &queued).await["error"]["operation"],
            "http_path"
        );
    }
}

#[tokio::test]
async fn credential_configuration_rotation_and_redaction() {
    use orion_ann_server::Authenticator;
    let file = tempfile::NamedTempFile::new().unwrap();
    let hash = format!("{:x}", Sha256::digest(ALICE));
    for invalid in [
        json!([]),
        json!({"unexpected":true}),
        json!([{"subject":"../alice","token_sha256":hash}]),
        json!([{"subject":"alice","token_sha256":"short"}]),
        json!([{"subject":"alice","token_sha256":"z".repeat(64)}]),
        json!([{"subject":"alice","token_sha256":hash},{"subject":"bob","token_sha256":hash}]),
        json!([{"subject":"alice","token_sha256":hash,"unexpected":true}]),
    ] {
        std::fs::write(file.path(), invalid.to_string()).unwrap();
        assert!(TokenAuthenticator::from_file(file.path()).is_err());
    }
    std::fs::write(
        file.path(),
        json!([
            {"subject":"alice","token_sha256":hash},
            {"subject":"alice","token_sha256":format!("{:x}",Sha256::digest(BOB))}
        ])
        .to_string(),
    )
    .unwrap();
    let auth = TokenAuthenticator::from_file(file.path()).unwrap();
    let debug = format!("{auth:?}");
    assert!(!debug.contains(&hash) && !debug.contains(ALICE));
    for token in [ALICE, BOB] {
        assert_eq!(
            auth.authenticate(token).await.unwrap().unwrap().subject(),
            "alice"
        );
    }
    assert!(auth.authenticate("invalid").await.unwrap().is_none());
}

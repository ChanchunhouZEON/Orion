<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
# Orion native HTTP service

A local prototype over `orion-ann::Index`. Independent transport modules handle
JSON, job admission, server-owned paths and HTTP status mapping. Core algorithms,
SIMD, prefetch and persisted formats are unchanged.

```bash
python3 orion-server/tools/create-credentials.py --subject alice --directory /tmp/orion-private-auth
cargo run --locked -p orion-ann-server -- \
  --listen 127.0.0.1:8080 --data-root /absolute/path/to/service-data \
  --auth-file /tmp/orion-private-auth/credentials.json
```

Use `--profile python` for optimized builds with panic unwinding. The default
release profile aborts on panic and cannot translate a panic into an HTTP error.

Read [REQUESTS.md](REQUESTS.md) for the versioned protocol and
[postman_collection.json](postman_collection.json) for importable requests.

```bash
cargo test --locked -p orion-ann-server
cargo build --locked -p orion-ann-server
python3 orion-server/tools/smoke-http.py --binary target/debug/orion-server \
  --nginx "$(command -v nginx)" --output /tmp/orion-http-evidence
```

The sibling `orion-explorer` provides a Native HTTP service entry in its source
catalog. `npm run dev` proxies `/api` to loopback port 8080. It can submit a build,
follow stages, load snapshots, search and replay a newly recorded native trace
through the existing WASM worker. Enter the generated `token.txt` value in the
Native workspace; it is retained in component memory only and cleared on
disconnect. Static production hosting needs an equivalent
same-origin proxy; the static bundle itself does not run the native service.

The `Native HTTP` job in `.github/workflows/ci.yml` repeats the Rust contract
tests and this curl script on Linux and macOS, including pushes to `feature/web`.
Request/response evidence is uploaded as a workflow artifact, also on failure.
The service is being developed on that branch; these checks do not merge it
into `main` or deploy a service.

This first stage has bearer authentication, per-subject namespaces and process-local bounded job history,
no uploads, no resumable/cancellable jobs, and phase-level rather than per-node
progress. Build/load work is bounded separately from search/trace work. Snapshot
files survive process exit; loaded indexes must be reopened. The operator owns
the data root and any trusted snapshot contents. Put file inputs under
`data-root/tenants/alice/` for subject `alice`; old unscoped prototype snapshots
are not automatically assigned to anyone.

`auth.rs` verifies credentials through an asynchronous `Authenticator` trait;
`request_context.rs` issues correlation IDs; `Registry` checks ownership; `Paths`
confines storage to that identity. A future database or OIDC provider replaces
only the authenticator. There is no relational database dependency.

nginx is a test/deployment dependency, not an embedded proxy. The fixture at
`proxy/nginx.conf.template` is rendered by the smoke script and terminates TLS
with a temporary test certificate. For remote hosting, configure a trusted TLS
certificate and the same header/cache/retry policy. Keep the Orion listener
private. No proxy or public listener is installed as a background system service.

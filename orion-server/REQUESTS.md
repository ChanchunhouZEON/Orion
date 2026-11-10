<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
# HTTP request specification · v1

This document describes the native HTTP prototype. JSON field names and Orion
error codes are shared by the Explorer, curl checks and Postman collection.
The `/v1` prefix versions the transport contract; snapshot identities retain
the runtime's SHA-256 format. No HTTP-specific graph or search implementation
is introduced.

## Execution and ownership

- The native process performs graph construction, calibration, verified loading,
  search and persistence. The browser's WASM module only replays traces.
- `POST /v1/indexes` and `POST /v1/indexes/load` return **202**, with a job ID.
  Acceptance is not success: poll the job until `complete` or `failed`.
- Stages are real transitions. Building currently includes base loading and
  graph preparation; no per-vector progress or completion percentage is exposed.
- Only a fully prepared index is published. Searches use shared `&Index` access;
  trace recording currently takes exclusive access because the native trace API
  requires `&mut Index`. Calibration finishes before publication.
- Jobs and loaded handles live in this process. Snapshots survive restart.
  Jobs are not resumable and disconnecting HTTP does not cancel native work.
- The service owns `jobs/<id>/cache`, mapping scratch and `snapshots/<id>` under
  `data-root/tenants/<subject>/`. The subject is derived only from authentication. Do not delete a job's cache: snapshots share those immutable
  objects. Deleting a job record or unloading an index never deletes files.

## Common transport rules

Default address: `http://127.0.0.1:8080`. Requests with bodies require
`Content-Type: application/json`. Unknown request fields are rejected. This is
an authenticated prototype with owner-scoped resources and operator-controlled storage.
The executable only accepts loopback binds; use the Explorer's same-origin
`/api` proxy. Cross-origin CORS is not enabled.

Default limits: 16 MiB JSON body, 1 build/load worker, 8 accepted active/queued
jobs, 8 loaded-or-reserved indexes, 128 retained job records, 8 simultaneous
search/trace requests and at most 64 Rayon threads per index. Each request is
limited to a 65,536 search-list size and 1,000,000 result ID slots. Capacity rejection
is immediate (429). A memory budget is a planning estimate, not an RSS cap.

File inputs are **relative to `data-root/tenants/<subject>/`**. Absolute paths,
`..` and symlinks escaping that root are rejected. The operator must not allow
untrusted users to mutate this root or place arbitrary snapshot manifests in it.
There is no upload endpoint in v1; place large files on the server first.

## Authentication and correlation

All requests except `GET /health` require `Authorization: Bearer <token>`.
Exactly one Authorization header is accepted. Unknown/duplicate/malformed keys
return 401 with `WWW-Authenticate: Bearer realm="orion"`. There is no anonymous
mode or proxy-only bypass. Authorization in cookies or URL parameters is not
supported. Use TLS for every connection outside loopback.

The executable requires `--auth-file`, a JSON array:

```json
[{"subject":"alice","token_sha256":"<64 hexadecimal SHA-256 characters>"}]
```

Generate a random 256-bit token using `tools/create-credentials.py`, keep its
plaintext only on the client, and store the digest server-side. These are API
keys, not passwords: SHA-256 alone is not suitable for hashing human passwords.
Keep credential configuration outside the data root and web document root,
readable/writable only by the operator. Configuration changes require restart;
multiple distinct keys can identify the same subject for rotation.

`Authenticator` is an async Rust interface; it returns `Result<Option<Principal>>`.
`Some` authenticates a stable subject, `None` rejects a key, and `Err` fails closed
with 503. Provider errors are not exposed because they may contain connection
secrets. Database/OIDC adapters can implement this boundary without changing
build, search or resource authorization. No SQL schema, user registration,
password login or database connection is part of this stage.

Each identity has full access to its own indexes, jobs and snapshots only.
Catalogs exclude other owners. Direct GET/search/trace/unload/delete for another
owner's ID returns the same 404 as an unknown ID. Snapshot loading resolves only
inside the authenticated subject's directory, including after restart. A foreign
snapshot load therefore fails as an asynchronous job with no published index.
There is no sharing/admin role in this prototype. Filesystem ownership is trusted:
HTTP users cannot write manifests, create symlinks or upload arbitrary files.

Every application response, including 401/404/413, has a server-generated
`X-Request-ID` and `Cache-Control: no-store`. Client-provided request IDs are never
reused. The submission response records the same ID in `job.request_id`; subsequent
polls get their own HTTP IDs while retaining the original job ID for correlation.
Structured request and job-completion logs contain these IDs, never credentials,
query arrays or URL query strings. Native `ErrorReport` bodies are unchanged.

The nginx fixture forwards Authorization, removes identity/Forwarded/request-ID
headers, disables caching and automatic upstream retries, and exposes `/api/`.
The server itself ignores proxy identity and client IP headers. It does not need
a trusted-proxy list because no authorization decision uses those headers.
The acceptance script tests HTTPS with an ephemeral certificate trusted via
`curl --cacert`; it never disables certificate verification. Proxy-generated
errors (TLS failure, upstream unavailable, ingress limits) are not Orion errors
and may not have Orion's JSON envelope or application request ID.

Limits are global admission bounds, not per-owner fairness or aggregate RSS
isolation. Authentication and ownership alone do not make arbitrary public
multi-tenant workloads safe to host without additional resource controls.

## Endpoints

| Method | Path | Response | Purpose |
|---|---|---:|---|
| GET | `/health` | 200 | Process health and API version |
| GET | `/v1/teapot` | 418 | Authenticated Easter egg |
| GET | `/v1/catalog` | 200 | Loaded indexes, jobs and persisted snapshot IDs |
| POST | `/v1/indexes` | 202 | Build, optionally calibrate, save and publish |
| POST | `/v1/indexes/load` | 202 | Verify and load an existing snapshot |
| GET | `/v1/jobs/{id}` | 200 | Current stage, transitions and outcome |
| DELETE | `/v1/jobs/{id}` | 204 | Remove a terminal job record only |
| GET | `/v1/indexes/{id}` | 200 | Native `Index::info()` |
| DELETE | `/v1/indexes/{id}` | 204 | Unload handle; in-flight readers keep ownership |
| POST | `/v1/indexes/{id}/search` | 200 | Adaptive batch search |
| POST | `/v1/indexes/{id}/trace` | 200 | One native query, bounded diagnostic trace |

### Teapot

`GET /v1/teapot` requires the same Bearer credentials as the other `/v1` routes.
With valid credentials it always returns **418**, `Content-Type: application/json`,
`Cache-Control: no-store` and a fresh server-generated `X-Request-ID`:

```json
{"message":"I'm a teapot"}
```

Missing or invalid credentials still return the normal **401** error. This
endpoint performs no index work and works through the proxy at `/api/v1/teapot`.
It is an Easter egg inspired by [RFC 2324](https://www.rfc-editor.org/rfc/rfc2324.html#section-2.3.2),
not a business error: [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110.html#section-15.5.19)
reserves 418 as “Unused”. Its dedicated message body is deliberately not a native
`ErrorReport`; no new Orion error code is introduced.

### Build

```json
{
  "base": {"kind": "file", "path": "base.u8bin"},
  "options": {
    "threads": 8,
    "auto_pin": true,
    "memory_budget_gib": 16,
    "storage_options": {"mode": "auto"}
  },
  "calibrations": [
    {"queries": [0, 1, 2, 3, 4, 5, 6, 7], "k": 3, "search_list_size": 24}
  ]
}
```

The example calibration assumes dimension 4: its flat array contains two rows.
Supply representative rows with the **actual base dimension**. The executable
curl/Postman fixtures generate a complete 128 × 13 dataset and matching queries.

`base` is one of:

```json
{"kind":"file","path":"base.fbin"}
{"kind":"f32","dimension":4,"values":[0.1,0.2,0.3,0.4]}
{"kind":"u8","dimension":4,"values":[1,2,3,4]}
```

Inline values are packed row-major and copied according to `BaseInput`'s native
contract. File shape comes from the native format/header. `options` defaults to
`IndexOptions::default()` and uses its exact serde field names and validation;
see [configuration](../orion-runtime/src/guide/configuration.md). The exceptions
are service-owned `cache_dir` (leave at default) and `storage_options.directory`
(omit). A supplied `graph.staged_file` must also be relative to the subject directory.
No dataset presets or shortcuts are added.

`calibrations` defaults to `[]` and accepts at most 16 entries. Each entry has
`queries: f32[]`, `k: usize`, and optional `search_list_size: usize|null`.
A successfully built but uncalibrated index is valid; searching an uncalibrated
k returns the native `ORION_INVALID_STATE`. To add calibrations in this first
prototype, build a new index with the desired entries; v1 has no live mutation API.

### Job response

```json
{
  "id": "job-123-456-1",
  "owner": "alice",
  "request_id": "req-123-456-1",
  "status": "complete",
  "stage": "complete",
  "stages": ["queued","building","calibrating","saving","complete"],
  "index_id": "job-123-456-1",
  "snapshot_id": "job-123-456-1",
  "error": null
}
```

`status` is `queued`, `running`, `complete` or `failed`. Build stages are
`queued → building → [calibrating] → saving → complete`; load stages are
`queued → loading → complete`. Failures append `failed`, retain the preceding
stages, and provide the unchanged native `ErrorReport` in `error`. A failed job
has no published index ID. Polling a failed job itself still returns HTTP 200.

### Load

```json
{
  "snapshot_id":"job-123-456-1",
  "base":{"kind":"file","path":"base.u8bin"},
  "options":{"threads":8,"auto_pin":true,"storage_options":{"mode":"mapped"}}
}
```

Only service snapshot IDs are accepted, not arbitrary manifest paths. Original
base data is still required and verified by `OpenOptions::open`; saved
calibrations are restored. `options` may contain `threads`, `memory_budget_gib`,
`auto_pin` and `storage_options`. Omitted threads use min(8, server thread limit).
Omitted storage policy uses Auto with service-owned mapping scratch.

### Search

```json
{
  "queries":{"kind":"u8","dimension":4,"values":[1,2,3,4]},
  "k":3,
  "search_list_size":24
}
```

Queries accept `u8` or `f32` inline batches, never a file path. Controls have the
same validation and calibration requirement as `QueryBatch` + `SearchOptions`.

```json
{"ids":[12,8,31],"counts":[3],"k":3}
```

Each query occupies exactly k entries in the packed `ids`. Only its `counts`
prefix is valid; unused entries are `4294967295` (`u32::MAX`). There are no scores
or QPS claims in this response. Query ordering is preserved.

### Trace

```json
{"query":[1,2,3,4],"k":3,"search_list_size":24}
```

The response is the native `SearchTrace` JSON, directly consumable by
`orion-wasm::TraceReplay`: metadata, bounded events, final `result_ids` and
`truncated`. Adaptive search is always enabled. The entire search finishes
before this response is sent; v1 does **not** stream individual events.
Recording adds overhead and is not a performance measurement.

### Error responses

```json
{
  "code":"ORION_INVALID_STATE",
  "operation":"search",
  "message":"calibrate this index for k=3 before adaptive search",
  "details":{"k":3},
  "causes":[]
}
```

Except for the documented teapot response, the JSON body is `Error::report()`
itself, not a separate HTTP error envelope.
Native validation messages, details and causes are preserved. HTTP decoding
errors use the same fields with operation `http_decode`.

| Condition | HTTP status |
|---|---:|
| Missing, invalid or ambiguous bearer credentials | 401 (`ORION_UNAUTHENTICATED`) |
| Authentication provider failed or exceeded 5 seconds | 503 (`ORION_INTERNAL`, generic message) |
| Resource missing or owned by another identity | 404 |
| Native invalid argument / dimension mismatch | 400 |
| Native invalid data / unsupported configuration | 422 |
| Missing calibration / incompatible cache | 409 |
| Capacity or native resource exhaustion | 429 |
| Native I/O, backend or internal error | 500 |
| Unknown route, index or job | 404 |
| Unsupported HTTP method | 405 |
| JSON decoding, content type or body size | Extractor status: 400 / 422 / 415 / 413 |

An unwinding worker panic becomes `ORION_INTERNAL`; abort/OOM cannot be caught.
Use the `python` Cargo profile for optimized builds that retain unwinding.

## Reproduce the protocol

From the workspace root:

```bash
cargo build --locked -p orion-ann-server
python3 orion-server/tools/smoke-http.py --binary target/debug/orion-server \
  --nginx "$(command -v nginx)" --output /tmp/orion-http-evidence
```

The script starts a temporary loopback service and nginx TLS proxy, executes
the same curl suite through both paths, checks owner isolation, forged headers,
request IDs, the authenticated 418 response, search/reload/trace parity and ownership after restart. It saves
request/response evidence and checks that no generated credential leaked into it.
Omit `--nginx` for direct-only debugging; CI always supplies it. Import `postman_collection.json` into Postman for
manual exploration. Supply `token` using a private local Postman environment;
never export real keys. This does not require the Postman app to be installed for the
curl verification above.

#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Real curl acceptance, direct and optionally through an isolated nginx process."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import tempfile
import time


class CurlClient:
    """Keep secrets out of process arguments and saved request evidence."""

    def __init__(self, url, output, tokens, ca_file=None):
        self.url, self.output, self.tokens = url.rstrip("/"), output, tokens
        self.serial, self.request_ids = 0, set()
        self.ca_file = ca_file
        output.mkdir(parents=True, exist_ok=False)

    def __call__(
        self, method, path, body=None, expected=200, identity="alice", spoof=False
    ):
        self.serial += 1
        name = f"{self.serial:03d}-{method.lower()}"
        target = self.output / f"{name}-response.json"
        headers = self.output / f"{name}-response.headers"
        # Temporary header files are mode 0600 and never copied to artifacts.
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="orion-curl-headers-"
        ) as secret:
            if identity is not None:
                secret.write(f"Authorization: Bearer {self.tokens[identity]}\n")
            if spoof:
                secret.write(
                    "X-Orion-Subject: alice\nX-Forwarded-User: alice\n"
                    "X-Forwarded-For: 127.0.0.1\nX-Request-ID: forged\n"
                )
            secret.flush()
            cmd = [
                "curl",
                "--silent",
                "--show-error",
                "--max-time",
                "30",
                "--request",
                method,
                "--header",
                "@" + secret.name,
                "--dump-header",
                str(headers),
                "--output",
                str(target),
                "--write-out",
                "%{http_code}",
                self.url + path,
            ]
            if self.ca_file:
                cmd += ["--cacert", str(self.ca_file)]
            if body is not None:
                payload = self.output / f"{name}-request.json"
                payload.write_text(json.dumps(body))
                cmd += [
                    "--header",
                    "Content-Type: application/json",
                    "--data-binary",
                    "@" + str(payload),
                ]
            status = int(subprocess.check_output(cmd, text=True))
        raw = target.read_text()
        value = json.loads(raw) if raw else None
        assert status == expected, (method, path, status, value)
        response_headers = dict(
            line.split(":", 1)
            for line in headers.read_text().lower().splitlines()
            if ":" in line
        )
        request_id = response_headers["x-request-id"].strip()
        assert re.fullmatch(r"req-[0-9-]+", request_id), request_id
        assert request_id not in self.request_ids
        self.request_ids.add(request_id)
        assert response_headers["cache-control"].strip() == "no-store"
        if status == 401:
            assert (
                response_headers["www-authenticate"].strip() == 'bearer realm="orion"'
            )
            assert value["code"] == "ORION_UNAUTHENTICATED"
        if status == 418:
            assert (
                response_headers["content-type"].strip().startswith("application/json")
            )
            assert value == {"message": "I'm a teapot"}
        if status == 202:
            assert value["request_id"] == request_id
            assert value["owner"] == identity
        (self.output / f"{name}-metadata.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "path": path,
                    "identity": identity,
                    "request_id": request_id,
                    "status": status,
                },
                indent=2,
            )
        )
        return value

    def wait(self, job, identity="alice", expected="complete"):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            result = self("GET", "/v1/jobs/" + job["id"], identity=identity)
            if result["status"] in ("failed", "complete"):
                assert result["status"] == expected, result
                assert result["request_id"] == job["request_id"]
                return result
            time.sleep(0.1)
        raise TimeoutError(f'Job {job["id"]} did not finish')


def fixtures():
    values = [(i * 17 + (i // 13) * 23) % 251 for i in range(128 * 13)]
    base = {"kind": "u8", "dimension": 13, "values": values}
    build = {
        "base": base,
        "options": {
            "threads": 2,
            "auto_pin": False,
            "graph": {"degree": 12, "build_l": 24, "max_extra": 8},
        },
        "calibrations": [{"queries": values[: 13 * 8], "k": 3, "search_list_size": 24}],
    }
    query = {
        "queries": {"kind": "u8", "dimension": 13, "values": values[:13]},
        "k": 3,
        "search_list_size": 24,
    }
    return base, build, query


def verify(client):
    base, build, query = fixtures()
    assert client("GET", "/health", identity=None)["status"] == "ok"
    client("GET", "/v1/teapot", identity=None, expected=401, spoof=True)
    client("GET", "/v1/teapot", identity="invalid", expected=401)
    teapot = client("GET", "/v1/teapot", expected=418, spoof=True)
    assert client("GET", "/v1/teapot", expected=418) == teapot
    client("GET", "/v1/catalog", identity=None, expected=401, spoof=True)
    client("POST", "/v1/indexes", build, 401, identity="invalid", spoof=True)
    before = client("GET", "/v1/catalog", identity="bob")
    assert before["indexes"] == [] and before["snapshots"] == []
    built = client.wait(client("POST", "/v1/indexes", build, 202))
    index = built["index_id"]
    first = client("POST", f"/v1/indexes/{index}/search", query)
    client(
        "POST",
        f"/v1/indexes/{index}/search",
        {**query, "search_list_size": 100000},
        429,
    )
    assert first["counts"] == [3] and len(first["ids"]) == 3
    trace_request = {
        "query": query["queries"]["values"],
        "k": 3,
        "search_list_size": 24,
    }
    trace = client("POST", f"/v1/indexes/{index}/trace", trace_request)
    assert trace["result_ids"] == first["ids"]
    (client.output / "trace.json").write_text(json.dumps(trace))
    # Every entry point must authorize the owner, even with forged proxy headers.
    for method, path, body in [
        ("GET", f"/v1/jobs/{index}", None),
        ("DELETE", f"/v1/jobs/{index}", None),
        ("GET", f"/v1/indexes/{index}", None),
        ("DELETE", f"/v1/indexes/{index}", None),
        ("POST", f"/v1/indexes/{index}/search", query),
        ("POST", f"/v1/indexes/{index}/trace", trace_request),
    ]:
        client(method, path, body, 404, identity="bob", spoof=True)
    bob = client("GET", "/v1/catalog", identity="bob", spoof=True)
    assert bob["indexes"] == [] and built["snapshot_id"] not in bob["snapshots"]
    assert not any(job["id"] == index for job in bob["jobs"])
    load = {
        "snapshot_id": built["snapshot_id"],
        "base": base,
        "options": {"threads": 2, "auto_pin": False},
    }
    denied = client.wait(
        client("POST", "/v1/indexes/load", load, 202, identity="bob"), "bob", "failed"
    )
    assert denied["index_id"] is None
    # Identity cannot be supplied through the build JSON either.
    client("POST", "/v1/indexes", {**build, "owner": "bob"}, 422)
    error = client("POST", f"/v1/indexes/{index}/search", {**query, "k": 2}, 409)
    assert error["code"] == "ORION_INVALID_STATE" and error["details"]["k"] == 2
    malformed = client("POST", "/v1/indexes", {"unexpected": True}, 422)
    assert malformed["operation"] == "http_decode"
    client("DELETE", "/v1/indexes/" + index, expected=204)
    restored = client.wait(client("POST", "/v1/indexes/load", load, 202))
    second = client("POST", f'/v1/indexes/{restored["index_id"]}/search', query)
    assert first == second
    catalog = client("GET", "/v1/catalog")
    assert any(i["id"] == restored["index_id"] for i in catalog["indexes"])
    assert built["snapshot_id"] in catalog["snapshots"]
    record = {
        "status": "passed",
        "requests": client.serial,
        "build_job": built,
        "load_job": restored,
        "search": first,
        "reload_search": second,
        "trace_events": len(trace["events"]),
        "missing_calibration": error,
        "transport": client.url,
        "teapot": {"status": 418, "body": teapot},
        "authorization": "owner isolation and spoof rejection verified",
    }
    (client.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "requests": client.serial,
                "ids": first["ids"],
                "evidence": str(client.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return record


@contextmanager
def running_process(command, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def wait_server(process, log_path):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        match = re.search(
            r"Orion HTTP listening on (http://127\.0\.0\.1:\d+)", log_path.read_text()
        )
        if match:
            return match[1]
        if process.poll() is not None:
            raise RuntimeError(log_path.read_text())
        time.sleep(0.1)
    raise TimeoutError("Service did not start")


@contextmanager
def proxy(nginx, upstream, directory, secrets_dir):
    directory.mkdir()
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    certificate, key = secrets_dir / "test-ca.pem", secrets_dir / "test-key.pem"
    cert_config = secrets_dir / "certificate.cnf"
    cert_config.write_text(
        "[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=ext\n"
        "[dn]\nCN=localhost\n[ext]\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "basicConstraints=critical,CA:TRUE\nkeyUsage=critical,digitalSignature,keyEncipherment,keyCertSign\n"
        "extendedKeyUsage=serverAuth\n"
    )
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-nodes",
            "-newkey",
            "rsa:2048",
            "-days",
            "1",
            "-config",
            str(cert_config),
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    key.chmod(0o600)
    template = (
        Path(__file__).resolve().parents[1] / "proxy/nginx.conf.template"
    ).read_text()
    config = (
        template.replace("@PORT@", str(port))
        .replace("@UPSTREAM@", upstream)
        .replace("@RUN_DIR@", str(directory))
        .replace("@CERT@", str(certificate))
        .replace("@KEY@", str(key))
    )
    config_path = directory / "nginx.conf"
    config_path.write_text(config)
    with running_process(
        [nginx, "-p", str(directory) + "/", "-c", str(config_path)],
        directory / "nginx.log",
    ) as process:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError((directory / "nginx.log").read_text())
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("nginx did not start")
        yield f"https://127.0.0.1:{port}/api", certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--binary", type=Path)
    source.add_argument(
        "--url",
        help="Existing service; supply ORION_TOKEN and ORION_OTHER_TOKEN with different subjects",
    )
    parser.add_argument(
        "--nginx",
        help="nginx executable: also exercise the real reverse proxy (requires --binary)",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.url and args.nginx:
        parser.error("--nginx requires --binary")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.url:
        tokens = {
            "alice": os.environ["ORION_TOKEN"],
            "bob": os.environ["ORION_OTHER_TOKEN"],
            "invalid": secrets.token_urlsafe(32),
        }
        verify(CurlClient(args.url, output / "existing", tokens))
        return
    tokens = {
        subject: secrets.token_urlsafe(32) for subject in ("alice", "bob", "invalid")
    }
    with tempfile.TemporaryDirectory(prefix="orion-http-smoke-") as temp:
        root = Path(temp)
        auth_file = root / "credentials.json"
        auth_file.write_text(
            json.dumps(
                [
                    {
                        "subject": subject,
                        "token_sha256": hashlib.sha256(
                            tokens[subject].encode()
                        ).hexdigest(),
                    }
                    for subject in ("alice", "bob")
                ]
            )
        )
        auth_file.chmod(0o600)
        command = [
            str(args.binary.resolve()),
            "--listen",
            "127.0.0.1:0",
            "--auth-file",
            str(auth_file),
            "--data-root",
            str(root / "data"),
        ]
        log_path = output / "server.log"
        with running_process(command, log_path) as server:
            url = wait_server(server, log_path)
            direct = verify(CurlClient(url, output / "direct", tokens))
            if args.nginx:
                with proxy(args.nginx, url, output / "nginx", root) as (
                    proxy_url,
                    ca_file,
                ):
                    proxied = verify(
                        CurlClient(proxy_url, output / "proxy", tokens, ca_file)
                    )
                    assert direct["search"] == proxied["search"]
                    assert direct["teapot"] == proxied["teapot"]
                # Proxy and service must correlate the same requests without logging keys.
                access = [
                    json.loads(line)
                    for line in (output / "nginx/proxy-access.jsonl")
                    .read_text()
                    .splitlines()
                ]
                assert any(
                    row["request_id"] == proxied["build_job"]["request_id"]
                    for row in access
                )
            service_log = log_path.read_text()
            assert direct["build_job"]["request_id"] in service_log
        # Restart with the same keys/storage: ownership cannot depend on RAM.
        restart_log = output / "restart.log"
        with running_process(command, restart_log) as server:
            client = CurlClient(
                wait_server(server, restart_log), output / "restart", tokens
            )
            base, _, query = fixtures()
            load = {
                "snapshot_id": direct["build_job"]["snapshot_id"],
                "base": base,
                "options": {"threads": 2, "auto_pin": False},
            }
            bob = client("GET", "/v1/catalog", identity="bob")
            assert bob["snapshots"] == []
            client.wait(
                client("POST", "/v1/indexes/load", load, 202, identity="bob"),
                "bob",
                "failed",
            )
            loaded = client.wait(client("POST", "/v1/indexes/load", load, 202))
            assert (
                client("POST", f'/v1/indexes/{loaded["index_id"]}/search', query)
                == direct["search"]
            )
        # Credentials must not appear anywhere in retained evidence.
        for file in output.rglob("*"):
            if file.is_file():
                content = file.read_bytes()
                assert not any(
                    token.encode() in content for token in tokens.values()
                ), file
    (output / "result.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "direct": True,
                "nginx_tls": bool(args.nginx),
                "restart_ownership": True,
                "credential_leak_check": "passed",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Verify that opening the production workspace does not enable benchmark APIs."""
import json
import subprocess

for manifest, experimental in [("Cargo.toml", False), ("benchmark/Cargo.toml", True)]:
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--locked", "--format-version", "1", "--manifest-path", manifest
    ]))
    names = {p["id"]: p["name"] for p in metadata["packages"]}
    members = {names[member] for member in metadata["workspace_members"]}
    assert ("benchmark" in members) == experimental, (manifest, members)
    features = {names[node["id"]]: node["features"] for node in metadata["resolve"]["nodes"]}
    for name in ("orion-ann", "orion-ann-core", "orion-ann-vector"):
        assert ("benchmark" in features[name]) == experimental, (manifest, name, features[name])
    # PQ is ordinary code, not an independent Cargo feature in either workspace.
    packages = {p["name"]: p for p in metadata["packages"]}
    for name in ("orion-ann", "orion-ann-core", "orion-ann-diskann"):
        assert "pq" not in packages[name]["features"], (manifest, name)

    # The Orion PQ implementation uses slices; ndarray remains an independent
    # DiskANN/Python dependency and must not leak back into the core API.
    assert all(d["name"] != "ndarray" for d in packages["orion-ann-core"]["dependencies"]), manifest

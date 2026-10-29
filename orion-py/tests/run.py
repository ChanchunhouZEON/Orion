#!/usr/bin/env python3
"""Build and test locally, without installing/publishing a wheel or touching data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-build", action="store_true")
parser.add_argument("--profile", default="dev", choices=["dev", "python"])
args = parser.parse_args()
if not args.no_build:
    subprocess.run(["cargo", "+1.92", "build", "--offline", "--locked", "--profile", args.profile,
                    "-p", "orion_py", "-p", "orion-cli"], cwd=root, check=True)
metadata = subprocess.check_output(["cargo", "+1.92", "metadata", "--offline", "--no-deps", "--format-version", "1"], cwd=root)
build = Path(json.loads(metadata)["target_directory"]) / ("debug" if args.profile == "dev" else args.profile)
lib = build / ("lib_native.dylib" if sys.platform == "darwin" else "lib_native.so")
with tempfile.TemporaryDirectory(prefix="orion-python-tests-") as tmp:
    package = Path(tmp) / "orion_py"
    shutil.copytree(root / "orion-py/python/orion_py", package)
    shutil.copyfile(lib, package / "_native.abi3.so")
    env = dict(os.environ, PYTHONPATH=tmp, ORION_TEST_CLI=str(build / "orion"))
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(root / "orion-py/tests"), "-v"],
                   cwd=tmp, env=env, check=True)

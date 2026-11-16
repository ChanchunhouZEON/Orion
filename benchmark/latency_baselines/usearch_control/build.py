# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Build only the private USearch adapter from pinned, external source."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

PIN = "7306bb446be5f0f0c529ec8acdc57361cef8a8a7"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN or (source / "VERSION").read_text().strip() != "2.23.0":
        raise ValueError("expected upstream USearch v2.23.0 pinned commit")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ):
        raise ValueError("upstream source must be clean")
    import usearch, simsimd

    if usearch.__version__ != "2.23.0" or not usearch.USES_SIMSIMD:
        raise ValueError("requires the measured USearch 2.23.0 / SimSIMD wheel")
    a.output.mkdir(parents=True, exist_ok=True)
    library = a.output.resolve() / (
        "liblatency_usearch" + (".dylib" if platform.system() == "Darwin" else ".so")
    )
    cpp = Path(__file__).resolve().parents[1] / "native/usearch.cpp"
    # Mirror wheel arithmetic settings and resolve SimSIMD from the same loaded
    # Python extension. Do not use -march=native to change this control's kernel.
    flags = [
        "-std=c++17",
        "-O3",
        "-ffast-math",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        "-pthread",
        "-DUSEARCH_USE_OPENMP=0",
        "-DUSEARCH_USE_SIMSIMD=1",
        "-DSIMSIMD_DYNAMIC_DISPATCH=1",
        "-DUSEARCH_USE_FP16LIB=" + str(int(usearch.USES_FP16LIB)),
    ]
    for name in [
        "NEON",
        "NEON_BF16",
        "NEON_F16",
        "NEON_I8",
        "SVE",
        "SVE_BF16",
        "SVE_F16",
        "SVE_I8",
        "SVE2",
        "HASWELL",
        "SKYLAKE",
        "ICE",
        "SAPPHIRE",
        "GENOA",
    ]:
        flags.append("-DSIMSIMD_TARGET_" + name + "=0")
    if platform.system() == "Darwin":
        flags += ["-undefined", "dynamic_lookup"]
    for folder in ["include", "simsimd/include", "fp16/include"]:
        flags += ["-isystem", str(source / folder)]
    command = [os.environ.get("CXX", "c++"), *flags, str(cpp), "-o", str(library)]
    subprocess.run(command, check=True)

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    record = dict(
        command=command,
        source_commit=head,
        source_sha256=digest(cpp),
        library=str(library),
        library_sha256=digest(library),
        usearch=usearch.__version__,
        simsimd=importlib.metadata.version("simsimd"),
        simsimd_path=simsimd.__file__,
        simsimd_sha256=digest(simsimd.__file__),
        compiler=subprocess.check_output([command[0], "--version"], text=True),
        submodules=subprocess.check_output(
            ["git", "-C", str(source), "submodule", "status"], text=True
        ),
    )
    (a.output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(library)


if __name__ == "__main__":
    main()

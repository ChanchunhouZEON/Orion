"""Shared helpers for artifact-only release validation; never publishes."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

PACKAGES = {
    'orion-ann-error', 'orion-ann-diagnostics', 'orion-ann', 'orion-ann-runtime', 'orion-ann-cli', 'orion-ann-parlay',
    'orion-ann-vector', 'orion-ann-platform', 'orion-ann-diskann', 'orion-ann-logger',
}


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('ORION_') and k not in
           {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CARGO_TARGET_DIR'}}
    # Environment values override repository-specific CPU and linker flags.
    env.update(RUSTFLAGS='', CARGO_ENCODED_RUSTFLAGS='',
               PIP_CACHE_DIR=str(Path(tempfile.gettempdir()) / 'orion-release-pip-cache'),
               PIP_DISABLE_PIP_VERSION_CHECK='1', RUSTUP_TOOLCHAIN='1.92')
    return env


def run(args, cwd, env, capture=False):
    print('+', ' '.join(map(str, args)), flush=True)
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True,
                            text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else None


def extract(archive, destination):
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unexpected archive member: {member.name}')
        source.extractall(destination, members=members, filter='data')
    return [m.name for m in members if m.isfile()]


def audit_names(names):
    for name in names:
        parts = PurePosixPath(name).parts
        assert not set(parts) & {'.git', 'target', '__pycache__', '.venv',
                                'visualizations', 'experiment'}, name
        assert 'docs/release/' not in name and not name.endswith(('.pyc', '.DS_Store')), name


def checksums(directory):
    directory = Path(directory)
    artifacts = sorted(p for p in directory.rglob('*')
                       if p.is_file() and (p.suffix in {'.crate', '.whl'} or p.name.endswith('.tar.gz')))
    assert artifacts, f'No artifacts in {directory}'
    records = {}
    for path in artifacts:
        with path.open('rb') as source:
            records[path.relative_to(directory).as_posix()] = hashlib.file_digest(source, 'sha256').hexdigest()
    (directory / 'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p, h in records.items()))
    (directory / 'artifact-hashes.json').write_text(json.dumps(records, indent=2) + '\n')
    return records

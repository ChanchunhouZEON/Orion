"""Prepare a self-contained Python workspace with an unchanged dependency set.

Maturin trims non-Python workspace members from an sdist but copies the original
Cargo.lock. Cargo can then reject --locked when an omitted member was the only
consumer of a registry dependency (for example serde-wasm-bindgen). Resolve only
that extracted workspace offline, accepting removals but no dependency changes,
then let maturin produce and build the final sdist with --locked as usual.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import tomllib

from common import audit_names, clean_env, extract, run


def validate_pruned_lock(before, after):
    """Reject additions or changes to retained packages, including checksums."""
    original = tomllib.loads(before)
    pruned = tomllib.loads(after)
    if original['version'] != pruned['version']:
        raise ValueError('Pruning changed the Cargo lockfile format')

    def packages(lock):
        return {(p['name'], p['version'], p.get('source')): p
                for p in lock['package']}

    old, new = packages(original), packages(pruned)
    for key, package in new.items():
        if old.get(key) != package:
            raise ValueError(f'Pruning added or changed a locked package: {key}')
    return sorted(f'{name}@{version}' for name, version, _ in old.keys() - new.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'source'
    if destination.exists():
        parser.error(f'Refusing to reuse an existing source workspace: {destination}')
    env = clean_env()
    with tempfile.TemporaryDirectory(prefix='orion-python-source-') as temporary:
        scratch = Path(temporary)
        run(['maturin', 'sdist', '--manifest-path', repo / 'orion-py/Cargo.toml',
             '--out', scratch], repo, env)
        archives = list(scratch.glob('*.tar.gz'))
        if len(archives) != 1:
            raise ValueError(f'Expected one source archive, found {archives}')
        extracted = scratch / 'extracted'
        extracted.mkdir()
        names = extract(archives[0], extracted)
        audit_names(names)
        source = extracted / names[0].split('/')[0]
        lockfile = source / 'Cargo.lock'
        before = lockfile.read_text()
        manifest = source / 'orion-py/Cargo.toml'
        # No registry refresh or version selection is allowed during pruning.
        run(['cargo', 'metadata', '--offline', '--format-version', '1',
             '--manifest-path', manifest], source, env, capture=True)
        removed = validate_pruned_lock(before, lockfile.read_text())
        # The resulting source workspace must now work without any lock changes.
        run(['cargo', 'metadata', '--offline', '--locked', '--format-version', '1',
             '--manifest-path', manifest], source, env, capture=True)
        shutil.copytree(source, destination)
        (output / 'source-lock.json').write_text(json.dumps({
            'status': 'passed', 'removed_packages': removed,
            'retained_packages': 'unchanged versions, sources, checksums and dependencies',
        }, indent=2) + '\n')
        print(f'Prepared {destination}; removed only {len(removed)} unused lock entries')


if __name__ == '__main__':
    main()

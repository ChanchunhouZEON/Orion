# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.


"""Read effective Linux memory limits; never modify cgroup or host settings."""
import json
import os
from pathlib import Path

MIB = 1024 * 1024


def mount_path(cgroup, root, mount):
    """Translate a membership path through the mount's subtree root."""
    cgroup, root, mount = Path(cgroup), Path(root), Path(mount)
    try:
        return mount / cgroup.relative_to(root)
    except ValueError:
        raise RuntimeError('cgroup membership is outside its visible mount root')


def discover():
    memberships = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy, controllers, path = line.split(':', 2)
        if hierarchy == '0' and not controllers:
            memberships.append(('v2', path))
        elif 'memory' in controllers.split(','):
            memberships.append(('v1', path))
    mounts = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        fields, tail = left.split(), right.split()
        if tail[0] == 'cgroup2':
            mounts.append(('v2', fields[3], fields[4]))
        elif tail[0] == 'cgroup' and 'memory' in tail[2].split(','):
            mounts.append(('v1', fields[3], fields[4]))
    for version, member in memberships:
        for kind, root, mount in mounts:
            if version == kind:
                return version, mount_path(member, root, mount), Path(mount)
    raise RuntimeError('No accessible memory cgroup; Slurm --mem alone is insufficient')


def ancestors(path, mount):
    path, mount = Path(path), Path(mount)
    if not path.is_relative_to(mount):
        raise RuntimeError('cgroup escaped mount')
    result = []
    while True:
        result.append(path)
        if path == mount:
            return result
        path = path.parent


def number(path):
    if not path.is_file():
        return None
    value = path.read_text().strip()
    if value == 'max':
        return None
    value = int(value)
    return value if value < (1 << 60) else None


def inspect(version, path, mount, requested, active_swap):
    dirs = ancestors(path, mount)
    filename = 'memory.max' if version == 'v2' else 'memory.limit_in_bytes'
    limits = [(number(p / filename), p) for p in dirs]
    finite = [(n, p) for n, p in limits if n is not None]
    if not finite:
        raise RuntimeError('memory limit is unlimited; refusing an uncontrolled experiment')
    effective, owner = min(finite, key=lambda pair: pair[0])
    if not requested <= effective <= requested + max(MIB, requested // 100):
        raise RuntimeError(f'effective cgroup limit {effective} differs from requested {requested}')
    swap_disabled = not active_swap
    if active_swap and version == 'v2':
        swap_disabled = any(number(p / 'memory.swap.max') == 0 for p in dirs)
    elif active_swap:
        # v1 memsw == memory bounds the combined charge but still permits
        # swapping within that bound. Do not call it a no-swap experiment.
        swap_disabled = False
    if not swap_disabled:
        raise RuntimeError('swap is active without a verified zero swap allowance; request a no-swap job cgroup')
    return {'version': version, 'membership': str(path), 'mount': str(mount),
            'limiting_cgroup': str(owner), 'requested_bytes': requested,
            'effective_limit_bytes': effective, 'swap_disabled': swap_disabled}


def verify(requested):
    if not Path('/proc/self/cgroup').is_file():
        raise RuntimeError('Controlled trials require Linux memory cgroups')
    version, path, mount = discover()
    active_swap = len(Path('/proc/swaps').read_text().splitlines()) > 1
    return inspect(version, path, mount, requested, active_swap)


def snapshot(info):
    result = {}
    for path in ancestors(info['membership'], info['mount']):
        values = {}
        for name in ['memory.current', 'memory.peak', 'memory.events', 'memory.stat',
                     'memory.pressure', 'memory.swap.current', 'io.stat',
                     'memory.usage_in_bytes', 'memory.max_usage_in_bytes',
                     'memory.failcnt', 'memory.memsw.usage_in_bytes']:
            p = path / name
            if p.is_file():
                values[name] = p.read_text().strip()
        result[str(path)] = values
    return result


def process_rss(pid):
    try:
        return {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                if line.startswith(('VmRSS:', 'VmHWM:', 'VmSwap:', 'Threads:'))}
    except (FileNotFoundError, ProcessLookupError):
        return {}

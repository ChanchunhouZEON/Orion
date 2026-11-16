#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Add bounded, thread-local GIL traces to an isolated CPython 3.9.19 tree.

Run once on an unmodified source tree. Never patches an installed interpreter.
The disabled path still has a TLS branch: compare enabled/disabled in this same
interpreter, not against Conda's differently optimized interpreter.
"""
from pathlib import Path
import argparse
import shutil


def patch(root):
    source = root / "Python/ceval_gil.h"
    original = source.read_text()
    if 'orion_gil_probe.h' in original:
        raise RuntimeError("source already instrumented")
    # Version-specific replacements fail closed instead of silently missing hooks.
    def replace(text, old, new):
        if text.count(old) != 1:
            raise RuntimeError(f"expected one occurrence: {old!r}")
        return text.replace(old, new)

    text = replace(original, '#include <errno.h>', '#include <errno.h>\n#include "orion_gil_probe.h"')
    before, take = text.split('static void\ntake_gil(PyThreadState *tstate)', 1)
    before = replace(before, '    MUTEX_LOCK(gil->mutex);\n    _Py_ANNOTATE_RWLOCK_RELEASED',
                     '    orion_probe_emit(7, 0);\n    MUTEX_LOCK(gil->mutex);\n    _Py_ANNOTATE_RWLOCK_RELEASED')
    before = replace(before, '    COND_SIGNAL(gil->cond);',
                     '    orion_probe_emit(8, 0);\n    COND_SIGNAL(gil->cond);')
    before = replace(before, '    MUTEX_UNLOCK(gil->mutex);',
                     '    MUTEX_UNLOCK(gil->mutex);\n    orion_probe_emit(9, 0);')
    before = replace(before, '            COND_WAIT(gil->switch_cond, gil->switch_mutex);',
                     '            orion_probe_emit(10, 0);\n            COND_WAIT(gil->switch_cond, gil->switch_mutex);\n            orion_probe_emit(11, 0);')
    take = replace(take, '    int err = errno;', '    int err = errno;\n    orion_probe_emit(1, 0);')
    take = replace(take, '    MUTEX_LOCK(gil->mutex);',
                   '    MUTEX_LOCK(gil->mutex);\n    orion_probe_emit(2, 0);')
    take = replace(take, '        COND_TIMED_WAIT(gil->cond, gil->mutex, interval, timed_out);',
                   '        orion_probe_emit(3, gil->switch_number);\n        COND_TIMED_WAIT(gil->cond, gil->mutex, interval, timed_out);\n        orion_probe_emit(4, timed_out);')
    take = replace(take, '            SET_GIL_DROP_REQUEST(interp);',
                   '            orion_probe_emit(5, 0);\n            SET_GIL_DROP_REQUEST(interp);')
    take = replace(take, '    _Py_atomic_store_relaxed(&gil->locked, 1);',
                   '    _Py_atomic_store_relaxed(&gil->locked, 1);\n    orion_probe_emit(6, (uintptr_t)tstate);')
    take = replace(take, '    errno = err;', '    orion_probe_emit(12, 0);\n    errno = err;')
    source.with_suffix('.h.original').write_text(original)
    source.write_text(before + 'static void\ntake_gil(PyThreadState *tstate)' + take)
    here = Path(__file__).parent
    shutil.copyfile(here / 'trace.h', root / 'Python/orion_gil_probe.h')
    shutil.copyfile(here / 'module.c', root / 'Modules/orion_gil_probe.c')
    setup = root / 'Modules/Setup.local'
    if setup.exists():
        raise RuntimeError('refusing to replace existing Setup.local')
    setup.write_text('_gilprobe orion_gil_probe.c\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    patch(parser.parse_args().source)

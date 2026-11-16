/* Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 * Experimental CPython instrumentation, not part of Orion's runtime.
 * Record layout: [CLOCK_MONOTONIC ns, request ID, event ID, argument] u64s.
 */
#include <stdint.h>
#include <time.h>

typedef struct { uint64_t ns, request, event, argument; } OrionGilEvent;
typedef struct {
    OrionGilEvent *events;
    size_t count, capacity, dropped;
    uint64_t request;
    int enabled;
} OrionGilTrace;
static _Thread_local OrionGilTrace orion_trace;

static inline void orion_probe_emit(uint64_t event, uint64_t argument) {
    if (!orion_trace.enabled) return;
    if (orion_trace.count == orion_trace.capacity) {
        ++orion_trace.dropped;
        return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    OrionGilEvent *p = &orion_trace.events[orion_trace.count++];
    p->ns = (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;
    p->request = orion_trace.request;
    p->event = event;
    p->argument = argument;
}

/* Only called under the GIL by the built-in module on the owning thread.
 * No allocation, Python objects or shared logging lock on the trace hot path.
 */
PyObject *_OrionGilStart(Py_ssize_t capacity) {
    if (capacity < 1 || capacity > 2000000) {
        PyErr_SetString(PyExc_ValueError, "capacity must be in 1..2000000");
        return NULL;
    }
    OrionGilEvent *p = calloc((size_t)capacity, sizeof(*p));
    if (!p) return PyErr_NoMemory();
    /* Prefault storage before enabling tracing. */
    for (size_t i = 0; i < (size_t)capacity; i += 64) p[i].event = 0;
    free(orion_trace.events);
    orion_trace = (OrionGilTrace){.events=p, .capacity=(size_t)capacity};
    PyObject *identity = PyLong_FromVoidPtr(PyThreadState_Get());
    if (!identity) { free(p); orion_trace.events = NULL; return NULL; }
    orion_trace.enabled = 1;
    return identity;
}
PyObject *_OrionGilFinish(void) {
    orion_trace.enabled = 0;
    PyObject *bytes = PyBytes_FromStringAndSize((char *)orion_trace.events,
                      (Py_ssize_t)(orion_trace.count * sizeof(OrionGilEvent)));
    size_t dropped = orion_trace.dropped;
    free(orion_trace.events);
    orion_trace = (OrionGilTrace){0};
    if (!bytes) return NULL;
    return Py_BuildValue("Nn", bytes, (Py_ssize_t)dropped);
}
void _OrionGilBegin(uint64_t id) {
    orion_trace.request = id;
    orion_probe_emit(13, 0);
}
void _OrionGilEnd(void) {
    orion_probe_emit(14, 0);
    orion_trace.request = 0;
}

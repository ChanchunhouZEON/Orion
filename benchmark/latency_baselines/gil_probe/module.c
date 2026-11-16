/* Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 * Private built-in control API for the isolated diagnostic interpreter.
 */
#include "Python.h"
#include <stdint.h>
extern PyObject *_OrionGilStart(Py_ssize_t);
extern PyObject *_OrionGilFinish(void);
extern void _OrionGilBegin(uint64_t);
extern void _OrionGilEnd(void);

static PyObject *start(PyObject *self, PyObject *arg) {
    Py_ssize_t capacity = PyLong_AsSsize_t(arg);
    if (capacity == -1 && PyErr_Occurred()) return NULL;
    return _OrionGilStart(capacity);
}
static PyObject *finish(PyObject *self, PyObject *unused) {
    return _OrionGilFinish();
}
static PyObject *begin(PyObject *self, PyObject *arg) {
    unsigned long long id = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) return NULL;
    _OrionGilBegin(id);
    Py_RETURN_NONE;
}
static PyObject *end(PyObject *self, PyObject *unused) {
    _OrionGilEnd();
    Py_RETURN_NONE;
}
static PyMethodDef methods[] = {
    {"start", start, METH_O, "Enable bounded TLS events, return tstate identity."},
    {"finish", finish, METH_NOARGS, "Disable and free TLS buffer; return bytes,dropped."},
    {"begin", begin, METH_O, "Tag following events with request ID."},
    {"end", end, METH_NOARGS, "End current request tag."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_gilprobe", "Private experimental GIL tracing.", -1, methods
};
PyMODINIT_FUNC PyInit__gilprobe(void) { return PyModule_Create(&module); }

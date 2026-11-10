# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""The CUDA driver calls spark_attic makes, through ctypes on libcuda.so.1 (no build step). Every call runs with the
device's primary context pushed, the context PyTorch and the CUDA runtime use, so it works from any thread."""
import ctypes
import os
import threading

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0


class CudaError(Exception):
    """A CUDA driver call failed."""


class _Location(ctypes.Structure):  # CUmemLocation
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _Prop(ctypes.Structure):  # CUmemAllocationProp, 32 bytes
    _fields_ = [("type", ctypes.c_int), ("handle_types", ctypes.c_int), ("location", _Location),
                ("win32_metadata", ctypes.c_void_p), ("alloc_flags", ctypes.c_ubyte * 8)]


class _Access(ctypes.Structure):  # CUmemAccessDesc, 12 bytes
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


_U64 = ctypes.c_ulonglong
_SIGNATURES = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    "cuCtxPushCurrent_v2": [ctypes.c_void_p],
    "cuCtxPopCurrent_v2": [ctypes.POINTER(ctypes.c_void_p)],
    "cuCtxSynchronize": [],
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    "cuMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(_Prop), ctypes.c_int],
    "cuMemImportFromShareableHandle": [ctypes.POINTER(_U64), ctypes.c_void_p, ctypes.c_int],
    "cuMemCreate": [ctypes.POINTER(_U64), ctypes.c_size_t, ctypes.POINTER(_Prop), _U64],
    "cuMemAddressReserve": [ctypes.POINTER(_U64), ctypes.c_size_t, ctypes.c_size_t, _U64, _U64],
    "cuMemMap": [_U64, ctypes.c_size_t, ctypes.c_size_t, _U64, _U64],
    "cuMemSetAccess": [_U64, ctypes.c_size_t, ctypes.POINTER(_Access), ctypes.c_size_t],
    "cuMemUnmap": [_U64, ctypes.c_size_t],
    "cuMemAddressFree": [_U64, ctypes.c_size_t],
    "cuMemRelease": [_U64],
    "cuMemsetD8_v2": [_U64, ctypes.c_ubyte, ctypes.c_size_t],
    "cuMemcpyHtoD_v2": [_U64, ctypes.c_void_p, ctypes.c_size_t],
    "cuMemcpyDtoH_v2": [ctypes.c_void_p, _U64, ctypes.c_size_t],
}

_lib = None
_contexts = {}
_lock = threading.Lock()


def _driver():
    global _lib
    with _lock:
        if _lib is None:
            try:
                lib = ctypes.CDLL(os.environ.get("SPARK_ATTIC_LIBCUDA") or "libcuda.so.1")
            except OSError as e:
                raise CudaError("no CUDA driver (libcuda.so.1): %s" % e)
            for name, args in _SIGNATURES.items():
                try:
                    f = getattr(lib, name)
                except AttributeError:
                    raise CudaError("the CUDA driver has no %s" % name)
                f.argtypes = args
                f.restype = ctypes.c_int
            rc = lib.cuInit(0)
            if rc:
                raise CudaError("cuInit: error %d" % rc)
            _lib = lib
        return _lib


def _check(rc, what):
    if rc:
        name = ctypes.c_char_p()
        _lib.cuGetErrorName(rc, ctypes.byref(name))
        raise CudaError("%s: %s" % (what, name.value.decode() if name.value else "error %d" % rc))


class _Current:
    """with _Current(device): the device's primary context is current inside, the caller's again after."""

    def __init__(self, device):
        lib = _driver()
        with _lock:
            ctx = _contexts.get(device)
            if ctx is None:
                dev = ctypes.c_int()
                _check(lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet(%d)" % device)
                ctx = ctypes.c_void_p()
                _check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value), "cuDevicePrimaryCtxRetain")
                _contexts[device] = ctx  # kept for the life of the process, as the CUDA runtime keeps it
        self.ctx = ctx

    def __enter__(self):
        _check(_lib.cuCtxPushCurrent_v2(self.ctx), "cuCtxPushCurrent")
        return _lib

    def __exit__(self, *exc):
        _lib.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))


def _prop(device, handle_types=CU_MEM_HANDLE_TYPE_NONE):
    p = _Prop()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.handle_types = handle_types
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = device
    return p


def granularity(device):
    with _Current(device) as lib:
        g = ctypes.c_size_t()
        p = _prop(device, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
        _check(lib.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p), CU_MEM_ALLOC_GRANULARITY_MINIMUM),
               "cuMemGetAllocationGranularity")
        return g.value


def import_fd(fd, device):
    """A lent slice's descriptor as a CUDA memory handle (the caller still closes fd)."""
    with _Current(device) as lib:
        h = _U64()
        _check(lib.cuMemImportFromShareableHandle(ctypes.byref(h), ctypes.c_void_p(fd),
                                                  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
               "cuMemImportFromShareableHandle")
        return h.value


def create(nbytes, device):
    """Ordinary device memory as a CUDA memory handle (nbytes a multiple of the granularity)."""
    with _Current(device) as lib:
        h = _U64()
        p = _prop(device)
        _check(lib.cuMemCreate(ctypes.byref(h), nbytes, ctypes.byref(p), 0), "cuMemCreate(%d MiB)" % (nbytes >> 20))
        return h.value


def release(handle, device):
    with _Current(device) as lib:
        _check(lib.cuMemRelease(handle), "cuMemRelease")


def map_parts(device, parts, align=2 << 20):
    """Reserve one address range for all parts [(handle, nbytes)], map them back to back, make it read-write for the
    device. Returns its address; on failure undoes what it did (the handles stay the caller's)."""
    total = sum(n for _, n in parts)
    with _Current(device) as lib:
        va = _U64()
        _check(lib.cuMemAddressReserve(ctypes.byref(va), total, align, 0, 0), "cuMemAddressReserve")
        done = []
        try:
            at = va.value
            for h, n in parts:
                _check(lib.cuMemMap(at, n, 0, h, 0), "cuMemMap")
                done.append((at, n))
                at += n
            acc = _Access()
            acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
            acc.location.id = device
            acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check(lib.cuMemSetAccess(va.value, total, ctypes.byref(acc), 1), "cuMemSetAccess")
        except CudaError:
            for a, n in done:
                lib.cuMemUnmap(a, n)
            lib.cuMemAddressFree(va.value, total)
            raise
        return va.value


def unmap_parts(device, va, sizes):
    """Undo map_parts: unmap each part, then free the range. Returns the first error, or None."""
    err = None
    with _Current(device) as lib:
        at = va
        for n in sizes:
            rc = lib.cuMemUnmap(at, n)
            err = err or (rc and "cuMemUnmap: error %d" % rc)
            at += n
        rc = lib.cuMemAddressFree(va, sum(sizes))
        return err or (rc and "cuMemAddressFree: error %d" % rc) or None


def memset(device, va, value, nbytes):
    with _Current(device) as lib:
        _check(lib.cuMemsetD8_v2(va, value, nbytes), "cuMemsetD8")
        _check(lib.cuCtxSynchronize(), "cuCtxSynchronize")


def write(device, va, data):
    with _Current(device) as lib:
        buf = ctypes.create_string_buffer(bytes(data), len(data))
        _check(lib.cuMemcpyHtoD_v2(va, buf, len(data)), "cuMemcpyHtoD")


def read(device, va, nbytes):
    with _Current(device) as lib:
        buf = ctypes.create_string_buffer(nbytes)
        _check(lib.cuMemcpyDtoH_v2(buf, va, nbytes), "cuMemcpyDtoH")
        return buf.raw

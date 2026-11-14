# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""spark_attic: borrow a GB10's display carveout from atticd as CUDA device memory. Pure Python (ctypes on libcuda),
no build step.

    import spark_attic
    m = spark_attic.borrow()             # the largest free run, on the current CUDA device; None if there is none
    if m is not None:
        x = m.tensor(torch.float16)      # a torch tensor over it (or m.ptr, m.size for anything else)

borrow() and Pool.borrow_or_none() fail open: no atticd, nothing free, or a failed step gives None and one log line
(logger "spark_attic"), never an exception. The memory stays lent until the Mapping is closed or the process exits.
A borrow() comes back zeroed: atticd hands a slice on with whatever its previous holder left in it.
Settings: SPARK_ATTIC_SOCK (/run/spark-attic/attic.sock), SPARK_ATTIC_KEY or SPARK_ATTIC_KEY_FILE (no key is compiled in), SPARK_ATTIC_TIMEOUT_MS.
"""
import logging
import os
import sys
import weakref

from . import _cuda
from ._cuda import CudaError
from ._proto import DEFAULT_SOCK, SLICE, Attic, AtticError, Settings, key_ok

__version__ = "0.1.0"
__all__ = ["Attic", "AtticError", "CudaError", "Lent", "Mapping", "Pool", "Settings", "borrow", "info", "SLICE",
           "DEFAULT_SOCK", "key_ok"]
log = logging.getLogger("spark_attic")
MIB = 1 << 20


def _round_up(n, to):
    return (n + to - 1) // to * to


def _current_device():
    """The CUDA device PyTorch uses now, if PyTorch is loaded and has CUDA up; else 0."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return torch.cuda.current_device()
    except Exception:
        pass
    return 0


def _release(device, va, parts, attic):
    try:
        if va is not None:
            err = _cuda.unmap_parts(device, va, [n for _, n, _ in parts])
            if err:
                log.warning("spark-attic: unmapping %d MiB: %s", sum(n for _, n, _ in parts) // MIB, err)
        for h, _, _ in parts:
            try:
                _cuda.release(h, device)
            except CudaError as e:
                log.warning("spark-attic: %s", e)
    finally:
        if attic is not None:
            attic.close()


class Mapping:
    """One contiguous device address range: optional ordinary device memory first, then attic slices.

    ptr, size: the range; attic_bytes: how much of its tail is display carveout; device: the CUDA device ordinal.
    """

    def __init__(self, device, va, parts, attic=None):
        self.device = device
        self.ptr = va
        self.size = sum(n for _, n, _ in parts)
        self.attic_bytes = sum(n for _, n, kind in parts if kind == "attic")
        self._fin = weakref.finalize(self, _release, device, va, list(parts), attic)
        self._fin.atexit = False  # at exit the driver and atticd take everything back by themselves

    @property
    def closed(self):
        return not self._fin.alive

    def close(self):
        """Unmap and give the slices back (tensors made by tensor() must be gone first)."""
        self._fin()

    def zero(self):
        _cuda.memset(self.device, self.ptr, 0, self.size)

    def write(self, offset, data):
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("outside the mapping")
        _cuda.write(self.device, self.ptr + offset, data)

    def read(self, offset, nbytes):
        if offset < 0 or nbytes < 0 or offset + nbytes > self.size:
            raise ValueError("outside the mapping")
        return _cuda.read(self.device, self.ptr + offset, nbytes)

    def tensor(self, dtype=None, shape=None, offset=0, nbytes=None):
        """A torch tensor over [offset, offset + nbytes) of the range (all of it by default), as dtype (uint8) and
        shape (flat). The tensor keeps this Mapping alive; zero copy, through __cuda_array_interface__."""
        import torch

        dtype = dtype or torch.uint8
        nbytes = self.size - offset if nbytes is None else nbytes
        if offset < 0 or nbytes <= 0 or offset + nbytes > self.size:
            raise ValueError("outside the mapping")
        raw = torch.as_tensor(_Owner(self, self.ptr + offset, nbytes), device="cuda:%d" % self.device)
        t = raw.view(dtype)
        return t if shape is None else t.view(shape)

    def __repr__(self):
        return "<spark_attic.Mapping %d MiB at 0x%x on cuda:%d, %d MiB of it display carveout%s>" % (
            self.size // MIB, self.ptr, self.device, self.attic_bytes // MIB, ", closed" if self.closed else "")


class _Owner:
    """What a tensor over a Mapping holds on to (PyTorch keeps a reference to the interface's object)."""

    def __init__(self, mapping, ptr, nbytes):
        self.mapping = mapping
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None, "stream": None}


class Lent:
    """One slice borrowed from atticd and imported as a CUDA memory handle, not mapped yet. Pool.map() takes it over;
    release() gives it back unmapped."""

    def __init__(self, device, handle, size):
        self.device = device
        self.handle = handle
        self.size = size

    def take(self):
        h, self.handle = self.handle, None
        if h is None:
            raise CudaError("this lent slice was mapped or released already")
        return h

    def release(self):
        if self.handle is not None:
            _cuda.release(self.take(), self.device)

    def __repr__(self):
        return "<spark_attic.Lent %d MiB on cuda:%d%s>" % (self.size // MIB, self.device,
                                                          "" if self.handle is not None else ", taken")


class Pool:
    """One atticd connection on one CUDA device. Its mappings share the connection; close() gives every slice back
    (close the mappings, and drop their tensors, first)."""

    def __init__(self, settings=None, device=None):
        self.device = _current_device() if device is None else device
        self.attic = Attic(settings)
        try:
            g = _cuda.granularity(self.device)
        except Exception:
            self.attic.close()
            raise
        if SLICE % g:
            self.attic.close()
            raise CudaError("this device maps in %d KiB steps; atticd lends 2 MiB slices" % (g >> 10))
        self.granularity = SLICE  # every part of a mapping is whole 2 MiB slices

    def info(self):
        return self.attic.info()

    def _slice(self, nbytes):
        fd = self.attic.alloc(nbytes)
        try:
            return _cuda.import_fd(fd, self.device)
        finally:
            os.close(fd)

    def lend(self, nbytes):
        """Borrow nbytes (whole 2 MiB slices) as one slice, imported but not mapped: a Lent for map()."""
        return Lent(self.device, self._slice(nbytes), nbytes)

    def map(self, front, attic_slices, owns_attic=False):
        """A Mapping of `front` bytes of ordinary device memory followed by the attic slices: each a size (a multiple
        of 2 MiB, borrowed now) or a Lent from lend(), which the mapping takes over, or releases if map() fails."""
        parts = []
        try:
            if front:
                parts.append((_cuda.create(_round_up(front, self.granularity), self.device),
                              _round_up(front, self.granularity), "device"))
            for s in attic_slices:
                if isinstance(s, Lent):
                    parts.append((s.take(), s.size, "attic"))
                else:
                    parts.append((self._slice(s), s, "attic"))
            va = _cuda.map_parts(self.device, [(h, n) for h, n, _ in parts], self.granularity)
        except Exception:
            _release(self.device, None, parts, None)
            for s in attic_slices:
                if isinstance(s, Lent):
                    s.release()
            raise
        return Mapping(self.device, va, parts, self.attic if owns_attic else None)

    def borrow(self, nbytes=None):
        """nbytes (rounded up to 2 MiB; default the largest free run) of display carveout as one Mapping, zeroed:
        atticd hands a slice on with whatever its previous holder left in it, and the documented API clears it."""
        if nbytes is None:
            nbytes = self.info()["largest"]
            if nbytes <= 0:
                raise AtticError("atticd has nothing free")
        m = self.map(0, [_round_up(nbytes, SLICE)])
        try:
            m.zero()
        except Exception:
            m.close()
            raise
        return m

    def borrow_or_none(self, nbytes=None):
        try:
            return self.borrow(nbytes)
        except (AtticError, CudaError, OSError) as e:
            log.warning("spark-attic: no display carveout: %s", e)
            return None

    def close(self):
        self.attic.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def borrow(nbytes=None, device=None, settings=None):
    """Fail open: a Mapping of nbytes (default: the largest free run) of display carveout on `device` (default: the
    current CUDA device), with its own atticd connection that close() ends; or None and one log line."""
    try:
        pool = Pool(settings, device)
    except (AtticError, CudaError, OSError) as e:
        log.warning("spark-attic: no display carveout: %s", e)
        return None
    try:
        if nbytes is None:
            nbytes = pool.info()["largest"]
            if nbytes <= 0:
                raise AtticError("atticd has nothing free")
        m = pool.map(0, [_round_up(nbytes, SLICE)], owns_attic=True)
        try:
            m.zero()  # atticd hands a slice on with the previous holder'"'"'s bytes in it; the API clears them
        except Exception:
            m.close()
            raise
        log.info("spark-attic: %d MiB of display carveout at 0x%x on cuda:%d", m.size // MIB, m.ptr, m.device)
        return m
    except (AtticError, CudaError, OSError) as e:
        pool.close()
        log.warning("spark-attic: no display carveout: %s", e)
        return None


def info(settings=None):
    """atticd's {"size", "free", "largest"}, or None (fail open)."""
    try:
        with Attic(settings) as a:
            return a.info()
    except AtticError as e:
        log.warning("spark-attic: %s", e)
        return None

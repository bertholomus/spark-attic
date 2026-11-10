# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""A stand-in torch for the hook tests (no GPU, no torch): tensors keep their bytes in host memory (torch.zeros, counted
as allocated while alive) or at a fake-CUDA address read and written through the fake libcuda (torch.as_tensor on an
array interface). torch.zeros goes through active TorchFunctionModes as real torch does."""
import ctypes
import functools
import threading
import types
import weakref

from spark_attic import _cuda

_modes = threading.local()


def _stack():
    if not hasattr(_modes, "stack"):
        _modes.stack = []
    return _modes.stack


class TorchFunctionMode:
    def __enter__(self):
        _stack().append(self)
        return self

    def __exit__(self, *exc):
        _stack().pop()


class DType:
    def __init__(self, name, size):
        self.name, self.itemsize = name, size

    def __repr__(self):
        return "torch." + self.name


class Device:
    def __init__(self, spec, index=None):
        if isinstance(spec, Device):
            spec, index = spec.type, spec.index
        elif ":" in spec:
            spec, index = spec.split(":")[0], int(spec.split(":")[1])
        self.type, self.index = spec, index

    def __repr__(self):
        return "%s:%s" % (self.type, self.index)


class Tensor:
    """Bytes in host memory (torch.zeros: "buf", counted as allocated while alive) or at a fake-CUDA address
    (torch.as_tensor on an array interface: "va")."""

    def __init__(self, kind, nbytes, dtype, shape=None, buf=None, va=None, owner=None, base=None):
        self.kind, self.nbytes, self.dtype, self.buf, self.va, self.owner = kind, nbytes, dtype, buf, va, owner
        self.shape = tuple(shape) if shape is not None else (nbytes // dtype.itemsize,)
        self.base = base  # a view keeps its source alive
        self.is_cuda = True

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self):
        return self.dtype.itemsize

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self.va if self.kind == "va" else ctypes.addressof(self.buf)

    def view(self, x):
        src = self.base or self
        if isinstance(x, DType):
            assert self.nbytes % x.itemsize == 0
            return Tensor(self.kind, self.nbytes, x, None, self.buf, self.va, self.owner, src)
        shape = tuple(x)
        t = Tensor(self.kind, self.nbytes, self.dtype, shape, self.buf, self.va, self.owner, src)
        assert t.numel() * self.dtype.itemsize == self.nbytes, (shape, self.nbytes)
        return t

    def raw(self):
        return bytes(self.buf) if self.kind == "buf" else _cuda.read(0, self.va, self.nbytes)

    def put(self, data):
        assert len(data) == self.nbytes
        if self.kind == "buf":
            ctypes.memmove(self.buf, data, len(data))
        else:
            _cuda.write(0, self.va, data)

    def copy_(self, src):
        self.put(src.raw())
        return self


class Scalar:
    def __init__(self, value, dtype):
        self.value, self.dtype = value, dtype

    def item(self):
        return self.value


class Parameter:
    def __init__(self, t):
        self._t = t

    @property
    def data(self):
        return self._t

    @data.setter
    def data(self, t):
        self._t = t

    def __getattr__(self, name):
        return getattr(self.__dict__["_t"], name)


class FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.int8, self.uint8, self.int64 = DType("int8", 1), DType("uint8", 1), DType("int64", 8)
        self.float16, self.bfloat16 = DType("float16", 2), DType("bfloat16", 2)
        self.device = Device
        self.allocated = 0
        cuda = types.ModuleType("torch.cuda")
        cuda.synchronize = lambda device=None: None
        cuda.empty_cache = lambda: None
        cuda.current_device = lambda: 0
        cuda.is_initialized = lambda: True
        cuda.memory_allocated = lambda device=None: self.allocated
        self.cuda = cuda
        self.overrides = types.ModuleType("torch.overrides")
        self.overrides.TorchFunctionMode = TorchFunctionMode
        dist = types.ModuleType("torch.distributed")
        dist.ReduceOp = types.SimpleNamespace(MIN="min")
        dist.all_reduce = self._all_reduce
        self.distributed = dist
        self.zeros = self._zeros  # one object, so `func is torch.zeros` holds as in torch

    def _free(self, n):
        self.allocated -= n

    def _zeros(self, *size, dtype=None, device=None):
        stack = _stack()
        if stack:  # the innermost mode handles it, with itself popped meanwhile
            mode = stack.pop()
            try:
                return mode.__torch_function__(self.zeros, (), size, {"dtype": dtype, "device": device})
            finally:
                stack.append(mode)
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list)) else tuple(size)
        dtype = dtype or self.uint8
        n = dtype.itemsize
        for d in shape:
            n *= d
        buf = (ctypes.c_char * n)()
        t = Tensor("buf", n, dtype, shape=shape, buf=buf)
        self.allocated += n
        weakref.finalize(buf, self._free, n)
        return t

    @staticmethod
    def inference_mode(mode=True):
        """As torch's decorator: its wrapper takes (*args, **kwargs) and keeps the function as __wrapped__."""
        def decorate(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorate

    def tensor(self, value, dtype=None, device=None):
        return Scalar(value, dtype)

    def _all_reduce(self, t, op=None, group=None):
        """The group stands in for the other ranks: group.reduce(op, value) gives the result."""
        t.value = group.reduce(op, t.value)

    def as_tensor(self, obj, device=None):
        cai = obj.__cuda_array_interface__
        assert cai["typestr"] == "|u1" and cai["version"] == 3 and str(device) == "cuda:0"
        return Tensor("va", cai["shape"][0], self.uint8, va=cai["data"][0], owner=obj)

    def param(self, nbytes, dtype, shape, seed):
        t = self.zeros(tuple(shape), dtype=dtype)
        assert t.nbytes == nbytes
        t.put(bytes((i * 31 + seed) & 0xFF for i in range(nbytes)))
        return Parameter(t)


TORCH = FakeTorch()


class Model:
    def __init__(self, named):
        self.named = named  # [(name, Parameter)], a tied one listed under each name

    def named_parameters(self, remove_duplicate=True):
        seen = set()
        for name, p in self.named:
            if remove_duplicate and id(p) in seen:
                continue
            seen.add(id(p))
            yield name, p

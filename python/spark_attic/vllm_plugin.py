# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""The vLLM plugin (entry point vllm.general_plugins, name spark_attic): a GB10's display carveout as more KV cache,
or as the home of named weights, in vLLM's GPU worker. No change to vLLM's files: it wraps the functions that make and
fill the KV cache as their modules load, in the processes that load them.

    SPARK_ATTIC_PLACE    kv (default), weights, weights,kv, or off
    SPARK_ATTIC_WEIGHTS  for weights: parameter names or globs, comma-separated (default embed_tokens)
    SPARK_ATTIC_MIB      at most this many MiB of carveout (default: all atticd has free)
    SPARK_ATTIC_LOG      0: no log handler of our own (default: our lines on stderr, logger spark_attic)
    and the client's SPARK_ATTIC_SOCK, SPARK_ATTIC_KEY or SPARK_ATTIC_KEY_FILE, SPARK_ATTIC_TIMEOUT_MS.

There is no pinned vLLM version. The plugin knows two shapes of vLLM's KV allocation and, in each worker process,
picks the one the running engine really has, by what its modules expose (the tables below; tests/test_vllm_plugin.py
proves each shape against vLLM's own source for every supported release, and scripts/attic-engine-drift.sh runs that
test against the newest release tags):

  0.24: Worker.determine_available_memory raises the KV budget; GPUModelRunner._allocate_kv_cache_tensors (V1) and
        attn_utils._allocate_kv_cache (V2) place the last KV tensors in the carveout. KVCacheTensor carries
        size/shared_by/offset/block_stride; shared_by lists what shares a backing.
  0.32: the same Worker hook; vllm.v1.worker.utils.allocate_kv_cache is the one allocation both runners go through,
        one backing tensor for the whole cache, so the carveout holds the tail of it. KVCacheTensor carries
        size/layers/layer_stride/offset/block_stride/host_resident.

kv: Worker.determine_available_memory borrows the carveout and adds it to what vLLM measured (or to
--kv-cache-memory-bytes, which then holds device memory), so vLLM sizes the KV pool larger by exactly that; the KV
allocation then puts the tail of the cache in the carveout and the rest where vLLM puts them. weights: after
Worker.load_model the named parameters move into the carveout, and vLLM's weight figure drops by what that freed, so
vLLM's own profile gives the KV pool that much more. weights,kv: weights first, the rest to the KV pool.

Fail open until the budget is raised, loud after it: anything seen before determine_available_memory (a shape the
plugin does not know, a mismatched signature, sleep mode, a KV connector, no atticd, nothing free) turns that
placement off and vLLM runs as it would without the plugin (one log line, logger "spark_attic"). Once the plugin has
added the carveout to the KV budget, a failed placement can no longer fail open: falling back would leave vLLM
allocating the KV cache above the memory it measured, so the plugin gives the carveout back and raises instead, and
vLLM stops coming up until the plugin is turned off (SPARK_ATTIC_PLACE=off).
"""
import functools
import inspect
import logging
import os
import sys
import threading

from . import MIB, Pool, Settings
from ._place import (Ledger, Mismatch, Placement, device_index, give_back, log_to_stderr, model_of, move_params,
                     params, tail_plan, when_imported)

log = logging.getLogger("spark_attic")
GIB = 1 << 30

WORKER = "vllm.v1.worker.gpu_worker"
RUNNER_V1 = "vllm.v1.worker.gpu_model_runner"
RUNNER_V2 = "vllm.v1.worker.gpu.model_runner"
ATTN_V2 = "vllm.v1.worker.gpu.attn_utils"
UTILS_V3 = "vllm.v1.worker.utils"
KV_IFACE = "vllm.v1.kv_cache_interface"

# The engine shapes the plugin knows, oldest first. SHAPE_RELEASE names the release each shape was read from, for
# log lines and for the upstream-source test.
SHAPES = ("0.24", "0.32")
SHAPE_RELEASE = {"0.24": "0.24.0", "0.32": "0.32.0"}

# The parameters, in order, of each function the plugin wraps, per shape. A key may appear in several shapes (the
# Worker hooks) with the same parameters.
EXPECTED = {
    "0.24": {
        (WORKER, "Worker.determine_available_memory"): ["self"],
        (WORKER, "Worker.load_model"): ["self", "load_dummy_weights"],
        (RUNNER_V1, "GPUModelRunner._allocate_kv_cache_tensors"): ["self", "kv_cache_config"],
        (ATTN_V2, "_allocate_kv_cache"): ["kv_cache_config", "shared_layers", "device"],
    },
    "0.32": {
        (WORKER, "Worker.determine_available_memory"): ["self"],
        (WORKER, "Worker.load_model"): ["self", "load_dummy_weights"],
        (UTILS_V3, "allocate_kv_cache"): ["kv_cache_config", "device", "layout", "kernel_block_sizes"],
    },
}
# KVCacheTensor's fields per shape; one we do not know may change what a KV tensor's bytes mean.
KV_TENSOR_FIELDS = {
    "0.24": frozenset({"size", "shared_by", "offset", "block_stride"}),
    "0.32": frozenset({"size", "layers", "layer_stride", "offset", "block_stride", "host_resident"}),
}


class _State:
    def __init__(self):
        self.registered = False
        self.placement = Placement(place=())
        self.settings = None
        self.pool = None          # one atticd connection for the life of the worker
        self.ledger = Ledger()    # what SPARK_ATTIC_MIB still allows
        self.kv = None            # the Lent for the KV cache, between determine_available_memory and the allocation
        self.kv_counted = 0       # bytes of carveout determine_available_memory added to what vLLM measured
        self.kv_done = False      # the KV allocation ran (it gets one try)
        self.shape = None         # which engine shape the KV placement runs as
        self.wrapped = set()      # the wrapped keys in this process
        self.errors = []          # why each wrap attempt did not wrap, for the placement's one line
        self.keep = []            # mappings that live as long as the worker
        self.lock = threading.RLock()


S = _State()


# ------------------------------------------------------------------------------------------------- registration

def register():
    """The entry point vLLM calls in each of its processes. Never raises: vLLM starts either way."""
    try:
        log_to_stderr()
        with S.lock:
            if S.registered:
                return
            S.registered = True
            S.placement = Placement.from_env(os.environ)
            S.ledger = Ledger(S.placement.cap)
            if not S.placement.place:
                log.info("spark-attic: vLLM plugin off (SPARK_ATTIC_PLACE=off)")
                return
            S.settings = Settings.from_env()
        when_imported(WORKER, _wrap_worker)
        if "kv" in S.placement.place:
            when_imported(RUNNER_V1, _wrap_runner_v1)
            when_imported(ATTN_V2, _wrap_attn_v2)
            when_imported(UTILS_V3, _wrap_utils)
    except Exception as e:
        S.placement = Placement(place=())
        log.warning("spark-attic: vLLM plugin off: %s", e)


def _try_wrap(owner, attr, key, make):
    """Wraps owner.attr when its parameters are ones a shape expects; records why not, and logs nothing: the KV
    placement decides once, in the worker, and says why in one line."""
    fn = getattr(owner, attr, None)
    if fn is None:
        why = "%s has no %s" % (key[0], key[1])
    elif getattr(fn, "__spark_attic__", False):
        S.wrapped.add(key)
        return True
    else:
        got = params(fn)
        wants = {tuple(EXPECTED[s][key]): s for s in SHAPES if key in EXPECTED[s]}
        if tuple(got) in wants:
            new = functools.wraps(fn)(make(fn))
            new.__spark_attic__ = True
            setattr(owner, attr, new)
            S.wrapped.add(key)
            return True
        known = " or ".join("(%s) from vLLM %s" % (", ".join(w), SHAPE_RELEASE[s]) for w, s in wants.items())
        why = "%s.%s takes (%s); the plugin knows %s" % (key[0], key[1], ", ".join(got), known)
    if why not in S.errors:  # the same hook re-runs per engine process: say it once
        S.errors.append(why)
    return False


def _wrap_worker(mod):
    worker = getattr(mod, "Worker", None)
    if worker is None:
        S.errors.append("%s has no Worker" % WORKER)
        return
    if "kv" in S.placement.place:
        _try_wrap(worker, "determine_available_memory", (WORKER, "Worker.determine_available_memory"),
                  _make_determine)
    if "weights" in S.placement.place:
        try:
            _wrap_load_model(worker)
        except Mismatch as e:
            log.warning("spark-attic: weights placement off: %s", e)


def _wrap_load_model(worker):
    """The weights wrap logs its own off line: nothing was raised yet, so failing open is safe."""
    fn = getattr(worker, "load_model", None)
    key = (WORKER, "Worker.load_model")
    if fn is None:
        raise Mismatch("%s has no %s" % key)
    if getattr(fn, "__spark_attic__", False):
        S.wrapped.add(key)
        return
    got = params(fn)
    if got != EXPECTED["0.24"][key]:
        raise Mismatch("%s.%s takes (%s); the plugin knows (%s) from vLLM %s" % (
            key[0], key[1], ", ".join(got), ", ".join(EXPECTED["0.24"][key]),
            " or ".join(SHAPE_RELEASE[s] for s in SHAPES)))
    new = functools.wraps(fn)(_make_load_model(fn))
    new.__spark_attic__ = True
    setattr(worker, "load_model", new)
    S.wrapped.add(key)


def _wrap_runner_v1(mod):
    runner = getattr(mod, "GPUModelRunner", None)
    if runner is None:
        S.errors.append("%s has no GPUModelRunner" % RUNNER_V1)
        return
    _try_wrap(runner, "_allocate_kv_cache_tensors", (RUNNER_V1, "GPUModelRunner._allocate_kv_cache_tensors"),
              _make_alloc_v1)


def _wrap_attn_v2(mod):
    _try_wrap(mod, "_allocate_kv_cache", (ATTN_V2, "_allocate_kv_cache"), _make_alloc_v2)


def _wrap_utils(mod):
    _try_wrap(mod, "allocate_kv_cache", (UTILS_V3, "allocate_kv_cache"), _make_alloc_v3)


# ------------------------------------------------------------------------------------------------------ helpers

def _pool(device):
    with S.lock:
        if S.pool is None:
            S.pool = Pool(S.settings, device_index(device))
        return S.pool


def _blocked(worker):
    """Why the carveout must stay out of this worker's memory, or None."""
    cfg = getattr(worker, "vllm_config", None)
    if cfg is None:
        return "the worker has no vllm_config"
    if getattr(getattr(cfg, "model_config", None), "enable_sleep_mode", False):
        return "sleep mode is on (vLLM's own CuMem pools hold the memory)"
    return None


def _module(name):
    return sys.modules.get(name)


def _kv_fields_unknown():
    """Field names of vLLM's live KVCacheTensor that no shape reads, or None when it is fine or unreadable."""
    try:
        from vllm.v1 import kv_cache_interface

        fields = frozenset(getattr(kv_cache_interface.KVCacheTensor, "__dataclass_fields__", {}))
    except Exception:
        return None
    known = set().union(*KV_TENSOR_FIELDS.values())
    return fields - known or None


def _kv_shape(worker):
    """Which shape this worker's KV allocation is, and that its every step is wrapped: (shape, None) or (None, why)."""
    why = _blocked(worker)
    if why:
        return None, why
    if getattr(worker.vllm_config, "kv_transfer_config", None) is not None:
        return None, "a KV connector is configured (it may lay the KV cache out itself)"
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return None, "the worker has no model runner"
    unknown = _kv_fields_unknown()
    if unknown:
        return None, "KVCacheTensor has fields the plugin does not know: %s" % ", ".join(sorted(unknown))
    kind = type(runner).__module__
    utils_alloc = getattr(_module(UTILS_V3), "allocate_kv_cache", None)
    if getattr(utils_alloc, "__spark_attic__", False):
        # 0.32: every runner calls the wrapped allocate_kv_cache through its own module's binding.
        if kind == RUNNER_V2 or kind.startswith("vllm.v1.worker.gpu."):
            attn = _module(ATTN_V2)
            if not getattr(getattr(attn, "allocate_kv_cache", None), "__spark_attic__", False) or not _calls(
                    getattr(attn, "init_kv_cache", None), "allocate_kv_cache"):
                return None, "vLLM's V2 KV allocation is not routed through %s.allocate_kv_cache" % UTILS_V3
            return "0.32", None
        mod = _module(RUNNER_V1)
        if mod is not None and getattr(mod, "allocate_kv_cache", None) is utils_alloc:
            return "0.32", None
        return None, "vLLM's V1 KV allocation is not routed through %s.allocate_kv_cache" % UTILS_V3
    if kind == RUNNER_V2 or kind.startswith("vllm.v1.worker.gpu."):
        attn = _module(ATTN_V2)
        if (ATTN_V2, "_allocate_kv_cache") not in S.wrapped or attn is None:
            return None, "vLLM's V2 KV allocation is not wrapped (%s)" % ("; ".join(S.errors) or "no hook wrapped")
        init = getattr(attn, "init_kv_cache", None)
        v2 = _module(RUNNER_V2)
        if v2 is None or getattr(v2, "init_kv_cache", None) is not init or not _calls(init, "_allocate_kv_cache"):
            return None, "the V2 model runner does not allocate through %s.init_kv_cache -> _allocate_kv_cache" % ATTN_V2
        return "0.24", None
    alloc = getattr(type(runner), "_allocate_kv_cache_tensors", None)
    if not getattr(alloc, "__spark_attic__", False) or not _calls(
            getattr(type(runner), "initialize_kv_cache_tensors", None), "_allocate_kv_cache_tensors"):
        return None, "model runner %s.%s allocates its KV cache in a way the plugin does not wrap (%s)" % (
            kind, type(runner).__name__, "; ".join(S.errors) or "no known allocation hook is wrapped")
    return "0.24", None


def _calls(fn, name):
    """fn's own code, under any wrappers that keep it as __wrapped__, names `name` (a call the plugin relies on is
    still there)."""
    code = getattr(inspect.unwrap(fn), "__code__", None)
    return code is not None and name in code.co_names


# ------------------------------------------------------------------------------------------------------- kv cache

def _make_determine(orig):
    def determine_available_memory(self):
        avail = orig(self)
        try:
            extra = _kv_lend(self)
        except Exception as e:
            log.warning("spark-attic: the KV cache stays as vLLM sized it: %s", e)
            return avail
        if extra:
            if getattr(self, "available_kv_cache_memory_bytes", None) == avail:
                self.available_kv_cache_memory_bytes = avail + extra
            log.info("spark-attic: KV cache memory %.2f GiB = vLLM's %.2f GiB + %d MiB display carveout",
                     (avail + extra) / GIB, avail / GIB, extra // MIB)
        return avail + extra
    return determine_available_memory


def _kv_lend(worker):
    """Borrow the carveout for the KV cache (once) and return its size; 0 when it must stay out."""
    shape, why = _kv_shape(worker)
    if why:
        log.warning("spark-attic: KV placement off: %s", why)
        return 0
    with S.lock:
        if S.kv_done:
            log.warning("spark-attic: vLLM sized the KV cache again after allocating it; no carveout this time")
            return 0
        if S.kv is not None:
            return S.kv.size
        pool = _pool(worker.device)
        n = S.ledger.room(pool)
        if n <= 0:
            log.warning("spark-attic: KV placement: atticd has nothing free")
            return 0
        try:
            S.kv = pool.lend(n)
        except Exception as e:
            log.warning("spark-attic: the KV cache stays as vLLM sized it: %s: %s", type(e).__name__, e)
            return 0
        S.ledger.spend(n)
        S.kv_counted = n
        S.shape = shape
        return n


def _unknown_fields(t):
    fields = set(getattr(type(t), "__dataclass_fields__", {}))
    known = KV_TENSOR_FIELDS["0.24"] | KV_TENSOR_FIELDS["0.32"]
    return fields - known or None


def _give_up(lent):
    """The placement failed after the budget was raised: the carveout goes back, unused."""
    try:
        lent.release()
    except Exception as e:
        log.warning("spark-attic: releasing the unused carveout: %s", e)
    give_back(S)


def _over_budget(what):
    """The line a post-raise failure leaves in the log. Falling back would leave the engine allocating a KV cache
    above the memory it measured, so the caller raises instead; this says why, and what to do."""
    log.error("spark-attic: %s, %d MiB more than the memory vLLM measured; not falling back: the KV budget was "
              "already raised, so vLLM would overcommit its device memory. Turn the plugin off "
              "(SPARK_ATTIC_PLACE=off) and restart vLLM, or fix the cause logged above.", what, S.kv_counted // MIB)


def _kv_tensors(kv_cache_config, device, zeros):
    """vLLM 0.24's raw KV tensors (layer name -> int8 tensor), as its own allocator lays them out, with the last of
    them in the lent carveout; zeros(nbytes) makes one the way vLLM does. None: vLLM allocates them all itself."""
    import torch

    with S.lock:
        lent, S.kv = S.kv, None
        if lent is None or S.kv_done:
            return None
        S.kv_done = True
    try:
        backings, packed = [], None  # one per KV tensor, except that all packed ones share the first one's
        for t in kv_cache_config.kv_cache_tensors:
            unknown = _unknown_fields(t)
            if unknown:
                raise Mismatch("KVCacheTensor has fields the plugin does not know: %s" % ", ".join(sorted(unknown)))
            if t.block_stride > 0:
                if packed is None:
                    packed = [t.size, []]
                    backings.append(packed)
                packed[1].extend(t.shared_by)
            else:
                backings.append([t.size, list(t.shared_by)])
        plan = tail_plan([size for size, _ in backings], lent.size)
        if plan.first == len(backings):
            raise Mismatch("no KV tensors to place")
        if plan.attic_used < lent.size:
            log.info("spark-attic: the KV cache needs only %d of the %d MiB lent", plan.attic_used // MIB,
                     lent.size // MIB)
        m = S.pool.map(plan.front, [lent])
    except Exception:
        _give_up(lent)
        raise
    try:
        m.zero()  # vLLM's own KV tensors start zeroed
        out = {}
        for i, (size, names) in enumerate(backings):
            if i < plan.first:
                t = zeros(size)
            else:
                t = m.tensor(torch.int8, offset=plan.offsets[i - plan.first], nbytes=size)
            for name in names:
                out[name] = t
    except Exception:
        m.close()
        _give_up(lent)
        _over_budget("the KV cache could not be placed in the carveout")
        raise
    S.keep.append(m)
    log.info("spark-attic: KV tensors %d-%d of %d in the display carveout (%d MiB), %d MiB of device memory glued "
             "before it", plan.first, len(backings) - 1, len(backings), plan.attic_used // MIB, plan.front // MIB)
    return out


def _alloc_or_raise(placement):
    """One allocation hook, both shapes: the placement runs; without a raised budget its failure fails open (one log
    line, the engine's own allocation runs), with one it raises, never leaving the engine above what it measured."""
    try:
        return placement()
    except Exception as e:
        if S.kv_counted:  # the budget was raised: never allocate it in device memory
            _over_budget("%s: %s" % (type(e).__name__, e))
            raise
        log.warning("spark-attic: %s: %s; vLLM allocates the whole KV cache itself", type(e).__name__, e)
        return None


def _make_alloc_v1(orig):
    def _allocate_kv_cache_tensors(self, kv_cache_config):
        import torch

        def zeros(n):
            return torch.zeros(n, dtype=torch.int8, device=self.device)
        out = _alloc_or_raise(lambda: _kv_tensors(kv_cache_config, self.device, zeros))
        return out if out is not None else orig(self, kv_cache_config)
    return _allocate_kv_cache_tensors


def _make_alloc_v2(orig):
    def _allocate_kv_cache(kv_cache_config, shared_layers, device):
        import torch

        def zeros(n):
            return torch.zeros(n, dtype=torch.int8, device=device)
        out = _alloc_or_raise(lambda: _kv_tensors(kv_cache_config, device, zeros))
        return out if out is not None else orig(kv_cache_config, shared_layers, device)
    return _allocate_kv_cache


def _kv_backing(kv_cache_config):
    """vLLM 0.32: the one backing tensor's size, after checking every field the plugin reads. Raises Mismatch when
    the layout is not the one the 0.32 shape knows (all tensors share one device backing)."""
    tensors = kv_cache_config.kv_cache_tensors
    if not tensors:
        raise Mismatch("the KV cache config has no tensors to place")
    sizes = {t.size for t in tensors}
    if len(sizes) != 1:
        raise Mismatch("the KV tensors do not share one backing allocation (%d sizes)" % len(sizes))
    for t in tensors:
        unknown = _unknown_fields(t)
        if unknown:
            raise Mismatch("KVCacheTensor has fields the plugin does not know: %s" % ", ".join(sorted(unknown)))
        if getattr(t, "host_resident", False):
            raise Mismatch("a KV cache group is host resident; the carveout is device memory")
    size = sizes.pop()
    if size <= 0:
        raise Mismatch("the KV cache backing is empty")
    return size


def _make_alloc_v3(orig):
    """vLLM 0.32: allocate_kv_cache makes one torch.zeros backing for the whole cache; a function mode answers exactly
    that call with a tensor over the mapping, so vLLM's own code builds every view on it."""
    def allocate_kv_cache(kv_cache_config, device, layout, kernel_block_sizes=None):
        import torch

        with S.lock:
            lent, S.kv = S.kv, None
        if lent is None:
            return orig(kv_cache_config, device, layout, kernel_block_sizes)
        S.kv_done = True
        try:
            size = _kv_backing(kv_cache_config)
            front = size - min(size, lent.size)
            m = S.pool.map(front, [lent])
            m.zero()  # the handed tensor stands in for vLLM's own torch.zeros: it starts zeroed
        except Exception as e:
            _give_up(lent)
            if S.kv_counted:  # the budget was raised: never allocate it in device memory
                _over_budget("%s: %s" % (type(e).__name__, e))
                raise
            log.warning("spark-attic: %s: %s; vLLM allocates the whole KV cache itself", type(e).__name__, e)
            return orig(kv_cache_config, device, layout, kernel_block_sizes)
        if size < lent.size:
            log.info("spark-attic: the KV cache needs only %d of the %d MiB lent", size // MIB, lent.size // MIB)
        handed, miss = [], [None]
        dev = device_index(device)

        class _BackingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is torch.zeros and not handed and miss[0] is None:
                    shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
                    n = 1
                    for d in shape:
                        n *= d
                    dtype = kwargs.get("dtype") or torch.uint8
                    where = kwargs.get("device")
                    if n * dtype.itemsize > m.size:
                        miss[0] = "the backing allocation is %d MiB, the mapping covers %d" % (
                            n * dtype.itemsize // MIB, m.size // MIB)
                    elif where is not None and device_index(where) != dev:
                        miss[0] = "the backing allocation lands on %s, not %s" % (where, device)
                    elif dtype != torch.int8:
                        miss[0] = "the backing allocation is %s, not int8" % dtype
                    else:
                        t = m.tensor(torch.int8, shape=(n,), nbytes=n)
                        handed.append(t)
                        return t
                return func(*args, **kwargs)

        mode = _BackingMode()
        try:
            with mode:
                out = orig(kv_cache_config, device, layout, kernel_block_sizes)
        except Exception as e:
            m.close()
            _give_up(lent)
            _over_budget("the KV cache could not be placed (%s: %s)" % (type(e).__name__, e))
            raise
        if not handed or miss[0]:
            m.close()
            _give_up(lent)
            reason = miss[0] or "no backing allocation seen"
            _over_budget("the KV cache was not allocated through one torch.zeros over the mapping (%s)" % reason)
            raise Mismatch(reason)
        S.keep.append(m)
        log.info("spark-attic: the KV cache backing (%d MiB) holds %d MiB of display carveout, %d MiB of device "
                 "memory before it", size // MIB, min(size, lent.size) // MIB, front // MIB)
        return out
    return allocate_kv_cache


# -------------------------------------------------------------------------------------------------------- weights

def _make_load_model(orig):
    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            _move_weights(self)
        except Exception as e:
            log.warning("spark-attic: weights stay in device memory: %s", e)
        return out
    return load_model


def _move_weights(worker):
    """Move the parameters SPARK_ATTIC_WEIGHTS names into the carveout and lower vLLM's weight figure by what that
    freed."""
    why = _blocked(worker)
    if why:
        log.warning("spark-attic: weights placement off: %s", why)
        return
    runner = getattr(worker, "model_runner", None)
    if not isinstance(getattr(runner, "model_memory_usage", None), (int, float)):
        raise Mismatch("the model runner has no model_memory_usage")
    model = model_of(runner)
    moved, freed = move_params(model, S.placement.weights, _pool(worker.device), S.ledger, S.keep,
                               device_index(worker.device))
    if not moved:
        return
    old = runner.model_memory_usage
    runner.model_memory_usage = max(old - freed, 0)
    log.info("spark-attic: %s moved to the display carveout (%d MiB); %d MiB of device memory freed, vLLM's weight "
             "figure %.2f -> %.2f GiB", ", ".join(n for n, _ in moved), sum(b for _, b in moved) // MIB,
             freed // MIB, old / GIB, runner.model_memory_usage / GIB)

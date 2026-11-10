# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""The vLLM plugin (entry point vllm.general_plugins, name spark_attic): a GB10's display carveout as more KV cache,
or as the home of named weights, in vLLM's GPU worker. No change to vLLM's files: it wraps four functions as their
modules load, in the processes that load them.

    SPARK_ATTIC_PLACE    kv (default), weights, weights,kv, or off
    SPARK_ATTIC_WEIGHTS  for weights: parameter names or globs, comma-separated (default embed_tokens)
    SPARK_ATTIC_MIB      at most this many MiB of carveout (default: all atticd has free)
    SPARK_ATTIC_LOG      0: no log handler of our own (default: our lines on stderr, logger spark_attic)
    and the client's SPARK_ATTIC_SOCK, SPARK_ATTIC_KEY or SPARK_ATTIC_KEY_FILE, SPARK_ATTIC_TIMEOUT_MS.

kv: Worker.determine_available_memory borrows the carveout and adds it to what vLLM measured (or to
--kv-cache-memory-bytes, which then holds device memory), so vLLM sizes the KV pool larger by exactly that; the KV
allocation then puts the last KV tensors (the last layers) in the carveout, the one before them glued from device
memory and what is left of it, and the rest where vLLM puts them. weights: after Worker.load_model the named
parameters move into the carveout, and vLLM's weight figure drops by what that freed, so vLLM's own profile gives the
KV pool that much more. weights,kv: weights first, the rest to the KV pool.

Pinned to vLLM 0.24.0 (upstream tag v0.24.0, ee0da84a; NGC nvidia/vllm:26.07-py3 ships 0.24.0+092c4842). Each wrap
first checks the parameters of what it wraps against EXPECTED. A mismatch, sleep mode, a KV connector, a model runner
it does not know, no atticd or nothing free: that placement stays off and vLLM runs as it would without the plugin
(fail open, one log line, logger "spark_attic").
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
VLLM_PINNED = "0.24.0"

WORKER = "vllm.v1.worker.gpu_worker"
RUNNER_V1 = "vllm.v1.worker.gpu_model_runner"
RUNNER_V2 = "vllm.v1.worker.gpu.model_runner"
ATTN_V2 = "vllm.v1.worker.gpu.attn_utils"

# The parameters, in order, of each function the plugin wraps, as vLLM v0.24.0 has them.
EXPECTED = {
    (WORKER, "Worker.determine_available_memory"): ["self"],
    (WORKER, "Worker.load_model"): ["self", "load_dummy_weights"],
    (RUNNER_V1, "GPUModelRunner._allocate_kv_cache_tensors"): ["self", "kv_cache_config"],
    (ATTN_V2, "_allocate_kv_cache"): ["kv_cache_config", "shared_layers", "device"],
}
# KVCacheTensor's fields in v0.24.0; one we do not know may change what a KV tensor's bytes mean.
KV_TENSOR_FIELDS = frozenset({"size", "shared_by", "offset", "block_stride"})


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
        self.wrapped = set()      # the EXPECTED keys wrapped in this process
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
    except Exception as e:
        S.placement = Placement(place=())
        log.warning("spark-attic: vLLM plugin off: %s", e)


def _wrap(owner, attr, key, make):
    fn = getattr(owner, attr, None)
    if fn is None:
        raise Mismatch("%s has no %s" % key)
    if getattr(fn, "__spark_attic__", False):
        S.wrapped.add(key)
        return
    got = params(fn)
    if got != EXPECTED[key]:
        raise Mismatch("%s.%s takes (%s); the plugin knows (%s) from vLLM %s" % (
            key[0], key[1], ", ".join(got), ", ".join(EXPECTED[key]), VLLM_PINNED))
    new = functools.wraps(fn)(make(fn))
    new.__spark_attic__ = True
    setattr(owner, attr, new)
    S.wrapped.add(key)


def _wrap_worker(mod):
    worker = getattr(mod, "Worker", None)
    if worker is None:
        raise Mismatch("%s has no Worker" % WORKER)
    if "kv" in S.placement.place:
        try:
            _wrap(worker, "determine_available_memory", (WORKER, "Worker.determine_available_memory"),
                  _make_determine)
        except Mismatch as e:
            log.warning("spark-attic: KV placement off: %s", e)
    if "weights" in S.placement.place:
        _wrap(worker, "load_model", (WORKER, "Worker.load_model"), _make_load_model)


def _wrap_runner_v1(mod):
    runner = getattr(mod, "GPUModelRunner", None)
    if runner is None:
        raise Mismatch("%s has no GPUModelRunner" % RUNNER_V1)
    _wrap(runner, "_allocate_kv_cache_tensors", (RUNNER_V1, "GPUModelRunner._allocate_kv_cache_tensors"),
          _make_alloc_v1)


def _wrap_attn_v2(mod):
    _wrap(mod, "_allocate_kv_cache", (ATTN_V2, "_allocate_kv_cache"), _make_alloc_v2)


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


def _kv_blocked(worker):
    why = _blocked(worker)
    if why:
        return why
    if getattr(worker.vllm_config, "kv_transfer_config", None) is not None:
        return "a KV connector is configured (it may lay the KV cache out itself)"
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return "the worker has no model runner"
    kind = type(runner).__module__
    if kind == RUNNER_V2 or kind.startswith("vllm.v1.worker.gpu."):
        attn, v2 = sys.modules.get(ATTN_V2), sys.modules.get(RUNNER_V2)
        if (ATTN_V2, "_allocate_kv_cache") not in S.wrapped or attn is None:
            return "vLLM's V2 KV allocation is not wrapped"
        init = getattr(attn, "init_kv_cache", None)
        if v2 is None or getattr(v2, "init_kv_cache", None) is not init or not _calls(init, "_allocate_kv_cache"):
            return "the V2 model runner does not allocate through %s.init_kv_cache -> _allocate_kv_cache" % ATTN_V2
        return None
    alloc = getattr(type(runner), "_allocate_kv_cache_tensors", None)
    if not getattr(alloc, "__spark_attic__", False) or not _calls(
            getattr(type(runner), "initialize_kv_cache_tensors", None), "_allocate_kv_cache_tensors"):
        return "model runner %s.%s allocates its KV cache in a way the plugin does not wrap" % (
            kind, type(runner).__name__)
    return None


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
    why = _kv_blocked(worker)
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
        S.kv = pool.lend(n)
        S.ledger.spend(n)
        S.kv_counted = n
        return n


def _kv_tensors(kv_cache_config, device, zeros):
    """vLLM's raw KV tensors (layer name -> int8 tensor), as its own allocator lays them out, with the last of them
    in the lent carveout; zeros(nbytes) makes one the way vLLM does. None: vLLM allocates them all itself."""
    import torch

    with S.lock:
        lent, S.kv = S.kv, None
        if lent is None or S.kv_done:
            return None
        S.kv_done = True
    try:
        backings, packed = [], None  # one per KV tensor, except that all packed ones share the first one's
        for t in kv_cache_config.kv_cache_tensors:
            fields = set(getattr(type(t), "__dataclass_fields__", {}))
            if fields - KV_TENSOR_FIELDS:
                raise Mismatch("KVCacheTensor has fields the plugin does not know: %s" % ", ".join(
                    sorted(fields - KV_TENSOR_FIELDS)))
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
        lent.release()
        give_back(S)
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
        give_back(S)
        raise
    S.keep.append(m)
    log.info("spark-attic: KV tensors %d-%d of %d in the display carveout (%d MiB), %d MiB of device memory glued "
             "before it", plan.first, len(backings) - 1, len(backings), plan.attic_used // MIB, plan.front // MIB)
    return out


def _alloc_or_fallback(make, fallback):
    try:
        out = make()
    except Exception as e:
        log.warning("spark-attic: %s: %s; vLLM allocates the whole KV cache in device memory, %d MiB more than the "
                    "memory it measured", type(e).__name__, e, S.kv_counted // MIB)
        out = None
    return fallback() if out is None else out


def _make_alloc_v1(orig):
    def _allocate_kv_cache_tensors(self, kv_cache_config):
        import torch

        def zeros(n):
            return torch.zeros(n, dtype=torch.int8, device=self.device)
        return _alloc_or_fallback(lambda: _kv_tensors(kv_cache_config, self.device, zeros),
                                  lambda: orig(self, kv_cache_config))
    return _allocate_kv_cache_tensors


def _make_alloc_v2(orig):
    def _allocate_kv_cache(kv_cache_config, shared_layers, device):
        import torch

        def zeros(n):
            return torch.zeros(n, dtype=torch.int8, device=device)
        return _alloc_or_fallback(lambda: _kv_tensors(kv_cache_config, device, zeros),
                                  lambda: orig(kv_cache_config, shared_layers, device))
    return _allocate_kv_cache


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

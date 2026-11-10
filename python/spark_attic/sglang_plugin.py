# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""The SGLang plugin (entry point sglang.srt.plugins, name spark_attic): a GB10's display carveout as more KV cache,
or as the home of named weights, in SGLang's model runner. It registers AROUND hooks with SGLang's own HookRegistry
(sglang.srt.plugins.hook_registry), which SGLang applies itself; no change to SGLang's files.

    SPARK_ATTIC_PLACE    kv (default), weights, weights,kv, or off
    SPARK_ATTIC_WEIGHTS  for weights: parameter names or globs, comma-separated (default embed_tokens)
    SPARK_ATTIC_MIB      at most this many MiB of carveout (default: all atticd has free)
    SPARK_ATTIC_LOG      0: no log handler of our own (default: our lines on stderr, logger spark_attic)
    and the client's SPARK_ATTIC_SOCK, SPARK_ATTIC_KEY or SPARK_ATTIC_KEY_FILE, SPARK_ATTIC_TIMEOUT_MS.

kv: when the KV pool will be the plain MHA pool (MHATokenToKVPool, NHD layout: dense attention, no MLA, SWA, Mamba,
FP4, memory saver, custom pool or speculative decoding), _profile_available_bytes borrows the carveout and adds it to
what SGLang measured, so its pool sizing gives the KV cache exactly that much more. MHATokenToKVPool._create_buffers
then gets its last K/V buffers (the V buffers of the last layers) from the carveout, the one before them glued from
device memory: a TorchFunctionMode answers exactly those torch.zeros calls, after checking each one's shape, dtype and
device against the plan, so SGLang's own code builds the pool around them. With more than one rank, the ranks first
agree on one amount over SGLang's world group (the smallest any of them can borrow, or none), as SGLang agrees on free
memory, so every rank's pool keeps the same size. --max-total-tokens caps the pool's device memory, not the pool: when
the cap binds, _apply_token_constraints lets the pool take the carveout's tokens (SGLang's own configurator turns its
bytes into tokens) on top of the cap, as the vLLM plugin adds the carveout on top of --kv-cache-memory-bytes. weights:
after ModelRunner.load_model the named parameters move into the carveout; SGLang measures free memory after the load,
so the KV pool gets that much more by itself (up to --max-total-tokens).

Pinned to SGLang 0.5.14 (upstream tag v0.5.14, 49e384ce; NGC nvidia/sglang:26.07-py3 ships 0.5.14+0e76cb3c). Each
hook's target, and each function in CALLS, must have the parameters listed there or that placement's hooks are not
registered. Anything else (another pool kind, no atticd, nothing free) leaves that placement off and SGLang runs as
without the plugin (fail open, one log line, logger "spark_attic").
"""
import logging
import os
import pkgutil
import threading

from . import MIB, Pool, Settings
from ._place import (Ledger, Mismatch, Placement, device_index, give_back, log_to_stderr, model_of, move_params,
                     params, tail_plan)

log = logging.getLogger("spark_attic")
GIB = 1 << 30
SGLANG_PINNED = "0.5.14"

MIXIN = "sglang.srt.model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin"
MHA = "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool"
RUNNER = "sglang.srt.model_executor.model_runner.ModelRunner"
CONFIGURATOR = "sglang.srt.model_executor.pool_configurator"
# What each hook wraps, and that function's parameters in SGLang v0.5.14.
TARGETS = {
    "profile": (MIXIN + "._profile_available_bytes", ["self", "pre_model_load_memory"]),
    "cap": (MIXIN + "._apply_token_constraints", ["self", "token_capacity"]),
    "pools": (MIXIN + "._init_pools", ["self"]),
    "buffers": (MHA + "._create_buffers", ["self"]),
    "load": (RUNNER + ".load_model", ["self"]),
}
# What the cap hook calls, likewise.
CALLS = {
    "configurator": (CONFIGURATOR + ".create_memory_pool_configurator", ["mr"]),
    "pool_sizes": (CONFIGURATOR + ".DefaultPoolConfigurator.calculate_pool_sizes",
                   ["self", "available_bytes", "page_size"]),
}


class _State:
    def __init__(self):
        self.registered = False
        self.placement = Placement(place=())
        self.settings = None
        self.ledger = Ledger()
        self.pool = None
        self.kv = None            # the Lent for the KV pool, between the profile and the buffers
        self.kv_counted = 0
        self.kv_done = False      # the pool was built: a later sizing gets no carveout
        self.hooked = set()       # TARGETS keys registered
        self.keep = []
        self.lock = threading.RLock()


S = _State()


# ------------------------------------------------------------------------------------------------- registration

def register():
    """The entry point SGLang calls in each of its processes. Never raises: SGLang starts either way."""
    try:
        log_to_stderr()
        with S.lock:
            if S.registered:
                return
            S.registered = True
            S.placement = Placement.from_env(os.environ)
            S.ledger = Ledger(S.placement.cap)
            if not S.placement.place:
                log.info("spark-attic: SGLang plugin off (SPARK_ATTIC_PLACE=off)")
                return
            S.settings = Settings.from_env()
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        hooks = {"profile": _profile_hook, "cap": _cap_hook, "pools": _pools_hook, "buffers": _buffers_hook,
                 "load": _load_hook}
        groups = []
        if "kv" in S.placement.place:
            groups.append(("KV", ("profile", "cap", "pools", "buffers"), ("configurator", "pool_sizes")))
        if "weights" in S.placement.place:
            groups.append(("weights", ("load",), ()))
        for what, keys, calls in groups:  # a placement's hooks go in all together or not at all
            try:
                for key in keys + calls:
                    _check(key)
            except Mismatch as e:
                log.warning("spark-attic: %s placement off: %s", what, e)
                continue
            for key in keys:
                HookRegistry.register(TARGETS[key][0], hooks[key], HookType.AROUND)
                S.hooked.add(key)
    except Exception as e:
        S.placement = Placement(place=())
        log.warning("spark-attic: SGLang plugin off: %s: %s", type(e).__name__, e)


def _check(key):
    target, want = TARGETS[key] if key in TARGETS else CALLS[key]
    try:
        fn = pkgutil.resolve_name(target)
    except (ImportError, AttributeError, ValueError) as e:
        raise Mismatch("%s: %s" % (target, e))
    got = params(fn)
    if got != want:
        raise Mismatch("%s takes (%s); the plugin knows (%s) from SGLang %s" % (
            target, ", ".join(got), ", ".join(want), SGLANG_PINNED))


def _pool(device):
    with S.lock:
        if S.pool is None:
            S.pool = Pool(S.settings, device_index(device))
        return S.pool


# ------------------------------------------------------------------------------------------------------- kv cache

def _kv_blocked(mr):
    """Why the KV pool of model runner `mr` must stay as SGLang sizes it, or None. The pool must come out as the
    plain MHATokenToKVPool (_init_pools' last branch); anything the plugin cannot read counts as no."""
    from sglang.srt.configs.model_config import is_deepseek_v4
    from sglang.srt.platforms import current_platform

    args = mr.server_args
    checks = (
        (str(mr.device) == "cuda", "the device is %s, not cuda" % mr.device),
        (mr.use_mla_backend is False, "the model uses MLA"),
        (mr.is_hybrid_swa is False, "the model has sliding-window layers"),
        (mr.mambaish_config is None, "the model has Mamba-like layers"),
        (mr.is_draft_worker is False and mr.spec_algorithm.is_none(), "speculative decoding is on"),
        (not is_deepseek_v4(mr.model_config.hf_config), "a DeepSeek V4 pool"),
        (not current_platform.is_out_of_tree() and args.attention_backend != "ascend", "an out-of-tree platform"),
        ("float4" not in str(mr.kv_cache_dtype), "an FP4 KV cache"),
        (not args.prefill_only_disable_kv_cache, "prefill-only without a KV cache"),
        (not args.enable_memory_saver, "the memory saver is on"),
        (args.disaggregation_mode in (None, "null"), "PD disaggregation"),
        ("SGLANG_MOONCAKE_CUSTOM_MEM_POOL" not in os.environ, "a custom KV memory pool"),
    )
    for ok, why in checks:
        if not ok:
            return why
    return None


def _profile_hook(original, self, pre_model_load_memory):
    avail = original(self, pre_model_load_memory)
    extra = _kv_lend(self)
    if extra:
        log.info("spark-attic: KV budget %.2f GiB = %.2f GiB SGLang measured + %d MiB display carveout",
                 (avail + extra) / GIB, avail / GIB, extra // MIB)
    return avail + extra


def _cap_hook(original, self, token_capacity):
    """--max-total-tokens holds the KV pool's device memory: when it binds, the pool takes the carveout's tokens on
    top of it. Without this the cap would bind on the measured memory plus the carveout, and the carveout would only
    stand in for device memory. The PP ranks still agree in SGLang's own code, on the raised cap."""
    with S.lock:
        lent = S.kv
    cap = self.server_args.max_total_tokens
    if lent is None or cap is None or cap >= token_capacity:
        return original(self, token_capacity)
    try:
        from sglang.srt.model_executor.pool_configurator import create_memory_pool_configurator

        extra = create_memory_pool_configurator(self).calculate_pool_sizes(S.kv_counted, 1).max_total_num_tokens
    except Exception as e:
        log.warning("spark-attic: max_total_tokens=%d caps the KV pool with the carveout in it, which then only "
                    "stands in for device memory: %s: %s", cap, type(e).__name__, e)
        return original(self, token_capacity)
    self.server_args.max_total_tokens = min(cap + extra, token_capacity)
    try:
        out = original(self, token_capacity)
    finally:
        self.server_args.max_total_tokens = cap
    log.info("spark-attic: max_total_tokens=%d holds the KV pool's device memory; the display carveout (%d MiB) adds "
             "%d tokens: %d", cap, S.kv_counted // MIB, out - cap, out)
    return out


def _kv_lend(mr):
    """Borrow the KV pool's share of the carveout; its size, or 0. Never raises. Every rank reaches the same
    _agree calls: whether KV placement is off depends only on the configuration, which the ranks share."""
    try:
        why = _kv_blocked(mr)
    except AttributeError as e:
        why = "SGLang's model runner is not the one the plugin knows (%s)" % e
    except Exception as e:
        why = "%s: %s" % (type(e).__name__, e)
    if why:
        log.warning("spark-attic: KV placement off: %s", why)
        return 0
    with S.lock:
        if S.kv is not None:  # sized again before the pool was built: the same amount
            return S.kv.size
        if S.kv_done:
            log.warning("spark-attic: SGLang sized the KV pool again after building it; no carveout this time")
            return 0
        try:
            n = S.ledger.room(_pool(mr.device))
        except Exception as e:
            log.warning("spark-attic: the KV pool stays as SGLang sized it: %s: %s", type(e).__name__, e)
            n = 0
        agreed = _agree(n)
        if agreed <= 0:
            if n > 0:
                log.warning("spark-attic: KV placement off: another rank has no display carveout to lend")
            elif S.pool is not None:
                log.warning("spark-attic: KV placement: atticd has nothing free")
            return 0
        lent = None
        try:
            lent = S.pool.lend(agreed)
        except Exception as e:
            log.warning("spark-attic: the KV pool stays as SGLang sized it: %s: %s", type(e).__name__, e)
        if _agree(agreed if lent is not None else 0) < agreed:  # every rank borrowed it, or none keeps it
            if lent is not None:
                lent.release()
                give_back(S)
                log.warning("spark-attic: KV placement off: another rank could not borrow %d MiB", agreed // MIB)
            return 0
        if agreed < n:
            log.info("spark-attic: KV placement: %d MiB, what every rank can lend (this one has %d MiB)",
                     agreed // MIB, n // MIB)
        S.kv = lent
        S.ledger.spend(agreed)
        S.kv_counted = agreed
        return agreed


def _agree(n):
    """The smallest n over SGLang's world group, which sizes the KV pool once for all its ranks (as
    get_available_gpu_memory agrees on free memory, over the same CPU group). A failure counts as 0 here; the other
    ranks still get their answer from the collective."""
    try:
        from sglang.srt.distributed.parallel_state import get_world_group

        group = get_world_group()
        if group.world_size <= 1:
            return n
        import torch

        t = torch.tensor(n, dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
        return int(t.item())
    except Exception as e:
        log.warning("spark-attic: the ranks could not agree on the carveout: %s: %s", type(e).__name__, e)
        return 0


def _pools_hook(original, self):
    try:
        return original(self)
    finally:
        with S.lock:
            lent, S.kv = S.kv, None
            if lent is not None:
                S.kv_done = True
        if lent is not None:  # the pool was not the kind the plugin fills
            lent.release()
            give_back(S)
            log.warning("spark-attic: the KV pool (%s) took no carveout; it is %d MiB over the memory SGLang "
                        "measured", type(getattr(self, "token_to_kv_pool", None)).__name__, S.kv_counted // MIB)


def _buffers_hook(original, self):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    with S.lock:
        lent = S.kv
        usable = (lent is not None and type(self) is MHATokenToKVPool and self.kv_cache_layout == "nhd"
                  and not self.enable_custom_mem_pool)
        if usable:
            S.kv, S.kv_done = None, True
    if not usable:
        return original(self)
    try:
        dev = device_index(self.device)
        layers = self.layer_num
        k = (self.size + self.page_size, self.head_num, self.head_dim)
        v = (self.size + self.page_size, self.head_num, self.v_head_dim)
        expected = [k] * layers + [v] * layers  # _create_buffers' torch.zeros calls, in order
        item = self.store_dtype.itemsize
        sizes = [_numel(s) * item for s in expected]
        plan = tail_plan(sizes, lent.size)
        m = S.pool.map(plan.front, [lent])
        m.zero()  # SGLang's own buffers start zeroed
    except Exception as e:
        lent.release()
        give_back(S)
        log.warning("spark-attic: %s: %s; SGLang allocates the whole KV pool in device memory, %d MiB more than the "
                    "memory it measured", type(e).__name__, e, S.kv_counted // MIB)
        return original(self)
    mode = _zeros_mode()(expected, sizes, self.store_dtype, dev, plan, m)
    try:
        with mode:
            out = original(self)
    finally:
        if mode.handed:
            S.keep.append(m)
        else:  # nothing points into the mapping: give it back now
            m.close()
            give_back(S)
    given = {id(t) for t in mode.handed}
    kept = sum(1 for t in list(self.k_buffer) + list(self.v_buffer) if id(t) in given)
    if mode.miss or kept != len(expected) - plan.first:
        log.warning("spark-attic: SGLang allocated its KV buffers differently than planned (%s); %d of %d buffers "
                    "are in the carveout, up to %d MiB of it unused and the pool that much over the memory SGLang "
                    "measured", mode.miss or "buffers replaced", kept, len(expected) - plan.first,
                    S.kv_counted // MIB)
    else:
        log.info("spark-attic: KV buffers %d-%d of %d (K then V, by layer) in the display carveout (%d MiB), %d MiB "
                 "of device memory glued before it", plan.first, len(expected) - 1, len(expected),
                 plan.attic_used // MIB, plan.front // MIB)
    return out


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


_MODE = None


def _zeros_mode():
    """The TorchFunctionMode class (made on first use, so this module imports without torch)."""
    global _MODE
    if _MODE is not None:
        return _MODE
    import torch

    class ZerosMode(torch.overrides.TorchFunctionMode):
        """Answers the planned torch.zeros calls, in order, with tensors over the mapping; every other call, and
        every call after one that does not match the plan, goes to torch."""

        def __init__(self, expected, sizes, dtype, dev, plan, mapping):
            super().__init__()
            self.expected, self.sizes, self.dtype, self.dev = expected, sizes, dtype, dev
            self.plan, self.mapping = plan, mapping
            self.i, self.miss, self.handed = 0, None, []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.zeros and self.miss is None and self.i < len(self.expected):
                i = self.i
                self.i += 1
                shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
                shape = tuple(kwargs.get("size", shape))
                if shape != self.expected[i] or kwargs.get("dtype") != self.dtype or not _same_device(
                        kwargs.get("device"), self.dev):
                    self.miss = "call %d: torch.zeros(%s, dtype=%s, device=%s)" % (
                        i, shape, kwargs.get("dtype"), kwargs.get("device"))
                elif i >= self.plan.first:
                    t = self.mapping.tensor(self.dtype, shape=shape, offset=self.plan.offsets[i - self.plan.first],
                                            nbytes=self.sizes[i])
                    self.handed.append(t)
                    return t
            return func(*args, **kwargs)

    _MODE = ZerosMode
    return _MODE


def _same_device(device, dev):
    if device is None:
        return False
    try:
        return device_index(device) == dev
    except Mismatch:
        return False


# -------------------------------------------------------------------------------------------------------- weights

def _load_hook(original, self):
    out = original(self)
    try:
        moved, freed = move_params(model_of(self), S.placement.weights, _pool(self.device), S.ledger, S.keep,
                                   device_index(self.device))
        if moved:
            log.info("spark-attic: %s moved to the display carveout (%d MiB); %d MiB of device memory freed",
                     ", ".join(n for n, _ in moved), sum(b for _, b in moved) // MIB, freed // MIB)
    except Exception as e:
        log.warning("spark-attic: weights stay in device memory: %s: %s", type(e).__name__, e)
    return out

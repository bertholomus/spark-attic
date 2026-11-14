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

There is no pinned SGLang version. The plugin knows two shapes of SGLang's KV sizing and, in each process, picks the
one the running engine really has, by what its modules expose (the tables below; tests/test_sglang_plugin.py proves
each shape against SGLang's own source for every supported release, and scripts/attic-engine-drift.sh runs that test
against the newest release tags):

  0.5.14: the model runner's KV methods live in model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin, the
          configurator is built from the model runner (create_memory_pool_configurator(mr)), and the plain MHA pool's
          _create_buffers makes one torch.zeros per layer, K then V.
  0.5.23: the same methods moved into mem_cache.kv_cache_configurator.KVCacheConfigurator (which carries what the
          model runner used to: device, server_args, the pool fields), the configurator is built from it
          (create_memory_pool_configurator(kvc)), the cap is read through get_schedule(), and _create_buffers makes
          one fused torch.zeros per layer, K and V in one allocation, so the pool keeps views of it.

The hook-signature shape says nothing about how the pool allocates its buffers: 0.5.21 has 0.5.23's signatures
but its plain MHA pool still allocates K and V as their own zeros calls, one per layer, K's for all layers first
(0.5.14's layout). So before the KV budget is raised the plugin reads the buffer layout from the pool class's own
_create_buffers source (_layout_rule: fused when a zeros call starts with the literal 2 of the fused (2, padded,
H, D) allocation, separate otherwise, None when neither can be read) and plans the placement against that; a
pattern the plugin cannot read keeps the KV placement off with one log line, before anything is raised.

kv: when the KV pool will be the plain MHA pool (MHATokenToKVPool, NHD layout: dense attention, no MLA, SWA, Mamba,
FP4 or other quantized store, memory saver, custom pool, HND or page-major layout, MiniMax-sparse model, post-capture
KV, unequal K and V head sizes, or speculative decoding), _profile_available_bytes borrows
the carveout and adds it to what SGLang measured, so its pool sizing gives the KV cache exactly that much more.
MHATokenToKVPool._create_buffers then gets its last buffers from the carveout, the one before them glued from device
memory: a TorchFunctionMode answers exactly those torch.zeros calls, after checking each one's shape, dtype and
device against the plan, so SGLang's own code builds the pool around them. With more than one rank, the ranks first
agree on one amount over SGLang's world group (the smallest any of them can borrow, or none), as SGLang agrees on
free memory, so every rank's pool keeps the same size. --max-total-tokens caps the pool's device memory, not the
pool: when the cap binds, _apply_token_constraints lets the pool take the carveout's tokens on top of the cap, as
the vLLM plugin adds the carveout on top of --kv-cache-memory-bytes. weights: after ModelRunner.load_model the named
parameters move into the carveout; SGLang measures free memory after the load, so the KV pool gets that much more by
itself (up to --max-total-tokens).

Fail open until the KV budget is raised, loud after it: anything seen before the profile hook (a shape the plugin
does not know, a mismatched signature, another pool kind, no atticd, nothing free) turns that placement off and
SGLang runs as without the plugin (one log line, logger "spark_attic"). Once the plugin has added the carveout to
the KV budget, a failed placement can no longer fail open: falling back would leave SGLang allocating the KV pool
above the memory it measured, so the plugin gives the carveout back and raises instead, and SGLang stops coming up
until the plugin is turned off (SPARK_ATTIC_PLACE=off).
"""
import ast
import inspect
import logging
import os
import pkgutil
import textwrap
import threading
from contextlib import contextmanager

from . import MIB, Pool, Settings
from ._place import (Ledger, Mismatch, Placement, device_index, give_back, log_to_stderr, model_of, move_params,
                     params, tail_plan)

log = logging.getLogger("spark_attic")
GIB = 1 << 30

MHA = "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool"
RUNNER = "sglang.srt.model_executor.model_runner.ModelRunner"
MIXIN = "sglang.srt.model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin"
CONF = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
CONFIGURATOR = "sglang.srt.model_executor.pool_configurator"

# The engine shapes the plugin knows, oldest first.
SHAPES = ("0.5.14", "0.5.23")
SHAPE_RELEASE = {"0.5.14": "0.5.14", "0.5.23": "0.5.23"}

# What each hook wraps, and that function's parameters, per shape.
TARGETS = {
    "0.5.14": {
        "profile": (MIXIN + "._profile_available_bytes", ["self", "pre_model_load_memory"]),
        "cap": (MIXIN + "._apply_token_constraints", ["self", "token_capacity"]),
        "pools": (MIXIN + "._init_pools", ["self"]),
        "buffers": (MHA + "._create_buffers", ["self"]),
        "load": (RUNNER + ".load_model", ["self"]),
    },
    "0.5.23": {
        "profile": (CONF + "._profile_available_bytes", ["self", "pre_model_load_memory"]),
        "cap": (CONF + "._apply_token_constraints", ["self", "token_capacity"]),
        "pools": (CONF + "._init_pools", ["self", "sizes", "req_to_token_pool", "token_to_kv_pool_allocator"]),
        "buffers": (MHA + "._create_buffers", ["self"]),
        "load": (RUNNER + ".load_model", ["self"]),
    },
}
# What the cap hook calls, likewise.
CALLS = {
    "0.5.14": {
        "configurator": (CONFIGURATOR + ".create_memory_pool_configurator", ["mr"]),
        "pool_sizes": (CONFIGURATOR + ".DefaultPoolConfigurator.calculate_pool_sizes",
                       ["self", "available_bytes", "page_size"]),
    },
    "0.5.23": {
        "configurator": (CONFIGURATOR + ".create_memory_pool_configurator", ["kvc"]),
        "pool_sizes": (CONFIGURATOR + ".DefaultPoolConfigurator.calculate_pool_sizes",
                       ["self", "available_bytes", "page_size"]),
    },
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
        self.shape = None         # which engine shape the KV hooks run as
        self.layout = None        # how the pool's _create_buffers allocates: separate | fused
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
        if "kv" in S.placement.place:
            shape, why = _kv_shape(("profile", "cap", "pools", "buffers"), ("configurator", "pool_sizes"))
            if shape is None:
                log.warning("spark-attic: KV placement off: %s", why)
            else:
                S.shape = shape
                for key in ("profile", "cap", "pools", "buffers"):
                    HookRegistry.register(TARGETS[shape][key][0], hooks[key], HookType.AROUND)
                    S.hooked.add(key)
        if "weights" in S.placement.place:
            try:
                _check("load", shape=S.shape or "0.5.14")
            except Mismatch as e:
                log.warning("spark-attic: weights placement off: %s", e)
            else:
                HookRegistry.register(TARGETS[S.shape or "0.5.14"]["load"][0], _load_hook, HookType.AROUND)
                S.hooked.add("load")
    except Exception as e:
        S.placement = Placement(place=())
        log.warning("spark-attic: SGLang plugin off: %s: %s", type(e).__name__, e)


def _kv_shape(keys, calls):
    """The shape whose every hook target and call exists with the parameters the plugin reads, or (None, why)."""
    errors = []
    for shape in SHAPES:
        try:
            for key in keys + calls:
                _check(key, shape=shape)
            return shape, None
        except Mismatch as e:
            errors.append("%s: %s" % (SHAPE_RELEASE[shape], e))
    return None, "SGLang is not a shape the plugin knows (%s)" % "; ".join(errors)


def _check(key, shape):
    table = TARGETS[shape] if key in TARGETS[shape] else CALLS[shape]
    target, want = table[key]
    try:
        fn = pkgutil.resolve_name(target)
    except (ImportError, AttributeError, ValueError) as e:
        raise Mismatch("%s: %s" % (target, e))
    got = params(fn)
    if got != want:
        raise Mismatch("%s takes (%s); the plugin knows (%s) from SGLang %s" % (
            target, ", ".join(got), ", ".join(want), SHAPE_RELEASE[shape]))


def _pool(device):
    with S.lock:
        if S.pool is None:
            S.pool = Pool(S.settings, device_index(device))
        return S.pool


# ------------------------------------------------------------------------------------------------------- kv cache

def _layout_rule(tree):
    """separate | fused | None, from the AST of a pool's _create_buffers (following a
    self._create_buffers_normal() dispatch is the caller's job): fused when some torch.zeros call's size
    starts with the literal 2 of SGLang's fused (2, padded, H, D) allocation (0.5.22+), separate when it
    allocates K and V as their own zeros calls (0.5.14, 0.5.21), None when the source shows neither."""
    fused = seen = False
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = n.func.id if isinstance(n.func, ast.Name) else (
            n.func.attr if isinstance(n.func, ast.Attribute) else "")
        if not name.endswith("zeros"):
            continue
        seen = True
        arg = n.args[0] if n.args else None
        if isinstance(arg, (ast.Tuple, ast.List)) and arg.elts \
                and isinstance(arg.elts[0], ast.Constant) and arg.elts[0].value == 2:
            fused = True
    if not seen:
        return None
    return "fused" if fused else "separate"


def _buffer_layout(pool_cls):
    """The layout the pool class really allocates, read from its own _create_buffers source before the
    KV budget is raised (the hook-signature shape says nothing about it: 0.5.21 has 0.5.23's signatures
    but 0.5.14's separate K/V). None when the source cannot be read or shows no known pattern."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(pool_cls._create_buffers)))
        if [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_create_buffers_normal"]:
            tree = ast.parse(textwrap.dedent(inspect.getsource(pool_cls._create_buffers_normal)))
        return _layout_rule(tree)
    except Exception:
        return None


def _screen(name, ok, why):
    """(ok(), why): an attribute the plugin cannot read counts as blocked, never as a placement."""
    try:
        return ok(), why
    except Exception as e:
        return False, "SGLang's %s cannot be read (%s: %s)" % (name, type(e).__name__, e)


def _kv_blocked(mr):
    """Why the KV pool of the sizing object `mr` (the model runner in 0.5.14, the KV configurator in 0.5.23) must
    stay as SGLang sizes it, or None. The pool must come out as the plain MHATokenToKVPool (_init_pools' last
    branch); anything the plugin cannot read counts as no."""
    from sglang.srt.configs.model_config import is_deepseek_v4, is_minimax_sparse
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
    if S.shape != "0.5.14":
        # 0.5.21+ builds another pool or layout for these, and the buffers hook needs equal K and V head sizes;
        # screened before the budget is raised, so the placement turns off instead of the pool raising.
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_memory

        checks += (
            _screen("SGLANG_USE_HND_KVCACHE", lambda: not envs.SGLANG_USE_HND_KVCACHE.get(), "an HND KV cache"),
            _screen("enable_page_major_kv_layout", lambda: not get_memory().enable_page_major_kv_layout,
                    "the page-major KV layout"),
            _screen("post_capture_kv_active", lambda: not mr.post_capture_kv_active, "post-capture KV sizing"),
            _screen("model_config.head_dim", lambda: mr.model_config.head_dim == mr.model_config.v_head_dim,
                    "K and V head sizes differ"),
            _screen("model_config.hf_config", lambda: not is_minimax_sparse(mr.model_config.hf_config),
                    "a MiniMax-sparse model"),
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
    stand in for device memory. A failure here leaves the pool no bigger than the cap: nothing over what SGLang
    measured, so this hook may fail open."""
    with S.lock:
        lent = S.kv
    cap = _user_cap(self)
    if lent is None or cap is None or cap >= token_capacity:
        return original(self, token_capacity)
    try:
        from sglang.srt.model_executor.pool_configurator import create_memory_pool_configurator

        page = 1 if S.shape == "0.5.14" else self.page_size
        extra = create_memory_pool_configurator(self).calculate_pool_sizes(S.kv_counted, page).max_total_num_tokens
    except Exception as e:
        log.warning("spark-attic: max_total_tokens=%d caps the KV pool with the carveout in it, which then only "
                    "stands in for device memory: %s: %s", cap, type(e).__name__, e)
        return original(self, token_capacity)
    with _held_cap(self, min(cap + extra, token_capacity)):
        out = original(self, token_capacity)
    log.info("spark-attic: max_total_tokens=%d holds the KV pool's device memory; the display carveout (%d MiB) adds "
             "%d tokens: %d", cap, S.kv_counted // MIB, out - cap, out)
    return out


def _user_cap(mr):
    """The --max-total-tokens the engine reads: server_args in 0.5.14, the schedule in 0.5.23."""
    if S.shape == "0.5.23":
        from sglang.srt.runtime_context import get_schedule

        return get_schedule().max_total_tokens
    return mr.server_args.max_total_tokens


@contextmanager
def _held_cap(mr, value):
    """Point the cap the engine reads at `value` for the block, each version through its own API:
    0.5.14 writes server_args; 0.5.23's schedule bag allows the write (a msgspec Struct); 0.5.21's
    runtime context made the bag read-only - its scoped .override(**kw) is the API for exactly this
    (set, run the block, restore, all validated)."""
    if S.shape == "0.5.14":
        saved, args = mr.server_args.max_total_tokens, mr.server_args
        args.max_total_tokens = value
        try:
            yield
        finally:
            args.max_total_tokens = saved
    else:
        from sglang.srt.runtime_context import get_schedule

        bag = get_schedule()
        try:
            hold = bag.override
        except AttributeError:
            # a schedule bag without the override primitive (a plain namespace): the direct write
            saved = bag.max_total_tokens
            bag.max_total_tokens = value
            try:
                yield
            finally:
                bag.max_total_tokens = saved
        else:
            with hold(max_total_tokens=value):
                yield


def _kv_lend(mr):
    """Borrow the KV pool's share of the carveout; its size, or 0. Never raises. Every rank reaches the same
    _agree calls: whether KV placement is off depends only on the configuration, which the ranks share."""
    try:
        why = _kv_blocked(mr)
    except AttributeError as e:
        why = "SGLang's model runner is not the one the plugin knows (%s)" % e
    except Exception as e:
        why = "%s: %s" % (type(e).__name__, e)
    if not why:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        S.layout = _buffer_layout(MHATokenToKVPool)
        if S.layout is None:
            why = ("the KV pool's _create_buffers does not show an allocation pattern the plugin knows "
                   "(separate K and V buffers, or the fused (2, ...) allocation)")
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


def _over_budget(what):
    """The line a post-raise failure leaves in the log. Falling back would leave the engine allocating a KV pool
    above the memory it measured, so the caller raises instead; this says why, and what to do."""
    log.error("spark-attic: %s, %d MiB more than the memory SGLang measured; not falling back: the KV budget was "
              "already raised, so SGLang would overcommit its device memory. Turn the plugin off "
              "(SPARK_ATTIC_PLACE=off) and restart SGLang, or fix the cause logged above.", what,
              S.kv_counted // MIB)


def _pools_hook(original, self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    with S.lock:
        lent, S.kv = S.kv, None
        if lent is not None:
            S.kv_done = True
    if lent is not None:  # the pool was not the kind the plugin fills
        lent.release()
        give_back(S)
        pool = getattr(out, "token_to_kv_pool", None) if S.shape == "0.5.23" else getattr(self, "token_to_kv_pool",
                                                                                          None)
        _over_budget("the KV pool (%s) took no carveout" % type(pool).__name__)
        raise Mismatch("the KV pool (%s) took no carveout" % type(pool).__name__)
    return out


def _buffers_hook(original, self):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    with S.lock:
        lent = S.kv
        usable = lent is not None and type(self) is MHATokenToKVPool and self.kv_cache_layout == "nhd" \
            and not self.enable_custom_mem_pool
        if S.shape != "0.5.14":
            usable = usable and not self.is_quantized_kv_cache and not self.post_capture_active \
                and self.head_dim == self.v_head_dim and not self.use_hnd
        if usable:
            S.kv, S.kv_done = None, True
    if not usable:
        return original(self)
    m = None
    try:
        dev = device_index(self.device)
        item = self.store_dtype.itemsize
        layers = self.layer_num
        if S.layout == "separate":
            # the pool allocates K and V as their own zeros calls per layer, K's for all layers first
            # (0.5.14's mixin pool, 0.5.21's plain NHD pool)
            k = (self.size + self.page_size, self.head_num, self.head_dim)
            v = (self.size + self.page_size, self.head_num, self.v_head_dim)
            expected = [k] * layers + [v] * layers  # _create_buffers' torch.zeros calls, in order
            sizes = [_numel(s) * item for s in expected]
        else:
            # the fused layout (0.5.22+): K and V of a layer in one (2, padded, H, D) allocation, and the
            # pool keeps buf[0, :rows] and buf[1, :rows] views of it
            rows = self.size + self.page_size
            padded = rows if self.use_hnd else -(-rows // self.page_size) * self.page_size
            fused = (2, padded, self.head_num, self.head_dim)
            expected = [fused] * layers
            sizes = [_numel(fused) * item] * layers
        plan = tail_plan(sizes, lent.size)
        m = S.pool.map(plan.front, [lent])
        m.zero()  # SGLang's own buffers start zeroed
    except Exception as e:
        if m is not None:
            m.close()
        lent.release()
        give_back(S)
        _over_budget("%s: %s" % (type(e).__name__, e))
        raise
    mode = _zeros_mode()(expected, sizes, self.store_dtype, dev, plan, m)
    try:
        with mode:
            out = original(self)
    except Exception as e:
        m.close()
        give_back(S)
        _over_budget("the KV buffers could not be placed (%s: %s)" % (type(e).__name__, e))
        raise
    if mode.handed:
        S.keep.append(m)
    else:  # nothing points into the mapping: give it back now
        m.close()
        give_back(S)
    if not mode.handed or mode.miss:
        _over_budget("SGLang allocated its KV buffers differently than planned (%s)" % (mode.miss or "no buffers"))
        raise Mismatch(mode.miss or "no buffers handed")
    if S.layout == "separate":
        given = {id(t) for t in mode.handed}
        kept = sum(1 for t in list(self.k_buffer) + list(self.v_buffer) if id(t) in given)
        if mode.miss or kept != len(expected) - plan.first:
            _over_budget("SGLang allocated its KV buffers differently than planned (%s); %d of %d buffers are in "
                         "the carveout" % (mode.miss or "buffers replaced", kept, len(expected) - plan.first))
            raise Mismatch(mode.miss or "buffers replaced")
        log.info("spark-attic: KV buffers %d-%d of %d (K then V, by layer) in the display carveout (%d MiB), %d MiB "
                 "of device memory glued before it", plan.first, len(expected) - 1, len(expected),
                 plan.attic_used // MIB, plan.front // MIB)
    elif S.layout == "fused":
        half = sizes[0] // 2
        for i in range(plan.first, layers):
            at = m.ptr + plan.offsets[i - plan.first]
            if self.k_buffer[i].data_ptr() != at or self.v_buffer[i].data_ptr() != at + half:
                _over_budget("SGLang's KV buffers %d do not point into the mapping (K at %x, V at %x, expected "
                             "%x and %x)" % (i, self.k_buffer[i].data_ptr(), self.v_buffer[i].data_ptr(), at,
                                             at + half))
                raise Mismatch("KV buffers outside the mapping")
        log.info("spark-attic: KV buffers %d-%d of %d (K and V fused, by layer) in the display carveout (%d MiB), "
                 "%d MiB of device memory glued before it", plan.first, layers - 1, layers,
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

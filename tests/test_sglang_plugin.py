#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_sglang_plugin.py: spark_attic's SGLang plugin on a host with no GPU and no SGLang. SGLang's own plugin loader
and hook registry (sglang/srt/plugins/__init__.py and hook_registry.py, linked from --sglang-src when a tree is given;
a pocket pair with the same register/apply shape otherwise) run over a stand-in SGLang tree in a temp dir, which
carries the functions the plugin hooks or calls with the parameters of the SGLang shape it stands for and a pocket
version of their logic. Both supported shapes run: the 0.5.14 tree (the model runner's KV mixin, one torch.zeros per K
and V buffer) and the 0.5.23 tree (the KV configurator, one fused torch.zeros per layer with K and V views of it). The
stand-in torch keeps tensor bytes in host memory or, for the carveout, in the fake libcuda's; atticd runs with the
fake RM shim, as in test_python.py.

    python3 -I tests/test_sglang_plugin.py --bin build/host [--sglang-src DIR]...

--sglang-src (repeatable) adds the check that a real SGLang source tree still has what the plugin hooks and reads for
its shape; without it that case is skipped, and every other case still runs (the stand-in tree then carries the pocket
plugin loader and hook registry).
"""
import argparse
import ast
import ctypes
import gc
import importlib
import importlib.metadata
import os
import pkgutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from test_server import FAILS, KEY, MIB, Rig, check, wait_free  # noqa: E402
from test_vllm_plugin import LOGS, SMALL, env_for, lent, pattern  # noqa: E402

from spark_attic import sglang_plugin  # noqa: E402
from fake_torch import TORCH, Model  # noqa: E402

SGLANG_SRCS = []


# --------------------------------------------------------------------------------------- a stand-in SGLang tree

PKG = "\n"
# A pocket SGLang plugin framework, for the runs without --sglang-src: the register/apply shape the plugin uses,
# so every case but case_upstream runs on the stand-in tree alone (case_upstream keeps the real pair honest).
POCKET_HOOK_REGISTRY = '''import functools
import pkgutil
from enum import Enum


class HookSource:
    def __init__(self, plugin_name, dist_name):
        self.plugin_name, self.dist_name = plugin_name, dist_name


class HookType(Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


class HookRegistry:
    _hooks = {}
    _patched = set()

    @classmethod
    def register(cls, target, hook, hook_type=HookType.AFTER):
        cls._hooks.setdefault(target, []).append((hook_type, hook))

    @classmethod
    def apply_hooks(cls):
        for target, hooks in cls._hooks.items():
            if target in cls._patched:
                continue
            obj_path, attr = target.rsplit(".", 1)
            obj = pkgutil.resolve_name(obj_path)
            original = getattr(obj, attr)
            for hook_type, hook in hooks:
                if hook_type is HookType.AROUND:
                    original = _around(original, hook)
            setattr(obj, attr, original)
            cls._patched.add(target)


def _around(original, hook):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        return hook(original, *args, **kwargs)
    return wrapper
'''
POCKET_PLUGINS = '''from importlib.metadata import entry_points

from sglang.srt.plugins.hook_registry import HookRegistry

GENERAL_PLUGINS_GROUP = "sglang.srt.plugins"
_loaded = False


def load_plugins():
    global _loaded
    if _loaded:
        return
    _loaded = True
    for ep in entry_points(group=GENERAL_PLUGINS_GROUP):
        try:
            ep.load()()
        except Exception:
            pass
    HookRegistry.apply_hooks()
'''
ENVIRON = '''import os


class _Var:
    def __init__(self, name):
        self.name = name

    def get(self):
        return os.environ.get(self.name)


class _Envs:
    def __getattr__(self, name):
        return _Var(name)


envs = _Envs()
'''
PARALLEL_STATE = '''import types

WORLD = types.SimpleNamespace(world_size=1, cpu_group=None)


def get_world_group():
    return WORLD
'''
MODEL_CONFIG = '''def is_deepseek_v4(config) -> bool:
    return getattr(config, "dsv4", False)


def is_minimax_sparse(config) -> bool:
    return (config.architectures or [None])[0] in ("MiniMaxM3SparseForCausalLM",
                                                   "MiniMaxM3SparseForConditionalGeneration")
'''
PLATFORMS = '''class _Platform:
    out_of_tree = False

    def is_out_of_tree(self):
        return self.out_of_tree


current_platform = _Platform()
'''
MIXIN = '''import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, OtherKVPool


class ModelRunnerKVCacheMixin:
    @torch.inference_mode()  # not in SGLang 0.5.14, but vLLM has such wrappers: the hook reads through them
    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        # SGLang's sum, pocket size: free memory now (on a GB10: what the system has available)
        return int(self.total_memory - torch.cuda.memory_allocated())

    def _init_pools(self):
        cls = OtherKVPool if self.pool_kind == "other" else MHATokenToKVPool
        self.token_to_kv_pool = cls(
            self.max_total_num_tokens, page_size=self.page_size, dtype=self.kv_cache_dtype, head_num=self.head_num,
            head_dim=self.head_dim, v_head_dim=self.head_dim, layer_num=self.num_effective_layers,
            device=self.device, enable_memory_saver=self.server_args.enable_memory_saver)

    def _apply_token_constraints(self, token_capacity: int) -> int:
        user_limit = self.server_args.max_total_tokens
        if user_limit is not None:
            token_capacity = min(token_capacity, user_limit)
        return token_capacity

    def _resolve_memory_pool_config(self, pre_model_load_memory: int):
        from sglang.srt.model_executor.pool_configurator import create_memory_pool_configurator

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(available_bytes, self.page_size)
        constrained = self._apply_token_constraints(config.max_total_num_tokens)
        if constrained != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(constrained, self.page_size)
        return config

    def init_memory_pool(self, pre_model_load_memory: int):
        self.max_total_num_tokens = self._resolve_memory_pool_config(pre_model_load_memory).max_total_num_tokens
        self._init_pools()
'''
RUNNER = '''import types

from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
import torch


class Spec:
    def __init__(self, name=None):
        self.name = name

    def is_none(self):
        return self.name is None


class ModelRunner(ModelRunnerKVCacheMixin):
    def __init__(self, model, total_memory, **over):
        self.model, self.total_memory = model, total_memory
        self.server_args = types.SimpleNamespace(attention_backend="flashinfer", prefill_only_disable_kv_cache=False,
                                                 enable_memory_saver=False, disaggregation_mode="null",
                                                 max_total_tokens=None)
        self.model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace())
        self.device, self.gpu_id = "cuda", 0
        self.use_mla_backend = self.is_hybrid_swa = self.is_draft_worker = False
        self.mambaish_config = None
        self.spec_algorithm = Spec()
        self.kv_cache_dtype = torch.bfloat16
        self.page_size, self.head_num, self.head_dim, self.num_effective_layers = 16, 8, 128, 4
        self.pool_kind = "mha"
        for k, v in over.items():
            setattr(self, k, v)

    def load_model(self):
        self.weight_bytes = sum(p.numel() * p.element_size() for _, p in self.model.named_parameters())
'''
POOL = '''from contextlib import nullcontext

import torch


class _NoSaver:
    def region(self, tag):
        return nullcontext()


class KVCache:
    def __init__(self, size, page_size, dtype, layer_num, device, enable_memory_saver):
        self.size, self.page_size, self.dtype, self.device = size, page_size, dtype, device
        self.store_dtype = dtype
        self.layer_num = layer_num
        self.memory_saver_adapter = _NoSaver()
        self.enable_custom_mem_pool, self.custom_mem_pool = False, None


class MHATokenToKVPool(KVCache):
    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver,
                 v_head_dim=None):
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver)
        self.head_num, self.head_dim = head_num, head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.kv_cache_layout = "nhd"
        self._create_buffers()

    def _create_buffers(self):
        with self.memory_saver_adapter.region("kv_cache"):
            with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool else nullcontext()):
                self.k_buffer = [
                    torch.zeros(K_SHAPE, dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros((self.size + self.page_size, self.head_num, self.v_head_dim),
                                dtype=self.store_dtype, device=self.device)
                    for _ in range(self.layer_num)
                ]
        self.data_ptrs = [x.data_ptr() for x in self.k_buffer + self.v_buffer]


class OtherKVPool(MHATokenToKVPool):
    """A pool kind the plugin does not fill: buffers of its own."""

    def _create_buffers(self):
        shape = (self.size + self.page_size, self.head_num, self.head_dim)
        self.k_buffer = [torch.zeros(shape, dtype=self.store_dtype, device=self.device) for _ in range(self.layer_num)]
        self.v_buffer = [torch.zeros(shape, dtype=self.store_dtype, device=self.device) for _ in range(self.layer_num)]
'''
K_SHAPE = "(self.size + self.page_size, self.head_num, self.head_dim)"
CONFIGURATOR = '''class MemoryPoolConfig:
    def __init__(self, max_total_num_tokens):
        self.max_total_num_tokens = max_total_num_tokens


class DefaultPoolConfigurator:
    def __init__(self, mr):
        self._cell_size = mr.num_effective_layers * 2 * mr.head_num * mr.head_dim * mr.kv_cache_dtype.itemsize

    def calculate_pool_sizes(self, available_bytes: int, page_size: int):
        return MemoryPoolConfig(available_bytes // self._cell_size // page_size * page_size)

    def calculate_pool_sizes_from_max_tokens(self, max_total_num_tokens: int, page_size: int):
        return MemoryPoolConfig(max_total_num_tokens // page_size * page_size)


def create_memory_pool_configurator(mr):
    return DefaultPoolConfigurator(mr)
'''
SIZES_PARAMS = "self, available_bytes: int, page_size: int"


# ----------------------------------------------------------------------------------------- the 0.5.23 stand-in tree

SCHEDULE_23 = '''import types

SCHEDULE = types.SimpleNamespace(max_total_tokens=None, page_size=16, mem_fraction_static=0.9)
MEMORY = types.SimpleNamespace(enable_page_major_kv_layout=False)


def get_schedule():
    return SCHEDULE


def get_memory():
    return MEMORY
'''
RUNNER_MIN_23 = '''import types


class Spec:
    def __init__(self, name=None):
        self.name = name

    def is_none(self):
        return self.name is None


class ModelRunner:
    def __init__(self, model=None):
        self.model = model

    def load_model(self):
        self.weight_bytes = sum(p.numel() * p.element_size() for _, p in self.model.named_parameters())
'''
CONF_23 = '''import types

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, OtherKVPool
from sglang.srt.model_executor.model_runner import Spec


class _InitializedPools:
    def __init__(self, token_to_kv_pool):
        self.token_to_kv_pool = token_to_kv_pool


class KVCacheConfigurator:
    def __init__(self, model, total_memory, **over):
        self.model, self.total_memory = model, total_memory
        self.server_args = types.SimpleNamespace(attention_backend="flashinfer", prefill_only_disable_kv_cache=False,
                                                 enable_memory_saver=False, disaggregation_mode="null")
        self.model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
                                                  head_dim=128, v_head_dim=128)
        self.device, self.gpu_id = "cuda", 0
        self.use_mla_backend = self.is_hybrid_swa = self.is_draft_worker = False
        self.mambaish_config = None
        self.spec_algorithm = Spec()
        self.kv_cache_dtype = torch.bfloat16
        self.page_size, self.head_num, self.head_dim, self.num_effective_layers = 16, 8, 128, 4
        self.pool_kind = "mha"
        self.post_capture_kv_active = False
        self.token_to_kv_pool = None
        for k, v in over.items():
            setattr(self, k, v)

    @torch.inference_mode()
    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        # SGLang's sum, pocket size: free memory now (on a GB10: what the system has available)
        return int(self.total_memory - torch.cuda.memory_allocated())

    def _apply_token_constraints(self, token_capacity: int) -> int:
        from sglang.srt.runtime_context import get_schedule

        user_limit = get_schedule().max_total_tokens
        if user_limit is not None:
            token_capacity = min(token_capacity, user_limit)
        return token_capacity

    def _resolve_memory_pool_config(self, pre_model_load_memory: int):
        from sglang.srt.model_executor.pool_configurator import create_memory_pool_configurator

        available_bytes = self._profile_available_bytes(pre_model_load_memory)
        configurator = create_memory_pool_configurator(self)
        config = configurator.calculate_pool_sizes(available_bytes, self.page_size)
        constrained = self._apply_token_constraints(config.max_total_num_tokens)
        if constrained != config.max_total_num_tokens:
            config = configurator.calculate_pool_sizes_from_max_tokens(constrained, self.page_size)
        return config

    def _init_pools(self, *, sizes, req_to_token_pool, token_to_kv_pool_allocator):
        cls = OtherKVPool if self.pool_kind == "other" else MHATokenToKVPool
        self.token_to_kv_pool = cls(
            self.max_total_num_tokens, page_size=self.page_size, dtype=self.kv_cache_dtype, head_num=self.head_num,
            head_dim=self.head_dim, v_head_dim=self.head_dim, layer_num=self.num_effective_layers,
            device=self.device, enable_memory_saver=self.server_args.enable_memory_saver)
        return _InitializedPools(self.token_to_kv_pool)

    def configure(self, *, pre_model_load_memory: int):
        self.max_total_num_tokens = self._resolve_memory_pool_config(pre_model_load_memory).max_total_num_tokens
        return self._init_pools(sizes=None, req_to_token_pool=None, token_to_kv_pool_allocator=None)
'''
POOL_23 = '''from contextlib import nullcontext

import torch


class _NoSaver:
    def region(self, tag):
        return nullcontext()


class KVCache:
    def __init__(self, size, page_size, dtype, layer_num, device, enable_memory_saver):
        self.size, self.page_size, self.dtype, self.device = size, page_size, dtype, device
        self.store_dtype = dtype
        self.layer_num = layer_num
        self.memory_saver_adapter = _NoSaver()
        self.enable_custom_mem_pool, self.custom_mem_pool = False, None


class MHATokenToKVPool(KVCache):
    def __init__(self, size, page_size, dtype, head_num, head_dim, layer_num, device, enable_memory_saver,
                 v_head_dim=None):
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver)
        self.head_num, self.head_dim = head_num, head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.kv_cache_layout = "nhd"
        self.use_hnd = False
        self.is_quantized_kv_cache = False
        self.post_capture_active = False
        self._create_buffers()

    def _kv_buffer_shapes(self):
        rows = self.size + self.page_size
        return (rows, self.head_num, self.head_dim), (rows, self.head_num, self.v_head_dim)

    def _create_buffers(self):
        with self.memory_saver_adapter.region("kv_cache"):
            with (torch.cuda.use_mem_pool(self.custom_mem_pool) if self.enable_custom_mem_pool else nullcontext()):
                k_shape, v_shape = self._kv_buffer_shapes()

                def _zeros(shape):
                    return torch.zeros(shape, dtype=self.store_dtype, device=self.device)

                if k_shape == v_shape:
                    # as in SGLang 0.5.23: K and V of a layer share one fused allocation, views of it
                    rows = k_shape[0]
                    padded = rows
                    self.k_buffer, self.v_buffer, self.fused = [], [], []
                    for _ in range(self.layer_num):
                        buf = _zeros((2, padded, *k_shape[1:]))
                        self.fused.append(buf)
                        self.k_buffer.append(buf[0, :rows])
                        self.v_buffer.append(buf[1, :rows])
                else:
                    self.k_buffer = [_zeros(k_shape) for _ in range(self.layer_num)]
                    self.v_buffer = [_zeros(v_shape) for _ in range(self.layer_num)]
        self.data_ptrs = [x.data_ptr() for x in self.k_buffer + self.v_buffer]


class OtherKVPool(MHATokenToKVPool):
    """A pool kind the plugin does not fill: buffers of its own."""

    def _create_buffers(self):
        shape = (self.size + self.page_size, self.head_num, self.head_dim)
        self.k_buffer = [torch.zeros(shape, dtype=self.store_dtype, device=self.device) for _ in range(self.layer_num)]
        self.v_buffer = [torch.zeros(shape, dtype=self.store_dtype, device=self.device) for _ in range(self.layer_num)]
'''
CONFIGURATOR_23 = '''class MemoryPoolConfig:
    def __init__(self, max_total_num_tokens):
        self.max_total_num_tokens = max_total_num_tokens


class DefaultPoolConfigurator:
    def __init__(self, kvc):
        self._cell_size = kvc.num_effective_layers * 2 * kvc.head_num * kvc.head_dim * kvc.kv_cache_dtype.itemsize

    def calculate_pool_sizes(self, available_bytes: int, page_size: int):
        return MemoryPoolConfig(available_bytes // self._cell_size // page_size * page_size)

    def calculate_pool_sizes_from_max_tokens(self, max_total_num_tokens: int, page_size: int):
        return MemoryPoolConfig(max_total_num_tokens // page_size * page_size)


def create_memory_pool_configurator(kvc):
    return DefaultPoolConfigurator(kvc)
'''


POOL_21 = POOL_23.replace('''
                if k_shape == v_shape:
                    # as in SGLang 0.5.23: K and V of a layer share one fused allocation, views of it
                    rows = k_shape[0]
                    padded = rows
                    self.k_buffer, self.v_buffer, self.fused = [], [], []
                    for _ in range(self.layer_num):
                        buf = _zeros((2, padded, *k_shape[1:]))
                        self.fused.append(buf)
                        self.k_buffer.append(buf[0, :rows])
                        self.v_buffer.append(buf[1, :rows])
                else:
                    self.k_buffer = [_zeros(k_shape) for _ in range(self.layer_num)]
                    self.v_buffer = [_zeros(v_shape) for _ in range(self.layer_num)]
''', '''
                # as in SGLang 0.5.21: K and V of a layer are their own allocations, K's for all layers first
                self.k_buffer = [_zeros(k_shape) for _ in range(self.layer_num)]
                self.v_buffer = [_zeros(v_shape) for _ in range(self.layer_num)]
''')
assert "fused" not in POOL_21.split("data_ptrs")[0].split("class OtherKVPool")[0]


def plugins_files():
    """SGLang's own plugin loader and hook registry when --sglang-src gives a tree, else the pocket pair with the
    same register/apply shape: every case but case_upstream runs without the upstream trees."""
    up = os.path.join(SGLANG_SRCS[0], "python", "sglang", "srt", "plugins") if SGLANG_SRCS else None
    if up is None:
        return {"sglang/srt/plugins/__init__.py": POCKET_PLUGINS,
                "sglang/srt/plugins/hook_registry.py": POCKET_HOOK_REGISTRY}
    return {"sglang/srt/plugins/__init__.py": ("link", os.path.join(up, "__init__.py")),
            "sglang/srt/plugins/hook_registry.py": ("link", os.path.join(up, "hook_registry.py"))}


def tree21():
    """The 0.5.21-shaped stand-in: 0.5.23's hook surface (the configurator owns the sizing chain) but
    the plain MHA pool allocates K and V separately, as 0.5.21's _create_buffers_normal does."""
    t = tree23()
    t["sglang/srt/mem_cache/memory_pool.py"] = POOL_21
    return t


def tree23():
    """The 0.5.23-shaped stand-in: the KV configurator owns the sizing chain, the pool fuses K and V."""
    return {
        "sglang/__init__.py": PKG, "sglang/srt/__init__.py": PKG, "sglang/srt/environ.py": ENVIRON,
        **plugins_files(),
        "sglang/srt/configs/__init__.py": PKG, "sglang/srt/configs/model_config.py": MODEL_CONFIG,
        "sglang/srt/platforms/__init__.py": PLATFORMS,
        "sglang/srt/distributed/__init__.py": PKG, "sglang/srt/distributed/parallel_state.py": PARALLEL_STATE,
        "sglang/srt/runtime_context.py": SCHEDULE_23,
        "sglang/srt/model_executor/__init__.py": PKG,
        "sglang/srt/model_executor/model_runner.py": RUNNER_MIN_23,
        "sglang/srt/model_executor/pool_configurator.py": CONFIGURATOR_23,
        "sglang/srt/mem_cache/__init__.py": PKG,
        "sglang/srt/mem_cache/kv_cache_configurator.py": CONF_23,
        "sglang/srt/mem_cache/memory_pool.py": POOL_23,
    }


def tree(k_shape=K_SHAPE, buffers_params="self", sizes_params=SIZES_PARAMS):
    return {
        "sglang/__init__.py": PKG, "sglang/srt/__init__.py": PKG, "sglang/srt/environ.py": ENVIRON,
        **plugins_files(),
        "sglang/srt/configs/__init__.py": PKG, "sglang/srt/configs/model_config.py": MODEL_CONFIG,
        "sglang/srt/platforms/__init__.py": PLATFORMS,
        "sglang/srt/distributed/__init__.py": PKG, "sglang/srt/distributed/parallel_state.py": PARALLEL_STATE,
        "sglang/srt/model_executor/__init__.py": PKG,
        "sglang/srt/model_executor/model_runner_kv_cache_mixin.py": MIXIN,
        "sglang/srt/model_executor/model_runner.py": RUNNER,
        "sglang/srt/model_executor/pool_configurator.py": CONFIGURATOR.replace(
            "def calculate_pool_sizes(%s):" % SIZES_PARAMS, "def calculate_pool_sizes(%s):" % sizes_params, 1),
        "sglang/srt/mem_cache/__init__.py": PKG,
        "sglang/srt/mem_cache/memory_pool.py": POOL.replace("K_SHAPE", k_shape).replace(
            "def _create_buffers(self):", "def _create_buffers(%s):" % buffers_params, 1),
    }


class World:
    """The other ranks of SGLang's world group: each all_reduce MIN gives the smaller of this rank's value and the
    next one in `peers`."""

    def __init__(self, size, peers):
        self.size, self.peers, self.seen = size, list(peers), []

    def reduce(self, op, value):
        assert op == "min"
        self.seen.append(value)
        return min(value, self.peers.pop(0))


class Sgl:
    """A fresh stand-in SGLang on sys.path, a fresh plugin state, and SGLang's own load_plugins() run over it."""

    def __init__(self, env, files=None, world=None, kind="0514"):
        for name in [n for n in sys.modules if n == "sglang" or n.startswith("sglang.")]:
            del sys.modules[name]
        self.kind = kind
        self.dir = tempfile.mkdtemp(prefix="attic-sglang-")
        for rel, src in (files or tree()).items():
            path = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if isinstance(src, tuple):
                os.symlink(src[1], path)
            else:
                with open(path, "w") as f:
                    f.write(src)
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()
        for k in [k for k in os.environ if k.startswith("SPARK_ATTIC_") and k != "SPARK_ATTIC_LIBCUDA"]:
            del os.environ[k]
        os.environ.pop("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", None)
        os.environ.update(env)
        sglang_plugin.S = sglang_plugin._State()
        LOGS.lines.clear()
        if world is not None:
            ps = importlib.import_module("sglang.srt.distributed.parallel_state")
            ps.WORLD.world_size, ps.WORLD.cpu_group = world.size, world
        importlib.import_module("sglang.srt.plugins").load_plugins()
        if kind == "0514":
            self.mixin = importlib.import_module("sglang.srt.model_executor.model_runner_kv_cache_mixin")
        else:
            self.conf = importlib.import_module("sglang.srt.mem_cache.kv_cache_configurator")
            self.runtime = importlib.import_module("sglang.srt.runtime_context")
        self.mr = importlib.import_module("sglang.srt.model_executor.model_runner")
        self.pool = importlib.import_module("sglang.srt.mem_cache.memory_pool")
        self.platforms = importlib.import_module("sglang.srt.platforms")

    def wrapped(self):
        """What SGLang's hook registry wrapped: its wrapper's code is in hook_registry.py (a decorator's is not)."""
        out = set()
        for shape in sglang_plugin.SHAPES:
            for key, (target, _) in sglang_plugin.TARGETS[shape].items():
                try:
                    fn = pkgutil.resolve_name(target)
                except Exception:
                    continue
                if getattr(fn, "__code__", None) and fn.__code__.co_filename.endswith("hook_registry.py"):
                    out.add(key)
        return out

    def runner(self, budget=100 * MIB, model=None, **over):
        """A model runner that measures `budget` bytes free once its model's weights are made."""
        gc.collect()  # what earlier cases left must not be freed in the middle of this one
        return self.mr.ModelRunner(model or Model([]), budget + TORCH.allocated, **over)

    def configurator(self, budget=100 * MIB, model=None, **over):
        """The 0.5.23 sizing object, which measures `budget` bytes free once its model's weights are made."""
        gc.collect()
        return self.conf.KVCacheConfigurator(model or Model([]), budget + TORCH.allocated, **over)

    def done(self):
        st = sglang_plugin.S
        for m in st.keep:
            m.close()
        st.keep.clear()
        if st.kv is not None:
            st.kv.release()
        if st.pool is not None:
            st.pool.close()
        sys.path.remove(self.dir)


def measured(r):
    """What SGLang itself measures, without the plugin."""
    return r._profile_available_bytes.__wrapped__(r, 0)


# ------------------------------------------------------------------------------------------------------- cases

def case_entry_point(rig):
    eps = [e for e in importlib.metadata.entry_points(group="sglang.srt.plugins") if e.name == "spark_attic"]
    check(len(eps) == 1 and eps[0].value == "spark_attic.sglang_plugin:register" and eps[0].load()
          is sglang_plugin.register, "the static dist-info registers the plugin: %r" % eps)
    import tomllib
    with open(os.path.join(ROOT, "python", "pyproject.toml"), "rb") as f:
        pp = tomllib.load(f)
    check(pp["project"]["entry-points"]["sglang.srt.plugins"]["spark_attic"] == eps[0].value,
          "pyproject.toml says the same")


def case_settings(rig):
    v = Sgl(env_for(None, PLACE="bogus"))
    try:
        check(sglang_plugin.S.placement.place == () and LOGS.has("SGLang plugin off") and not v.wrapped(),
              "a bad setting: plugin off, logged, nothing hooked")
    finally:
        v.done()
    v = Sgl(env_for(None, PLACE="off"))
    try:
        check(LOGS.has("SPARK_ATTIC_PLACE=off") and not v.wrapped(), "off: nothing hooked")
    finally:
        v.done()


def case_kv(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig))
    try:
        check(v.wrapped() == {"profile", "cap", "pools", "buffers"}, "kv: SGLang's registry applied the four KV "
              "hooks, load_model left alone: %r" % v.wrapped())
        r = v.runner()
        r.load_model()
        check(measured(r) == 100 * MIB, "SGLang measures 100 MiB")
        got = r._profile_available_bytes(0)
        check(got == 162 * MIB and lent(rig) == 62 * MIB, "100 MiB measured + the 62 MiB lent: %d MiB" % (got // MIB))
        r.init_memory_pool(0)
        check(r.max_total_num_tokens == 10368, "sized again before the pool: the same 162 MiB, 10368 tokens, not "
              "6400: %d" % r.max_total_num_tokens)
        p = r.token_to_kv_pool
        kinds = [t.kind for t in p.k_buffer + p.v_buffer]
        check(kinds == ["buf"] * 5 + ["va"] * 3, "K 0-3 and V 0 where SGLang puts them, V 1-3 in the mapping: %r"
              % kinds)
        m = sglang_plugin.S.keep[0]
        vas = [t.va for t in p.v_buffer[1:]]
        nb = (10368 + 16) * 8 * 128 * 2
        check(m.size == 66 * MIB and m.attic_bytes == 62 * MIB and vas == [m.ptr, m.ptr + 22 * MIB, m.ptr + 44 * MIB]
              and p.data_ptrs[5:] == vas, "4 MiB device + 62 MiB attic, buffers at 0, 22, 44 MiB: %r %r" % (m, vas))
        check(all(t.shape == (10384, 8, 128) and t.dtype is TORCH.bfloat16 and t.nbytes == nb
                  for t in p.k_buffer + p.v_buffer), "every buffer has SGLang's shape and dtype")
        check(m.read(0, 4096) == bytes(4096) and m.read(m.size - 4096, 4096) == bytes(4096), "the mapping is zeroed")
        seam = TORCH.zeros(nb)
        data = pattern(8192, 7)
        ctypes.memmove(ctypes.addressof(seam.buf) + 4 * MIB - 4096, data, 8192)
        p.v_buffer[1].copy_(seam)
        check(m.read(4 * MIB - 4096, 8192) == data, "a write across the device/carveout seam reads back")
        check(LOGS.has("KV buffers 5-7 of 8"), "logged")
        LOGS.lines.clear()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("after building it"),
              "sized again after the pool was built: no carveout")
    finally:
        v.done()
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "closed: all given back: %r" % inf)


def case_cap(rig):
    """--max-total-tokens holds the device memory; the carveout's tokens come on top of it."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig))
    try:
        r = v.runner()
        r.server_args.max_total_tokens = 4000
        r.init_memory_pool(0)
        check(r.max_total_num_tokens == 7968 and r.server_args.max_total_tokens == 4000,
              "a binding cap of 4000 tokens + the 62 MiB carveout's 3968 tokens of 16 KiB: %d; the cap restored: %r"
              % (r.max_total_num_tokens, r.server_args.max_total_tokens))
        check(LOGS.has("max_total_tokens=4000 holds the KV pool's device memory; the display carveout (62 MiB) adds "
                       "3968 tokens: 7968"), "logged")
        p = r.token_to_kv_pool
        m = sglang_plugin.S.keep[0]
        kinds = [t.kind for t in p.k_buffer + p.v_buffer]
        check(kinds == ["buf"] * 4 + ["va"] * 4 and m.attic_bytes == 62 * MIB, "V 0-3 in the mapping, all 62 MiB "
              "of the slice used: %r %r" % (kinds, m))
        check((7968 + 16) * 16384 - m.attic_bytes == (4000 + 16) * 16384,
              "so device memory holds exactly what the cap alone would")
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)
    v = Sgl(env_for(rig))
    try:
        r = v.runner()
        r.server_args.max_total_tokens = 20000
        r.init_memory_pool(0)
        check(r.max_total_num_tokens == 10368 and not LOGS.has("holds the KV pool's device memory"),
              "a cap over the measured memory + the carveout does not bind: 10368 tokens, as without a cap")
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)
    v = Sgl(env_for(rig))
    try:
        cfg = importlib.import_module("sglang.srt.model_executor.pool_configurator").DefaultPoolConfigurator
        sizes = cfg.calculate_pool_sizes

        def page_one_fails(self, available_bytes, page_size):  # SGLang's own sizing (page 16) still works
            if page_size == 1:
                raise RuntimeError("no sizes")
            return sizes(self, available_bytes, page_size)
        cfg.calculate_pool_sizes = page_one_fails
        r = v.runner()
        r.server_args.max_total_tokens = 4000
        r.init_memory_pool(0)
        p = r.token_to_kv_pool
        check(r.max_total_num_tokens == 4000 and r.server_args.max_total_tokens == 4000
              and LOGS.has("which then only stands in for device memory: RuntimeError: no sizes"),
              "the carveout's tokens cannot be counted: the cap binds as set, logged: %d" % r.max_total_num_tokens)
        check(sum(1 for t in p.k_buffer + p.v_buffer if t.kind == "va") == 8, "and the carveout holds that pool")
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_kv_other_pool(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig))
    try:
        r = v.runner(pool_kind="other")
        raised = None
        try:
            r.init_memory_pool(0)
        except Exception as e:
            raised = e
        check(raised is not None and not sglang_plugin.S.keep,
              "a pool kind the plugin does not fill: the engine stops coming up, it does not allocate 62 MiB more "
              "than SGLang measured")
        check(LOGS.has("took no carveout") and LOGS.has("not falling back"), "one line says why and what to do")
        inf = wait_free(rig.sock, 62 * MIB)
        check(inf and inf["free"] == 62 * MIB, "the unused slice went back at once: %r" % inf)
    finally:
        v.done()


def case_kv_miss(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree(k_shape="(self.size, self.head_num, self.head_dim)"))
    try:
        r = v.runner()
        raised = None
        try:
            r.init_memory_pool(0)
        except Exception as e:
            raised = e
        pool = getattr(r, "token_to_kv_pool", None)
        check(raised is not None and (pool is None or all(t.kind == "buf" for t in pool.k_buffer)),
              "an unplanned torch.zeros after the raise: the engine stops, nothing was substituted")
        check(LOGS.has("differently than planned (call 0: torch.zeros((10368, 8, 128)") and LOGS.has(
            "not falling back"), "logged with the call, and why the engine stops")
        inf = wait_free(rig.sock, 62 * MIB)
        check(not sglang_plugin.S.keep and inf and inf["free"] == 62 * MIB,
              "nothing points into the mapping: closed, the slice back at once: %r" % inf)
    finally:
        v.done()


def case_blocked(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig))
    try:
        def no_mamba(r):
            del r.mambaish_config

        def custom_pool(r):
            os.environ["SGLANG_MOONCAKE_CUSTOM_MEM_POOL"] = "NVLINK"

        def out_of_tree(r):
            v.platforms.current_platform.out_of_tree = True

        for why, setup in (
                ("uses MLA", lambda r: setattr(r, "use_mla_backend", True)),
                ("sliding-window", lambda r: setattr(r, "is_hybrid_swa", True)),
                ("speculative decoding", lambda r: setattr(r, "spec_algorithm", v.mr.Spec("EAGLE"))),
                ("DeepSeek V4", lambda r: setattr(r.model_config.hf_config, "dsv4", True)),
                ("memory saver", lambda r: setattr(r.server_args, "enable_memory_saver", True)),
                ("PD disaggregation", lambda r: setattr(r.server_args, "disaggregation_mode", "decode")),
                ("custom KV memory pool", custom_pool),
                ("out-of-tree platform", out_of_tree),
                ("not the one the plugin knows", no_mamba)):
            r = v.runner()
            setup(r)
            LOGS.lines.clear()
            check(r._profile_available_bytes(0) == measured(r) and LOGS.has(why), "%s: KV placement off" % why)
            os.environ.pop("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", None)
            v.platforms.current_platform.out_of_tree = False
        check(lent(rig) == 0, "nothing lent")
    finally:
        v.done()


def case_hnd(rig):
    """An HND KV cache is not the plain nhd pool: the placement turns off before the budget is raised, with one
    log line, and the engine starts."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        os.environ["SGLANG_USE_HND_KVCACHE"] = "1"
        r = v.configurator()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("KV placement off: an HND KV cache"),
              "an HND KV cache: no carveout added to the KV budget, one log line")
        raised = None
        try:
            r.configure(pre_model_load_memory=0)
        except Exception as e:
            raised = e
        check(raised is None and r.max_total_num_tokens == 6400 and lent(rig) == 0
              and not LOGS.has("not falling back"),
              "and the engine starts: the pool as SGLang sized it (%d tokens), no raise" % r.max_total_num_tokens)
    finally:
        os.environ.pop("SGLANG_USE_HND_KVCACHE", None)
        v.done()


def case_page_major(rig):
    """The page-major pool class (--enable-page-major-kv-layout) is not the plain MHATokenToKVPool: off before the
    budget is raised, with one log line, and the engine starts."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        v.runtime.MEMORY.enable_page_major_kv_layout = True
        r = v.configurator()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("KV placement off: the page-major KV layout"),
              "the page-major KV layout: no carveout added to the KV budget, one log line")
        raised = None
        try:
            r.configure(pre_model_load_memory=0)
        except Exception as e:
            raised = e
        check(raised is None and r.max_total_num_tokens == 6400 and lent(rig) == 0
              and not LOGS.has("not falling back"),
              "and the engine starts: the pool as SGLang sized it (%d tokens), no raise" % r.max_total_num_tokens)
    finally:
        v.done()


def case_minimax_sparse(rig):
    """A MiniMax-sparse model gets its own pool kind: off before the budget is raised, one log line, the engine
    starts."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        r = v.configurator()
        r.model_config.hf_config.architectures = ["MiniMaxM3SparseForCausalLM"]
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("KV placement off: a MiniMax-sparse model"),
              "a MiniMax-sparse model: no carveout added to the KV budget, one log line")
        raised = None
        try:
            r.configure(pre_model_load_memory=0)
        except Exception as e:
            raised = e
        check(raised is None and r.max_total_num_tokens == 6400 and lent(rig) == 0
              and not LOGS.has("not falling back"),
              "and the engine starts: the pool as SGLang sized it (%d tokens), no raise" % r.max_total_num_tokens)
    finally:
        v.done()


def case_post_capture(rig):
    """Post-capture KV sizing keeps buffers of its own (and an attribute the plugin cannot read counts as blocked):
    off before the budget is raised, one log line, the engine starts."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        r = v.configurator(post_capture_kv_active=True)
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("KV placement off: post-capture KV sizing"),
              "post-capture KV sizing: no carveout added to the KV budget, one log line")
        raised = None
        try:
            r.configure(pre_model_load_memory=0)
        except Exception as e:
            raised = e
        check(raised is None and r.max_total_num_tokens == 6400 and lent(rig) == 0
              and not LOGS.has("not falling back"),
              "and the engine starts: the pool as SGLang sized it (%d tokens), no raise" % r.max_total_num_tokens)
        del r.post_capture_kv_active
        LOGS.lines.clear()
        check(r._profile_available_bytes(0) == measured(r)
              and LOGS.has("post_capture_kv_active cannot be read"),
              "an attribute the plugin cannot read counts as blocked, not as a placement")
    finally:
        v.done()


def case_unequal_head_dims(rig):
    """K and V head sizes that differ are not the buffers the fused pool plans: off before the budget is raised, one
    log line, and the engine starts."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        r = v.configurator()
        r.model_config.v_head_dim = 64
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("KV placement off: K and V head sizes differ"),
              "unequal K and V head sizes: no carveout added to the KV budget, one log line")
        raised = None
        try:
            r.configure(pre_model_load_memory=0)
        except Exception as e:
            raised = e
        check(raised is None and r.max_total_num_tokens == 6400 and lent(rig) == 0
              and not LOGS.has("not falling back"),
              "and the engine starts: the pool as SGLang sized it (%d tokens), no raise" % r.max_total_num_tokens)
    finally:
        v.done()


def case_weights(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig, PLACE="weights"))
    try:
        check(v.wrapped() == {"load"}, "weights: only load_model hooked: %r" % v.wrapped())
        nb = 1281 * 2048 * 2
        embed = TORCH.param(nb, TORCH.float16, (1281, 2048), 3)
        other = TORCH.param(MIB, TORCH.bfloat16, (512, 1024), 9)
        before = embed.data.raw()
        r = v.runner(model=Model([("model.embed_tokens.weight", embed), ("model.layers.0.w", other),
                                  ("lm_head.weight", embed)]))
        r.load_model()
        check(embed.data.kind == "va" and embed.data.shape == (1281, 2048) and embed.data.dtype is TORCH.float16
              and embed.data.raw() == before, "embed_tokens moved, shape, dtype and bytes kept")
        check(other.data.kind == "buf", "other weights stay")
        check(len(sglang_plugin.S.keep) == 1 and lent(rig) == 6 * MIB, "tied lm_head moved once: one 6 MiB slice")
        r.init_memory_pool(0)
        check(r.max_total_num_tokens == (100 * MIB + nb) // 16384 // 16 * 16,
              "SGLang measures after the load, so the pool grows by what moved: %d tokens" % r.max_total_num_tokens)
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_weights_and_kv(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig, PLACE="weights,kv", MIB="40", WEIGHTS="lm_head,embed_tokens"))
    try:
        check(v.wrapped() == {"profile", "cap", "pools", "buffers", "load"}, "both: all five hooks")
        embed = TORCH.param(4 * MIB, TORCH.float16, (2048, 1024), 1)
        head = TORCH.param(40 * MIB, TORCH.float16, (20480, 1024), 2)
        r = v.runner(model=Model([("model.embed_tokens.weight", embed), ("lm_head.weight", head)]))
        r.load_model()
        check(embed.data.kind == "va" and head.data.kind == "buf" and LOGS.has("lm_head.weight stays"),
              "under a 40 MiB cap the 40 MiB lm_head stays, embed_tokens moves")
        got = r._profile_available_bytes(0)
        check(got == 104 * MIB + 36 * MIB and lent(rig) == 40 * MIB,
              "the KV pool takes the rest of the cap, 36 MiB, on top of 104 measured: %d MiB" % (got // MIB))
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_fail_open(rig):
    v = Sgl(env_for(None, PLACE="weights,kv"))
    try:
        embed = TORCH.param(2 * MIB, TORCH.float16, (1024, 1024), 4)
        r = v.runner(model=Model([("model.embed_tokens.weight", embed)]))
        r.load_model()
        check(embed.data.kind == "buf" and LOGS.has("weights stay in device memory"), "no atticd: weights stay")
        check(r._profile_available_bytes(0) == 100 * MIB and LOGS.has("stays as SGLang sized it"),
              "no atticd: the KV pool as SGLang measured it")
        r.init_memory_pool(0)
        p = r.token_to_kv_pool
        check(r.max_total_num_tokens == 6400 and all(t.kind == "buf" for t in p.k_buffer + p.v_buffer),
              "and SGLang builds it all itself")
    finally:
        v.done()


def case_drift(rig):
    rig.start(env=SMALL)
    v = Sgl(env_for(rig, PLACE="weights,kv"), tree(buffers_params="self, page_size=None"))
    try:
        check(v.wrapped() == {"load"} and LOGS.has("KV placement off")
              and LOGS.has("takes (self, page_size); the plugin knows (self)"),
              "a changed signature: the whole KV group stays off, weights still on: %r" % v.wrapped())
        r = v.runner()
        check(r._profile_available_bytes(0) == 100 * MIB and lent(rig) == 0, "no carveout for the KV pool")
    finally:
        v.done()
    v = Sgl(env_for(rig), tree(sizes_params="self, available_bytes: int"))
    try:
        check(not v.wrapped() and LOGS.has("calculate_pool_sizes takes (self, available_bytes); the plugin knows "
                                           "(self, available_bytes, page_size)"),
              "a function the cap hook calls changed: the KV group stays off too: %r" % v.wrapped())
    finally:
        v.done()


def case_kv23(rig):
    """The 0.5.23 shape: the configurator sizes, the pool fuses K and V, the carveout holds the tail."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        check(v.wrapped() == {"profile", "cap", "pools", "buffers"},
              "kv on 0.5.23: SGLang's registry applied the four KV hooks: %r" % v.wrapped())
        r = v.configurator()
        check(measured(r) == 100 * MIB, "SGLang measures 100 MiB")
        got = r._profile_available_bytes(0)
        check(got == 162 * MIB and lent(rig) == 62 * MIB, "100 MiB measured + the 62 MiB lent: %d MiB" % (got // MIB))
        r.configure(pre_model_load_memory=0)
        check(r.max_total_num_tokens == 10368, "sized with the carveout: 10368 tokens, not 6400: %d"
              % r.max_total_num_tokens)
        p = r.token_to_kv_pool
        rows = 10368 + 16
        fused_bytes = 2 * rows * 8 * 128 * 2
        check([t.kind for t in p.fused] == ["buf"] * 2 + ["va"] * 2,
              "fused buffers 0-1 where SGLang puts them, 2-3 in the mapping: %r" % [t.kind for t in p.fused])
        m = sglang_plugin.S.keep[0]
        check(m.attic_bytes == 62 * MIB and p.k_buffer[2].data_ptr() == m.ptr
              and p.v_buffer[2].data_ptr() == m.ptr + fused_bytes // 2
              and p.k_buffer[3].data_ptr() == m.ptr + 42 * MIB,
              "the fused buffers' K and V views land in the mapping: %r" % m)
        check(all(t.shape == (rows, 8, 128) and t.dtype is TORCH.bfloat16 for t in p.k_buffer + p.v_buffer),
              "every buffer keeps SGLang's shape and dtype")
        check(m.read(0, 4096) == bytes(4096) and m.read(m.size - 4096, 4096) == bytes(4096), "the mapping is zeroed")
        seam = TORCH.zeros(fused_bytes // 2)
        data = pattern(8192, 7)
        ctypes.memmove(ctypes.addressof(seam.buf), data, 8192)
        p.k_buffer[3].copy_(seam)
        check(m.read(42 * MIB, 8192) == data, "a write through SGLang's own view reads back in the carveout")
        check(LOGS.has("KV buffers 2-3 of 4 (K and V fused, by layer)"), "logged")
        LOGS.lines.clear()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("after building it"),
              "sized again after the pool was built: no carveout")
    finally:
        v.done()
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "closed: all given back: %r" % inf)


def case_kv21(rig):
    """0.5.21: 0.5.23's hook signatures, but the pool allocates K and V separately. The buffers hook
    follows the pool's own layout (read from its source before the budget is raised), so the placement
    works exactly as on 0.5.14 - and a plugin that planned the fused layout here would raise after the
    budget was raised (window 3's defect: the node's sglang-kv.log, Mismatch call 0)."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree21(), kind="0523")
    try:
        check(v.wrapped() == {"profile", "cap", "pools", "buffers"},
              "kv on 0.5.21: the registry applied the four KV hooks (the 0.5.23 surface): %r" % v.wrapped())
        r = v.configurator()
        got = r._profile_available_bytes(0)
        check(got == 162 * MIB and lent(rig) == 62 * MIB, "100 MiB measured + the 62 MiB lent: %d MiB" % (got // MIB))
        r.configure(pre_model_load_memory=0)
        check(r.max_total_num_tokens == 10368, "sized with the carveout: 10368 tokens, not 6400: %d"
              % r.max_total_num_tokens)
        p = r.token_to_kv_pool
        rows = 10368 + 16
        check([t.kind for t in p.k_buffer] == ["buf"] * 4,
              "every K buffer where SGLang puts it (device memory): %r" % [t.kind for t in p.k_buffer])
        check([t.kind for t in p.v_buffer] == ["buf", "va", "va", "va"],
              "the tail of the K-then-V order in the carveout: %r" % [t.kind for t in p.v_buffer])
        check(all(t.shape == (rows, 8, 128) and t.dtype is TORCH.bfloat16 for t in p.k_buffer + p.v_buffer),
              "every buffer keeps SGLang's shape and dtype")
        m = sglang_plugin.S.keep[0]
        check(m.attic_bytes == 62 * MIB and p.v_buffer[1].data_ptr() == m.ptr
              and p.v_buffer[2].data_ptr() == m.ptr + 22 * MIB and p.v_buffer[3].data_ptr() == m.ptr + 44 * MIB,
              "the V buffers land in the mapping in order (v1 across the seam): %r" % m)
        check(m.read(0, 4096) == bytes(4096) and m.read(m.size - 4096, 4096) == bytes(4096),
              "the mapping is zeroed")
        seam = TORCH.zeros(rows * 8 * 128 * 2)
        data = pattern(8192, 7)
        ctypes.memmove(ctypes.addressof(seam.buf), data, 8192)
        p.v_buffer[3].copy_(seam)
        check(m.read(44 * MIB, 8192) == data, "a write through SGLang's own V view reads back in the carveout")
        check(LOGS.has("KV buffers 5-7 of 8 (K then V, by layer)"), "logged")
        LOGS.lines.clear()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("after building it"),
              "sized again after the pool was built: no carveout")
    finally:
        v.done()
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "closed: all given back: %r" % inf)


def case_cap23(rig):
    """0.5.23: --max-total-tokens lives in the schedule; the carveout's tokens come on top of it."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        r = v.configurator()
        v.runtime.SCHEDULE.max_total_tokens = 4000
        r.configure(pre_model_load_memory=0)
        check(r.max_total_num_tokens == 7968 and v.runtime.SCHEDULE.max_total_tokens == 4000,
              "a binding cap of 4000 tokens + the 62 MiB carveout's 3968 tokens of 16 KiB: %d; the cap restored: %r"
              % (r.max_total_num_tokens, v.runtime.SCHEDULE.max_total_tokens))
        check(LOGS.has("max_total_tokens=4000 holds the KV pool's device memory"), "logged")
        p = r.token_to_kv_pool
        check(sum(1 for t in p.fused if t.kind == "va") == 2, "the last two fused buffers in the carveout")
        m = sglang_plugin.S.keep[0]
        check(m.attic_bytes == 62 * MIB and m.size - m.attic_bytes == 2 * MIB,
              "device memory holds %d MiB; the carveout stands in for the rest of the pool's bytes"
              % ((m.size - m.attic_bytes) // MIB))
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        r = v.configurator()
        v.runtime.SCHEDULE.max_total_tokens = 20000
        r.configure(pre_model_load_memory=0)
        check(r.max_total_num_tokens == 10368 and not LOGS.has("holds the KV pool's device memory"),
              "a cap over the measured memory + the carveout does not bind")
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_s2_23(rig):
    """0.5.23: the budget was raised and the pool is not the kind the plugin fills: the engine stops."""
    rig.start(env=SMALL)
    v = Sgl(env_for(rig), tree23(), kind="0523")
    try:
        r = v.configurator(pool_kind="other")
        raised = None
        try:
            r.configure(pre_model_load_memory=0)
        except Exception as e:
            raised = e
        check(raised is not None and not sglang_plugin.S.keep,
              "a pool kind the plugin does not fill: the engine stops coming up, it does not allocate 62 MiB more")
        check(LOGS.has("took no carveout") and LOGS.has("not falling back"), "one line says why and what to do")
        inf = wait_free(rig.sock, 62 * MIB)
        check(inf and inf["free"] == 62 * MIB, "the unused slice went back at once: %r" % inf)
    finally:
        v.done()


def case_ranks(rig):
    rig.start(env=SMALL)
    w = World(2, [40 * MIB, 40 * MIB])
    v = Sgl(env_for(rig), world=w)
    try:
        r = v.runner()
        r.init_memory_pool(0)
        check(w.seen == [62 * MIB, 40 * MIB] and r.max_total_num_tokens == (140 * MIB) // 16384 // 16 * 16
              and lent(rig) == 40 * MIB and LOGS.has("40 MiB, what every rank can lend (this one has 62 MiB)"),
              "two ranks, the other can lend 40 MiB: both take 40: %r %d" % (w.seen, r.max_total_num_tokens))
        kinds = [t.kind for t in r.token_to_kv_pool.k_buffer + r.token_to_kv_pool.v_buffer]
        check(kinds.count("va") >= 2, "and fill it: %r" % kinds)
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)
    w = World(2, [62 * MIB, 0])
    v = Sgl(env_for(rig), world=w)
    try:
        r = v.runner()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("another rank could not borrow 62 MiB")
              and w.seen == [62 * MIB, 62 * MIB], "the other rank could not borrow: this one gives its slice back")
        inf = wait_free(rig.sock, 62 * MIB)
        check(inf and inf["free"] == 62 * MIB and sglang_plugin.S.kv is None, "given back: %r" % inf)
    finally:
        v.done()
    w = World(2, [0])
    v = Sgl(env_for(rig), world=w)
    try:
        r = v.runner()
        check(r._profile_available_bytes(0) == measured(r) and LOGS.has("another rank has no display carveout")
              and lent(rig) == 0, "the other rank has none: nobody borrows")
    finally:
        v.done()


def _upstream_helpers(src):
    """The AST readers the upstream cases share, bound to one source tree's python/ directory."""
    import types as _types

    def parse(mod):
        path = os.path.join(src, *mod.split("."))
        with open(os.path.join(path, "__init__.py") if os.path.isdir(path) else path + ".py") as f:
            return ast.parse(f.read())

    def find(t, dotted):
        node = t
        for part in dotted.split("."):
            node = next((n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                         and n.name == part), None)
            if node is None:
                return None
        return node

    def params(fn):
        a = fn.args
        return [x.arg for x in a.posonlyargs + a.args + ([a.vararg] if a.vararg else []) + a.kwonlyargs
                + ([a.kwarg] if a.kwarg else [])]

    def names(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}

    def calls(node):
        return {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}

    def stores(node):
        return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}

    def split(target):
        """module, qualified name: the module is the longest prefix with a source file."""
        parts = target.split(".")
        for i in range(len(parts) - 1, 0, -1):
            path = os.path.join(src, *parts[:i])
            if os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py")):
                return ".".join(parts[:i]), ".".join(parts[i:])
        raise ValueError(target)

    return _types.SimpleNamespace(parse=parse, find=find, params=params, names=names, calls=calls, stores=stores,
                                  split=split)


def case_upstream(rig):
    """SGLang's own source (--sglang-src, repeatable) still has what the plugin hooks and reads for its shape."""
    if not SGLANG_SRCS:
        print("  skip: no --sglang-src")
        return
    for root in SGLANG_SRCS:
        src = os.path.join(os.path.abspath(root), "python")
        tag = os.path.basename(root.rstrip("/"))
        h = _upstream_helpers(src)
        shape = "0.5.23" if os.path.isfile(os.path.join(src, "sglang", "srt", "mem_cache",
                                                        "kv_cache_configurator.py")) else "0.5.14"
        check(shape in sglang_plugin.TARGETS, "%s: a shape the plugin knows (%s)" % (tag, shape))
        for key, (target, want) in (list(sglang_plugin.TARGETS[shape].items())
                                    + list(sglang_plugin.CALLS[shape].items())):
            fn = h.find(h.parse(h.split(target)[0]), h.split(target)[1])
            check(fn is not None and h.params(fn) == want, "%s: %s: (%s) as the plugin expects: %r"
                  % (tag, target, ", ".join(want), fn and h.params(fn)))
            decos = [ast.unparse(d) for d in fn.decorator_list] if fn else []
            check(set(decos) <= {"torch.inference_mode()"}, "%s: %s: decorators that keep its parameters, read "
                  "through: %r" % (tag, target, decos))

        # the plugin's entry point group, and SGLang loads it before it builds the model runner
        pl = h.parse("sglang.srt.plugins")
        groups = {ast.unparse(n.targets[0]): n.value.value for n in pl.body if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Constant)}
        check(groups.get("GENERAL_PLUGINS_GROUP") == "sglang.srt.plugins", "%s: the plugin group: %r" % (tag, groups))
        check("HookRegistry.apply_hooks" in h.calls(h.find(pl, "load_plugins")), "%s: load_plugins applies the hooks"
              % tag)
        rsp = h.find(h.parse("sglang.srt.managers.scheduler"), "run_scheduler_process")
        check("load_plugins" in h.calls(rsp), "%s: the scheduler process loads plugins first thing" % tag)
        if shape == "0.5.14":
            _upstream_0514(src, h, tag)
        else:
            _upstream_0523(src, h, tag)


def _upstream_0514(src, h, tag):
    """The 0.5.14 source has the mixin chain and the one-zeros-per-buffer pool the plugin plans around."""
    parse, find, names, calls, stores, split = h.parse, h.find, h.names, h.calls, h.stores, h.split
    mixin = parse("sglang.srt.model_executor.model_runner_kv_cache_mixin")
    m = "ModelRunnerKVCacheMixin."
    check("self._resolve_memory_pool_config" in calls(find(mixin, m + "init_memory_pool"))
          and "self._profile_available_bytes" in calls(find(mixin, m + "_resolve_memory_pool_config"))
          and "self._init_pools" in calls(find(mixin, m + "_apply_memory_pool_config")),
          "init_memory_pool sizes with _profile_available_bytes, then builds with _init_pools")
    rmp = ast.unparse(find(mixin, m + "_resolve_memory_pool_config"))
    at = [rmp.find(x) for x in ("available_bytes = self._profile_available_bytes(",
                                "configurator = create_memory_pool_configurator(self)",
                                "config = configurator.calculate_pool_sizes(available_bytes",
                                "constrained = self._apply_token_constraints(config.max_total_num_tokens)",
                                "config = configurator.calculate_pool_sizes_from_max_tokens(constrained")]
    check(-1 not in at and at == sorted(at), "_resolve_memory_pool_config: the profiled bytes become tokens in the "
          "configurator, then the cap, then tokens again: %r" % at)
    atc = find(mixin, m + "_apply_token_constraints")
    check("user_limit = self.server_args.max_total_tokens" in ast.unparse(atc)
          and "token_capacity = min(token_capacity, user_limit)" in ast.unparse(atc),
          "_apply_token_constraints takes the smaller of the capacity and server_args.max_total_tokens")
    pc = parse("sglang.srt.model_executor.pool_configurator")
    cps = find(pc, "DefaultPoolConfigurator.calculate_pool_sizes")
    check("max_total_num_tokens = available_bytes // self._cell_size" in ast.unparse(cps),
          "the default configurator's tokens are bytes // cell size, so page size 1 counts the carveout's tokens")
    mpc = find(pc, "MemoryPoolConfig")
    check("max_total_num_tokens" in {n.target.id for n in mpc.body if isinstance(n, ast.AnnAssign)},
          "MemoryPoolConfig.max_total_num_tokens")
    check("return DefaultPoolConfigurator(mr)" in ast.unparse(find(pc, "create_memory_pool_configurator")),
          "the plain MHA pool gets the default configurator")
    check({"MHATokenToKVPool", "NoOpMHATokenToKVPool", "is_float4_e2m1fn_x2", "prefill_only_disable_kv_cache",
           "use_mla_backend", "is_hybrid_swa", "mambaish_config"} <= names(find(mixin, m + "_init_pools")),
          "_init_pools picks the plain MHA pool on the conditions the plugin checks")
    check({"get_world_group", "cpu_group"} <= names(find(mixin, m + "_profile_available_bytes")),
          "SGLang agrees on free memory over the world group's CPU group, as the plugin agrees on the carveout")
    runner = parse("sglang.srt.model_executor.model_runner")
    mr = find(runner, "ModelRunner")
    check([ast.unparse(b) for b in mr.bases] == ["ModelRunnerKVCacheMixin"] and not {
        n.name for n in mr.body if isinstance(n, ast.FunctionDef)} & {"_profile_available_bytes", "_init_pools"},
        "ModelRunner takes both from the mixin")
    need = {"device", "use_mla_backend", "is_hybrid_swa", "mambaish_config", "is_draft_worker", "spec_algorithm",
            "model_config", "kv_cache_dtype", "server_args"}
    have = stores(mr) | {n.name for n in mr.body if isinstance(n, ast.FunctionDef)}
    check(need <= have, "ModelRunner has what the plugin reads; missing: %r" % (need - have))
    sa = find(parse("sglang.srt.server_args"), "ServerArgs")
    fields = {n.target.id for n in sa.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    need = {"attention_backend", "prefill_only_disable_kv_cache", "enable_memory_saver", "disaggregation_mode",
            "max_total_tokens"}
    check(need <= fields, "ServerArgs has the fields the plugin reads; missing: %r" % (need - fields))

    # the plain MHA pool: K then V buffers, one torch.zeros each, as the plugin plans them
    pool = parse("sglang.srt.mem_cache.memory_pool")
    cb = find(pool, "MHATokenToKVPool._create_buffers")
    ifs = [n for n in ast.walk(cb) if isinstance(n, ast.If)
           and ast.unparse(n.test) == "self.kv_cache_layout == 'vectorized_5d'"]
    got = [(ast.unparse(a.targets[0]), ast.unparse(a.value)) for a in (ifs[0].orelse if ifs else [])
           if isinstance(a, ast.Assign)]
    z = ("[torch.zeros((self.size + self.page_size, self.head_num, self.%s), dtype=self.store_dtype, "
         "device=self.device) for _ in range(self.layer_num)]")
    check(got == [("self.k_buffer", z % "head_dim"), ("self.v_buffer", z % "v_head_dim")],
          "the nhd branch of _create_buffers: %r" % got)
    zeros = [n for n in ast.walk(cb) if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.zeros"]
    check(ifs and all(any(c is n for n in ast.walk(ifs[0])) for c in zeros), "no other torch.zeros before them")
    init = find(pool, "MHATokenToKVPool.__init__")
    layouts = [ast.unparse(a.value) for a in ast.walk(init) if isinstance(a, ast.Assign)
               and ast.unparse(a.targets[0]) == "self.kv_cache_layout"]
    check(layouts[:1] == ["'nhd'"] and "self._create_buffers" in calls(init)
          and {"head_num", "head_dim", "v_head_dim", "kv_cache_layout"} <= stores(init),
          "MHATokenToKVPool defaults to nhd and builds its buffers in __init__: %r" % layouts)
    need = {"size", "page_size", "device", "store_dtype", "layer_num", "enable_custom_mem_pool"}
    have = stores(find(pool, "KVCache.__init__"))
    check(need <= have, "KVCache sets what the plugin reads; missing: %r" % (need - have))

    # the helpers _kv_blocked imports, and the custom pool switch
    check(find(parse("sglang.srt.configs.model_config"), "is_deepseek_v4") is not None, "is_deepseek_v4")
    plat = parse("sglang.srt.platforms")
    check("current_platform" in names(plat) or any(isinstance(n, ast.Constant) and n.value == "current_platform"
                                                   for n in ast.walk(plat)), "sglang.srt.platforms.current_platform")
    check(any(isinstance(n, ast.FunctionDef) and n.name == "is_out_of_tree"
              for n in ast.walk(parse("sglang.srt.platforms.device_mixin"))), "Platform.is_out_of_tree")
    check("SGLANG_MOONCAKE_CUSTOM_MEM_POOL" in names(parse("sglang.srt.mem_cache.utils")),
          "the custom pool switch is SGLANG_MOONCAKE_CUSTOM_MEM_POOL")
    check({"get_world_group"} <= {f.name for f in parse("sglang.srt.distributed.parallel_state").body
                                  if isinstance(f, ast.FunctionDef)}, "parallel_state.get_world_group")


def _upstream_0523(src, h, tag):
    """The 0.5.23 source has the configurator chain and the fused-buffer pool the plugin plans around."""
    parse, find, names, calls, stores = h.parse, h.find, h.names, h.calls, h.stores

    # the sizing chain: profile -> config_from_budget (pool sizes, then the cap), in the configurator
    conf = parse("sglang.srt.mem_cache.kv_cache_configurator")
    c = "KVCacheConfigurator."
    rmp = find(conf, c + "_resolve_memory_pool_config")
    at = [ast.unparse(rmp).find(x) for x in ("available_bytes = self._profile_available_bytes(",
                                             "config = self.config_from_budget(available_bytes)")]
    cfb = find(conf, c + "config_from_budget")
    at += [ast.unparse(cfb).find(x) for x in ("configurator = create_memory_pool_configurator(self)",
                                              "config = configurator.calculate_pool_sizes(",
                                              "max_tokens = self._apply_token_constraints(config.max_total_num_tokens")]
    check(-1 not in at and at == sorted(at), "%s: _resolve_memory_pool_config: the profiled bytes go through "
          "config_from_budget: pool sizes in the configurator, then the cap: %r" % (tag, at))
    atc = find(conf, c + "_apply_token_constraints")
    check("get_schedule().max_total_tokens" in ast.unparse(atc),
          "%s: _apply_token_constraints reads the cap through get_schedule()" % tag)
    confure = find(conf, c + "configure")
    check("self._resolve_memory_pool_config" in calls(confure) and "self._init_pools" in calls(confure),
          "%s: configure sizes with _profile_available_bytes, then builds with _init_pools" % tag)
    check("token_to_kv_pool" in names(find(conf, c + "_init_pools")),
          "%s: _init_pools builds the token pool" % tag)
    pc = parse("sglang.srt.model_executor.pool_configurator")
    cps = find(pc, "DefaultPoolConfigurator.calculate_pool_sizes")
    check("max_total_num_tokens = (\n            available_bytes // self._cell_size" in ast.unparse(cps)
          or "available_bytes // self._cell_size" in ast.unparse(cps),
          "%s: the default configurator's tokens are bytes // cell size, so the carveout's bytes count" % tag)
    check(any(ast.unparse(n) == "create_memory_pool_configurator(self)" for n in ast.walk(cfb)
              if isinstance(n, ast.Call)),
          "%s: the configurator is built from the KVCacheConfigurator" % tag)

    # what the plugin reads off the configurator
    fields = {n.target.id for n in find(conf, "KVCacheConfigurator").body
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    need = {"device", "gpu_id", "model_config", "server_args", "kv_cache_dtype", "page_size", "spec_algorithm",
            "is_draft_worker", "is_hybrid_swa", "use_mla_backend", "mambaish_config"}
    check(need <= fields, "%s: KVCacheConfigurator has what the plugin reads; missing: %r" % (tag, need - fields))
    check(find(parse("sglang.srt.model_executor.model_runner"), "ModelRunner.load_model") is not None,
          "%s: ModelRunner.load_model (the weights hook)" % tag)

    # the plain MHA pool: the plugin reads the buffer layout from this very source, so the rule must
    # name what the tag does - the fused (2, padded, ...) allocation (0.5.22+) or separate K and V
    # (0.5.14, 0.5.21: its hook signatures match 0.5.23's, so only the source tells the layouts apart)
    pool = parse("sglang.srt.mem_cache.memory_pool")
    cbn = find(pool, "MHATokenToKVPool._create_buffers_normal")
    fused = [n for n in ast.walk(cbn) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "_zeros" and "_zeros((2, padded, *k_shape[1:]))" in ast.unparse(n)]
    text = ast.unparse(cbn)
    separate = "torch.zeros(k_shape" in text and "torch.zeros(v_shape" in text
    check(sglang_plugin._layout_rule(cbn) == ("fused" if fused else "separate"),
          "%s: the plugin's layout rule reads this pool as %s" % (tag, "fused" if fused else "separate"))
    if fused:
        check(len(fused) == 1, "%s: one fused _zeros((2, padded, *k_shape[1:])) per layer: %d" % (tag, len(fused)))
        check("self.k_buffer = [buf[0, :rows] for buf in fused]" in text
              and "self.v_buffer = [buf[1, :rows] for buf in fused]" in text,
              "%s: the pool keeps buf[0, :rows] and buf[1, :rows] views of the fused allocation" % tag)
    else:
        check(separate, "%s: the pool allocates K and V separately (the buffers hook follows that layout, "
                        "as on 0.5.21)" % tag)
        check("fused" not in text, "%s: no fused K/V branch the hook would plan against" % tag)
    cb = find(pool, "MHATokenToKVPool._create_buffers")
    check("self._create_buffers_normal()" in ast.unparse(cb) and "self.is_quantized_kv_cache" in ast.unparse(cb)
          and "self.post_capture_active" in ast.unparse(cb),
          "%s: _create_buffers dispatches on is_quantized_kv_cache and post_capture_active" % tag)
    init = find(pool, "MHATokenToKVPool.__init__")
    check({"head_num", "head_dim", "v_head_dim", "kv_cache_layout", "use_hnd"} <= stores(init)
          and "self._create_buffers" in calls(init),
          "%s: MHATokenToKVPool sets the layout in __init__ and builds its buffers there" % tag)
    need = {"size", "page_size", "device", "store_dtype", "layer_num", "enable_custom_mem_pool"}
    have = stores(find(pool, "KVCache.__init__"))
    check(need <= have, "%s: KVCache sets what the plugin reads; missing: %r" % (tag, need - have))

    # the helpers the plugin imports, and where the cap lives
    check(find(parse("sglang.srt.configs.model_config"), "is_deepseek_v4") is not None, "%s: is_deepseek_v4" % tag)
    check("current_platform" in names(parse("sglang.srt.platforms")),
          "%s: sglang.srt.platforms.current_platform" % tag)
    check(any(isinstance(n, ast.FunctionDef) and n.name == "get_schedule"
              for n in parse("sglang.srt.runtime_context").body), "%s: runtime_context.get_schedule" % tag)
    check({"get_world_group"} <= {f.name for f in parse("sglang.srt.distributed.parallel_state").body
                                  if isinstance(f, ast.FunctionDef)}, "%s: parallel_state.get_world_group" % tag)


CASES = [case_entry_point, case_settings, case_kv, case_kv23, case_cap, case_cap23, case_kv_other_pool,
         case_kv_miss, case_s2_23, case_blocked, case_hnd, case_page_major, case_minimax_sparse, case_post_capture,
         case_unequal_head_dims, case_weights, case_weights_and_kv, case_fail_open, case_drift, case_ranks,
         case_upstream]
CASES.insert(CASES.index(case_cap23), case_kv21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/host")
    ap.add_argument("--sglang-src", action="append", default=[])
    a = ap.parse_args()
    SGLANG_SRCS.extend(os.path.abspath(s) for s in a.sglang_src if os.path.isdir(s))
    if not SGLANG_SRCS:
        # An export tree carries no upstream cache; the Makefile passes $(SGLANG_UPSTREAM), which is
        # empty there. That is not drift: case_upstream skips itself, and every other case runs on the
        # stand-in tree alone (plugins_files() gives it the pocket plugin loader and hook registry).
        print("test_sglang_plugin: no --sglang-src tree given: case_upstream is skipped")
    for src in SGLANG_SRCS:
        if not os.path.isfile(os.path.join(src, "python", "sglang", "srt", "plugins", "hook_registry.py")):
            print("test_sglang_plugin: %s is not an SGLang source tree" % src)
            return 2
    bindir = os.path.abspath(a.bin)
    os.environ["SPARK_ATTIC_LIBCUDA"] = os.path.join(bindir, "fakecuda", "libcuda.so.1")
    sys.modules["torch"] = TORCH
    os.umask(0o022)
    for case in CASES:
        rig = Rig(bindir, "sp-" + case.__name__[5:])
        before = len(FAILS)
        try:
            case(rig)
        except Exception as e:  # a crash is a failure, with atticd's log
            import traceback
            traceback.print_exc()
            FAILS.append("%s: %r" % (case.__name__, e))
            print("  FAIL %s raised %r" % (case.__name__, e))
        if len(FAILS) > before:
            print("  plugin log:\n    " + "\n    ".join(LOGS.lines))
            if os.path.exists(rig.log):
                print("  atticd log:\n" + open(rig.log).read())
        print("%s %s" % ("ok  " if len(FAILS) == before else "FAIL", case.__name__))
        rig.done()
    print("test_sglang_plugin: %d cases, %d failures" % (len(CASES), len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

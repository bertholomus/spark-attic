#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_vllm_plugin.py: spark_attic's vLLM plugin on a host with no GPU and no vLLM. A stand-in vLLM tree (files in a
temp dir, imported for real, so the after-import hooks run as in vLLM) carries the four functions the plugin wraps,
with vLLM 0.24.0's parameters and a pocket version of their logic; a stand-in torch keeps tensor bytes in host memory
or, for the carveout, in the fake libcuda's. atticd runs with the fake RM shim, as in test_python.py.

    python3 -I tests/test_vllm_plugin.py --bin build/host [--vllm-src cache/upstream/vllm-0.24.0]

--vllm-src adds the check that vLLM's own source still has what the plugin wraps (parameters, KVCacheTensor's fields,
the calls the plugin relies on); without it that case is skipped.
"""
import argparse
import ast
import ctypes
import importlib
import importlib.metadata
import logging
import os
import sys
import tempfile
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from test_server import FAILS, KEY, MIB, Rig, check, wait_free  # noqa: E402

import spark_attic  # noqa: E402
from spark_attic import _cuda, _place, vllm_plugin  # noqa: E402
from fake_torch import TORCH, Device, Model  # noqa: E402

SMALL = {"ATTIC_FAKE_REGION": "0x280000000:0x4000000"}  # 62 MiB free in one run
VLLM_SRC = None


class Logs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def has(self, text):
        return any(text in line for line in self.lines)


LOGS = Logs()
logging.getLogger("spark_attic").addHandler(LOGS)
logging.getLogger("spark_attic").setLevel(logging.INFO)


# ----------------------------------------------------------------------------------------- a stand-in vLLM tree

PKG = "\n"
KV_IFACE = '''
from dataclasses import dataclass, field

@dataclass
class KVCacheTensor:
    size: int
    shared_by: list
    offset: int = 0
    block_stride: int = 0

@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list
    kv_cache_groups: list = field(default_factory=list)
'''
ALLOC_BODY = '''
    out, packed = {}, None
    for t in kv_cache_config.kv_cache_tensors:
        if t.block_stride > 0:
            if packed is None:
                packed = torch.zeros(t.size, dtype=torch.int8, device=device)
            tensor = packed
        else:
            tensor = torch.zeros(t.size, dtype=torch.int8, device=device)
        for name in t.shared_by:
            out[name] = tensor
    return out
'''
RUNNER_COMMON = '''
    def __init__(self, device, model=None):
        self.device, self.model, self.model_memory_usage = device, model, 0

    def load_model(self, load_dummy_weights=False):
        self.model_memory_usage = sum(p.numel() * p.element_size() for _, p in self.model.named_parameters())
'''
WORKER_SRC = '''import torch


class Worker:
    def __init__(self, vllm_config, device, model_runner, budget):
        self.vllm_config, self.device, self.model_runner, self.budget = vllm_config, device, model_runner, budget

    def load_model(self, *, load_dummy_weights=False):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

    @torch.inference_mode()  # as in vLLM 0.24.0: the wrapper itself takes (*args, **kwargs)
    def determine_available_memory(self):
        # vLLM's sum, pocket size: what it may use, less the weights it measured at load
        self.available_kv_cache_memory_bytes = self.budget - int(self.model_runner.model_memory_usage)
        return int(self.available_kv_cache_memory_bytes)
'''


def tree(attn_params="kv_cache_config, shared_layers, device", extra_field=""):
    return {
        "vllm/__init__.py": PKG, "vllm/v1/__init__.py": PKG, "vllm/v1/worker/__init__.py": PKG,
        "vllm/v1/worker/gpu/__init__.py": PKG,
        "vllm/v1/kv_cache_interface.py": KV_IFACE.replace("    block_stride: int = 0",
                                                          "    block_stride: int = 0" + extra_field),
        "vllm/v1/worker/gpu_worker.py": WORKER_SRC,
        "vllm/v1/worker/gpu_model_runner.py": "import torch\n\nclass GPUModelRunner:" + RUNNER_COMMON + (
            "\n    def _allocate_kv_cache_tensors(self, kv_cache_config):\n        device = self.device"
            + ALLOC_BODY.replace("\n    ", "\n        ") +
            "\n    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):\n"
            "        return self._allocate_kv_cache_tensors(kv_cache_config)\n"),
        "vllm/v1/worker/gpu/attn_utils.py": "import torch\n\ndef _allocate_kv_cache(%s):%s\n"
        "def init_kv_cache(runner_kv_caches, forward_context, kv_cache_config, attn_groups, device, cache_dtype,\n"
        "                  kernel_block_sizes, vllm_config):\n"
        "    return _allocate_kv_cache(kv_cache_config, {}, device)\n" % (attn_params, ALLOC_BODY),
        "vllm/v1/worker/gpu/model_runner.py": "from vllm.v1.worker.gpu.attn_utils import init_kv_cache\n\n"
        "class GPUModelRunner:" + RUNNER_COMMON +
        "\n    def initialize_kv_cache(self, kv_cache_config):\n"
        "        return init_kv_cache([], {}, kv_cache_config, [], self.device, 'auto', [], None)\n",
    }


class Vllm:
    """A fresh stand-in vLLM on sys.path, a fresh plugin state, register() called as vLLM would."""

    def __init__(self, env, files=None):
        for name in [n for n in sys.modules if n == "vllm" or n.startswith("vllm.")]:
            del sys.modules[name]
        self.dir = tempfile.mkdtemp(prefix="attic-vllm-")
        for rel, src in (files or tree()).items():
            path = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(textwrap.dedent(src) if not src.startswith("import") else src)
        sys.path.insert(0, self.dir)
        importlib.invalidate_caches()
        for k in [k for k in os.environ if k.startswith("SPARK_ATTIC_") and k != "SPARK_ATTIC_LIBCUDA"]:
            del os.environ[k]
        os.environ.update(env)
        vllm_plugin.S = vllm_plugin._State()
        LOGS.lines.clear()
        vllm_plugin.register()
        self.gw = importlib.import_module("vllm.v1.worker.gpu_worker")
        self.r1 = importlib.import_module("vllm.v1.worker.gpu_model_runner")
        self.au = importlib.import_module("vllm.v1.worker.gpu.attn_utils")
        self.r2 = importlib.import_module("vllm.v1.worker.gpu.model_runner")
        self.kv = importlib.import_module("vllm.v1.kv_cache_interface")

    def worker(self, runner, budget, sleep=False, connector=None):
        cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(enable_sleep_mode=sleep),
                                    kv_transfer_config=connector)
        return self.gw.Worker(cfg, Device("cuda:0"), runner, budget)

    def config(self, sizes, packed=()):
        ts = [self.kv.KVCacheTensor(size=s, shared_by=["layer.%d" % i], block_stride=(s if i in packed else 0))
              for i, s in enumerate(sizes)]
        return self.kv.KVCacheConfig(num_blocks=1, kv_cache_tensors=ts)

    def done(self):
        st = vllm_plugin.S
        for m in st.keep:
            m.close()
        st.keep.clear()
        if st.kv is not None:
            st.kv.release()
        if st.pool is not None:
            st.pool.close()
        sys.path.remove(self.dir)


def env_for(rig, **kw):
    e = {"SPARK_ATTIC_SOCK": rig.sock if rig else "/nonexistent/attic.sock", "SPARK_ATTIC_KEY": KEY}
    e.update({("SPARK_ATTIC_" + k): v for k, v in kw.items()})
    return e


def lent(rig):
    return spark_attic.Attic(spark_attic.Settings(sock=rig.sock, key=KEY)).status()["lent"]


def pattern(n, seed):
    return bytes((i * 131 + seed) & 0xFF for i in range(n))


# ------------------------------------------------------------------------------------------------------- cases

def case_tail_plan(rig):
    p = _place.tail_plan([27 * MIB] * 6, 62 * MIB)
    check((p.first, p.front, p.offsets, p.attic_used) == (3, 22 * MIB, (0, 28 * MIB, 56 * MIB), 62 * MIB),
          "six 27 MiB tensors, 62 MiB lent: the last three, the first of them split 22 + 6: %r" % (p,))
    p = _place.tail_plan([10 * MIB] * 3, 62 * MIB)
    check((p.first, p.front, p.offsets, p.attic_used) == (0, 0, (0, 10 * MIB, 20 * MIB), 30 * MIB),
          "all fit, 30 of 62 MiB used: %r" % (p,))
    p = _place.tail_plan([20 * MIB] * 3, 40 * MIB)
    check((p.first, p.front, p.offsets, p.attic_used) == (1, 0, (0, 20 * MIB), 40 * MIB), "an exact fit: %r" % (p,))
    p = _place.tail_plan([100 * MIB], 62 * MIB)
    check((p.first, p.front, p.offsets, p.attic_used) == (0, 38 * MIB, (0,), 62 * MIB), "one big tensor: %r" % (p,))
    p = _place.tail_plan([5 * MIB + 1, 3 * MIB], 4 * MIB)
    check((p.first, p.front, p.offsets, p.attic_used) == (1, 0, (0,), 4 * MIB),
          "2 MiB boundaries: 5 MiB + 1 takes 6, 3 takes 4: %r" % (p,))
    p = _place.tail_plan([], 4 * MIB)
    check((p.first, p.front, p.offsets, p.attic_used) == (0, 0, (), 0), "nothing to place: %r" % (p,))


def case_settings(rig):
    P = _place.Placement.from_env
    check(P({}) == _place.Placement(("kv",), ("embed_tokens",), 0), "default: kv, embed_tokens, no cap: %r" % (P({}),))
    check(P({"SPARK_ATTIC_PLACE": "weights, kv"}).place == ("kv", "weights"), "weights,kv")
    check(P({"SPARK_ATTIC_PLACE": "off"}).place == (), "off")
    for bad in ({"SPARK_ATTIC_PLACE": "kv,bogus"}, {"SPARK_ATTIC_PLACE": ","}, {"SPARK_ATTIC_MIB": "1.5"}):
        try:
            P(bad)
            check(False, "refused: %r" % bad)
        except ValueError:
            pass
    p = P({"SPARK_ATTIC_PLACE": "weights", "SPARK_ATTIC_WEIGHTS": "embed_tokens, lm_head", "SPARK_ATTIC_MIB": "512"})
    check(p.weights == ("embed_tokens", "lm_head") and p.cap == 512 * MIB, "weights and cap: %r" % (p,))
    m = _place.name_matches
    check(m("model.embed_tokens.weight", ["embed_tokens"]) and m("lm_head.weight", ["lm_head"])
          and m("model.layers.3.mlp.down_proj.weight", ["model.layers.*.mlp.*"])
          and not m("model.embed_tokens_x.weight", ["embed_tokens"]) and not m("model.norm.weight", ["embed_tokens"]),
          "parameter names: a dotted run or a glob")
    os.environ["SPARK_ATTIC_PLACE"] = "bogus"
    vllm_plugin.S = vllm_plugin._State()
    LOGS.lines.clear()
    vllm_plugin.register()  # must not raise
    check(vllm_plugin.S.placement.place == () and LOGS.has("vLLM plugin off"), "a bad setting: plugin off, logged")
    del os.environ["SPARK_ATTIC_PLACE"]


def case_entry_point(rig):
    eps = [e for e in importlib.metadata.entry_points(group="vllm.general_plugins") if e.name == "spark_attic"]
    check(len(eps) == 1 and eps[0].value == "spark_attic.vllm_plugin:register" and eps[0].load()
          is vllm_plugin.register, "the static dist-info registers the plugin: %r" % eps)
    import tomllib
    with open(os.path.join(ROOT, "python", "pyproject.toml"), "rb") as f:
        pp = tomllib.load(f)
    check(pp["project"]["entry-points"]["vllm.general_plugins"]["spark_attic"] == eps[0].value
          and pp["project"]["version"] == spark_attic.__version__, "pyproject.toml says the same")


def case_after_import(rig):
    d = tempfile.mkdtemp(prefix="attic-mod-")
    with open(os.path.join(d, "attic_probe_mod.py"), "w") as f:
        f.write("VALUE = 41\n")
    with open(os.path.join(d, "attic_probe_bad.py"), "w") as f:
        f.write("VALUE = 1\n")
    sys.path.insert(0, d)
    importlib.invalidate_caches()
    seen = []
    _place.when_imported("attic_probe_mod", lambda m: seen.append(m.VALUE))
    check(seen == [], "nothing runs before the import")
    import attic_probe_mod  # noqa: F401
    check(seen == [41], "the hook runs once the module has executed: %r" % seen)
    importlib.reload(attic_probe_mod)
    check(seen == [41], "and only once")
    _place.when_imported("attic_probe_mod", lambda m: seen.append("now"))
    check(seen == [41, "now"], "a module already loaded: the hook runs at once")

    def boom(m):
        raise RuntimeError("hook trouble")
    LOGS.lines.clear()
    _place.when_imported("attic_probe_bad", boom)
    import attic_probe_bad
    check(attic_probe_bad.VALUE == 1 and LOGS.has("hook trouble"), "a failing hook never stops the import")
    sys.path.remove(d)


def case_kv_v2(rig):
    rig.start(env=SMALL)
    v = Vllm(env_for(rig))
    try:
        check(getattr(v.gw.Worker.determine_available_memory, "__spark_attic__", False)
              and getattr(v.au._allocate_kv_cache, "__spark_attic__", False)
              and getattr(v.r1.GPUModelRunner._allocate_kv_cache_tensors, "__spark_attic__", False)
              and not getattr(v.gw.Worker.load_model, "__spark_attic__", False), "kv: three wraps, load_model left alone")
        w = v.worker(v.r2.GPUModelRunner(Device("cuda:0")), 100 * MIB)
        got = w.determine_available_memory()
        check(got == 162 * MIB and w.available_kv_cache_memory_bytes == 162 * MIB,
              "100 MiB measured + the 62 MiB lent: %d MiB" % (got // MIB))
        check(lent(rig) == 62 * MIB, "atticd lent 62 MiB")
        check(w.determine_available_memory() == 162 * MIB, "asked again before the allocation: the same")
        out = w.model_runner.initialize_kv_cache(v.config([27 * MIB] * 6))
        kinds = [out["layer.%d" % i].kind for i in range(6)]
        check(kinds == ["buf"] * 3 + ["va"] * 3, "layers 0-2 where vLLM puts them, 3-5 in the mapping: %r" % kinds)
        m = vllm_plugin.S.keep[0]
        vas = [out["layer.%d" % i].va for i in (3, 4, 5)]
        check(m.size == 84 * MIB and m.attic_bytes == 62 * MIB and vas == [m.ptr, m.ptr + 28 * MIB, m.ptr + 56 * MIB]
              and all(out["layer.%d" % i].nbytes == 27 * MIB for i in range(6)),
              "22 MiB device + 62 MiB attic, tensors at 0, 28, 56 MiB: %r %r" % (m, vas))
        t3 = out["layer.3"]
        check(m.read(0, 4096) == bytes(4096) and m.read(m.size - 4096, 4096) == bytes(4096), "the mapping is zeroed")
        seam = TORCH.zeros(27 * MIB)
        data = pattern(8192, 5)
        ctypes.memmove(ctypes.addressof(seam.buf) + 22 * MIB - 4096, data, 8192)
        t3.copy_(seam)
        check(m.read(22 * MIB - 4096, 8192) == data, "a write across the device/carveout seam reads back")
        check(w.determine_available_memory() == 100 * MIB and LOGS.has("after allocating"),
              "sized again after the allocation: no carveout")
    finally:
        v.done()
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "closed: all given back: %r" % inf)


def case_kv_v1_packed(rig):
    rig.start(env=SMALL)
    v = Vllm(env_for(rig, PLACE="kv"))
    try:
        r = v.r1.GPUModelRunner(Device("cuda:0"))
        w = v.worker(r, 100 * MIB)
        check(w.determine_available_memory() == 162 * MIB, "V1: 100 + 62 MiB")
        cfg = v.config([20 * MIB] * 4 + [30 * MIB] * 2, packed=(0, 1, 2, 3))
        out = r.initialize_kv_cache_tensors(cfg, [16])
        m = vllm_plugin.S.keep[0]
        packed = out["layer.0"]
        check(all(out["layer.%d" % i] is packed for i in range(4)), "the packed tensors share one backing")
        check(packed.kind == "va" and packed.va == m.ptr and out["layer.4"].va == m.ptr + 20 * MIB
              and out["layer.5"].va == m.ptr + 50 * MIB and m.size == 80 * MIB,
              "backings 20/30/30 MiB, 62 lent: all three in the mapping, 18 MiB device first: %r" % m)
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_weights(rig):
    rig.start(env=SMALL)
    v = Vllm(env_for(rig, PLACE="weights"))
    try:
        check(getattr(v.gw.Worker.load_model, "__spark_attic__", False)
              and not getattr(v.gw.Worker.determine_available_memory, "__spark_attic__", False)
              and not getattr(v.au._allocate_kv_cache, "__spark_attic__", False), "weights: only load_model wrapped")
        nb = 1281 * 2048 * 2
        embed = TORCH.param(nb, TORCH.float16, (1281, 2048), 3)
        other = TORCH.param(MIB, TORCH.bfloat16, (512, 1024), 9)
        before = embed.data.raw()
        model = Model([("model.embed_tokens.weight", embed), ("model.layers.0.w", other), ("lm_head.weight", embed)])
        w = v.worker(v.r2.GPUModelRunner(Device("cuda:0"), model), 100 * MIB)
        w.load_model()
        check(embed.data.kind == "va" and embed.data.shape == (1281, 2048) and embed.data.dtype is TORCH.float16
              and embed.data.raw() == before, "embed_tokens moved, shape, dtype and bytes kept")
        check(other.data.kind == "buf", "other weights stay")
        check(len(vllm_plugin.S.keep) == 1 and lent(rig) == 6 * MIB, "tied lm_head moved once: one 6 MiB slice")
        check(w.model_runner.model_memory_usage == MIB, "vLLM's weight figure drops by what was freed: %d"
              % w.model_runner.model_memory_usage)
        check(w.determine_available_memory() == 99 * MIB, "so its own sum gives the KV cache that much more")
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_weights_and_kv(rig):
    rig.start(env=SMALL)
    v = Vllm(env_for(rig, PLACE="weights,kv", MIB="40", WEIGHTS="lm_head,embed_tokens"))
    try:
        embed = TORCH.param(4 * MIB, TORCH.float16, (2048, 1024), 1)
        head = TORCH.param(40 * MIB, TORCH.float16, (20480, 1024), 2)
        model = Model([("model.embed_tokens.weight", embed), ("lm_head.weight", head)])
        w = v.worker(v.r2.GPUModelRunner(Device("cuda:0"), model), 100 * MIB)
        w.load_model()
        check(embed.data.kind == "va" and head.data.kind == "buf" and LOGS.has("lm_head.weight stays"),
              "under a 40 MiB cap the 40 MiB lm_head stays, embed_tokens moves")
        got = w.determine_available_memory()
        check(got == 100 * MIB - 40 * MIB + 36 * MIB and lent(rig) == 40 * MIB,
              "the KV cache takes the rest of the cap, 36 MiB: %d MiB" % (got // MIB))
    finally:
        v.done()
    wait_free(rig.sock, 62 * MIB)


def case_fail_open(rig):
    v = Vllm(env_for(None, PLACE="weights,kv"))
    try:
        embed = TORCH.param(2 * MIB, TORCH.float16, (1024, 1024), 4)
        w = v.worker(v.r2.GPUModelRunner(Device("cuda:0"), Model([("model.embed_tokens.weight", embed)])), 100 * MIB)
        w.load_model()
        check(embed.data.kind == "buf" and LOGS.has("weights stay in device memory"), "no atticd: weights stay")
        check(w.determine_available_memory() == 98 * MIB and LOGS.has("stays as vLLM sized it"),
              "no atticd: the KV cache as vLLM measured it")
        out = w.model_runner.initialize_kv_cache(v.config([10 * MIB] * 3))
        check(all(t.kind == "buf" for t in out.values()), "and vLLM allocates it all itself")
    finally:
        v.done()


def case_blocked(rig):
    rig.start(env=SMALL)
    v = Vllm(env_for(rig))
    try:
        for why, w in (("sleep mode", v.worker(v.r2.GPUModelRunner(Device("cuda:0")), 100 * MIB, sleep=True)),
                       ("KV connector", v.worker(v.r2.GPUModelRunner(Device("cuda:0")), 100 * MIB, connector={})),
                       ("does not wrap", v.worker(types.SimpleNamespace(), 100 * MIB))):
            if why == "does not wrap":
                w.model_runner.model_memory_usage = 0
            LOGS.lines.clear()
            check(w.determine_available_memory() == 100 * MIB and LOGS.has(why), "%s: KV placement off" % why)
        check(lent(rig) == 0, "nothing lent")
    finally:
        v.done()


def case_drift(rig):
    rig.start(env=SMALL)
    v = Vllm(env_for(rig), tree(attn_params="kv_cache_config, device, shared_layers=None"))
    try:
        check(not getattr(v.au._allocate_kv_cache, "__spark_attic__", False)
              and LOGS.has("takes (kv_cache_config, device, shared_layers)"), "a changed signature is not wrapped")
        w = v.worker(v.r2.GPUModelRunner(Device("cuda:0")), 100 * MIB)
        check(w.determine_available_memory() == 100 * MIB and LOGS.has("V2 KV allocation is not wrapped"),
              "so the V2 runner gets no carveout")
        check(lent(rig) == 0, "nothing lent")
    finally:
        v.done()
    v = Vllm(env_for(rig), tree(extra_field="\n    stream: int = 0"))
    try:
        r = v.r1.GPUModelRunner(Device("cuda:0"))
        w = v.worker(r, 100 * MIB)
        check(w.determine_available_memory() == 162 * MIB, "V1, 62 MiB counted in")
        out = r.initialize_kv_cache_tensors(v.config([27 * MIB] * 6), [16])
        check(all(t.kind == "buf" for t in out.values()) and LOGS.has("fields the plugin does not know: stream")
              and LOGS.has("62 MiB more than the memory it measured"),
              "KVCacheTensor grew a field: vLLM allocates it all, and the log says by how much it overshoots")
        inf = wait_free(rig.sock, 62 * MIB)
        check(inf and inf["free"] == 62 * MIB and vllm_plugin.S.pool is None,
              "nothing else borrowed: the connection closed, the unused slice back at once: %r" % inf)
    finally:
        v.done()


# Decorators that keep a function's parameters (functools.wraps), so the plugin reads its parameters through them
KEEP_PARAMS = {"torch.inference_mode()"}


def case_upstream(rig):
    """vLLM's own source (--vllm-src) still has what the plugin wraps."""
    if not VLLM_SRC:
        print("  skip: no --vllm-src")
        return

    def parse(mod):
        with open(os.path.join(VLLM_SRC, *mod.split(".")) + ".py") as f:
            return ast.parse(f.read())

    def find(t, dotted):
        node = t
        for part in dotted.split("."):
            node = next((n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                         and n.name == part), None)
            if node is None:
                return None
        return node

    for (mod, qual), want in vllm_plugin.EXPECTED.items():
        fn = find(parse(mod), qual)
        a = fn.args if fn else None
        got = [x.arg for x in a.posonlyargs + a.args + ([a.vararg] if a.vararg else []) + a.kwonlyargs
               + ([a.kwarg] if a.kwarg else [])] if a else None
        check(got == want, "%s.%s: (%s) as the plugin expects: %r" % (mod, qual, ", ".join(want), got))
        decos = [ast.unparse(d) for d in fn.decorator_list] if fn else []
        check(set(decos) <= KEEP_PARAMS, "%s.%s: decorators that keep its parameters, read through: %r"
              % (mod, qual, decos))
    kv = find(parse("vllm.v1.kv_cache_interface"), "KVCacheTensor")
    fields = {n.target.id for n in kv.body if isinstance(n, ast.AnnAssign)}
    check(fields == vllm_plugin.KV_TENSOR_FIELDS, "KVCacheTensor's fields: %r" % fields)

    def names(fn):
        return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    check("_allocate_kv_cache" in names(find(parse(vllm_plugin.ATTN_V2), "init_kv_cache")),
          "attn_utils.init_kv_cache calls _allocate_kv_cache")
    check("_allocate_kv_cache_tensors" in names(find(parse(vllm_plugin.RUNNER_V1),
                                                     "GPUModelRunner.initialize_kv_cache_tensors")),
          "V1 initialize_kv_cache_tensors calls _allocate_kv_cache_tensors")
    v2 = parse(vllm_plugin.RUNNER_V2)
    check(any(isinstance(n, ast.ImportFrom) and n.module == vllm_plugin.ATTN_V2
              and "init_kv_cache" in [a.name for a in n.names] for n in ast.walk(v2)),
          "the V2 runner imports init_kv_cache from attn_utils")
    check("init_kv_cache" in names(find(v2, "GPUModelRunner.initialize_kv_cache")),
          "and calls it from initialize_kv_cache")
    w = find(parse(vllm_plugin.WORKER), "Worker.determine_available_memory")
    check("available_kv_cache_memory_bytes" in names(w), "Worker keeps available_kv_cache_memory_bytes")
    for runner in (vllm_plugin.RUNNER_V1, vllm_plugin.RUNNER_V2):
        lm = find(parse(runner), "GPUModelRunner.load_model")
        check(lm is not None and "model_memory_usage" in names(lm), "%s.load_model sets model_memory_usage" % runner)


CASES = [case_tail_plan, case_settings, case_entry_point, case_after_import, case_kv_v2, case_kv_v1_packed,
         case_weights, case_weights_and_kv, case_fail_open, case_blocked, case_drift, case_upstream]


def main():
    global VLLM_SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/host")
    ap.add_argument("--vllm-src")
    a = ap.parse_args()
    VLLM_SRC = a.vllm_src and os.path.isdir(a.vllm_src) and os.path.abspath(a.vllm_src)
    bindir = os.path.abspath(a.bin)
    os.environ["SPARK_ATTIC_LIBCUDA"] = os.path.join(bindir, "fakecuda", "libcuda.so.1")
    sys.modules["torch"] = TORCH
    os.umask(0o022)
    for case in CASES:
        rig = Rig(bindir, "vp-" + case.__name__[5:])
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
    print("test_vllm_plugin: %d cases, %d failures" % (len(CASES), len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

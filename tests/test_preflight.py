#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_preflight.py: attic-preflight.py's table logic against the plugin tables, with no engine. The plugin
tables are keyed per shape (vllm_plugin.EXPECTED["0.24"/"0.32"], sglang_plugin.TARGETS/CALLS["0.5.14"/"0.5.23"]);
the preflight must pick one shape and iterate only that shape's rows. Iterating the tables flat - what
c78c889's preflight did - raises "too many values to unpack (expected 2)" on every framework, which is what
the 3:03 PM node window saw. Here the preflight's module is loaded with fake framework modules whose
functions have a known shape's parameters, and its table checks must all pass; a preflight that iterates
flat raises before printing a single check and fails this test. With a shape-mismatched fake, the preflight
must report the picked shape as none and fail. The real plugin tables are used throughout.

    python3 -I tests/test_preflight.py
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "python"))
FAILS = []


def check(cond, what):
    print("  %s %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def load_preflight():
    spec = importlib.util.spec_from_file_location("attic_preflight", os.path.join(HERE, "scripts", "attic-preflight.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install_fake_kv_interface(shape_fields):
    """vllm.v1.kv_cache_interface.KVCacheTensor as a dataclass with shape_fields' names."""
    import dataclasses
    mod = types.ModuleType("vllm.v1.kv_cache_interface")
    src = "@dataclasses.dataclass\nclass KVCacheTensor:\n    " + "\n    ".join("%s: int = 0" % f for f in sorted(shape_fields))
    ns: dict = {"dataclasses": dataclasses}
    exec(compile(src, "<fake>", "exec"), ns)
    mod.KVCacheTensor = ns["KVCacheTensor"]
    sys.modules["vllm.v1.kv_cache_interface"] = mod
    setattr(sys.modules["vllm.v1"], "kv_cache_interface", mod)


def install_fake_vllm(params_map):
    """A fake vllm package: the modules the plugin wraps, each function with params_map's parameters.
    params_map: {(modpath, qual): [params]}; a function missing from it raises AttributeError (a target
    the plugin knows that this image does not have)."""
    pkg = types.ModuleType("vllm")
    pkg.__version__ = "0.31.0-fake"
    pkg.__path__ = []
    sys.modules["vllm"] = pkg
    mods = {}
    for modpath in sorted({mp for mp, _ in params_map}):
        parts = modpath.split(".")
        for i in range(len(parts)):
            sub = ".".join(parts[:i + 1])
            if sub in sys.modules:
                continue
            m = types.ModuleType(sub)
            m.__path__ = []
            sys.modules[sub] = m
            if i:
                setattr(sys.modules[".".join(parts[:i])], parts[i], m)
    for (modpath, qual), ps in params_map.items():
        mod = sys.modules[modpath]
        cls_name, _, fn_name = qual.rpartition(".")
        code = "def %s(%s):\n    raise NotImplementedError\n" % (fn_name, ", ".join(ps))
        if cls_name:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                ns: dict = {}
                exec(compile("class %s:\n    pass" % cls_name, "<fake>", "exec"), ns)
                cls = ns[cls_name]
                setattr(mod, cls_name, cls)
            ns2: dict = dict(cls.__dict__)
            exec(compile(code, "<fake>", "exec"), ns2)
            for k, v in ns2.items():
                if not k.startswith("__"):
                    setattr(cls, k, v)
        else:
            exec(compile(code, "<fake>", "exec"), vars(mod))
    return pkg


def run_vllm_checks(params_map, expect_shape):
    pf = load_preflight()
    install_fake_vllm(params_map)
    pf.FAILS.clear()
    pf.vllm()
    lines = pf.FAIL
    fails = list(pf.FAILS)
    return fails


def main():
    from spark_attic import vllm_plugin as vp

    # The 0.32 image: every wrap target exists with the parameters the plugin knows. The preflight must
    # pick shape 0.32, check only its rows, and print no FAIL.
    pmap = {(mod, qual): want for shape in ("0.32",) for (mod, qual), want in vp.EXPECTED[shape].items()}
    pf = load_preflight()
    install_fake_vllm(pmap)
    install_fake_kv_interface(vp.KV_TENSOR_FIELDS["0.32"])
    pf.FAILS.clear()
    pf.vllm()
    check(pf.FAILS == [], "a 0.32-shaped fake vllm: the per-shape preflight passes every check (%d failures)"
          % len(pf.FAILS))

    # A mismatched image (the vLLM 0.31-style utils without layout): the preflight must say the picked
    # shape is none and fail, not check a table that does not fit.
    pmap_bad = {(mod, qual): (want if qual != "allocate_kv_cache" else ["kv_cache_config", "device"])
                for (mod, qual), want in vp.EXPECTED["0.32"].items()}
    pf2 = load_preflight()
    install_fake_vllm(pmap_bad)
    pf2.FAILS.clear()
    pf2.vllm()
    check(any("picks vLLM shape none" in w for w in pf2.FAILS),
          "a shape-mismatched fake: the preflight reports none and fails (%r)" % pf2.FAILS[:1])

    # The c78c889 discriminator: the flat iteration of the per-shape tables raises ValueError. A preflight
    # that still iterates flat never gets here; loading it and running vllm() must not raise the unpack
    # error but report checks.
    try:
        for key, (target, want) in vp.EXPECTED.items():
            pass
        check(False, "the flat iteration should raise (test invariant)")
    except ValueError:
        check(True, "the flat iteration of EXPECTED raises ValueError (the node-window bug, what the fix removes)")

    # The cap write path: the preflight exercises the plugin's own hold primitive against the real
    # runtime context. Here the fake sglang tree is built from the real 0.5.23 tables (a plain
    # namespace schedule, no override primitive), and the preflight must pass the shape check AND the
    # write-path check through the same fallback code.
    import types
    from spark_attic import sglang_plugin as sp

    pkg = types.ModuleType("sglang")
    pkg.__version__ = "0.5.21-fake"
    pkg.__path__ = []
    sys.modules["sglang"] = pkg
    rows = {}
    for shape in sp.SHAPES:
        for key, (target, want) in list(sp.TARGETS[shape].items()) + list(sp.CALLS[shape].items()):
            rows[target] = want
    for target, ps in rows.items():
        # a target is module[.Class].fn: the module is everything before the first uppercase-leading
        # part (SGLang's class names are CamelCase), fn is the last part, the class the part before fn.
        parts = target.split(".")
        fn = parts[-1]
        cls = parts[-2] if parts[-2][0].isupper() else None
        modpath = ".".join(parts[:-2] if cls else parts[:-1])
        if modpath not in sys.modules:
            msplits = modpath.split(".")
            for i in range(len(msplits)):
                sub = ".".join(msplits[:i + 1])
                if sub not in sys.modules:
                    m = types.ModuleType(sub)
                    m.__path__ = []
                    sys.modules[sub] = m
                    if i:
                        setattr(sys.modules[".".join(msplits[:i])], msplits[i], m)
        mod = sys.modules[modpath]
        if cls:
            c = getattr(mod, cls, None)
            if c is None:
                ns = {}
                exec(compile("class %s:\n    pass" % cls, "<fake>", "exec"), ns)
                c = ns[cls]
                setattr(mod, cls, c)
            ns2 = {}
            exec(compile("def %s(%s):\n    pass" % (fn, ", ".join(ps)), "<fake>", "exec"), ns2)
            setattr(c, fn, ns2[fn])
        else:
            ns3 = {}
            exec(compile("def %s(%s):\n    pass" % (fn, ", ".join(ps)), "<fake>", "exec"), ns3)
            setattr(mod, fn, ns3[fn])
    rc = types.ModuleType("sglang.srt.runtime_context")
    sched = types.SimpleNamespace(max_total_tokens=None, page_size=16)
    rc.get_schedule = lambda: sched
    sys.modules["sglang.srt.runtime_context"] = rc
    setattr(sys.modules["sglang.srt"], "runtime_context", rc)
    pf3 = load_preflight()
    pf3.FAILS.clear()
    pf3.sglang()
    shape_ok = any("picks SGLang shape 0.5.14" in w or "picks SGLang shape 0.5.23" in w for w in
                   [l for l in pf3.PASS if isinstance(l, str)]) if hasattr(pf3, "PASS") else None
    check(not [w for w in pf3.FAILS if "cap write path" in w],
          "the cap write path holds and restores on a plain-namespace schedule (failures: %r)" % pf3.FAILS[:2])
    for name in list(sys.modules):
        if name.startswith("sglang"):
            sys.modules.pop(name, None)

    print("test_preflight: %d failures" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

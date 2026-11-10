#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""attic-preflight.py vllm|sglang: run in the framework's image before node-test-2.sh starts its servers. One PASS or
FAIL line per check; exit 0 when all pass.

  1. the client: atticd's info, a 64 MiB slice as a torch tensor that round-trips data, and 32 MiB of device memory
     glued before a 64 MiB slice in one address range, round-tripped across the seam;
  2. the plugin's entry point is visible in this image;
  3. every function the plugin wraps, hooks or calls has the parameters the plugin knows, and the calls it relies on
     are there.

Needs atticd on SPARK_ATTIC_SOCK with SPARK_ATTIC_KEY or SPARK_ATTIC_KEY_FILE, and python/ on PYTHONPATH.
"""
import importlib
import importlib.metadata
import os
import sys

os.environ["SPARK_ATTIC_PLACE"] = "off"  # importing the framework may load its plugins: ours must wrap nothing here
FAILS = []


def result(ok, what):
    print("%s %s" % ("PASS" if ok else "FAIL", what), flush=True)
    if not ok:
        FAILS.append(what)


def client():
    import torch

    import spark_attic

    torch.cuda.init()
    print("torch %s, CUDA %s, %s, spark_attic %s" % (torch.__version__, torch.version.cuda,
                                                     torch.cuda.get_device_name(), spark_attic.__version__))
    info = spark_attic.info()
    result(bool(info) and info.get("largest", 0) >= 64 << 20, "atticd info: %s" % info)
    m = spark_attic.borrow(64 << 20)
    result(m is not None, "borrow 64 MiB: %r" % m)
    if m is not None:
        t = m.tensor(torch.float32)
        ref = torch.arange(t.numel(), device="cuda", dtype=torch.float32)
        t.copy_(ref)
        torch.cuda.synchronize()
        result(torch.equal(t, ref) and t.data_ptr() == m.ptr, "a float32 tensor over the slice round-trips 64 MiB")
        del t
        m.close()
    pool = spark_attic.Pool()
    try:
        g = pool.map(32 << 20, [64 << 20])
        x = g.tensor(torch.int32)
        gen = torch.Generator(device="cuda").manual_seed(1)
        ref = torch.randint(-2 ** 31, 2 ** 31 - 1, (x.numel(),), dtype=torch.int32, device="cuda", generator=gen)
        x.copy_(ref)
        torch.cuda.synchronize()
        result(torch.equal(x, ref) and g.attic_bytes == 64 << 20,
               "32 MiB device memory + a 64 MiB slice in one range round-trip across the seam: %r" % g)
        del x
        g.close()
    finally:
        pool.close()


def entry_point(group):
    eps = [e.value for e in importlib.metadata.entry_points(group=group) if e.name == "spark_attic"]
    result(len(eps) == 1, "entry point %s: %s" % (group, eps))


def resolve(mod, qual):
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def vllm():
    import vllm as v

    from spark_attic import vllm_plugin as p
    from spark_attic._place import params

    print("vllm %s" % v.__version__)
    entry_point("vllm.general_plugins")
    for (mod, qual), want in p.EXPECTED.items():
        try:
            got = params(resolve(mod, qual))  # what the plugin reads, through torch.inference_mode and the like
        except Exception as e:
            got = "%s: %s" % (type(e).__name__, e)
        result(got == want, "%s.%s(%s): %s" % (mod, qual, ", ".join(want), got))
    from vllm.v1.kv_cache_interface import KVCacheTensor

    fields = set(getattr(KVCacheTensor, "__dataclass_fields__", {}))
    result(fields == p.KV_TENSOR_FIELDS, "KVCacheTensor's fields: %s" % sorted(fields))
    init = resolve(p.ATTN_V2, "init_kv_cache")
    result(p._calls(init, "_allocate_kv_cache") and resolve(p.RUNNER_V2, "init_kv_cache") is init,
           "the V2 runner allocates through %s.init_kv_cache -> _allocate_kv_cache" % p.ATTN_V2)
    result(p._calls(resolve(p.RUNNER_V1, "GPUModelRunner.initialize_kv_cache_tensors"), "_allocate_kv_cache_tensors"),
           "the V1 runner's initialize_kv_cache_tensors calls _allocate_kv_cache_tensors")


def sglang():
    import sglang as s

    from spark_attic import sglang_plugin as p

    print("sglang %s" % getattr(s, "__version__", "?"))
    entry_point("sglang.srt.plugins")
    for key, (target, want) in list(p.TARGETS.items()) + list(p.CALLS.items()):
        try:
            p._check(key)
            result(True, "%s(%s)" % (target, ", ".join(want)))
        except Exception as e:
            result(False, "%s: %s" % (target, e))


def main():
    fw = sys.argv[1] if len(sys.argv) > 1 else ""
    if fw not in ("vllm", "sglang"):
        print("usage: attic-preflight.py vllm|sglang")
        return 2
    for step in (client, vllm if fw == "vllm" else sglang):
        try:
            step()
        except Exception as e:
            result(False, "%s: %s: %s" % (step.__name__, type(e).__name__, e))
    print("preflight %s: %d failures" % (fw, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""What the vLLM and SGLang hooks share: running a function right after a module loads, the placement settings, and
the tail plan that puts the last of a list of buffers into one lent slice."""
import fnmatch
import importlib.abc
import inspect
import logging
import os
import sys
import threading
from dataclasses import dataclass

from ._proto import SLICE

log = logging.getLogger("spark_attic")
MIB = 1 << 20
PLACES = ("kv", "weights")


def round_up(n, to):
    return (n + to - 1) // to * to


class Mismatch(Exception):
    """The framework is not the one a hook was written against; the hook stays off."""


def params(fn):
    """The names of fn's parameters as a call to it takes them, read through wrappers that keep the function as
    __wrapped__ (functools.wraps: torch.inference_mode, tracing decorators), whose own are (*args, **kwargs)."""
    return [p.name for p in inspect.signature(fn).parameters.values()]


# -------------------------------------------------------------------------------------------- after a module loads

class _AfterImport(importlib.abc.MetaPathFinder):
    """Runs hooks[name](module) right after `name` executes, whoever imports it. Finds nothing itself: it asks the
    finders after it for the spec and wraps the loader's exec_module."""

    def __init__(self):
        self.hooks = {}
        self.lock = threading.Lock()

    def find_spec(self, name, path=None, target=None):
        with self.lock:
            fns = self.hooks.get(name)
        if not fns:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        with self.lock:
            fns = self.hooks.pop(name, None)
        if not fns:
            return spec
        exec_module = loader.exec_module

        def run(module):
            del loader.exec_module  # once: a reload runs the module alone
            exec_module(module)
            for fn in fns:
                _call(fn, module)

        loader.exec_module = run
        return spec


_finder = _AfterImport()


def _call(fn, module):
    try:
        fn(module)
    except Mismatch as e:
        log.warning("spark-attic: off for %s: %s", module.__name__, e)
    except Exception as e:  # a hook must never stop the framework from loading
        log.warning("spark-attic: off for %s: %s: %s", module.__name__, type(e).__name__, e)


def when_imported(name, fn):
    """fn(module) once `name` is loaded: now if it already is, else right after it loads."""
    module = sys.modules.get(name)
    if module is not None:
        _call(fn, module)
        return
    with _finder.lock:
        _finder.hooks.setdefault(name, []).append(fn)
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)


# ---------------------------------------------------------------------------------------------------- settings

def log_to_stderr():
    """Our log lines on stderr, beside the framework's: vLLM and SGLang set up only their own loggers, so the INFO
    lines (how much carveout went where) would be dropped. Not when the logger has a handler already or
    SPARK_ATTIC_LOG=0."""
    if os.environ.get("SPARK_ATTIC_LOG") == "0" or log.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s spark_attic %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


@dataclass(frozen=True)
class Placement:
    place: tuple = ("kv",)          # where the carveout goes: "kv", "weights", or both (weights first)
    weights: tuple = ("embed_tokens",)  # for "weights": parameter names or globs
    cap: int = 0                    # at most this many bytes of carveout; 0 for all atticd has free

    @classmethod
    def from_env(cls, env, default="kv"):
        """SPARK_ATTIC_PLACE (kv, weights, weights,kv or off), SPARK_ATTIC_WEIGHTS, SPARK_ATTIC_MIB."""
        raw = (env.get("SPARK_ATTIC_PLACE") or default).replace(" ", "").lower()
        if raw in ("off", "none", "0"):
            return cls(place=())
        place = tuple(dict.fromkeys(p for p in raw.split(",") if p))
        bad = [p for p in place if p not in PLACES]
        if bad or not place:
            raise ValueError("SPARK_ATTIC_PLACE=%s: use kv, weights, weights,kv or off" % raw)
        weights = tuple(w.strip() for w in (env.get("SPARK_ATTIC_WEIGHTS") or "embed_tokens").split(",")
                        if w.strip())
        cap = env.get("SPARK_ATTIC_MIB") or "0"
        if not cap.isdigit():
            raise ValueError("SPARK_ATTIC_MIB=%s is not a whole number of MiB" % cap)
        return cls(place=tuple(p for p in PLACES if p in place), weights=weights, cap=int(cap) * MIB)


def name_matches(name, patterns):
    """A parameter name matches a pattern that equals it, globs it, or is a run of its dotted parts
    ("embed_tokens" matches "model.embed_tokens.weight")."""
    dotted = "." + name + "."
    return any(fnmatch.fnmatchcase(name, p) or ("." + p.strip(".") + ".") in dotted for p in patterns)


# ------------------------------------------------------------------------------------------------------ the plan

@dataclass(frozen=True)
class TailPlan:
    """Buffers [first, n) go into one mapping, back to back at offsets[i - first]: `front` bytes of device memory
    first, then the lent slice. Buffers before `first` stay where the framework puts them."""
    first: int
    front: int
    offsets: tuple
    attic_used: int


def tail_plan(sizes, attic):
    """The last buffers of sizes[] that fit in `attic` bytes (whole 2 MiB slices), whole, plus the one before them
    split across device memory and what is left of the slice. Each buffer starts on a 2 MiB boundary, as the caching
    allocator places a buffer this large, so the device part is whole 2 MiB pages too."""
    aligned = [round_up(s, SLICE) for s in sizes]
    tail, i = 0, len(sizes)
    while i > 0 and tail + aligned[i - 1] <= attic:
        tail += aligned[i - 1]
        i -= 1
    first, front = i, 0
    if i > 0 and attic > tail:  # what is left of the slice goes to the end of the buffer before the tail
        first = i - 1
        front = aligned[first] - (attic - tail)
    offsets, at = [], 0
    for s in aligned[first:]:
        offsets.append(at)
        at += s
    return TailPlan(first=first, front=front, offsets=tuple(offsets), attic_used=min(attic, at - front))


# ------------------------------------------------------------------------------------- what the hooks do at runtime

class Ledger:
    """How much carveout one process may still take: atticd's largest free run, under SPARK_ATTIC_MIB."""

    def __init__(self, cap=0):
        self.cap, self.spent = cap, 0

    def room(self, pool):
        n = pool.info()["largest"]
        if self.cap:
            n = min(n, max(self.cap - self.spent, 0))
        return n - n % SLICE

    def spend(self, n):
        self.spent += n


def give_back(state):
    """Close the plugin's atticd connection when nothing borrowed through it is in use: atticd counts a slice as
    lent until its connection closes, so this hands an unused slice back at once."""
    with state.lock:
        if state.pool is not None and not state.keep and state.kv is None:
            state.pool.close()
            state.pool = None


def device_index(device):
    """The CUDA ordinal of a torch.device, a "cuda" / "cuda:N" string or an int; "cuda" alone is the current one."""
    if isinstance(device, int):
        return device
    kind, index = getattr(device, "type", None), getattr(device, "index", None)
    if kind is None:
        kind, _, n = str(device).partition(":")
        index = int(n) if n else None
    if kind != "cuda":
        raise Mismatch("device %s is not a CUDA device" % (device,))
    if index is not None:
        return index
    import torch
    return torch.cuda.current_device()


def model_of(holder):
    """The nn.Module a model runner holds (unwrapping compile and graph wrappers)."""
    model = getattr(holder, "model", None)
    for _ in range(4):
        if model is None or hasattr(model, "named_parameters"):
            break
        model = model.unwrap() if hasattr(model, "unwrap") else getattr(model, "model", None)
    if model is None or not hasattr(model, "named_parameters"):
        raise Mismatch("the model runner has no model module")
    return model


def move_params(model, patterns, pool, ledger, keep, dev):
    """Move the parameters of `model` that `patterns` name (each tied parameter once) into the carveout, one slice
    each, while the ledger has room. The mappings go to `keep`. Returns ([(name, slice bytes)], device bytes freed)."""
    import gc
    import torch

    want, seen = [], set()
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) not in seen and name_matches(name, patterns):
            seen.add(id(p))
            want.append((name, p))
    if not want:
        log.warning("spark-attic: weights placement: no parameter matches %s", ",".join(patterns))
        return [], 0
    torch.cuda.synchronize(dev)
    gc.collect()
    before = torch.cuda.memory_allocated(dev)
    moved = []
    for name, p in want:
        n = p.numel() * p.element_size()
        need = round_up(n, SLICE)
        if not p.is_cuda or not p.is_contiguous():
            log.warning("spark-attic: %s stays: not a contiguous CUDA tensor", name)
            continue
        room = ledger.room(pool)
        if need > room:
            log.warning("spark-attic: %s stays: it needs %d MiB, %d MiB of carveout is left", name, need // MIB,
                        room // MIB)
            continue
        m = pool.map(0, [need])
        try:
            t = m.tensor(p.dtype, shape=tuple(p.shape), nbytes=n)
            t.copy_(p.data)
            torch.cuda.synchronize(dev)
            p.data = t
        except Exception:
            m.close()
            raise
        keep.append(m)
        ledger.spend(need)
        moved.append((name, need))
    if not moved:
        return [], 0
    gc.collect()
    torch.cuda.empty_cache()
    return moved, max(before - torch.cuda.memory_allocated(dev), 0)

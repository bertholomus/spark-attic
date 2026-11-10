#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""attic-bench.py: what node-test-2.sh measures on one running server (OpenAI-compatible /v1/completions), and its
summary. Standard library only.

    attic-bench.py --port 8000 [--model m] [--reps 3]          one JSON object on stdout
    attic-bench.py --summary DIR --frameworks "vllm sglang" --arms "off:off kv:kv weights:weights off2:off" \
                   --bytes-per-token N
    attic-bench.py --kv-bytes-per-token MODEL/config.json      K and V bytes a token takes in the KV cache

Replies: 8 prompts at temperature 0, one at a time (the same batch every run), 200 tokens each: text and finish
reason. Decode: 512 tokens with ignore_eos, streamed. Decode tok/s = (tokens - 1) / (last token - first token), at 1
stream and at 8 streams sent together (aggregate: every stream's tokens after its first, over the time from the first
first token to the last token; and the median stream), 3 reps each, after a warmup.

The summary reads DIR/<fw>-<arm>.json, DIR/<fw>-<arm>.kv (the server's KV pool log lines) and DIR/status-<fw>-<arm>.json
(what atticd lent while the arm ran) and prints one line per arm, then the KV pool check. Arms are ATTIC_ARMS specs,
name:SPARK_ATTIC_PLACE[:VAR=value]; a bare name is its own placement. Each arm runs with the same pinned KV budget, so
every arm without kv placement must have the same pool (the base), and an arm with kv placement alone must have the
base plus what atticd lent it, within TOL (block and page rounding). It exits 1 when a placement arm's replies differ
from the first arm's while the off arms agree, when the off arms' replies differ from each other (a baseline that does
not repeat cannot show equal replies: the arms run again), or when the pool check fails or cannot be made.
"""
import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.request

REPLY_PROMPTS = [
    "The history of the printing press begins",
    "def merge_sort(xs):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
    "Explain, step by step, why the sky is blue during the day and red at sunset.",
    "A short story about a lighthouse keeper who finds a message in a bottle:\n\n",
    "SELECT customers.name, COUNT(orders.id) AS n\nFROM customers\n",
    "The three laws of thermodynamics are",
    "// A small HTTP server in Go that answers /health with 200\npackage main\n\nimport (\n",
    "Translate into French: The weather is lovely today, so we will walk to the market.\n",
]
DECODE_PROMPTS = [
    "Write a long, detailed essay about the ocean.",
    "Write a long story about a city in the year 3000.",
    "Explain in great detail how a compiler works.",
    "Write a long letter to a friend about a journey across mountains.",
    "Describe, at length, the life cycle of a star.",
    "Write a detailed guide to growing tomatoes.",
    "Tell a long fairy tale about a dragon and a baker.",
    "Explain at length how the internet routes a packet.",
]
REPLY_TOKENS, DECODE_TOKENS, STREAMS = 200, 512, 8
MIB = 1 << 20
TOL = 8 * MIB  # a pool may differ from base + lent by block and page rounding, no more
POOL = (re.compile(r"GPU KV cache size: ([\d,]+) tokens"),  # vLLM
        re.compile(r"max_total_num_tokens=(\d+)"))  # SGLang
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def post(port, body, timeout=900):
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def reply(port, model, prompt):
    with post(port, {"model": model, "prompt": prompt, "max_tokens": REPLY_TOKENS, "temperature": 0, "seed": 0}) as r:
        out = json.loads(r.read())
    c = out["choices"][0]
    return {"text": c["text"], "finish_reason": c.get("finish_reason"),
            "tokens": (out.get("usage") or {}).get("completion_tokens")}


def decode(port, model, prompt, tokens, barrier=None):
    """One streamed request: when its first and last text arrived, and how many tokens it made."""
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": 0, "seed": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    if barrier is not None:
        barrier.wait()
    t0 = time.perf_counter()
    first = last = None
    chunks, usage = 0, None
    with post(port, body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for c in obj.get("choices") or []:
                if c.get("text"):
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    chunks += 1
    n = (usage or {}).get("completion_tokens") or chunks
    return {"t0": t0, "first": first, "last": last, "tokens": n, "chunks": chunks}


def rate(r):
    span = (r["last"] or 0) - (r["first"] or 0)
    return (r["tokens"] - 1) / span if span > 0 and r["tokens"] > 1 else 0.0


def streams(port, model, n, tokens):
    barrier = threading.Barrier(n)
    out = [None] * n
    errs = []

    def run(i):
        try:
            out[i] = decode(port, model, DECODE_PROMPTS[i % len(DECODE_PROMPTS)], tokens, barrier)
        except Exception as e:  # reported, and the rep counts as failed
            errs.append("%s: %s" % (type(e).__name__, e))
            barrier.abort()
    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise RuntimeError("; ".join(errs))
    span = max(r["last"] for r in out) - min(r["first"] for r in out)
    return {"agg": sum(r["tokens"] - 1 for r in out) / span if span > 0 else 0.0,
            "stream_median": statistics.median(rate(r) for r in out),
            "tokens": [r["tokens"] for r in out],
            "ttft": statistics.median(r["first"] - r["t0"] for r in out)}


def bench(port, model, reps):
    decode(port, model, "Hello", 16)  # warmup: one stream, then all of them (CUDA graphs for both batch sizes)
    streams(port, model, STREAMS, 32)
    out = {"replies": [reply(port, model, p) for p in REPLY_PROMPTS], "s1": [], "s8": []}
    for _ in range(reps):
        r = decode(port, model, DECODE_PROMPTS[0], DECODE_TOKENS)
        out["s1"].append({"rate": rate(r), "tokens": r["tokens"], "ttft": r["first"] - r["t0"]})
    for _ in range(reps):
        out["s8"].append(streams(port, model, STREAMS, DECODE_TOKENS))
    return out


# ------------------------------------------------------------------------------------------------------ summary

def med(xs):
    return statistics.median(xs) if xs else 0.0


def pct(x, base):
    return "%+.1f%%" % ((x / base - 1) * 100) if base else "n/a"


def arm_specs(arms):
    """[(name, set of placements)] from ATTIC_ARMS specs, name:place[:VAR=value]; a bare name is its own placement
    (off2 is off)."""
    out = []
    for spec in arms:
        parts = spec.split(":")
        place = parts[1] if len(parts) > 1 else ("off" if parts[0].startswith("off") else parts[0])
        out.append((parts[0], set(place.split(",")) - {""}))
    return out


def pool_tokens(text):
    """The KV pool's size in tokens from the server's log lines, or None."""
    for pat in POOL:
        found = pat.findall(text)
        if found:
            return int(found[-1].replace(",", ""))
    return None


def read(path, how):
    try:
        with open(path) as f:
            return how(f)
    except (OSError, ValueError):
        return None


def num(n):
    return "{:,}".format(n) if n is not None else "?"


def pools(d, fw, specs, bpt):
    """The KV pool check of one framework: (lines, fail, {arm: tokens or None}) over the arms whose log was read."""
    toks = {}
    for name, _ in specs:
        text = read(os.path.join(d, "%s-%s.kv" % (fw, name)), lambda f: f.read())
        if text is not None:
            toks[name] = pool_tokens(text)
    lines, fail = [], False
    if not toks:
        return lines, fail, toks

    def bad(why):
        lines.append("%s KV pool FAIL: %s" % (fw, why))
        return True
    unread = [n for n in toks if toks[n] is None]
    if unread:
        fail = bad("no pool size in the log of %s" % ", ".join(unread))
    base_arms = [n for n, place in specs if "kv" not in place and n in toks]
    seen = {toks[n] for n in base_arms} - {None}
    if len(seen) != 1:
        if not base_arms:
            fail = bad("no arm without kv placement ran: no base to compare with")
        elif len(seen) > 1:
            fail = bad("the base moved: %s" % ", ".join("%s %s tokens" % (n, num(toks[n])) for n in base_arms))
        return lines, True, toks
    base = seen.pop()
    lines.append("%s KV pool base %s tokens (%s)" % (fw, num(base), ", ".join(base_arms)))
    if not bpt:
        return lines, bad("no --bytes-per-token: the kv arms' gain is not checked"), toks
    for name, place in specs:
        if "kv" not in place or toks.get(name) is None:
            continue
        gain = (toks[name] - base) * bpt
        st = read(os.path.join(d, "status-%s-%s.json" % (fw, name)), json.load)
        lent = st.get("lent") if isinstance(st, dict) else None
        what = "%s %s tokens = base + %s tokens = %+.1f MiB; atticd lent %s" % (
            name, num(toks[name]), num(toks[name] - base), gain / MIB,
            "%.1f MiB" % (lent / MIB) if isinstance(lent, int) else "?")
        if place != {"kv"}:
            lines.append("%s KV pool %s (not checked: atticd's count holds the %s too)" % (
                fw, what, ", ".join(sorted(place - {"kv"}))))
        elif not isinstance(lent, int) or lent <= 0:
            fail = bad("%s: atticd lent nothing, or its status was not read" % what)
        elif abs(gain - lent) > TOL:
            fail = bad("%s: more than %d MiB apart" % (what, TOL // MIB))
        else:
            lines.append("%s KV pool ok: %s" % (fw, what))
    return lines, fail, toks


def summary(d, frameworks, arms, bpt=None):
    lines, fail = [], False
    specs = arm_specs(arms)
    names = [n for n, _ in specs]
    for fw in frameworks:
        runs = {}
        for arm in names:
            p = os.path.join(d, "%s-%s.json" % (fw, arm))
            if os.path.exists(p):
                with open(p) as f:
                    runs[arm] = json.load(f)
        plines, pfail, toks = pools(d, fw, specs, bpt)
        if not runs and not toks:
            lines.append("%s: no runs" % fw)
            continue
        fail = fail or pfail
        if runs:
            base_arm = next(a for a in names if a in runs)
            offs = [n for n, place in specs if n in runs and place == {"off"}]
            base = runs[base_arm]
            same = lambda a: [x["text"] for x in runs[a]["replies"]] == [x["text"] for x in base["replies"]]  # noqa
            repro = len(offs) >= 2 and all(same(a) for a in offs)
            b1 = med([x["rate"] for x in base["s1"]])
            b8 = med([x["agg"] for x in base["s8"]])
        for arm, place in specs:
            pool = "KV pool %s tokens" % num(toks.get(arm))
            if arm not in runs:
                lines.append("%s %-8s missing | %s" % (fw, arm, pool))
                continue
            r = runs[arm]
            n = sum(1 for x, y in zip(r["replies"], base["replies"]) if x["text"] == y["text"])
            if arm == base_arm:
                eq = "baseline"
            else:
                eq = "%s (%d/%d)" % ("equal" if n == len(base["replies"]) else "DIFFER", n, len(base["replies"]))
                if n != len(base["replies"]) and place != {"off"} and repro:
                    fail = True
                    eq += " FAIL"
            s1 = med([x["rate"] for x in r["s1"]])
            s8 = med([x["agg"] for x in r["s8"]])
            s8m = med([x["stream_median"] for x in r["s8"]])
            lines.append("%s %-8s replies %s | decode 1 stream %.1f tok/s (%s) | 8 streams %.1f tok/s (%s), median "
                         "stream %.1f | %s" % (fw, arm, eq, s1, pct(s1, b1), s8, pct(s8, b8), s8m, pool))
            kv = read(os.path.join(d, "%s-%s.kv" % (fw, arm)),
                      lambda f: [x[x.index("spark-attic:"):].strip()[:220] for x in f if "spark-attic:" in x])
            lines.extend("    " + x for x in dict.fromkeys(kv or []))  # one line per process, the same text
        if runs and len(offs) >= 2:
            fail = fail or not repro
            lines.append("%s the off arms (%s) %s" % (fw, ", ".join(offs), "agree: the baseline is reproducible"
                                                      if repro else "DIFFER: replies are not reproducible here, so "
                                                      "equality says nothing about the carveout: FAIL, run the arms "
                                                      "again"))
        lines.extend(plines)
    return lines, fail


def kv_bytes_per_token(config):
    """K and V bytes one token takes in all layers of a dense-attention model, from its Hugging Face config.json, with
    the KV cache in the model's dtype (both frameworks' default): layers x 2 x KV heads x head dim x dtype bytes."""
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    dtype = config.get("torch_dtype") or config.get("dtype")
    if dtype not in DTYPE_BYTES:
        raise ValueError("dtype %r" % dtype)
    return config["num_hidden_layers"] * 2 * kv_heads * head_dim * DTYPE_BYTES[dtype]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="m")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--summary")
    ap.add_argument("--frameworks", default="vllm sglang")
    ap.add_argument("--arms", default="off:off kv:kv weights:weights off2:off")
    ap.add_argument("--bytes-per-token", type=int)
    ap.add_argument("--kv-bytes-per-token", metavar="CONFIG")
    a = ap.parse_args()
    if a.kv_bytes_per_token:
        try:
            with open(a.kv_bytes_per_token) as f:
                print(kv_bytes_per_token(json.load(f)))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print("attic-bench.py: no KV bytes per token from %s: %s: %s" % (a.kv_bytes_per_token, type(e).__name__,
                                                                            e), file=sys.stderr)
            return 2
        return 0
    if a.summary:
        lines, fail = summary(a.summary, a.frameworks.split(), a.arms.split(), a.bytes_per_token)
        print("\n".join(lines))
        return 1 if fail else 0
    json.dump(bench(a.port, a.model, a.reps), sys.stdout)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

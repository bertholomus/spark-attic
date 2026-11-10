#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_bench.py: scripts/attic-bench.py against a stand-in OpenAI-compatible server (standard library), so the
node test's measurements and summary are right before they run on a node.

    python3 -I tests/test_bench.py
"""
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("attic_bench", os.path.join(HERE, "..", "scripts", "attic-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
FAILS = []
SEEN = []
TOKEN_S = 0.002  # the stand-in makes one token every 2 ms: 500 tok/s per stream


def check(ok, what):
    print("%s %s" % ("ok  " if ok else "FAIL", what))
    if not ok:
        FAILS.append(what)


class Server(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        n = body["max_tokens"]
        if not body.get("stream"):
            text = "reply to %s" % body["prompt"][:12]
            out = json.dumps({"choices": [{"text": text, "finish_reason": "length"}],
                              "usage": {"completion_tokens": n}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(0.01)  # prefill
        for i in range(n):
            time.sleep(TOKEN_S)
            self.wfile.write(b"data: " + json.dumps({"choices": [{"text": "t%d " % i}]}).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"completion_tokens": n}}).encode() +
                         b"\n\ndata: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main():
    http.server.ThreadingHTTPServer.request_queue_size = 64  # all 8 streams connect at once
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    bench.DECODE_TOKENS = 100
    out = bench.bench(port, "m", 2)
    srv.shutdown()

    check(len(out["replies"]) == 8 and all(r["finish_reason"] == "length" and r["tokens"] == 200
                                           for r in out["replies"]), "8 replies, 200 tokens each")
    replies = [b for b in SEEN if not b.get("stream")]
    check(all(b["temperature"] == 0 and b["model"] == "m" for b in replies), "replies at temperature 0")
    streamed = [b for b in SEEN if b.get("stream")]
    check(all(b["ignore_eos"] and b["stream_options"] == {"include_usage": True} and b["temperature"] == 0
              for b in streamed), "decode requests: ignore_eos, usage in the stream, temperature 0")
    check(len(streamed) == 1 + 8 + 2 + 2 * 8, "warmup 1 + 8, then 2 reps at 1 stream and 2 at 8: %d" % len(streamed))
    s1 = [x["rate"] for x in out["s1"]]
    check(all(250 < r < 520 and x["tokens"] == 100 for r, x in zip(s1, out["s1"])),
          "1 stream: (tokens - 1) over first-to-last token, about 500 tok/s: %r" % s1)
    s8 = [x["agg"] for x in out["s8"]]
    check(all(8 * 250 < a < 8 * 520 and x["tokens"] == [100] * 8 for a, x in zip(s8, out["s8"])),
          "8 streams: about 8 x 500 tok/s together: %r" % s8)
    check(all(250 < x["stream_median"] < 520 and x["ttft"] > 0.005 for x in out["s8"]), "median stream and TTFT")

    d = tempfile.mkdtemp(prefix="attic-bench-")
    bpt, mib = 114688, 1 << 20  # Qwen3-1.7B's KV bytes a token
    lent = 2044 * mib  # what NODE-TEST-1's atticd lends: 18,688 such tokens

    def arm(fw, name, data, tokens=None, attic=None, note=""):
        """An arm's bench JSON, its KV log lines (the pool size as the framework logs it) and atticd's status."""
        if data is not None:
            with open(os.path.join(d, "%s-%s.json" % (fw, name)), "w") as f:
                json.dump(data, f)
        if tokens is not None:
            with open(os.path.join(d, "%s-%s.kv" % (fw, name)), "w") as f:
                f.write(note + ("INFO GPU KV cache size: {:,} tokens\n".format(tokens) if fw == "vllm" else
                                "[x] max_total_num_tokens=%d, chunked_prefill_size=8192\n" % tokens))
        if attic is not None:
            with open(os.path.join(d, "status-%s-%s.json" % (fw, name)), "w") as f:
                json.dump({"state": "ready", "lent": attic}, f)

    def run(frameworks=("vllm",), arms=("off:off", "kv:kv", "weights:weights", "off2:off"), per_token=bpt):
        lines, fail = bench.summary(d, list(frameworks), list(arms), per_token)
        return "\n".join(lines), fail

    differ = json.loads(json.dumps(out))
    differ["replies"][3]["text"] += "!"
    slower = json.loads(json.dumps(out))
    for x in slower["s1"]:
        x["rate"] /= 2
    note = "x spark_attic INFO spark-attic: KV cache memory 18.02 GiB = 16.02 GiB vLLM's budget + 2044 MiB carveout\n"
    arm("vllm", "off", out, 150000, 0)
    arm("vllm", "kv", slower, 150000 + 18688, lent, note + note)
    arm("vllm", "weights", differ, 150000, 594 * mib)
    arm("vllm", "off2", out, 150000, 0)
    text, fail = run(("vllm", "sglang"))
    print(text)
    check(fail and "weights  replies DIFFER (7/8) FAIL" in text and "kv       replies equal (8/8)" in text
          and "(-50.0%)" in text and "KV pool 168,688 tokens" in text and "agree: the baseline is reproducible" in text
          and "sglang: no runs" in text, "summary: equal, DIFFER with reproducible off arms is a FAIL, tok/s vs off")
    check("vllm KV pool base 150,000 tokens (off, weights, off2)" in text and "vllm KV pool ok: kv 168,688 tokens = "
          "base + 18,688 tokens = +2044.0 MiB; atticd lent 2044.0 MiB" in text
          and text.count("spark-attic: KV cache memory") == 1, "pools: the base, the kv arm's gain is what atticd "
          "lent, the plugin's line once")
    arm("vllm", "weights", out)
    text, fail = run()
    check(not fail and "FAIL" not in text, "equal replies and every pool as it should be: no FAIL")
    arm("vllm", "kv", None, 150000 + 18688 - 64)
    check(not run()[1], "64 tokens (7 MiB) short of what was lent: rounding, no FAIL")
    for tokens, why in ((150000 + 18688 - 80, "80 tokens (8.75 MiB) short of what was lent"),
                        (150000, "no gain at all, NODE-TEST-2's SGLang symptom"),
                        (150000 + 2 * 18688, "twice what was lent")):
        arm("vllm", "kv", None, tokens)
        text, fail = run()
        check(fail and "vllm KV pool FAIL: kv %s tokens" % "{:,}".format(tokens) in text
              and "more than 8 MiB apart" in text, "%s: FAIL" % why)
    arm("vllm", "kv", None, 150000 + 18688)
    arm("vllm", "off2", None, 153467)
    text, fail = run()
    check(fail and "vllm KV pool FAIL: the base moved: off 150,000 tokens, weights 150,000 tokens, off2 153,467 "
          "tokens" in text, "an off arm with another pool: the base moved, FAIL")
    arm("vllm", "off2", None, 150000)
    with open(os.path.join(d, "vllm-kv.kv"), "w") as f:
        f.write(note)
    text, fail = run()
    check(fail and "no pool size in the log of kv" in text, "a kv arm with no pool size in its log: FAIL")
    arm("vllm", "kv", None, 150000 + 18688)
    os.unlink(os.path.join(d, "status-vllm-kv.json"))
    text, fail = run()
    check(fail and "atticd lent nothing, or its status was not read" in text, "atticd's status unread: FAIL")
    arm("vllm", "kv", None, None, lent)
    text, fail = run(per_token=None)
    check(fail and "no --bytes-per-token" in text, "no bytes per token: the gain cannot be checked, FAIL")
    text, fail = run(arms=("kv:kv",))
    check(fail and "no arm without kv placement ran" in text, "no base arm: FAIL")
    arm("vllm", "off2", differ)
    text, fail = run()
    check(fail and "are not reproducible" in text and "FAIL, run the arms again" in text,
          "off arms that differ: FAIL, run again")
    arm("sglang", "off", out, 150000, 0)
    arm("sglang", "both", out, 150000 + 13000, lent)
    text, fail = run(("sglang",), ("off:off", "both:weights,kv:SPARK_ATTIC_MIB=2044"))
    check(not fail and "sglang KV pool both 163,000 tokens = base + 13,000 tokens" in text
          and "(not checked: atticd's count holds the weights too)" in text,
          "SGLang's pool line, and a kv arm with weights too: reported, not checked")

    for cfg, want in (({"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
                        "hidden_size": 2048, "torch_dtype": "bfloat16"}, 114688),
                      ({"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256, "dtype": "float16"},
                       2048)):
        check(bench.kv_bytes_per_token(cfg) == want, "KV bytes per token: %d" % want)
    p = os.path.join(d, "config.json")
    script = os.path.join(HERE, "..", "scripts", "attic-bench.py")
    for cfg, rc, want in (({"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                            "head_dim": 128, "torch_dtype": "bfloat16"}, 0, "114688"),
                          ({"num_hidden_layers": 28, "num_attention_heads": 16, "head_dim": 128,
                            "torch_dtype": "int8"}, 2, "ValueError: dtype 'int8'"),
                          ({"num_attention_heads": 16}, 2, "KeyError")):
        with open(p, "w") as f:
            json.dump(cfg, f)
        r = subprocess.run([sys.executable, "-I", script, "--kv-bytes-per-token", p], capture_output=True, text=True)
        check(r.returncode == rc and want in r.stdout + r.stderr, "--kv-bytes-per-token: exit %d, %r"
              % (r.returncode, (r.stdout + r.stderr).strip()))
    print("test_bench: %d failures" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

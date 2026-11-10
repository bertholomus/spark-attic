#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_validate.py: attic-validate.sh on the host, with the fake RM shim and the fake libcuda. A build is enabled
(VALIDATED, then atticd told) only when attic-check passes; never on a failed check, another driver, or a carveout
still in use.

    python3 -I tests/test_validate.py --bin build/host
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "spark-attic-1"
FAILS = []


def check(cond, what):
    print("  %s %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def info(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.settimeout(10)
    s.connect(path)
    s.send(('{"key":"%s","op":"info"}' % KEY).encode())
    r = json.loads(s.recv(4096))
    s.close()
    return r


class Node:
    """A pretend node: its own /var/lib/attic, /run/spark-attic and driver version file."""

    def __init__(self, bindir):
        self.bin = bindir
        self.dir = tempfile.mkdtemp(prefix="attic-validate-")
        os.chmod(self.dir, 0o755)
        self.lib = os.path.join(self.dir, "lib")
        self.run = os.path.join(self.dir, "run")
        os.makedirs(self.run)
        self.version = os.path.join(self.dir, "version")
        self.driver("999.0.1")
        for name, text in (("meminfo", "MemTotal: 129798784 kB\nMemFree: 64000000 kB\n"),
                           ("iomem", "80000000-27fffffff : System RAM\n280000000-2ffffffff : reserved\n")):
            with open(os.path.join(self.dir, name), "w") as f:
                f.write(text)
        self.cand = os.path.join(self.dir, "cand", "999.0.1")
        os.makedirs(self.cand)
        shutil.copyfile(os.path.join(bindir, "fakeshim.so"), os.path.join(self.cand, "rmshim.so"))
        sha = hashlib.sha256(open(os.path.join(self.cand, "rmshim.so"), "rb").read()).hexdigest()
        with open(os.path.join(self.cand, "BUILD-INFO"), "w") as f:
            f.write("driver=999.0.1\nshim_sha256=%s\nattic_commit=test\n" % sha)
        self.prod = None
        self.env = dict(os.environ, ATTIC_TEST="1", ATTIC_BIN=bindir, ATTIC_LIB=self.lib, ATTIC_RUN=self.run,
                        ATTIC_VERSION_FILE=self.version, ATTIC_OTHER_SOCKETS=os.path.join(self.dir, "other.sock"),
                        ATTIC_CHECK_ARGS="--meminfo %s/meminfo --iomem %s/iomem" % (self.dir, self.dir),
                        ATTIC_DAEMON_ARGS="--meminfo %s/meminfo --iomem %s/iomem" % (self.dir, self.dir),
                        LD_LIBRARY_PATH=os.path.join(bindir, "fakecuda"),
                        ATTIC_FAKE_REGION="0x280000000:0x8000000", ATTIC_FAKE_HELD="0", ATTIC_HUP="true",
                        ATTIC_KEY_FILE=os.path.join(self.dir, "no-key-file"))

    def driver(self, v):
        with open(self.version, "w") as f:
            f.write("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  %s  Release Build  (test)\n" % v)

    def start_prod(self, *key):
        """The node's own atticd, before any VALIDATED: it refuses until told. `key`: its key flags (--key KEY)."""
        sock = os.path.join(self.run, "attic.sock")
        os.makedirs(os.path.join(self.lib, "artifacts"), exist_ok=True)
        os.chmod(self.lib, 0o755)
        os.chmod(os.path.join(self.lib, "artifacts"), 0o755)
        self.prod = subprocess.Popen([os.path.join(self.bin, "atticd"), "--socket", sock, *(key or ("--key", KEY)),
                                      "--artifacts", os.path.join(self.lib, "artifacts"), "--version-file", self.version,
                                      "--meminfo", os.path.join(self.dir, "meminfo"), "--iomem",
                                      os.path.join(self.dir, "iomem")],
                                     env=self.env, stderr=open(os.path.join(self.dir, "prod.log"), "a"))
        for _ in range(100):
            if os.path.exists(sock):
                break
            time.sleep(0.05)
        self.env["ATTIC_HUP"] = "kill -HUP %d" % self.prod.pid
        return sock

    def validate(self, modes="heap list", **extra):
        env = dict(self.env, **extra)
        p = subprocess.run([os.path.join(HERE, "scripts", "attic-validate.sh"), self.cand, modes], env=env,
                           capture_output=True, text=True, timeout=600)
        return p.returncode, p.stderr

    def validated(self):
        v = os.path.join(self.lib, "artifacts", "999.0.1", "VALIDATED")
        return open(v).read() if os.path.exists(v) else None

    def done(self):
        if self.prod and self.prod.poll() is None:
            self.prod.send_signal(signal.SIGTERM)
            self.prod.wait(10)
        shutil.rmtree(self.dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/host")
    bindir = os.path.abspath(ap.parse_args().bin)
    os.umask(0o022)

    n = Node(bindir)
    try:
        sock = n.start_prod()
        check(info(sock)["ok"] is False, "before validation the node's atticd lends nothing")
        rc, err = n.validate()
        v = n.validated() or ""
        check(rc == 0 and "modes=heap,list" in v and "driver=999.0.1" in v, "both modes pass: VALIDATED heap,list (%d) %s" % (rc, err[-400:]))
        shim = os.path.join(n.lib, "artifacts", "999.0.1", "rmshim.so")
        sha = hashlib.sha256(open(shim, "rb").read()).hexdigest() if os.path.exists(shim) else "missing"
        check(("shim_sha256=%s" % sha) in v, "VALIDATED names the installed shim's digest")
        time.sleep(0.5)
        r = info(sock)
        check(r["ok"] is True and r["size"] == 128 << 20, "told by SIGHUP, the node's atticd lends: %r" % r)
        rc, err = n.validate()
        check(rc == 4 and "is ready to lend" in err, "a ready atticd must be stopped first: exit 4 (%d)" % rc)
        holder = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        holder.connect(sock)
        holder.send(('{"key":"%s","op":"alloc","size":%d}' % (KEY, 2 << 20)).encode())
        msg, anc, _, _ = holder.recvmsg(4096, socket.CMSG_SPACE(4))
        check(json.loads(msg)["ok"] is True and len(anc) == 1, "a client borrows 2 MiB")
        os.unlink(os.path.join(n.lib, "artifacts", "999.0.1", "VALIDATED"))
        os.kill(n.prod.pid, signal.SIGHUP)
        time.sleep(0.5)
        check(info(sock)["ok"] is False, "VALIDATED removed, then SIGHUP: no new lends")
        rc, err = n.validate()
        check(rc == 4 and "still lends" in err, "a refusing atticd whose client still holds memory: exit 4 (%d)" % rc)
        holder.close()
    finally:
        n.done()

    n = Node(bindir)
    try:
        keyf = os.path.join(n.dir, "key")  # as the unit runs it: --key-file /etc/spark-attic/key, a random key
        with open(keyf, "w") as f:
            f.write("attic_0123456789abcdef01234567\n")
        n.start_prod("--key-file", keyf)
        rc, err = n.validate()
        check(rc == 4 and "gave no status" in err and n.validated() is None,
              "an atticd whose status it cannot read (not its key): in use, exit 4, nothing enabled (%d)" % rc)
        rc, err = n.validate(ATTIC_KEY_FILE=keyf)
        check(rc == 0, "with its key file, a refusing atticd that lends nothing is idle: validated (%d) %s" % (rc, err[-300:]))
        time.sleep(0.5)
        rc, err = n.validate(ATTIC_KEY_FILE=keyf)
        check(rc == 4 and "is ready to lend" in err, "then ready, read with its key file: exit 4 (%d)" % rc)
        rc, err = n.validate(ATTIC_KEY="attic_0123456789abcdef01234567")
        check(rc == 4 and "is ready to lend" in err, "ATTIC_KEY works as well: exit 4 (%d)" % rc)
    finally:
        n.done()

    n = Node(bindir)
    try:
        rc, err = n.validate(ATTIC_FAKECUDA_CORRUPT="1")
        cands = os.listdir(os.path.join(n.lib, "candidates", "999.0.1"))
        check(rc == 1 and n.validated() is None, "a check that fails enables nothing (%d)" % rc)
        check(any(f.startswith("FAILED-") for f in cands), "and its report is kept as FAILED-*: %r" % cands)
        rc, err = n.validate(modes="heap list", ATTIC_FAKE_HELD=str(2 << 20))
        v = n.validated() or ""
        check(rc == 0 and "modes=heap\n" in v, "RM holding part of it: heap passes, list does not (%d) %r" % (rc, v))
        n.driver("999.0.2")
        rc, err = n.validate()
        check(rc == 3 and "runs driver 999.0.2" in err, "a node on another driver: exit 3 (%d)" % rc)
        n.driver("999.0.1")
        other = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        other.bind(os.path.join(n.dir, "other.sock"))
        other.listen(1)
        rc, err = n.validate()
        check(rc == 4 and "another carveout lender" in err, "another lender listening: exit 4 (%d)" % rc)
        other.close()
    finally:
        n.done()
    print("test_validate: %d failures" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

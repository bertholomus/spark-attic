#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_server.py: the no-GPU host test. Runs the real atticd against the fake RM shim (fakeshim.so) and talks to it
the way our TensorFold client does (one JSON request a SOCK_SEQPACKET message, the slice's
descriptor beside the alloc reply). Each case starts its own atticd in a fresh directory.

    python3 -I tests/test_server.py --bin build/host
"""
import argparse
import array
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

SLICE = 2 << 20
MIB = 1 << 20
KEY = "spark-attic-1"
DRIVER = "999.0.1"
FAILS = []


def check(cond, what):
    if not cond:
        FAILS.append(what)
        print("  FAIL", what)
    return cond


def info_req(key=KEY):
    return '{"key":"%s","op":"info"}' % key  # byte for byte what our TensorFold client sends


def alloc_req(size, key=KEY):
    return '{"key":"%s","op":"alloc","size":%d}' % (key, size)


def connect(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.settimeout(10)
    s.connect(path)
    return s


def call(s, body):
    """One request, one reply: (reply dict or None when the daemon closed, [fds])."""
    try:
        s.send(body.encode() if isinstance(body, str) else body)
        msg, anc, _flags, _addr = s.recvmsg(4096, socket.CMSG_SPACE(4 * 4))
    except (ConnectionResetError, BrokenPipeError):
        return None, []
    fds = []
    for level, kind, data in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds += list(array.array("i", data[: len(data) - len(data) % 4]))
    if not msg:
        return None, fds
    check(len(msg) <= 4096, "a reply fits our client's 4096 bytes")
    return json.loads(msg), fds


def ask(path, body):
    s = connect(path)
    try:
        r, fds = call(s, body)
        for fd in fds:
            os.close(fd)
        return r
    finally:
        s.close()


class Rig:
    def __init__(self, bindir, name):
        self.bin = bindir
        self.dir = tempfile.mkdtemp(prefix="attic-%s-" % name)
        os.chmod(self.dir, 0o755)
        self.sock = os.path.join(self.dir, "attic.sock")
        self.art = os.path.join(self.dir, "art")
        self.version = os.path.join(self.dir, "version")
        self.iomem = os.path.join(self.dir, "iomem")
        self.ledger = os.path.join(self.dir, "ledger")
        self.log = os.path.join(self.dir, "atticd.log")
        self.proc = None
        self.set_driver(DRIVER)
        self.artifact(DRIVER, modes="heap,list")
        with open(self.iomem, "w") as f:  # the fake carveout 0x280000000+2 GiB is reserved, not System RAM
            f.write("80000000-27fffffff : System RAM\n280000000-2ffffffff : reserved\n300000000-1fffffffff : System RAM\n")

    def set_driver(self, version):
        with open(self.version, "w") as f:
            f.write("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  %s  Release Build  (test)\n" % version)

    def artifact(self, driver, modes="heap", sha=None, validated=True, vdriver=None):
        d = os.path.join(self.art, driver)
        os.makedirs(d, exist_ok=True)
        os.chmod(self.art, 0o755)
        os.chmod(d, 0o755)
        shim = os.path.join(d, "rmshim.so")
        shutil.copyfile(os.path.join(self.bin, "fakeshim.so"), shim)
        os.chmod(shim, 0o644)
        digest = hashlib.sha256(open(shim, "rb").read()).hexdigest()
        v = os.path.join(d, "VALIDATED")
        if validated:
            with open(v, "w") as f:
                f.write("driver=%s\nshim_sha256=%s\nmodes=%s\nchecked=2026-11-02T20:00:00Z\nhost=test\n"
                        % (vdriver or driver, sha or digest, modes))
            os.chmod(v, 0o644)
        elif os.path.exists(v):
            os.unlink(v)
        return d

    def args(self, *extra):
        return [os.path.join(self.bin, "atticd"), "--socket", self.sock, "--key", KEY, "--artifacts", self.art,
                "--version-file", self.version, "--iomem", self.iomem, "--state", self.ledger, *extra]

    def start(self, *extra, env=None):
        e = dict(os.environ)
        e.update(env or {})
        self.proc = subprocess.Popen(self.args(*extra), stderr=open(self.log, "a"), env=e)
        for _ in range(100):
            try:
                connect(self.sock).close()
                return self
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError("atticd exited %d: %s" % (self.proc.returncode, open(self.log).read()))
                time.sleep(0.05)
        raise RuntimeError("atticd did not listen")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(10)
        self.proc = None

    def done(self):
        self.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def holder(path, size, close_first=False, let_go=False):
    """A separate process that takes `size` bytes and keeps the descriptor (as a CUDA import would): its pid and the
    line it reports ("ok" when the reply and the descriptor's size were right). close_first: it closes its socket and
    keeps the descriptor; let_go: it closes the descriptor, then its socket (a client done with the memory), and lives
    on."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            s = connect(path)
            rep, fds = call(s, alloc_req(size))
            good = rep and rep.get("ok") is True and rep.get("size") == size and len(fds) == 1
            good = good and os.fstat(fds[0]).st_size == size
            if let_go:
                os.close(fds[0])
            if close_first or let_go:
                s.close()
            os.write(w, (b"ok\n" if good else ("bad %r %r\n" % (rep, fds)).encode()))
            time.sleep(3600)
        finally:
            os._exit(0)
    os.close(w)
    line = os.read(r, 4096).decode().strip()
    os.close(r)
    return pid, line


def kill(pid):
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)


def wait_free(path, want, seconds=8.0):
    end = time.time() + seconds
    last = None
    while time.time() < end:
        last = ask(path, info_req())
        if last and last.get("free") == want:
            return last
        time.sleep(0.2)
    return last


# ---------------------------------------------------------------------------------------------------------- cases

def case_heap_lend_and_reclaim(rig):
    rig.start()
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "ready" and st["mode"] == "heap" and st["driver"] == DRIVER, "status: ready, heap, driver")
    inf = ask(rig.sock, info_req())
    want = 2046 * MIB  # 2 GiB carveout, 2 MiB of it RM's own: what GB10 shows
    check(inf == {"ok": True, "size": want, "free": want, "largest": want}, "info: 2,046 MiB free in one run: %r" % inf)
    pid, line = holder(rig.sock, want)
    check(line == "ok", "alloc: the largest run, one descriptor of that size: %s" % line)
    inf = ask(rig.sock, info_req())
    check(inf["free"] == 0 and inf["largest"] == 0, "while held, nothing free: %r" % inf)
    r = ask(rig.sock, alloc_req(SLICE))
    check(r["ok"] is False and "free" in r["error"], "a second client cannot have it: %r" % r)
    kill(pid)
    inf = wait_free(rig.sock, want)
    check(inf and inf["free"] == want, "the holder killed: all of it free again: %r" % inf)


def case_heap_drain(rig):
    rig.start()
    want = 2046 * MIB
    pid, line = holder(rig.sock, 1024 * MIB, close_first=True)
    check(line == "ok", "drain holder got its slice")
    time.sleep(0.5)
    inf = ask(rig.sock, info_req())
    check(inf["free"] == want - 1024 * MIB, "socket closed, process alive: not lent again: %r" % inf)
    kill(pid)
    inf = wait_free(rig.sock, want)
    check(inf and inf["free"] == want, "process gone: free again: %r" % inf)
    # a client that lets go before it closes its socket gets the memory counted free at once, while it lives
    pid, line = holder(rig.sock, 1024 * MIB, let_go=True)
    check(line == "ok", "a second holder got its slice")
    inf = wait_free(rig.sock, want, seconds=1.0)
    check(inf and inf["free"] == want, "descriptor closed, then socket: free again in a second, process alive: %r" % inf)
    r = ask(rig.sock, alloc_req(want))
    check(r["ok"] is True, "and all of it can be lent again: %r" % r)
    kill(pid)


def case_requests(rig):
    rig.start("--max-slices", "2")
    s = connect(rig.sock)
    r, fds = call(s, '{"key":"%s","op":"alloc","size":3145728}' % KEY)
    check(r["ok"] is False and not fds, "3 MiB is not whole slices")
    r, fds = call(s, '{"key":"%s","op":"alloc"}' % KEY)
    check(r["ok"] is False and not fds, "alloc needs a size")
    r, fds = call(s, '{"key":"%s","op":"what"}' % KEY)
    check(r["ok"] is False, "unknown op refused")
    held = []
    for i in range(2):
        r, fds = call(s, alloc_req(SLICE))
        check(r["ok"] is True and len(fds) == 1, "small alloc %d" % i)
        held += fds
    r, fds = call(s, alloc_req(SLICE))
    check(r["ok"] is False and not fds, "--max-slices 2 holds")
    # a descriptor sent to atticd is dropped, and it still answers
    a, b = socket.socketpair()
    s.sendmsg([info_req().encode()], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [a.fileno()]))])
    msg = s.recv(4096)
    check(json.loads(msg)["ok"] is True, "a request carrying a descriptor is still served")
    a.close()
    b.close()
    r, fds = call(s, "{" + " " * 2000 + "}")
    check(r["ok"] is False and "longer" in r["error"], "a request over 1024 bytes is refused: %r" % r)
    for fd in held:
        os.close(fd)
    s.close()
    # three bad requests in a row and atticd hangs up
    s = connect(rig.sock)
    r1, _ = call(s, info_req("wrong"))
    r2, _ = call(s, "not json")
    check(r1["ok"] is False and r1["error"] == "wrong key" and r2["ok"] is False, "a wrong key and junk are refused")
    r3, _ = call(s, info_req("wrong"))
    r4, _ = call(s, info_req())
    check(r4 is None, "after three bad requests the connection is closed")
    s.close()
    check(ask(rig.sock, info_req())["ok"] is True, "and atticd still serves others")


def case_refusals(rig):
    def refusing(expect, *extra, env=None):
        rig.start(*extra, env=env)
        st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
        inf = ask(rig.sock, info_req())
        good = st["state"] == "refusing" and expect in st["why"] and inf["ok"] is False and expect in inf["error"]
        check(good, "refuses (%s): %r" % (expect, st.get("why")))
        r = ask(rig.sock, alloc_req(SLICE))
        check(r["ok"] is False, "and lends nothing")
        rig.stop()

    os.unlink(os.path.join(rig.art, DRIVER, "VALIDATED"))
    refusing("no VALIDATED")
    rig.artifact(DRIVER, sha="0" * 64)
    refusing("does not match its VALIDATED digest")
    rig.artifact(DRIVER, modes="heap")
    refusing("has not passed mode list", "--mode", "list")
    rig.artifact(DRIVER, modes="heap,list")
    rig.set_driver("999.0.2")  # a driver no artifact was built for
    refusing("no VALIDATED")
    rig.artifact("999.0.2")  # a shim built for 999.0.1 filed under 999.0.2
    refusing("was built for driver 999.0.1")
    rig.set_driver(DRIVER)
    os.chmod(os.path.join(rig.art, DRIVER), 0o775)
    refusing("writable by group or others")
    os.chmod(os.path.join(rig.art, DRIVER), 0o755)
    refusing("not the carveout", env={"ATTIC_FAKE_RAM": "1"})
    refusing("fake: open refused", env={"ATTIC_FAKE_FAIL": "open"})
    refusing("carveout region 2 (2 MiB) of a type the shim for driver 999.0.1 does not know",
             env={"ATTIC_FAKE_EXTRA": "uefi,other"})
    refusing("RM uses part of the display carveout", "--mode", "list")  # RM holds 2 MiB: list mode stands down
    with open(rig.iomem, "w") as f:
        f.write("80000000-2fffffffff : System RAM\n")
    refusing("System RAM", "--mode", "list", env={"ATTIC_FAKE_HELD": "0"})
    with open(rig.iomem, "w") as f:  # what a non-root reader sees
        f.write("00000000-00000000 : System RAM\n00000000-00000000 : reserved\n")
    refusing("no addresses", "--mode", "list", env={"ATTIC_FAKE_HELD": "0"})
    os.unlink(rig.version)
    refusing("no NVIDIA driver loaded")


def case_export_failure(rig):
    rig.start(env={"ATTIC_FAKE_FAIL": "export"})
    r = ask(rig.sock, alloc_req(64 * MIB))
    check(r["ok"] is False and "export" in r["error"], "a failed export is refused: %r" % r)
    inf = ask(rig.sock, info_req())
    check(inf["free"] == 2046 * MIB, "and nothing stays taken: %r" % inf)


def case_list_mode(rig):
    env = {"ATTIC_FAKE_HELD": "0"}
    rig.start("--mode", "list", env=env)
    whole = 2048 * MIB
    inf = ask(rig.sock, info_req())
    check(inf["ok"] is True and inf["size"] == whole and inf["free"] == whole, "list mode: 2,048 MiB: %r" % inf)
    check("base" not in inf or os.geteuid() == 0, "the physical base only goes to root")
    pid, line = holder(rig.sock, 1024 * MIB)
    check(line == "ok", "list alloc")
    text = open(rig.ledger).read()
    check((" %d " % pid) in text, "the ledger names the holder: %r" % text)
    rig.stop()  # atticd restarts while the holder lives
    rig.start("--mode", "list", env=env)
    inf = ask(rig.sock, info_req())
    check(inf["free"] == whole - 1024 * MIB, "after a restart the holder's pages stay out: %r" % inf)
    kill(pid)
    inf = wait_free(rig.sock, whole)
    check(inf and inf["free"] == whole, "holder gone: free again: %r" % inf)


def case_sighup(rig):
    rig.start()
    v = os.path.join(rig.art, DRIVER, "VALIDATED")
    os.rename(v, v + ".off")
    rig.proc.send_signal(signal.SIGHUP)
    time.sleep(0.3)
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and ask(rig.sock, info_req())["ok"] is False, "SIGHUP without VALIDATED stops lending")
    os.rename(v + ".off", v)
    rig.proc.send_signal(signal.SIGHUP)
    time.sleep(0.3)
    check(ask(rig.sock, info_req())["ok"] is True, "SIGHUP with VALIDATED back lends again")


def case_candidate_keep_probe(rig):
    cand = rig.artifact(DRIVER, validated=False)
    rig.start("--candidate", cand, "--keep-mib", "100")
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "ready" and st["candidate"] is True, "a candidate runs unvalidated on its own socket")
    check(ask(rig.sock, info_req())["size"] == 1946 * MIB, "--keep-mib 100 keeps 100 MiB back")
    rig.stop()
    p = subprocess.run([os.path.join(rig.bin, "atticd"), "--candidate", cand], capture_output=True)
    check(p.returncode == 2, "a candidate refuses the default socket")
    p = subprocess.run(rig.args("--candidate", cand, "--probe"), capture_output=True, text=True)
    check(p.returncode == 0 and json.loads(p.stdout)["state"] == "ready", "--probe: ready, exit 0")
    p = subprocess.run(rig.args("--probe"), capture_output=True, text=True)
    check(p.returncode == 3 and json.loads(p.stdout)["state"] == "refusing", "--probe: refusing, exit 3")


def case_socket_path(rig):
    with open(rig.sock, "w") as f:
        f.write("not a socket")
    p = subprocess.run(rig.args(), capture_output=True, timeout=10)
    check(p.returncode == 1 and open(rig.sock).read() == "not a socket", "a file at the socket path is left alone")
    os.unlink(rig.sock)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    stale.bind(rig.sock)
    stale.close()  # a socket file no one listens on: atticd replaces it
    rig.start()
    check(ask(rig.sock, info_req())["ok"] is True, "a stale socket is replaced")
    p = subprocess.run(rig.args(), capture_output=True, timeout=10)
    check(p.returncode == 1, "a second atticd on a live socket exits")
    check(oct(os.stat(rig.sock).st_mode & 0o777) == "0o660", "socket mode 0660")


# The status file's schema 1, as STATUS-FILE.md documents it: fields are only ever added at the end.
STATUS_SCHEMA = (("name", str), ("version", str), ("state", str), ("why", str), ("driver", str), ("mode", str),
                 ("validated", bool), ("validated_at", str), ("size_mib", int), ("free_mib", int),
                 ("largest_mib", int), ("lent_mib", int), ("draining_mib", int), ("clients", int), ("every_s", int),
                 ("pid", int), ("updated", int))


def status_file(path, until=lambda st: True, seconds=8.0):
    """The keyless status file as JSON once `until` holds for it (None when it never does, or there is none)."""
    end = time.time() + seconds
    while True:
        try:
            with open(path) as f:
                st = json.load(f)
            if until(st):
                return st
        except (OSError, ValueError):
            pass
        if time.time() > end:
            return None
        time.sleep(0.1)


def case_status_file(rig):
    path = os.path.join(rig.dir, "status.json")
    old = os.umask(0o077)
    try:
        rig.start("--status-every", "2")
    finally:
        os.umask(old)
    st = status_file(path)
    want = 2046
    check(st is not None and {k: st.get(k) for k in ("name", "state", "why", "driver", "mode", "validated",
                                                     "validated_at", "size_mib", "free_mib", "largest_mib", "lent_mib",
                                                     "draining_mib", "every_s")} == {
        "name": "spark-attic", "state": "ready", "why": "", "driver": DRIVER, "mode": "heap", "validated": True,
        "validated_at": "2026-11-02T20:00:00Z", "size_mib": want, "free_mib": want, "largest_mib": want,
        "lent_mib": 0, "draining_mib": 0, "every_s": 2} and st["pid"] == rig.proc.pid
          and abs(st["updated"] - time.time()) < 5, "status.json beside the socket: ready, validated, sizes in MiB: %r"
          % st)
    check(list(st) == [k for k, _ in STATUS_SCHEMA] and all(type(st[k]) is t for k, t in STATUS_SCHEMA),
          "schema 1 (STATUS-FILE.md, read by dashboards): exactly these fields, in this order and of these types: %r"
          % st)
    raw = open(path).read()
    check(oct(os.stat(path).st_mode & 0o777) == "0o644" and KEY not in raw and "key" not in raw and "base" not in raw,
          "0644 under umask 077, no key and no address in it")
    pid, line = holder(rig.sock, 1024 * MIB)
    st = status_file(path, lambda s: s["lent_mib"] == 1024)
    check(line == "ok" and st and st["free_mib"] == want - 1024 and st["clients"] == 1, "a client holds 1 GiB: %r" % st)
    kill(pid)
    st = status_file(path, lambda s: s["lent_mib"] == 0 and s["draining_mib"] == 0 and s["free_mib"] == want)
    check(st is not None, "given back, the file says so")
    first = status_file(path)["updated"]
    check(status_file(path, lambda s: s["updated"] > first, 7.0) is not None,
          "rewritten every --status-every seconds while nothing changes, so a stale file means atticd is gone")
    p = subprocess.run(rig.args("--status-every", "1"), capture_output=True, timeout=10)
    check(p.returncode == 2, "--status-every 1 is refused: atticd's clock ticks every 2 s")
    p = subprocess.run(rig.args(), capture_output=True, timeout=10)
    check(p.returncode == 1 and status_file(path)["pid"] == rig.proc.pid, "a second atticd on the live socket leaves "
          "the first one's file alone")
    v = os.path.join(rig.art, DRIVER, "VALIDATED")
    os.rename(v, v + ".off")
    rig.proc.send_signal(signal.SIGHUP)
    st = status_file(path, lambda s: s["state"] == "refusing")
    check(st and st["validated"] is False and st["validated_at"] == "" and "VALIDATED" in st["why"],
          "SIGHUP without VALIDATED: refusing, not validated, why: %r" % st)
    os.rename(v + ".off", v)
    rig.stop()
    check(not os.path.exists(path) and not os.path.exists(path + ".new"), "removed when atticd stops")
    rig.set_driver("999.0.2")
    rig.start()
    st = status_file(path)
    check(st and st["state"] == "refusing" and st["validated"] is False and st["driver"] == "999.0.2"
          and st["size_mib"] == 0 and st["why"] and all(type(st[k]) is t for k, t in STATUS_SCHEMA),
          "a driver with no artifact: refusing, with why, the same schema: %r" % st)
    rig.stop()
    rig.set_driver(DRIVER)
    rig.start("--no-status-file")
    check(ask(rig.sock, info_req())["ok"] is True and not os.path.exists(path), "--no-status-file: none")
    rig.stop()
    rig.start("--candidate", os.path.join(rig.art, DRIVER))
    check(ask(rig.sock, info_req())["ok"] is True and not os.path.exists(path),
          "a candidate (attic-validate.sh runs one beside the real socket) writes none")
    rig.stop()
    rig.start("--status-file", os.path.join(rig.dir, "no-such-dir", "status.json"))
    time.sleep(0.5)
    check(ask(rig.sock, info_req())["ok"] is True and open(rig.log).read().count("cannot write the status file") == 1,
          "an unwritable status file: one warning, lending goes on")


def run_check(rig, mode, extra_env=None):
    """attic-check against rig's atticd with the fake libcuda (no GPU) and a steady MemFree: (rc, report, process)."""
    meminfo = os.path.join(rig.dir, "meminfo")
    with open(meminfo, "w") as f:
        f.write("MemTotal:       129798784 kB\nMemFree:        64000000 kB\n")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(rig.bin, "fakecuda"))
    env.update(extra_env or {})
    report = os.path.join(rig.dir, "report-%s.json" % mode)
    if os.path.exists(report):
        os.unlink(report)
    p = subprocess.run([os.path.join(rig.bin, "attic-check"), "--socket", rig.sock, "--key", KEY, "--mode", mode,
                        "--report", report, "--meminfo", meminfo, "--iomem", rig.iomem],
                       env=env, capture_output=True, text=True, timeout=180)
    rep = json.load(open(report)) if os.path.exists(report) else None
    return p.returncode, rep, p


def case_attic_check(rig):
    small = {"ATTIC_FAKE_REGION": "0x280000000:0x10000000"}  # 256 MiB keeps the host run quick
    cand = rig.artifact(DRIVER, validated=False)
    rig.start("--candidate", cand, env=small)
    rc, rep, p = run_check(rig, "heap")
    check(rc == 0 and rep and rep["pass"] is True, "attic-check passes heap on the fake: %s %s" % (p.stdout[-1500:], p.stderr[-500:]))
    ph = rep["phases"] if rep else {}
    names = ("status", "full", "double", "isolation", "reclaim", "drain", "refusal")
    check(all(ph.get(k, {}).get("pass") is True for k in names), "every phase passed: %r" % sorted(ph))
    check(ph.get("full", {}).get("size") == 254 * MIB and ph.get("full", {}).get("mismatches") == 0,
          "full: 254 MiB, every word twice, no mismatch")
    q = subprocess.run([os.path.join(rig.bin, "attic-check"), "--status", "--socket", rig.sock, "--key", KEY],
                       capture_output=True, text=True)
    check(q.returncode == 0 and json.loads(q.stdout)["state"] == "ready", "attic-check --status prints the status")
    rc, rep, p = run_check(rig, "list")
    check(rc == 1 and rep and rep["phases"]["status"]["pass"] is False, "a check for list against a heap daemon fails")
    rig.stop()
    rig.start("--candidate", cand, env=small)
    rc, rep, p = run_check(rig, "heap", {"ATTIC_FAKECUDA_CORRUPT": "1"})
    check(rc == 1 and rep and rep["pass"] is False and rep["phases"]["full"]["pass"] is False,
          "a slice that reads back wrong fails the check")
    rig.stop()
    rig.start("--candidate", cand, env=small)
    rc, rep, p = run_check(rig, "heap", {"ATTIC_FAKECUDA_NOIMPORT": "1"})
    check(rc == 1 and rep and rep["phases"]["full"]["pass"] is False and "cuMemImportFromShareableHandle" in
          rep["phases"]["full"].get("why", ""), "a slice CUDA will not import fails the check")
    rig.stop()
    rig.start("--candidate", cand, "--mode", "list", env=dict(small, ATTIC_FAKE_HELD="0"))
    rc, rep, p = run_check(rig, "list")
    check(rc == 0 and rep and rep["pass"] is True, "attic-check passes list on the fake: %s" % p.stdout[-1500:])


def scripted_status(path, body):
    """A daemon that answers one status request with body, then stops listening (attic-check's later phases find no
    one): for a reply our atticd never gives."""
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    srv.bind(path)
    srv.listen(1)

    def run():
        c, _ = srv.accept()
        srv.close()
        with c:
            if c.recv(4096):
                c.send(body)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def case_region_kinds(rig):
    """RM lists other carveout regions beside the display one (GB10: uefi). A type the shim knows is lent beside; any
    other is refused, since NVIDIA renumbers that enum between drivers (carveout_kind.h), and attic-check fails a
    daemon that reports one anyway."""
    small = {"ATTIC_FAKE_REGION": "0x280000000:0x10000000"}
    rig.start(env=dict(small, ATTIC_FAKE_EXTRA="reserved,uefi"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    kinds = [r["kind"] for r in st.get("regions", [])]
    check(st["state"] == "ready" and kinds == ["display", "display-reserved", "uefi"],
          "display-reserved and uefi beside the display region: ready (%r)" % kinds)
    check(ask(rig.sock, info_req())["size"] == 254 * MIB, "only the display region is lent")
    rc, rep, p = run_check(rig, "heap")
    check(rc == 0 and rep and rep["pass"] is True, "attic-check passes it: %s" % p.stdout[-1500:])
    rig.stop()
    rig.start(env=dict(small, ATTIC_FAKE_EXTRA="other"))
    rc, rep, p = run_check(rig, "heap")
    check(rc == 1 and rep and rep["phases"]["status"]["pass"] is False, "attic-check fails the daemon that refuses it")
    rig.stop()
    for kind, good in (("uefi", True), ("other", False)):  # a daemon that says ready beside such a region
        body = ('{"ok":true,"name":"spark-attic","version":"scripted","state":"ready","why":"","driver":"%s",'
                '"mode":"heap","size":%d,"free":%d,"largest":%d,"lent":0,"draining":0,"clients":0,"regions":['
                '{"kind":"display","size":%d},{"kind":"%s","size":%d}]}'
                % (DRIVER, 254 * MIB, 254 * MIB, 254 * MIB, 256 * MIB, kind, SLICE))
        t = scripted_status(rig.sock, body.encode())
        rc, rep, p = run_check(rig, "heap")
        t.join(5)
        stp = rep["phases"]["status"] if rep else {}
        check(stp.get("pass") is good and ("does not know" in stp.get("regions_check", "")) is not good,
              "a ready daemon with a%s %s region: the status phase %s (%r)"
              % ("n" if kind[0] in "aeiou" else "", kind, "passes" if good else "fails", stp.get("regions_check")))


CASES = [case_heap_lend_and_reclaim, case_heap_drain, case_requests, case_refusals, case_export_failure, case_list_mode,
         case_sighup, case_candidate_keep_probe, case_socket_path, case_status_file, case_attic_check,
         case_region_kinds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/host")
    a = ap.parse_args()
    bindir = os.path.abspath(a.bin)
    os.umask(0o022)
    for case in CASES:
        rig = Rig(bindir, case.__name__[5:])
        before = len(FAILS)
        try:
            case(rig)
        except Exception as e:  # a crash is a failure, with atticd's log
            FAILS.append("%s: %r" % (case.__name__, e))
            print("  FAIL %s raised %r" % (case.__name__, e))
        if len(FAILS) > before:
            print("  atticd log:\n" + open(rig.log).read() if os.path.exists(rig.log) else "")
        print("%s %s" % ("ok  " if len(FAILS) == before else "FAIL", case.__name__))
        rig.done()
    print("test_server: %d cases, %d failures" % (len(CASES), len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

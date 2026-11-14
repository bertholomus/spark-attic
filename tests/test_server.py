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
    # A 2 GiB first allocation touches 1023 fresh 2 MiB memfds, and the fake shim dirties every one
    # (a slice arrives with its previous holder's bytes in it): the first reply can take longer than
    # a busy second. The client waits a minute before it calls a reply lost.
    s.settimeout(60)
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
        self.meminfo = os.path.join(self.dir, "meminfo")
        self.ledger = os.path.join(self.dir, "ledger")
        self.log = os.path.join(self.dir, "atticd.log")
        self.proc = None
        self.set_driver(DRIVER)
        self.artifact(DRIVER, modes="heap,list")
        with open(self.iomem, "w") as f:  # the fake carveout 0x280000000+2 GiB is reserved, not System RAM
            f.write("80000000-27fffffff : System RAM\n280000000-2ffffffff : reserved\n300000000-1fffffffff : System RAM\n")
        with open(self.meminfo, "w") as f:  # a steady MemFree, so the heap probe never reads the host live
            f.write("MemTotal:       129798784 kB\nMemFree:        64000000 kB\n")

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
                "--version-file", self.version, "--iomem", self.iomem, "--meminfo", self.meminfo,
                "--state", self.ledger, *extra]

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
            # Do not carry the parent's inheritable fds into the holder's hour of life: a long-lived child
            # that inherits the heavy-slot lock (or any fd the parent holds) pins it for an hour.
            import os as _os
            for fd in range(3, 256):
                if fd == w:
                    continue
                try:
                    _os.close(fd)
                except OSError:
                    pass
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
        except BaseException as e:  # a silent child death is un-diagnosable: the failure names the error
            try:
                os.write(w, ("%s: %s\n" % (type(e).__name__, e)).encode())
            except OSError:
                pass
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
    cand = rig.artifact(DRIVER, validated=False)  # list mode only runs as a candidate (S4); a candidate runs
    refusing("RM uses part of the display carveout", "--mode", "list", "--candidate", cand)  # unvalidated, so the
    rig.artifact(DRIVER, modes="heap,list")  # modes gate is the node check's, not the daemon's
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
    refusing("RM uses part of the display carveout", "--mode", "list", "--candidate", cand)
    # RM holds 2 MiB: list mode stands down
    with open(rig.iomem, "w") as f:
        f.write("80000000-2fffffffff : System RAM\n")
    refusing("System RAM", "--mode", "list", "--candidate", cand, env={"ATTIC_FAKE_HELD": "0"})
    with open(rig.iomem, "w") as f:  # what a non-root reader sees
        f.write("00000000-00000000 : System RAM\n00000000-00000000 : reserved\n")
    refusing("no addresses", "--mode", "list", "--candidate", cand, env={"ATTIC_FAKE_HELD": "0"})
    os.unlink(rig.version)
    refusing("no NVIDIA driver loaded")

    # no key: atticd refuses to start (no compiled-in default key, S6)
    rig.stop()
    nokey = rig.args()
    i = nokey.index("--key")
    del nokey[i:i + 2]  # drop the --key pair the rig always adds
    p = subprocess.run(nokey, capture_output=True, timeout=10)
    check(p.returncode == 2 and "no key" in p.stderr.decode(), "no --key and no --key-file: it does not start: %r"
          % p.stderr.decode().strip().splitlines()[-1:])
    kf = os.path.join(rig.dir, "empty-key")
    open(kf, "w").write("\n")
    p = subprocess.run(rig.args("--key-file", kf), capture_output=True, timeout=10)
    check(p.returncode == 2 and "no key" in p.stderr.decode(), "an empty key file: the same")
    rig.set_driver(DRIVER)  # the case ends with the version file removed; restore it
    rig.start()
    check(ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)["state"] == "ready",
          "an explicit --key starts as before (the tests always pass one)")
    rig.stop()
    p = subprocess.run(rig.args("--keep-mib", "3"), capture_output=True, timeout=10)
    check(p.returncode == 2 and "not a whole number of 2 MiB slices" in p.stderr.decode(),
          "--keep-mib 3: refused with why (M2): %r" % p.stderr.decode().strip().splitlines()[-1:])
    p = subprocess.run(rig.args("--keep-mib", "abc"), capture_output=True, timeout=10)
    check(p.returncode == 2, "--keep-mib abc: refused")

    # N5: a persistently failing RM client is reopened, not served refusals forever
    failfile = os.path.join(rig.dir, "shim-fail")
    open(failfile, "w").write("")
    rig.start(env={"ATTIC_FAKE_FAIL_FILE": failfile})
    check(ask(rig.sock, info_req())["ok"] is True, "healthy shim: ready")
    pid, line = holder(rig.sock, 8 * MIB)
    check(line == "ok", "an alloc works")
    kill(pid)  # the holder is gone before the trouble: nothing is out when the strikes land
    check(wait_free(rig.sock, 2046 * MIB) is not None, "and the slice is fully back")
    open(failfile, "w").write("heap carveout")  # allocs fail, and the rebuilt client fails its bring-up too
    for i in range(3):
        r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
        for fd in fds:
            os.close(fd)
        check(r["ok"] is False, "shim failing: alloc %d refused" % (i + 1))
    st = status_file(os.path.join(rig.dir, "status.json"),
                     lambda s_: s_["state"] == "refusing" and "reopening" in s_["why"], 12.0)
    check(st is not None, "three failures in a row: the RM client is torn down and reopened (N5): %r"
          % (st or {}).get("why"))
    st = status_file(os.path.join(rig.dir, "status.json"),
                     lambda s_: s_["state"] == "refusing" and "carveout" in s_["why"], 12.0)
    check(st is not None, "the fresh client's bring-up fails while the shim fails: still refusing, why: %r"
          % (st or {}).get("why"))
    open(failfile, "w").write("")
    rig.proc.send_signal(signal.SIGHUP)  # the daemon says "next try in 300 s, or on SIGHUP": nudge it
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s_: s_["state"] == "ready", 12.0)
    check(st is not None, "the shim healthy again: a fresh RM client, ready")
    r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
    for fd in fds:
        os.close(fd)
    check(r["ok"] is True, "and it lends again")
    check("closing and reopening it" in open(rig.log).read(), "the rebuild is logged")
    rig.stop()


def case_export_failure(rig):
    rig.start(env={"ATTIC_FAKE_FAIL": "export"})
    r = ask(rig.sock, alloc_req(64 * MIB))
    check(r["ok"] is False and "export" in r["error"], "a failed export is refused: %r" % r)
    inf = ask(rig.sock, info_req())
    check(inf["free"] == 2046 * MIB, "and nothing stays taken: %r" % inf)


def case_list_mode(rig):
    env = {"ATTIC_FAKE_HELD": "0"}
    # not released (S4): a real list-mode daemon refuses; only a candidate runs list, for the node check
    rig.start("--mode", "list", env=env)
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and "mode list is not released" in st["why"],
          "list mode is not released: refusing, why: %r" % st["why"])
    rig.stop()
    cand = rig.artifact(DRIVER, validated=False)
    rig.start("--mode", "list", "--candidate", cand, env=env)
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "ready" and st["candidate"] is True,
          "as a candidate it still runs list (the node check needs it)")
    rig.stop()
    rig.start("--mode", "list", "--candidate", cand, env=env)
    whole = 2048 * MIB
    inf = ask(rig.sock, info_req())
    check(inf["ok"] is True and inf["size"] == whole and inf["free"] == whole, "list mode: 2,048 MiB: %r" % inf)
    check("base" not in inf or os.geteuid() == 0, "the physical base only goes to root")
    pid, line = holder(rig.sock, 1024 * MIB)
    check(line == "ok", "list alloc")
    text = open(rig.ledger).read()
    check((" %d " % pid) in text, "the ledger names the holder: %r" % text)
    check(oct(os.stat(rig.ledger).st_mode & 0o777) == "0o600",
          "the ledger is root-only (it holds the physical base addresses, N1): %r"
          % oct(os.stat(rig.ledger).st_mode & 0o777))
    rig.stop()  # atticd restarts while the holder lives
    rig.start("--mode", "list", "--candidate", cand, env=env)
    inf = ask(rig.sock, info_req())
    check(inf["free"] == whole - 1024 * MIB, "after a restart the holder's pages stay out: %r" % inf)
    kill(pid)
    inf = wait_free(rig.sock, whole)
    check(inf and inf["free"] == whole, "holder gone: free again: %r" % inf)
    rig.stop()
    rig.start("--mode", "list", env=env)
    check(ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)["state"] == "refusing",
          "and a real daemon still refuses list (the release decision, S4)")
    rig.stop()


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
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(rig.bin, "fakecuda"))
    env.update(extra_env or {})
    report = os.path.join(rig.dir, "report-%s.json" % mode)
    if os.path.exists(report):
        os.unlink(report)
    p = subprocess.run([os.path.join(rig.bin, "attic-check"), "--socket", rig.sock, "--key", KEY, "--mode", mode,
                        "--report", report, "--meminfo", rig.meminfo, "--iomem", rig.iomem],
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


def modeset_holders(rig, holders, fill=0):
    """A fake proc tree for --modeset-dir: {pid: ("argv0 arg", holds)} where holds says the pid has an fd
    open on the modeset device (a symlink whose target names it, as /proc/<pid>/fd does), or None for no
    directory. The node's real /proc works as-is; nvidia-persistenced holding it is the idle state. fill:
    entries with no fd dir, for the rest of a real /proc (kernel threads, non-pid files); they go after the
    pids so the holder lands past entry 256 of a budgeted pass. readdir order is not creation order on every
    filesystem, so when the fillers come back first the tree is rebuilt with them first; the caller checks
    where the holder landed."""
    d = os.path.join(rig.dir, "proc")

    def pids():
        for pid, (cmd, holds) in holders.items():
            os.makedirs(os.path.join(d, str(pid), "fd"), exist_ok=True)
            with open(os.path.join(d, str(pid), "cmdline"), "wb") as f:
                f.write(cmd.replace(" ", "\x00").encode())
            if holds:
                os.symlink("/dev/nvidia-modeset", os.path.join(d, str(pid), "fd", "3"))

    def fillers():
        for i in range(fill):
            os.makedirs(os.path.join(d, "%d" % (900000 + i)))  # a kernel thread: no fd dir, so no holder

    def build(fillers_first):
        shutil.rmtree(d, ignore_errors=True)
        if holders is None:
            return
        if fillers_first:
            fillers()
            pids()
        else:
            pids()
            fillers()

    build(False)
    if holders and fill:
        at = [i for i, name in enumerate(os.listdir(d)) if name in [str(p) for p in holders]]
        if at and min(at) < fill:  # this filesystem lists entries in creation order: swap the two halves
            build(True)
    return d


def pmon_file(rig, lines):
    """A canned nvidia-smi pmon -c 1 -s u output: a list of lines like "0 4321 G 12 3 0 0 0 0"."""
    p = os.path.join(rig.dir, "pmon")
    if lines is None:
        if os.path.exists(p):
            os.unlink(p)
        return p + "-missing"
    with open(p, "w") as f:
        f.write("gpu pid type sm mem enc dec jpg ofa\n")
        for line in lines:
            f.write(line + "\n")
    return p


DISPLAY_WHYS = {
    "modeset": "a display is in use (pid ",
    "pmon": "a display is in use (pid ",
    "drm": "a display is in use (card0-eDP-1)",
}


def case_display3(rig):
    """B3-R: a display in use is any of the three signals - a modeset holder other than
    nvidia-persistenced, a graphics (type G) process, a connected DRM connector. Stock DGX OS with
    nvidia-drm modeset=0 has no connector dirs at all, so the first two carry the verdict there; the
    connector check stays for KMS systems. A desktop started after atticd came up stops new lends at
    once (the borrow path re-checks, not just the tick). Fails on 675705a: only the connector existed."""
    # (a) persistenced alone is the idle state: lends
    rig.start("--modeset-dir", modeset_holders(rig, {100: ("nvidia-persistenced --keep", True)}),
              "--pmon-file", pmon_file(rig, None), "--drm-dir", os.path.join(rig.dir, "drm-none"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "ready", "only nvidia-persistenced holds modeset: the idle state, lends (%r)" % st["why"])
    pid, line = holder(rig.sock, 8 * MIB)
    check(line == "ok", "a client borrows")
    # a desktop starts while lending: Xorg opens modeset - new lends stop at once
    os.makedirs(os.path.join(rig.dir, "proc", "200", "fd"), exist_ok=True)
    os.symlink("/dev/nvidia-modeset", os.path.join(rig.dir, "proc", "200", "fd", "3"))
    with open(os.path.join(rig.dir, "proc", "200", "cmdline"), "wb") as f:
        f.write(b"/usr/lib/xorg/Xorg\x00:0\x00")
    st = status_file(os.path.join(rig.dir, "status.json"),
                     lambda s_: "a display is in use (pid 200" in s_["why"], 12.0)
    check(st is not None and st["state"] == "refusing" and st["lent_mib"] == 8,
          "a desktop started while lending: refusing, why names the holder, the slice stays: %r" % st)
    r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
    check(r is not None and r["ok"] is False and "display is in use" in r["error"],
          "a borrow asked after the desktop started is refused at once (no tick wait)")
    for fd in fds:
        os.close(fd)
    kill(pid)
    os.unlink(os.path.join(rig.dir, "proc", "200", "cmdline"))
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s_: s_["state"] == "ready", 12.0)
    check(st is not None, "the desktop is gone: lending again")
    rig.stop()

    # (a2) a non-persistenced holder whose cmdline does not name modeset still counts (the fd route):
    # the fake tree names holders directly - atticd itself holding the device would count.
    rig.start("--modeset-dir", modeset_holders(rig, {300: ("gnome-shell --display", True)}),
              "--pmon-file", pmon_file(rig, None), "--drm-dir", os.path.join(rig.dir, "drm-none"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and "pid 300" in st["why"], "gnome-shell holds modeset: refusing (%r)" % st["why"])
    rig.stop()

    # (b) a graphics (type G) process on the GPU
    rig.start("--modeset-dir", modeset_holders(rig, None), "--pmon-file",
              pmon_file(rig, ["0 4321 G 12 3 0 0 0 0", "0 500 C 90 10 0 0 0 0"]),
              "--drm-dir", os.path.join(rig.dir, "drm-none"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and "pid 4321" in st["why"] and "type G" in st["why"],
          "a type G process on the GPU: refusing, why names it (%r)" % st["why"])
    rig.stop()

    # (b2) a compute (type C) process is not a display: lends
    rig.start("--modeset-dir", modeset_holders(rig, None), "--pmon-file",
              pmon_file(rig, ["0 500 C 90 10 0 0 0 0"]),
              "--drm-dir", os.path.join(rig.dir, "drm-none"))
    check(ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)["state"] == "ready",
          "only compute processes: lends")
    rig.stop()

    # (c) the connector check still fires on KMS systems (the old case keeps working)
    rig.start("--modeset-dir", modeset_holders(rig, None), "--pmon-file", pmon_file(rig, None),
              "--drm-dir", drm(rig, {"card0-eDP-1": "connected"}))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and DISPLAY_WHYS["drm"] in st["why"],
          "a connected connector on a KMS system: refusing (%r)" % st["why"])
    rig.stop()

    # unreadable signals are not evidence: a missing pmon file, an empty modeset dir, no drm tree
    rig.start("--modeset-dir", modeset_holders(rig, None), "--pmon-file", pmon_file(rig, None),
              "--drm-dir", os.path.join(rig.dir, "drm-none"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "ready", "every probe unreadable: lends, no signal can be read")
    rig.stop()

    # --probe-display: one line per signal, read-only, no key, exit 1 when any fires
    p = subprocess.run(rig.args("--probe-display", "--modeset-dir",
                                modeset_holders(rig, {200: ("Xorg :0", True)}),
                                "--pmon-file", pmon_file(rig, ["0 4321 G 1 1 0 0 0 0"]),
                                "--drm-dir", drm(rig, {"card0-eDP-1": "connected"})),
                       capture_output=True, text=True)
    lines = p.stdout.splitlines()
    check(p.returncode == 1 and len(lines) == 4 and lines[0].startswith("IN-USE modeset-holder")
          and lines[1].startswith("IN-USE graphics-process") and lines[2].startswith("IN-USE drm-connector")
          and lines[3].startswith("IN-USE:"), "--probe-display: one line per signal, exit 1: %r" % lines)
    p = subprocess.run(rig.args("--probe-display", "--modeset-dir", modeset_holders(rig, None),
                                "--pmon-file", pmon_file(rig, None),
                                "--drm-dir", os.path.join(rig.dir, "drm-none")),
                       capture_output=True, text=True)
    check(p.returncode == 0 and all(l.startswith("idle") for l in p.stdout.splitlines()),
          "--probe-display: all idle, exit 0: %r" % p.stdout.splitlines())
    rig.stop()


def case_display4(rig):
    """B3-R-OSCILLATION: the display verdict is held across ticks, not re-derived from one instant. On a
    real GB10 desktop the samples are sparser than the 2 s tick (the 256-pid modeset pass finds the holder
    only on the ticks whose pass reaches it; pmon runs every 30 s), and at adcbc74 the ready flag flipped
    with them: 12 of 20 borrows succeeded while a desktop was up the whole time. Here the fake signal is
    intermittent on purpose - visible on one tick, invisible on the next while the desktop process is
    still alive. The flip halves are 1.2 s, shorter than the 2 s tick, so no two consecutive ticks can
    ever both read the same state: the readings strictly alternate, which is the shape the node showed.
    Every borrow in the window must be refused, and the log must not flap. The same hold is proven for a
    pmon-only signal (the holder gone from /proc, the G sample appearing and vanishing), the block lifts
    only after a streak of complete idle ticks, and a returning desktop blocks again. Fails on adcbc74."""
    pmon = pmon_file(rig, [])  # a readable fresh sample that reads idle; the flip rewrites it below
    tree = modeset_holders(rig, {200: ("/usr/lib/xorg/Xorg :0", True)})
    cmdline = os.path.join(tree, "200", "cmdline")
    rig.start("--modeset-dir", tree, "--pmon-file", pmon, "--drm-dir", os.path.join(rig.dir, "drm-none"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing", "a desktop holds modeset at bring-up: refusing (%r)" % st["why"])

    def borrow_window(seconds):
        ok = refused = other = 0
        end = time.time() + seconds
        while time.time() < end:
            r = ask(rig.sock, alloc_req(8 * MIB))
            if r and r.get("ok") is True:
                ok += 1
            elif r is not None and "display is in use" in r.get("error", ""):
                refused += 1
            else:
                other += 1
            time.sleep(0.3)
        return ok, refused, other

    def flip_modeset(stop):
        while not stop.is_set():
            with open(cmdline, "wb") as f:
                f.write(b"/usr/lib/xorg/Xorg\x00:0\x00")
            time.sleep(1.2)
            if stop.is_set():
                break
            os.unlink(cmdline)
            time.sleep(1.2)

    def flip_pmon(stop):
        while not stop.is_set():
            pmon_file(rig, ["0 4321 G 12 3 0 0 0 0"])
            time.sleep(1.2)
            if stop.is_set():
                break
            pmon_file(rig, [])  # a fresh sample that reads idle
            time.sleep(1.2)

    # (1) the modeset signal intermittent, the desktop process alive the whole time: the block must hold
    stop1 = threading.Event()
    t1 = threading.Thread(target=flip_modeset, args=(stop1,), daemon=True)
    t1.start()
    ok, refused, other = borrow_window(9.6)
    stop1.set()
    t1.join()
    check(ok == 0 and refused >= 20 and other == 0,
          "an intermittent modeset signal, desktop alive: all %d borrows refused, %d lent, %d odd"
          % (refused, ok, other))
    log = open(rig.log).read()
    check(log.count("lending stopped") == 0 and log.count("lending again") == 0,
          "the verdict holds without flapping the log (%d stopped, %d again lines)"
          % (log.count("lending stopped"), log.count("lending again")))

    # (2) the same hold for a pmon-only signal: the holder's process is gone from /proc (an empty, readable
    # tree - the real node's /proc always reads), the G sample appears and vanishes every 1.2 s
    stop2 = threading.Event()
    shutil.rmtree(tree, ignore_errors=True)
    os.makedirs(tree, exist_ok=True)
    t2 = threading.Thread(target=flip_pmon, args=(stop2,), daemon=True)
    t2.start()
    ok, refused, other = borrow_window(8.4)
    stop2.set()
    t2.join()
    check(ok == 0 and refused >= 15 and other == 0,
          "an intermittent pmon signal, no modeset holder: all %d borrows refused, %d lent, %d odd"
          % (refused, ok, other))
    log = open(rig.log).read()
    check(log.count("lending stopped") == 0 and log.count("lending again") == 0,
          "the pmon-only hold flaps nothing either (%d stopped, %d again)"
          % (log.count("lending stopped"), log.count("lending again")))

    # (3) the control: both signals read idle for good - the block lifts after the idle streak and a borrow
    # goes through (the latch is not a permanent refusal)
    if os.path.exists(cmdline):
        os.unlink(cmdline)
    pmon_file(rig, [])
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s_: s_["state"] == "ready", 14.0)
    check(st is not None, "signals idle for good: lending again after the idle streak (%r)" % st)
    r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
    check(r is not None and r["ok"] is True, "a borrow goes through once the display is gone: %r" % r)
    for fd in fds:
        os.close(fd)

    # (4) a desktop coming back blocks again, within a tick, from the lending state
    os.makedirs(os.path.join(tree, "200", "fd"), exist_ok=True)
    os.symlink("/dev/nvidia-modeset", os.path.join(tree, "200", "fd", "3"))
    with open(cmdline, "wb") as f:
        f.write(b"/usr/lib/xorg/Xorg\x00:0\x00")
    st = status_file(os.path.join(rig.dir, "status.json"),
                     lambda s_: "a display is in use" in s_["why"], 8.0)
    check(st is not None and st["state"] == "refusing", "the desktop is back: blocked again (%r)" % st)
    r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
    check(r is not None and r["ok"] is False and "display is in use" in r["error"],
          "a borrow is refused while the returned desktop is held: %r" % r)
    for fd in fds:
        os.close(fd)
    log = open(rig.log).read()
    check(log.count("lending stopped") == 1 and log.count("lending nothing") == 1,
          "one stopped line for the returning desktop, one bring-up refusal, nothing between (%d/%d)"
          % (log.count("lending stopped"), log.count("lending nothing")))
    rig.stop()


def case_display_beyond_budget(rig):
    """A display holder past the first 256 entries of a real-sized /proc: the one-shot readings - --probe-display
    and bring-up - walk the whole tree, not just the budget the tick keeps. Before the fix both read only the
    first 256 entries, miss the holder and call the display idle: --probe-display prints an idle modeset-holder
    and bring-up lends over a desktop that is up (pmon reads idle, the holder is not a type G process)."""
    tree = modeset_holders(rig, {300: ("/usr/lib/xorg/Xorg :0", True)}, fill=260)
    order = os.listdir(tree)
    check(len(order) > 256 and order.index("300") > 256,
          "the fake tree is real-sized: %d entries, the holder at %d" % (len(order), order.index("300")))
    idle = pmon_file(rig, [])
    p = subprocess.run(rig.args("--probe-display", "--modeset-dir", tree, "--pmon-file", idle,
                                "--drm-dir", os.path.join(rig.dir, "drm-none")), capture_output=True, text=True)
    lines = p.stdout.splitlines()
    check(p.returncode == 1 and lines and lines[0].startswith("IN-USE modeset-holder") and "pid 300" in lines[0],
          "--probe-display reads the whole tree: the holder past entry 256 is found: %r" % lines)
    rig.start("--modeset-dir", tree, "--pmon-file", idle, "--drm-dir", os.path.join(rig.dir, "drm-none"))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and "pid 300" in st["why"],
          "bring-up reads the whole tree too: refusing, why names the holder (%r)" % st["why"])
    rig.stop()

    # the fillers are not holders: a full tree with only persistenced in it reads idle
    tree = modeset_holders(rig, {100: ("nvidia-persistenced --keep", True)}, fill=260)
    p = subprocess.run(rig.args("--probe-display", "--modeset-dir", tree, "--pmon-file", idle,
                                "--drm-dir", os.path.join(rig.dir, "drm-none")), capture_output=True, text=True)
    lines = p.stdout.splitlines()
    check(p.returncode == 0 and lines and lines[0].startswith("idle modeset-holder"),
          "--probe-display: a full tree with only nvidia-persistenced reads idle: %r" % lines)


def drm(rig, connectors):
    """A fake DRM tree: {"card0-eDP-1": "connected"} names each connector's status. None means no directory."""
    d = os.path.join(rig.dir, "drm")
    shutil.rmtree(d, ignore_errors=True)
    if connectors is not None:
        for name, status in connectors.items():
            os.makedirs(os.path.join(d, name), exist_ok=True)
            with open(os.path.join(d, name, "status"), "w") as f:
                f.write(status + "\n")
    return d


def lent_mib(sock):
    return ask(sock, '{"key":"%s","op":"status"}' % KEY)["lent"] // MIB


DISPLAY_WHY = "a display is in use (card0-eDP-1): atticd lends the memory a display needs"


def case_display(rig):
    """A connected display needs the scanout memory atticd lends: bring-up refuses, a display plugged in while
    lending stops new lends, unplugged lends again (fake tree over --drm-dir; the node proves a real one)."""
    rig.start("--drm-dir", drm(rig, {"card0-eDP-1": "connected", "card0-DP-2": "disconnected"}))
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and DISPLAY_WHY in st["why"], "connected display: refusing, why: %r" % st["why"])
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "refusing" and st["size"] == 0, "status still answers, nothing to lend")
    r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
    check(r is not None and r["ok"] is False and DISPLAY_WHY in r["error"], "alloc refused with the same reason")
    for fd in fds:
        os.close(fd)
    check(lent_mib(rig.sock) == 0, "nothing lent")
    p = subprocess.run(rig.args("--drm-dir", os.path.join(rig.dir, "drm"), "--probe"), capture_output=True, text=True)
    check(p.returncode == 3 and json.loads(p.stdout)["state"] == "refusing" and DISPLAY_WHY in json.loads(
        p.stdout)["why"], "--probe: exit 3, why")
    st = status_file(os.path.join(rig.dir, "status.json"))
    check(st and st["state"] == "refusing" and DISPLAY_WHY in st["why"], "the status file says the same: %r" % st)
    open(os.path.join(rig.dir, "drm", "card0-eDP-1", "status"), "w").write("disconnected\n")
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s: s["state"] == "ready", 12.0)
    check(st is not None, "unplugged: lending again within a few ticks")
    st = ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)
    check(st["state"] == "ready" and st["size"] == 2046 * MIB, "and the full carveout is free again: %r" % st["why"])
    check("a display is in use" in open(rig.log).read(), "the refusal is logged")

    # a display plugged in while lending: new lends stop at once, the holder keeps its slice
    drm(rig, {"card0-DP-1": "disconnected"})
    rig.stop()
    rig.start("--drm-dir", os.path.join(rig.dir, "drm"))
    check(ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)["size"] == 2046 * MIB, "a disconnected connector lends")
    pid, line = holder(rig.sock, 100 * MIB)
    check(line == "ok" and lent_mib(rig.sock) == 100, "a client holds 100 MiB")
    open(os.path.join(rig.dir, "drm", "card0-DP-1", "status"), "w").write("connected\n")
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s: "a display is in use" in s["why"], 12.0)
    check(st and st["state"] == "refusing" and st["lent_mib"] == 100,
          "plugged in while lending: refusing, the 100 MiB stays with its holder: %r" % st)
    r, fds = call(connect(rig.sock), alloc_req(8 * MIB))
    check(r is not None and r["ok"] is False and "display is in use" in r["error"], "a new alloc is refused")
    for fd in fds:
        os.close(fd)
    kill(pid)
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s: s["lent_mib"] == 0)
    check(st and st["state"] == "refusing", "the holder is gone: the slice drains, still refusing")
    open(os.path.join(rig.dir, "drm", "card0-DP-1", "status"), "w").write("disconnected\n")
    st = status_file(os.path.join(rig.dir, "status.json"), lambda s: s["state"] == "ready", 12.0)
    check(st is not None and st["free_mib"] == 2046, "unplugged: lending again, everything free")
    rig.stop()

    # no DRM at all (the headless default): lends
    rig.start("--drm-dir", os.path.join(rig.dir, "drm-none"))
    check(ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)["size"] == 2046 * MIB,
          "no DRM directory: lends, as the tests always have")
    rig.stop()
    rig.start("--drm-dir", drm(rig, {"card0-eDP-1": "connected-unknown", "card0-eDP-2": "unknown"}))
    check(ask(rig.sock, '{"key":"%s","op":"status"}' % KEY)["size"] == 2046 * MIB,
          "only non-connected statuses: lends")
    rig.stop()


CASES = [case_heap_lend_and_reclaim, case_heap_drain, case_requests, case_refusals, case_export_failure, case_list_mode,
         case_sighup, case_candidate_keep_probe, case_socket_path, case_status_file, case_display, case_display3,
         case_display4, case_display_beyond_budget, case_attic_check,
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

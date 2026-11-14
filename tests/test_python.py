#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_python.py: the Python client spark_attic on a host with no GPU, against the real atticd (fake RM shim) and the
fake libcuda.so.1 (tests/fakecuda.c), plus a scripted daemon for the protocol's edges. torch is not needed: tensor()
is checked against a stand-in that records what the client hands to torch.as_tensor.

    python3 -I tests/test_python.py --bin build/host
"""
import argparse
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
from test_server import FAILS, KEY, MIB, SLICE, Rig, check, wait_free  # noqa: E402

import spark_attic  # noqa: E402
from spark_attic import AtticError, Settings  # noqa: E402

SMALL = {"ATTIC_FAKE_REGION": "0x280000000:0x4000000"}  # a 64 MiB carveout, 2 MiB of it RM's own


class Logs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def last(self):
        return self.lines[-1] if self.lines else ""


LOGS = Logs()
logging.getLogger("spark_attic").addHandler(LOGS)
logging.getLogger("spark_attic").setLevel(logging.INFO)


def settings(rig, key=KEY):
    return Settings(sock=rig.sock, key=key, timeout=5.0)


def fds_open():
    return len(os.listdir("/proc/self/fd"))


def pattern(n, seed):
    return bytes((i * 131 + seed) & 0xFF for i in range(n))


# ---------------------------------------------------------------------------------------------------------- cases

def case_settings(rig):
    try:
        Settings.from_env({})
        check(False, "no key in the environment: from_env refuses")
    except AtticError as e:
        check("no key" in str(e), "no key in the environment: from_env refuses with why: %s" % e)
    s = Settings.from_env({"SPARK_ATTIC_SOCK": "/x/a.sock", "SPARK_ATTIC_KEY": "k1", "SPARK_ATTIC_TIMEOUT_MS": "250"})
    check((s.sock, s.key, s.timeout) == ("/x/a.sock", "k1", 0.25), "from the environment: %r" % (s,))
    try:
        Settings.from_env({"SPARK_ATTIC_SOCK": "", "SPARK_ATTIC_KEY": ""})
        check(False, "empty means unset: still no key")
    except AtticError as e:
        check("no key" in str(e), "empty means unset: still no key: %s" % e)
    kf = os.path.join(rig.dir, "key")
    with open(kf, "w") as f:
        f.write("attic_0123abcd\n")
    s = Settings.from_env({"SPARK_ATTIC_KEY_FILE": kf})
    check(s.key == "attic_0123abcd", "the key file, trimmed: %r" % s.key)
    s = Settings.from_env({"SPARK_ATTIC_KEY_FILE": kf, "SPARK_ATTIC_KEY": "direct"})
    check(s.key == "direct", "SPARK_ATTIC_KEY over the key file")
    for bad in ("a b", 'a"b', "a\\b", "x" * 128, "été"):
        try:
            Settings.from_env({"SPARK_ATTIC_KEY": bad})
            check(False, "key %r refused" % bad)
        except AtticError:
            pass
    try:
        Settings.from_env({"SPARK_ATTIC_KEY_FILE": os.path.join(rig.dir, "none")})
        check(False, "a missing key file refused")
    except AtticError as e:
        check("SPARK_ATTIC_KEY_FILE" in str(e), "a missing key file names itself: %s" % e)


def case_fail_open(rig):
    rig.start(env=SMALL)
    n = fds_open()
    m = spark_attic.borrow(settings=Settings(sock=os.path.join(rig.dir, "none.sock"), key=KEY))
    check(m is None and "no atticd at" in LOGS.last(), "no daemon: None and why: %s" % LOGS.last())
    m = spark_attic.borrow(settings=settings(rig, key="not_the_key"))
    check(m is None and "wrong key" in LOGS.last(), "wrong key: None and why: %s" % LOGS.last())
    old = os.environ.get("SPARK_ATTIC_KEY")
    os.environ["SPARK_ATTIC_KEY"] = "a b"
    try:
        m = spark_attic.borrow()
        check(m is None and "not one we can send" in LOGS.last(), "a bad key in the environment: None: %s"
              % LOGS.last())
    finally:
        os.environ.pop("SPARK_ATTIC_KEY") if old is None else os.environ.__setitem__("SPARK_ATTIC_KEY", old)
    m = spark_attic.borrow(1 << 30, settings=settings(rig))
    check(m is None and "refused" in LOGS.last(), "more than is free: None and why: %s" % LOGS.last())
    os.environ["ATTIC_FAKECUDA_NOIMPORT"] = "1"
    try:
        m = spark_attic.borrow(4 * MIB, settings=settings(rig))
        check(m is None and "cuMemImportFromShareableHandle" in LOGS.last(), "an import that fails: None: %s"
              % LOGS.last())
    finally:
        del os.environ["ATTIC_FAKECUDA_NOIMPORT"]
    inf = wait_free(rig.sock, spark_attic.info(settings(rig))["size"])
    check(inf and inf["free"] == inf["size"], "nothing stays lent after the failures: %r" % inf)
    check(fds_open() == n, "no descriptor leaked: %d then %d" % (n, fds_open()))


def case_unit(rig):
    """systemd/attic.service: the hardening is there and the pieces atticd needs are not locked out (S5)."""
    unit = {}
    section = None
    for line in open(os.path.join(ROOT, "systemd", "attic.service")):
        line = line.strip()
        if line.startswith("["):
            section = line
        elif "=" in line and section == "[Service]":
            k, v = line.split("=", 1)
            unit.setdefault(k, []).append(v)
    check(unit.get("ExecStart") == ["/usr/local/sbin/atticd --socket /run/spark-attic/attic.sock --key-file "
                                    "/etc/spark-attic/key --mode heap"], "ExecStart: key file, not a key: %r"
          % unit.get("ExecStart"))
    hardening = ["NoNewPrivileges", "PrivateTmp", "ProtectHome", "ProtectSystem", "ReadWritePaths",
                 "ProtectKernelTunables", "ProtectKernelModules", "ProtectKernelLogs", "ProtectControlGroups",
                 "ProtectClock", "ProtectHostname", "RestrictSUIDSGID", "LockPersonality", "RestrictAddressFamilies"]
    missing = [k for k in hardening if k not in unit]
    check(not missing, "the hardening lines are there (S5); missing: %r" % missing)
    check(unit.get("ProtectSystem") == ["strict"] and unit.get("ReadWritePaths") == ["-/run/spark-attic"],
          "the whole filesystem read-only except the runtime directory")
    check(unit.get("RestrictAddressFamilies") == ["AF_UNIX"], "AF_UNIX is the only socket family")
    check("User" not in unit and "CapabilityBoundingSet" not in unit and "DevicePolicy" not in unit
          and "SystemCallFilter" not in unit and "ProtectProc" not in unit,
          "and what is left off stays off (INSTALL section 5 says why): %r"
          % [k for k in ("User", "CapabilityBoundingSet", "DevicePolicy", "SystemCallFilter", "ProtectProc")
             if k in unit])


def case_borrow(rig):
    rig.start(env=SMALL)
    s = settings(rig)
    inf = spark_attic.info(s)
    check(inf and inf["largest"] == 62 * MIB, "info: the largest run, 62 MiB: %r" % inf)
    m = spark_attic.borrow(settings=s)
    check(m is not None and m.size == 62 * MIB and m.attic_bytes == m.size, "borrow(): the largest run: %r" % m)
    st = spark_attic.Attic(s).status()
    check(st["lent"] == 62 * MIB, "atticd counts it lent: %r" % st.get("lent"))
    for off in (0, 30 * MIB + 12345, m.size - 4096):
        data = pattern(4096, off & 0xFF)
        m.write(off, data)
        check(m.read(off, 4096) == data, "written and read back at %d" % off)
    try:
        m.read(m.size - 1, 2)
        check(False, "a read past the end refused")
    except ValueError:
        pass
    m.close()  # closed without zero(): this holder's bytes stay in the slice
    check(m.closed and "closed" in repr(m), "closed: %r" % m)
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "close() gives it back: %r" % inf)
    # the slice still holds the previous holder's bytes: Pool.map does not clear, so a raw borrow reads the
    # pattern back (this is what makes the zeroing check below mean anything)
    raw = spark_attic.Pool(settings=s).map(0, [62 * MIB])
    check(raw.read(0, 4096) == pattern(4096, 0) and raw.read(30 * MIB + 12345, 4096) == pattern(4096, 57)
          and raw.read(m.size - 4096, 4096) == pattern(4096, 0),
          "the same slice raw (Pool.map): the previous holder's bytes are still there (S3)")
    raw.close()
    m = spark_attic.borrow(settings=s)  # borrow() zeroes the slice before returning it
    check(m is not None and m.read(0, 4096) == bytes(4096) and m.read(30 * MIB + 12345, 4096) == bytes(4096)
          and m.read(m.size - 4096, 4096) == bytes(4096),
          "borrow() zeroes the slice: the previous holder's bytes are gone (S3)")
    m.zero()
    check(m.read(m.size - 64, 64) == bytes(64), "zero() clears it")
    m.close()
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "the re-borrow is given back: %r" % inf)
    m = spark_attic.borrow(5 * MIB, settings=s)
    check(m is not None and m.size == 6 * MIB, "5 MiB asked: 6 MiB (whole slices): %r" % m)
    del m  # dropped without close(): the finalizer gives it back
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "a dropped mapping is given back: %r" % inf)


def case_pool(rig):
    rig.start(env=SMALL)
    s = settings(rig)
    with spark_attic.Pool(s) as pool:
        a = pool.map(6 * MIB, [8 * MIB])  # glue: 6 MiB of device memory, then an attic slice
        b = pool.borrow(4 * MIB)
        check(a.size == 14 * MIB and a.attic_bytes == 8 * MIB, "glue: 6 MiB device + 8 MiB attic: %r" % a)
        check(pool.info()["free"] == 50 * MIB, "the pool holds 12 MiB of the attic: %r" % pool.info())
        da, db = pattern(1 << 20, 7), pattern(1 << 20, 99)
        a.write(6 * MIB - (1 << 19), da)  # across the seam
        b.write(0, db)
        check(a.read(6 * MIB - (1 << 19), 1 << 20) == da, "glue: one range across device memory and the attic")
        check(b.read(0, 1 << 20) == db, "a second slice holds its own data")
        a.close()
        b.close()
        check(pool.info()["free"] == 50 * MIB, "closed mappings stay lent to the pool until it closes")
    inf = wait_free(rig.sock, 62 * MIB)
    check(inf and inf["free"] == 62 * MIB, "the pool closed: all of it back: %r" % inf)
    os.environ["ATTIC_FAKECUDA_NOCREATE"] = "1"
    try:
        with spark_attic.Pool(s) as pool:
            try:
                pool.map(2 * MIB, [2 * MIB])
                check(False, "a glue whose device memory fails raises")
            except spark_attic.CudaError as e:
                check("cuMemCreate" in str(e), "names cuMemCreate: %s" % e)
            check(pool.info()["free"] == 62 * MIB, "and borrowed nothing")
    finally:
        del os.environ["ATTIC_FAKECUDA_NOCREATE"]


def case_tensor(rig):
    rig.start(env=SMALL)
    seen = []

    class FakeTensor:
        def __init__(self, obj, device):
            self.obj, self.device, self.views = obj, device, []

        def view(self, x):
            self.views.append(x)
            return self

    fake = types.ModuleType("torch")
    fake.uint8, fake.float16 = "uint8", "float16"
    fake.as_tensor = lambda obj, device=None: seen.append(FakeTensor(obj, device)) or seen[-1]
    sys.modules["torch"] = fake
    try:
        m = spark_attic.borrow(8 * MIB, settings=settings(rig))
        t = m.tensor(fake.float16, (1024, 2048), offset=2 * MIB, nbytes=4 * MIB)
        cai = t.obj.__cuda_array_interface__
        check(cai["data"] == (m.ptr + 2 * MIB, False) and cai["shape"] == (4 * MIB,) and cai["typestr"] == "|u1"
              and cai["version"] == 3 and cai["stream"] is None, "the array interface: %r" % cai)
        check(t.device == "cuda:0" and t.views == ["float16", (1024, 2048)], "device, dtype, shape: %r %r"
              % (t.device, t.views))
        check(t.obj.mapping is m, "the tensor's owner keeps the mapping alive")
        try:
            m.tensor(offset=7 * MIB, nbytes=2 * MIB)
            check(False, "a tensor past the end refused")
        except ValueError:
            pass
        m.close()
    finally:
        del sys.modules["torch"]


class Scripted:
    """A one-connection SEQPACKET daemon that answers each request with the next scripted reply: (bytes, n fds)."""

    def __init__(self, path, replies):
        self.path, self.replies, self.got = path, list(replies), []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.srv.bind(path)
        self.srv.listen(1)
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        c, _ = self.srv.accept()
        with c:
            for body, nfds in self.replies:
                req = c.recv(4096)
                if not req:
                    return
                self.got.append(req)
                fds = [os.open("/dev/null", os.O_RDONLY) for _ in range(nfds)]
                if body is None:
                    return
                socket.send_fds(c, [body], fds) if fds else c.send(body)
                for fd in fds:
                    os.close(fd)

    def close(self):
        self.srv.close()
        self.t.join(5)


def case_protocol(rig):
    path = os.path.join(rig.dir, "scripted.sock")
    n = fds_open()
    d = Scripted(path, [(b'{"ok":true,"size":4194304,"free":2097152,"largest":2097152}', 0),
                        (b'{"ok":true,"size":2097152}', 1)])
    a = spark_attic.Attic(Settings(sock=path, key=KEY))
    check(a.info() == {"size": 4 * MIB, "free": 2 * MIB, "largest": 2 * MIB}, "info parsed")
    fd = a.alloc(SLICE)
    os.close(fd)
    a.close()
    d.close()
    check(d.got == [b'{"key":"spark-attic-1","op":"info"}', b'{"key":"spark-attic-1","op":"alloc","size":2097152}'],
          "requests byte for byte as the engine's client sends them: %r" % d.got)
    bad = [(b'not json', 0, "not JSON"), (b'[1]', 0, "not a JSON object"), (b'{"ok":false,"error":"nope"}', 0,
           "refused: nope"), (b'{"ok":true,"size":2097152}', 0, "0 descriptors"),
           (b'{"ok":true,"size":2097152}', 2, "did not fit"), (b'{"ok":true,"size":4194304}', 1, "we asked for"),
           (None, 0, "closed the connection")]
    for body, nfds, want in bad:
        os.unlink(path)
        d = Scripted(path, [(body, nfds)])
        a = spark_attic.Attic(Settings(sock=path, key=KEY, timeout=5.0))
        try:
            a.alloc(SLICE)
            check(False, "reply %r refused" % body)
        except AtticError as e:
            check(want in str(e), "reply %r: %s" % (body, e))
        a.close()
        d.close()
    os.unlink(path)
    d = Scripted(path, [])
    a = spark_attic.Attic(Settings(sock=path, key=KEY))
    try:
        a.alloc(3 * MIB)
        check(False, "3 MiB is not whole slices")
    except AtticError as e:
        check("2 MiB slices" in str(e), "whole slices only: %s" % e)
    a.close()
    d.close()
    check(fds_open() == n, "no descriptor leaked: %d then %d" % (n, fds_open()))


CASES = [case_settings, case_unit, case_fail_open, case_borrow, case_pool, case_tensor, case_protocol]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/host")
    a = ap.parse_args()
    bindir = os.path.abspath(a.bin)
    os.environ["SPARK_ATTIC_LIBCUDA"] = os.path.join(bindir, "fakecuda", "libcuda.so.1")
    os.umask(0o022)
    for case in CASES:
        rig = Rig(bindir, "py-" + case.__name__[5:])
        before = len(FAILS)
        try:
            case(rig)
        except Exception as e:  # a crash is a failure, with atticd's log
            FAILS.append("%s: %r" % (case.__name__, e))
            print("  FAIL %s raised %r" % (case.__name__, e))
        if len(FAILS) > before and os.path.exists(rig.log):
            print("  atticd log:\n" + open(rig.log).read())
        print("%s %s" % ("ok  " if len(FAILS) == before else "FAIL", case.__name__))
        rig.done()
    print("test_python: %d cases, %d failures" % (len(CASES), len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

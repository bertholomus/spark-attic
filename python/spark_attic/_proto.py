# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""atticd's protocol: AF_UNIX SOCK_SEQPACKET, one flat JSON object a message, a lent slice's descriptor beside the
alloc reply (SCM_RIGHTS). The requests are byte for byte the ones our TensorFold client sends."""
import json
import os
import socket
from dataclasses import dataclass

DEFAULT_SOCK = "/run/spark-attic/attic.sock"
SLICE = 2 << 20  # atticd lends whole 2 MiB slices
KEY_MAX = 127
REPLY_MAX = 4096


class AtticError(Exception):
    """atticd is not there, refused, or answered something we do not understand."""


def key_ok(key):
    """1 to 127 printable bytes, no space, quote or backslash: a key we can send without escaping."""
    return 0 < len(key) <= KEY_MAX and all(0x21 <= ord(c) <= 0x7E and c not in '"\\' for c in key)


@dataclass(frozen=True)
class Settings:
    sock: str = DEFAULT_SOCK
    key: str = ""  # no compiled-in default: set it, or let from_env read the environment
    timeout: float = 5.0  # seconds, for connect and every reply

    @classmethod
    def from_env(cls, env=None):
        """SPARK_ATTIC_SOCK, SPARK_ATTIC_KEY (or SPARK_ATTIC_KEY_FILE), SPARK_ATTIC_TIMEOUT_MS; empty means unset."""
        env = os.environ if env is None else env
        key = env.get("SPARK_ATTIC_KEY") or ""
        if not key and env.get("SPARK_ATTIC_KEY_FILE"):
            try:
                with open(env["SPARK_ATTIC_KEY_FILE"]) as f:
                    key = f.read().strip()
            except OSError as e:
                raise AtticError("SPARK_ATTIC_KEY_FILE %s: %s" % (env["SPARK_ATTIC_KEY_FILE"], e.strerror))
        if not key:
            raise AtticError("no key: set SPARK_ATTIC_KEY or SPARK_ATTIC_KEY_FILE (atticd runs with none compiled "
                             "in: it refuses to start without one)")
        if not key_ok(key):
            raise AtticError("the attic key is not one we can send (1 to %d printable bytes, no space, quote or "
                             "backslash)" % KEY_MAX)
        try:
            timeout = int(env.get("SPARK_ATTIC_TIMEOUT_MS") or 5000) / 1000.0
        except ValueError:
            raise AtticError("SPARK_ATTIC_TIMEOUT_MS is not a number of milliseconds")
        return cls(sock=env.get("SPARK_ATTIC_SOCK") or DEFAULT_SOCK, key=key, timeout=max(timeout, 0.001))


class Attic:
    """One connection to atticd. The slices it lends stay lent to this connection until close(); atticd counts them
    free again once the socket is closed and the memory is unmapped everywhere."""

    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        if not key_ok(self.settings.key):
            raise AtticError("the attic key is not one we can send")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC)
        s.settimeout(self.settings.timeout)
        try:
            s.connect(self.settings.sock)
        except OSError as e:
            s.close()
            raise AtticError("no atticd at %s (%s)" % (self.settings.sock, e.strerror or e))
        self._sock = s

    def _ask(self, req, want_fd=False):
        if self._sock is None:
            raise AtticError("this attic connection is closed")
        try:
            self._sock.send(json.dumps(req, separators=(",", ":")).encode())
            msg, fds, flags, _ = socket.recv_fds(self._sock, REPLY_MAX, 1)
        except OSError as e:
            raise AtticError("atticd at %s: %s" % (self.settings.sock, e.strerror or e))
        try:
            if not msg:
                raise AtticError("atticd closed the connection")
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise AtticError("atticd's reply did not fit")
            try:
                reply = json.loads(msg)
            except ValueError:
                raise AtticError("atticd's reply is not JSON")
            if not isinstance(reply, dict):
                raise AtticError("atticd's reply is not a JSON object")
            if reply.get("ok") is not True:
                raise AtticError("atticd refused: %s" % reply.get("error", "no reason given"))
            if len(fds) != (1 if want_fd else 0):
                raise AtticError("atticd sent %d descriptors with its reply" % len(fds))
        except AtticError:
            for fd in fds:
                os.close(fd)
            raise
        return reply, (fds[0] if want_fd else None)

    def info(self):
        """{"size", "free", "largest"} in bytes: all atticd lends, what is free now, the largest slice it can lend."""
        reply, _ = self._ask({"key": self.settings.key, "op": "info"})
        for k in ("size", "free", "largest"):
            if not isinstance(reply.get(k), int) or reply[k] < 0:
                raise AtticError("atticd's info reply has no %s" % k)
        return {k: reply[k] for k in ("size", "free", "largest")}

    def status(self):
        return self._ask({"key": self.settings.key, "op": "status"})[0]

    def alloc(self, nbytes):
        """Borrow nbytes (whole 2 MiB slices) as one slice; returns its descriptor, which the caller closes."""
        if nbytes <= 0 or nbytes % SLICE:
            raise AtticError("%d bytes is not a whole number of 2 MiB slices" % nbytes)
        reply, fd = self._ask({"key": self.settings.key, "op": "alloc", "size": nbytes}, want_fd=True)
        if reply.get("size") != nbytes:
            os.close(fd)
            raise AtticError("atticd lent %r bytes, we asked for %d" % (reply.get("size"), nbytes))
        return fd

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

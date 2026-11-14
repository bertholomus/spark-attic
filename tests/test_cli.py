#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
"""test_cli.py: scripts/spark-attic on the host, against fake roots. The status file's four
states (ok, missing, stale, unreadable), --json, the header (plain versus colour, art versus
no art), doctor over a fake install.sh --check, uninstall (leaves nothing; keeps the key),
demo dispatch and version.

    python3 -I tests/test_cli.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(HERE, "scripts", "spark-attic")
NAME = "spark-attic"
FAILS = []


def check(cond, what):
    print("  %s %s" % ("ok  " if cond else "FAIL", what))
    if not cond:
        FAILS.append(what)


def run(args, env=None, root=None, stdin=None):
    e = dict(os.environ)
    e.pop("NO_COLOR", None)
    e.setdefault("TERM", "xterm-256color")
    e.setdefault("ATTIC_COLOR", "never")
    if env:
        e.update(env)
    cmd = ["python3", CLI] + args
    if root:
        cmd += ["--root", root]
    return subprocess.run(cmd, env=e, capture_output=True, text=True,
                          input=stdin, timeout=60)


STATUS = {"name": NAME, "version": "0.1.0", "state": "ready", "why": "", "driver": "999.0.1",
          "mode": "heap", "validated": True, "validated_at": "2026-11-12T00:00:00Z",
          "size_mib": 2044, "free_mib": 0, "largest_mib": 0, "lent_mib": 2044,
          "draining_mib": 0, "clients": 1, "every_s": 10, "pid": 1, "updated": 0}


def write_status(root, rec=None, stale=False):
    d = os.path.join(root, "run", "spark-attic")
    os.makedirs(d, exist_ok=True)
    rec = dict(STATUS if rec is None else rec)
    rec["updated"] = int(time.time()) - (3600 if stale else 0)
    with open(os.path.join(d, "status.json"), "w") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    t = tempfile.mkdtemp(prefix="test-cli-")

    # ---- version
    r = run(["version"])
    check(r.returncode == 0 and r.stdout.strip() == "%s 0.1.0" % NAME,
          "version: one line, rc 0 (%r)" % r.stdout.strip())
    r = run([])
    check(r.returncode == 0 and r.stdout.strip() == "%s 0.1.0" % NAME, "no args: version")

    # ---- status: ok
    root = os.path.join(t, "ok")
    write_status(root)
    r = run(["status"], root=root)
    check(r.returncode == 0 and "state     ready" in r.stdout and
          "lent      2,044 MiB" in r.stdout and "driver    999.0.1, validated" in r.stdout,
          "status ok: state, lent, driver in words")
    check("\x1b[" not in r.stdout and r.stdout.startswith("spark-attic 0.1.0\n"),
          "status piped: no escapes, the header is the plain text line")
    r = run(["status", "--json"], root=root)
    rec = json.loads(r.stdout)
    check(r.returncode == 0 and rec.get("state") == "ready" and rec.get("lent_mib") == 2044,
          "status --json: the status file as JSON")

    # ---- status: missing / stale / unreadable
    root = os.path.join(t, "missing")
    r = run(["status"], root=root)
    check(r.returncode == 1 and "no status file" in r.stdout and
          "systemctl show -p ActiveState" in r.stdout,
          "status missing: rc 1, where to look, what to run")
    r = run(["status", "--json"], root=root)
    check(r.returncode == 1 and json.loads(r.stdout) == {"error": "missing"},
          "status --json missing: the error object")
    root = os.path.join(t, "stale")
    write_status(root, stale=True)
    r = run(["status"], root=root)
    check(r.returncode == 1 and "stale" in r.stdout and "no numbers" in r.stdout.splitlines()[2],
          "status stale: rc 1, shown stale without numbers")
    root = os.path.join(t, "bad")
    os.makedirs(os.path.join(root, "run", "spark-attic"))
    with open(os.path.join(root, "run", "spark-attic", "status.json"), "w") as f:
        f.write('{"name":"someone-else","state":"ready"}')
    r = run(["status"], root=root)
    check(r.returncode == 1 and "unreadable" in r.stdout,
          "status unreadable: a foreign file is unknown, never shown as ours")

    # ---- the header: plain vs colour vs --plain, art vs no art
    share = os.path.join(t, "share", NAME)
    os.makedirs(share)
    for f in ("header-art.txt", "header-col.txt"):
        src = os.path.join(HERE, "share", NAME, f)
        if os.path.exists(src):
            with open(src, encoding="utf-8") as fh:
                data = fh.read()
            with open(os.path.join(share, f), "w", encoding="utf-8") as fh:
                fh.write(data)
    env = {"ATTIC_SHARE_DIR": share, "ATTIC_COLOR": "always"}
    r = run(["status"], env=env, root=os.path.join(t, "ok"))
    has_art = os.path.exists(os.path.join(share, "header-art.txt"))
    if has_art:
        check(r.stdout.startswith("\x1b[") and "spark-attic 0.1.0" in r.stdout,
              "colour TTY: the header is the wordmark and the title")
        check("⣿" in r.stdout or "⢀" in r.stdout, "colour TTY: the braille wordmark is printed")
        check("\x1b[38;5;255m" in r.stdout and "\x1b[38;5;179m" in r.stdout,
              "colour TTY: the paper and gold colours")
        check("\x1b[38;5;202m" not in r.stdout,
              "colour TTY: the header uses the Bertholomus colours only")
    else:
        # this tree carries no art (the exported tree: the graphics gate); the plain header
        check(r.stdout.splitlines()[0] == "spark-attic 0.1.0",
              "colour asked, art absent: the plain text header is the first line")
        check("\x1b[" not in r.stdout, "colour asked, art absent: no escapes at all")
    r = run(["status", "--plain"], env=env, root=os.path.join(t, "ok"))
    check(r.stdout.startswith("spark-attic 0.1.0") or "state" in r.stdout,
          "--plain beats ATTIC_COLOR=always")
    envn = {"ATTIC_SHARE_DIR": share, "ATTIC_COLOR": "always", "NO_COLOR": "1"}
    r = run(["status"], env=envn, root=os.path.join(t, "ok"))
    check("\x1b[" not in r.stdout, "NO_COLOR beats ATTIC_COLOR=always")
    empty = os.path.join(t, "empty-share")
    os.makedirs(empty)
    r = run(["status"], env={"ATTIC_SHARE_DIR": empty, "ATTIC_COLOR": "always"},
            root=os.path.join(t, "ok"))
    check(r.stdout.splitlines()[0] == "spark-attic 0.1.0",
          "colour on, art files absent: the plain text header (the exported tree's case)")

    # ---- doctor: fake install.sh --check feeding the preflight records
    fakebin = os.path.join(t, "fakebin")
    os.makedirs(fakebin)
    fake_install = os.path.join(fakebin, "install.sh")
    with open(fake_install, "w") as f:
        f.write('#!/bin/bash\n'
                'while [ $# -gt 0 ]; do case $1 in --format) shift; [ "$1" = json ] && JSON=1;;'
                ' --root) shift; R=$1;; esac; shift; done\n'
                'echo \'{"id":"arch","state":"pass","what":"this machine is aarch64"}\'\n'
                'echo \'{"id":"driver","state":"fail","what":"no NVIDIA driver is loaded",'
                '"fix":"install the driver"}\'\n'
                'echo \'{"id":"summary","problems":1}\'\n')
    os.chmod(fake_install, 0o755)
    root = os.path.join(t, "doctor")
    write_status(root)
    os.makedirs(os.path.join(root, "etc", "spark-attic"))
    with open(os.path.join(root, "etc", "spark-attic", "key"), "w") as f:
        f.write("attic_test\n")
    os.chmod(os.path.join(root, "etc", "spark-attic", "key"), 0o600)
    env = {"ATTIC_INSTALL": fake_install, "ATTIC_COLOR": "never"}
    r = run(["doctor"], env=env, root=root)
    check(r.returncode > 0 and "no NVIDIA driver is loaded" in r.stdout and
          "install the driver" in r.stdout,
          "doctor: the preflight failure with its fix, rc = problems")
    count_line = [l for l in r.stdout.splitlines() if l.strip().startswith("doctor:")][-1]
    check(" problems" in count_line and int(count_line.strip().split()[1]) >= 1,
          "doctor: the count line names the problems (%r)" % count_line)
    r = run(["doctor", "--json"], env=env, root=root)
    tail = "\n".join(r.stdout.splitlines()[r.stdout.splitlines().index("{"):])
    doc = json.loads(tail)
    check(doc["problems"] >= 1 and
          any(c["id"] == "driver" and c["state"] == "fail" for c in doc["checks"]) and
          any(c["id"] == "key" and c["state"] == "pass" for c in doc["checks"]),
          "doctor --json: problems and per-check records")

    # ---- doctor without install.sh
    r = run(["doctor"], env={"ATTIC_INSTALL": os.path.join(t, "absent"), "ATTIC_COLOR": "never"},
            root=root)
    check("install.sh is not where" in r.stdout and r.returncode > 0,
          "doctor without install.sh: says so, counts the problem")

    # ---- uninstall: nothing left, key and logs kept
    root = os.path.join(t, "uninstall")
    for d in ("usr/local/sbin", "usr/local/bin", "usr/local/share/spark-attic/scripts",
              "etc/systemd/system", "etc/spark-attic", "etc/sysctl.d", "var/lib/attic/artifacts",
              "var/log/spark-attic", "run/spark-attic"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    files = {"usr/local/sbin/atticd": "#!/bin/sh\n",
             "usr/local/sbin/attic-check": "#!/bin/sh\n",
             "usr/local/bin/spark-attic": "#!/usr/bin/env python3\n",
             "usr/local/share/spark-attic/header-art.txt": "art\n",
             "usr/local/share/spark-attic/scripts/attic-validate.sh": "#!/bin/bash\n",
             "etc/systemd/system/attic.service": "[Unit]\n",
             "etc/sysctl.d/99-spark-attic-minfree.conf": "# sysctl\n",
             "var/lib/attic/version": "0.1.0\n",
             "var/log/spark-attic/install.log": "log\n"}
    for rel, text in files.items():
        with open(os.path.join(root, rel), "w") as f:
            f.write(text)
    with open(os.path.join(root, "etc", "spark-attic", "key"), "w") as f:
        f.write("attic_keepme\n")
    os.chmod(os.path.join(root, "etc", "spark-attic", "key"), 0o600)
    env = {"ATTIC_COLOR": "never",
           "ATTIC_FAKE_STATE": os.path.join(root, ".systemctl"),
           "PATH": os.path.join(fakebin) + ":" + os.environ["PATH"]}
    with open(os.path.join(fakebin, "systemctl"), "w") as f:
        f.write('#!/bin/bash\ncase "$1 $2" in\n'
                '  "is-active --quiet") [ -f "$ATTIC_FAKE_STATE/active" ]; exit $((1 - $?)) ;;\n'
                '  "is-active attic.service") [ -f "$ATTIC_FAKE_STATE/active" ]; exit $((1 - $?)) ;;\n'
                '  "is-enabled attic.service") [ -f "$ATTIC_FAKE_STATE/enabled" ]; exit $((1 - $?)) ;;\n'
                '  "daemon-reload"*) exit 0 ;;\n'
                '  "stop attic.service") rm -f "$ATTIC_FAKE_STATE/active"; exit 0 ;;\n'
                '  "disable attic.service"|"disable --now") rm -f "$ATTIC_FAKE_STATE/enabled" "$ATTIC_FAKE_STATE/active"; exit 0 ;;\n'
                '  *) exit 0 ;;\nesac\n')
    os.chmod(os.path.join(fakebin, "systemctl"), 0o755)
    os.makedirs(os.path.join(root, ".systemctl"), exist_ok=True)
    open(os.path.join(root, ".systemctl", "active"), "w").close()
    open(os.path.join(root, ".systemctl", "enabled"), "w").close()
    r = run(["uninstall", "--yes"], env=env, root=root)
    check(r.returncode == 0, "uninstall: rc 0")
    check(not os.path.exists(os.path.join(root, ".systemctl", "enabled")),
          "uninstall: the unit is disabled (its enable link is gone), not just stopped")
    left = [rel for rel in files if os.path.exists(os.path.join(root, rel))
            and rel != "var/log/spark-attic/install.log"]
    check(not left, "uninstall: no unit, binary, share file, artifact or sysctl file left (%s)"
          % left)
    check(os.path.exists(os.path.join(root, "etc", "spark-attic", "key")) and
          os.path.exists(os.path.join(root, "var", "log", "spark-attic", "install.log")),
          "uninstall: the key and the logs stay")
    r = run(["uninstall", "--yes"], env=env, root=root)
    check(r.returncode == 0 and "nothing to remove" in r.stdout,
          "a second uninstall: nothing to remove, rc 0")

    # ---- uninstall --purge: the key and the logs go too
    root = os.path.join(t, "purge")
    for d in ("usr/local/bin", "etc/spark-attic", "var/log/spark-attic"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    with open(os.path.join(root, "usr/local/bin/spark-attic"), "w") as f:
        f.write("#!/usr/bin/env python3\n")
    with open(os.path.join(root, "etc/spark-attic/key"), "w") as f:
        f.write("attic_gone\n")
    with open(os.path.join(root, "var/log/spark-attic/install.log"), "w") as f:
        f.write("log\n")
    r = run(["uninstall", "--purge", "--yes"], env=env, root=root)
    check(r.returncode == 0 and not os.path.exists(os.path.join(root, "etc/spark-attic")) and
          not os.path.exists(os.path.join(root, "var/log/spark-attic")),
          "uninstall --purge: the key and the logs are gone too")

    # ---- uninstall asks on a terminal, refuses without --yes when piped
    root = os.path.join(t, "ask")
    os.makedirs(os.path.join(root, "usr/local/bin"))
    with open(os.path.join(root, "usr/local/bin/spark-attic"), "w") as f:
        f.write("#!/usr/bin/env python3\n")
    r = run(["uninstall"], env=env, root=root, stdin=None)
    check(r.returncode == 1 and "refusing to uninstall without --yes" in r.stdout,
          "uninstall piped without --yes: refused, nothing changed")

    # ---- demo dispatch: a --demo line reaches install.sh (verified end to end in test_install.sh)
    r = run(["demo", "--fail", "driver"], env={"ATTIC_INSTALL": os.path.join(HERE, "install.sh"),
                                               "ATTIC_COLOR": "never",
                                               "DEMO_PACE": "0"})
    check(r.returncode == 1 and "no NVIDIA driver is loaded" in r.stdout and
          r.stdout.startswith("DEMO:"),
          "demo dispatch: install.sh --demo --fail driver runs (first line DEMO)")

    # ---- no --root: the real paths under /, whatever the working directory (node window 3 defect:
    # status read ./run/spark-attic/status.json relative to the cwd). A cwd that holds a decoy tree
    # must not be read.
    decoy = os.path.join(t, "decoy")
    os.makedirs(os.path.join(decoy, "run", "spark-attic"), exist_ok=True)
    with open(os.path.join(decoy, "run", "spark-attic", "status.json"), "w") as f:
        json.dump(dict(STATUS, version="9.9.9-decoy"), f)
    import importlib.util
    spec = importlib.util.spec_from_loader("sa_cli", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(open(CLI).read(), CLI, "exec"), mod.__dict__)
    here = os.getcwd()
    try:
        os.chdir(decoy)
        check(mod.p("", "run", "spark-attic", "status.json") == "/run/spark-attic/status.json",
              "no --root: paths are absolute under / (not the working directory)")
        check(mod.p(os.path.join(t, "r"), "etc") == os.path.join(t, "r", "etc"),
              "--root DIR: paths stay under DIR")
    finally:
        os.chdir(here)

    print("test_cli: %d failures" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

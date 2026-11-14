# Installing spark-attic

> Headless boxes are the tested case: atticd refuses to lend while a
> display is connected, and lends again when it goes away (README, Supported).

Three parts: build on any Linux machine, install and validate on each GB10 node, then let your engines borrow.
Everything that touches the node runs as root.

## 1. What you need

- **The node:** an NVIDIA GB10 (tested on an ASUS Ascent GX10 with DGX OS 7.6.0) running Linux aarch64 with
  systemd and the NVIDIA driver loaded. Nothing else may lend the display carveout while you validate (see section 4).
- **A build host:** Linux (x86_64 or aarch64) with git, `make`, a C compiler and Python 3.11 or later for the host
  tests (tested with 3.14), network access to github.com, and zig 0.17.0 (it cross-compiles for the node, glibc 2.28 and
  later). The node itself needs no compiler.
- **Engines (optional):** vLLM 0.24.0, 0.31.0 or 0.32.0, SGLang 0.5.14 or 0.5.23 (NGC 26.07 ships the older pair, 26.10 ships vLLM 0.31.0), Python
  3.9+. Other releases work too as long as the engine still has the code shape a plugin knows: on one it does not,
  the KV placement turns itself off with one log line and the engine runs as without spark-attic.
- **A display:** none needed. atticd lends nothing while a DRM connector reports `connected` and resumes when it
  does not (README, Supported); `--drm-dir` points the check at a different connector tree (tests).

## 2. Build (on the build host)

The published artifact is built from an export (`release/export.sh OUT`, then `release/build-release.sh`
in that export): the export strips the private working tree's notes, tests and node scripts, and
`build-release.sh` refuses to run in the private tree. What the release binaries are proven against
splits two ways, and the docs keep it split: a *measured* claim ran on a GB10 node; a
*source-checked* claim (further engine releases, the drift test) was verified by reading upstream
source only. Do not reword one into the other.


```
git clone https://github.com/bertholomus/spark-attic && cd spark-attic
git clone -q --depth 1 --branch v0.24.0 https://github.com/vllm-project/vllm cache/upstream/vllm-0.24.0
git clone -q --depth 1 --branch v0.5.14 https://github.com/sgl-project/sglang cache/upstream/sglang-0.5.14
make check                                                       # host tests, no GPU needed
make OUT=build/aarch64 CC="zig cc -target aarch64-linux-gnu.2.28"  # atticd and attic-check for the node
scripts/attic-build.sh 580.178.04                                # the driver shim for that driver
```

`make check` builds a host copy of everything and runs it against a fake driver and a fake CUDA. The plugin tests
also run against vLLM's and SGLang's own source at the tags the plugins were written for (the two clones above), so a
release of either that moved something the plugins rely on shows up here.

`attic-build.sh` fetches tag `580.178.04` of github.com/NVIDIA/open-gpu-kernel-modules (shallow, the exact tag
checked), compiles `src/rmshim.c` against its headers with every warning an error, and writes
`build/artifacts/580.178.04/rmshim.so` and `BUILD-INFO` (driver, NVIDIA commit, shim sha256). A driver structure
that changed fails here, never on a node. Use the version the node runs: `head -n1 /proc/driver/nvidia/version` on
the node prints it. Set `ATTIC_ZIG` if zig is not on your `PATH`.

Stage one folder for the node:

```
mkdir -p stage/bin stage/scripts stage/systemd stage/artifacts stage/sysctl
cp build/aarch64/atticd build/aarch64/attic-check stage/bin/
cp scripts/attic-validate.sh stage/scripts/ && cp systemd/attic.service stage/systemd/
cp -r build/artifacts/580.178.04 stage/artifacts/
cp sysctl/99-spark-attic-minfree.conf stage/sysctl/               # optional, see section 3
cp -r python stage/                                              # for your engine images
scp -r stage <node>:/root/spark-attic
Stage into `/root` (or another folder only root can reach), never into `/tmp`: a fixed folder there can be created
first by another local user, who would then own the folder root installs from. `/root` is root-only (0700), so nobody
else can put anything in front of these files.
```

`scripts/attic-update.sh 580.178.04 <node>` builds the shim and the tools, stages them on the node and runs step 4
there in one go (it needs ssh to the node and `sudo -n` there). It does not install anything: do step 3 once.

`release/build-release.sh 0.1.0` makes the whole release artifact instead of a hand-staged
folder: `spark-attic-0.1.0-aarch64.tar.gz` with atticd, attic-check, the driver shims for the
compatibility list, the scripts, the Python package, the unit, the docs and SHA256SUMS — the
tarball step 3 installs. The build is reproducible (two runs, one sha256). The node needs no
compiler and no zig either way.

## 3. Install (on the node, as root)

One command:

```
curl -fsSL https://github.com/bertholomus/spark-attic/releases/download/v0.1.0/install.sh | sudo bash
```

or, for an offline copy of the release tarball (the curl form fetches and verifies the same
tarball from the release URL):

```
install.sh --from spark-attic-0.1.0-aarch64.tar.gz
```

What it does, in order:

- **Preflight, read-only** (`install.sh --check` runs only this): an aarch64 machine, a GB10
  device tree, root, systemd, the NVIDIA driver loaded, a shim for that driver in the
  artifact, `atticd --probe-display` clean, no other carveout lender listening, disk space.
  Each failure is one sentence with what to do, and nothing is written.
- **Fetch and verify**: the release tarball for the version, checked against its
  `SHA256SUMS`; a mismatch refuses the install.
- **Install**: the binaries to `/usr/local/sbin`, the CLI to `/usr/local/bin/spark-attic`,
  the share files (docs, licences, the terminal art when the release carries it) to
  `/usr/local/share/spark-attic`, the Python package beside them, the unit - and the sysctl
  drop-in of section 3 only when you pass `--with-sysctl`.
- **The key**: created root-only (0600) when none exists; an existing key is never overwritten.
- **Validate** the running driver (`attic-validate.sh`), then **enable and start**
  `attic.service` only if that passed.
- **Report**: the status file read back — the MiB lent and, when ready, one next step each
  for vLLM, SGLang and TensorFold.

Idempotent: a second run on an installed node changes nothing; a newer artifact upgrades in
place (the service is stopped for the check and started again); an older artifact is refused.
Any failure after the first write rolls back what that run changed and says what. Every step
is logged to `/var/log/spark-attic/install.log`.

Try it without a node: in a clone of this repo, `make demo` runs the same flow against a simulated
GB10 in a throwaway directory (any Linux box with a C compiler; no root, nothing installed, a Y/N
confirm before the install step; `bash install.sh --demo --fail display|driver|lender|validate`
for the refusal paths after one `make demo`).

### Install by hand

The manual path, if you would rather run each step (the layout matches what install.sh puts
down):

```
cd /root/spark-attic
install -m 0755 bin/atticd /usr/local/sbin/atticd
install -m 0755 bin/attic-check /usr/local/bin/attic-check
install -d -m 0755 /var/lib/attic /var/lib/attic/artifacts /etc/spark-attic
(umask 077; od -An -N12 -tx1 /dev/urandom | tr -d ' \n' | sed 's/^/attic_/' > /etc/spark-attic/key)
install -m 0644 systemd/attic.service /etc/systemd/system/attic.service
systemctl daemon-reload
```

The unit runs `atticd --socket /run/spark-attic/attic.sock --key-file /etc/spark-attic/key --mode heap`. Edit its
`ExecStart` for other settings (`atticd --help`): `--socket-group NAME` lets members of a group connect (the socket
is 0660), `--keep-mib N` keeps the last N MiB unlent.

**Optional, for DGX OS on the 64 KiB-page kernel** (`uname -r` ends in `-64k`). There the kernel's rule for
transparent huge pages sets `vm.min_free_kbytes` to 6,466,368 at boot, a 9,964 MiB reserve that MemAvailable never
counts. This file sets the kernel's value without that rule, 45,560, from the next boot. spark-attic does not need
it; README.md (Measured on stock DGX OS) has what it changes. This file is opt-in: the installer
writes it only when you run it with `--with-sysctl` (no default on, and `--no-sysctl` remains for
clarity).

```
install -m 0644 sysctl/99-spark-attic-minfree.conf /etc/sysctl.d/99-spark-attic-minfree.conf
```

## 4. Validate the driver (on the node, as root)

Nothing else may lend the display carveout while you validate or run spark-attic. The installer's
preflight refuses to run while a second lender is listening on the sockets named in
`ATTIC_OTHER_SOCKETS` (a list of Unix-socket paths; set it before the install when your node runs a
lender whose socket is not the default). An empty list checks nothing and the preflight says so in
one line - a warning, not a pass.

```
scripts/attic-validate.sh artifacts/580.178.04
```

It refuses (exit 3) unless the node runs that driver, and (exit 4) while anything else may hold the carveout: our
atticd when it is ready or still lending (stop it: `systemctl stop attic.service`), an atticd it cannot ask (it reads
the key from `/etc/spark-attic/key`; set `ATTIC_KEY` or `ATTIC_KEY_FILE` otherwise), or another carveout lender
listening on its socket (stop that lender). Then it runs a candidate atticd on a socket of its own and `attic-check`
against it, once per mode: every word of the largest slice written and read back through CUDA with two patterns, a
second client refused the same memory, two slices that keep their own data, a killed holder's slice coming back,
and wrong keys and malformed requests refused.

Only a mode that passes is enabled: the shim is installed as `/var/lib/attic/artifacts/580.178.04/` with a
`VALIDATED` file that names the driver, the shim's sha256 and the modes that passed. On 580.178.04, heap passes and
list is refused (the driver uses part of the region), so the exit code is 0 with `modes=heap`. If no mode passes,
nothing is enabled (exit 1) and the reports stay in `/var/lib/attic/candidates/580.178.04/`.

## 5. Start

```
systemctl enable --now attic.service        # install.sh does this when validate passed
spark-attic status                          # state, MiB lent, clients, driver, validated
cat /run/spark-attic/status.json            # the same file, keyless (STATUS-FILE.md)
attic-check --status --key "$(cat /etc/spark-attic/key)"   # the same, with the region's details
```

`spark-attic doctor` runs the preflight plus the live checks (the unit, the key's mode, the
VALIDATED record) and prints a fix line for everything it finds; its exit code is the number
of problems.

`status.json` is readable by anyone and holds no key ([STATUS-FILE.md](STATUS-FILE.md)). When atticd lends nothing,
its `why` says why.

The unit is hardened: no new privileges, no writable /tmp or /home, the whole filesystem read-only except
`/run/spark-attic` (its socket, ledger and status file), no kernel tunables or logs, no module loading, no
clock or hostname changes, and `AF_UNIX` as the only socket family. Left off on purpose, each because a node
fact is missing and a wrong guess breaks bring-up:

- **Root stays, and so do all capabilities** (`User=`, `CapabilityBoundingSet=`): the RM client needs root, and
  which exact capabilities its ioctls need is a node fact.
- **`DevicePolicy=closed`**: the shim opens the NVIDIA device nodes; the exact set a driver opens is a node fact.
- **`MemoryDenyWriteExecute=`**: the driver maps memory into the daemon's address space through its own ioctls;
  whether any of it is writable-plus-executable differs per driver.
- **`SystemCallFilter=`**: the shim's syscall use behind the RM ioctls is not a fixed set across drivers.
- **`ProtectProc=invisible`**: atticd reads `/proc/<pid>/stat` of its clients' processes (any user) to know when a
  drained slice is really free; hiding other users' processes would break that.

## 6. Engines

**vLLM and SGLang.** Put the Python package into the image (`pip install --no-deps ./python`, or put `python/` on
`PYTHONPATH`: its static `dist-info` registers both entry points without an install). The vLLM plugin
(`vllm.general_plugins`) and the SGLang hook (`sglang.srt.plugins`) load by themselves. The container needs atticd's
socket and key:

```
docker run --gpus all --ipc=host \
  -v /run/spark-attic:/run/spark-attic -v /etc/spark-attic/key:/etc/spark-attic/key:ro \
  -v /usr/local/share/spark-attic/python:/opt/spark-attic/python:ro \
  -e PYTHONPATH=/opt/spark-attic/python -e SPARK_ATTIC_KEY_FILE=/etc/spark-attic/key \
  nvcr.io/nvidia/vllm:26.07-py3 vllm serve <model> ...
```

`/usr/local/share/spark-attic/python` is the installed copy (the container reads it as root
through the mount); for a container that outlives an uninstall, copy it somewhere permanent
and mount that. The NGC vLLM and SGLang 26.07 images set no `PYTHONPATH` of their own; on an image that
does, put `/opt/spark-attic/python` in front of it.

The server's log then has a `spark-attic:` line saying what went into the carveout (for `kv`: "KV cache memory ... +
2044 MiB display carveout"). Set `SPARK_ATTIC_PLACE=off` to compare without it. A container that does not run as root
needs atticd started with `--socket-group <group>`, `--group-add <gid>` on `docker run`, and the key file readable by
that group (`chgrp <group> /etc/spark-attic/key; chmod 0640 /etc/spark-attic/key`).

On kernels with 64 KiB pages, vLLM 0.24.0's default lazy safetensors load can stall; add
`--safetensors-load-strategy eager` to `vllm serve`. This is vLLM's, with or without spark-attic.

**TensorFold.** The carveout source proposed to TensorFold (not in a TensorFold release yet) borrows from atticd with
`TF_CUDA_CARVEOUT=1 TF_CUDA_CARVEOUT_SOURCE=attic`, `TENSORFOLD_ATTIC_SOCK=/run/spark-attic/attic.sock` (its default)
and `TENSORFOLD_ATTIC_KEY` set to the key (else it reads `/etc/spark-attic/key`). In a container, mount
`/run/spark-attic` and the key as above.

**Anything else.** `python/spark_attic` (`spark_attic.borrow()`) or the protocol in README.md.

## 7. Driver updates

When the node moves to a new driver, atticd lends nothing ("no VALIDATED artifact" in its `why`) until that driver
is built and validated. From the build host, once the node runs it:

```
scripts/attic-update.sh 580.196.07 <node>
```

atticd on the new driver already lends nothing, so the check can run beside it; on success atticd gets a SIGHUP and
lends again. Exit codes are attic-validate.sh's, plus 5 when the build failed (for example a driver NVIDIA has not
tagged). It only enables what passed, so it is safe to run on every driver change.

On the node, the same move through the CLI (a newer spark-attic release or a new driver's
artifact; it runs install.sh, which validates before enabling):

```
spark-attic update --from spark-attic-0.2.0-aarch64.tar.gz
```

## 8. Uninstall

```
spark-attic uninstall        # stops the service, removes everything install.sh put down
```

It keeps the key (`/etc/spark-attic`) and the logs (`/var/log/spark-attic`) so a reinstall
keeps its identity; `spark-attic uninstall --purge` removes those too. Afterwards no unit,
socket, binary or artifact is left. By hand, the same thing:

```
systemctl disable --now attic.service
rm -f /usr/local/sbin/atticd /usr/local/bin/attic-check /etc/systemd/system/attic.service
rm -rf /var/lib/attic /etc/spark-attic /run/spark-attic
systemctl daemon-reload
```

Memory a running process still holds comes back when that process exits. If you installed the optional file of
section 3, remove it too; the kernel's own value returns at the next boot:

```
rm -f /etc/sysctl.d/99-spark-attic-minfree.conf
```

## Troubleshooting

`spark-attic doctor` runs every check below and prints a fix line for each problem; start
there.

| What you see | What it means |
|---|---|
| `"state":"refusing"`, why: driver X has no VALIDATED artifact | This driver is not validated yet: section 7. |
| why: ... has not passed mode M on driver X | That mode was not validated on this driver (on 580.178.04 only `heap` is). |
| why: ... does not match its VALIDATED digest | The shim changed after it was validated: validate it again (section 4). |
| why: ... was built for driver X; Y runs | The artifact folder holds another driver's shim: build and validate for Y. |
| why: RM reports no display carveout on this GPU | Nothing to lend: not a GB10, or the driver reserved no carveout. |
| `attic-validate.sh` exit 4 | Something may hold the carveout: stop atticd or the other lender, or give the key. |
| A plugin logs "plugin off" | `SPARK_ATTIC_PLACE=off`, or a setting it could not read; the engine runs without it. |
| A plugin logs that a placement stays off | atticd unreachable, nothing free, or a layout it does not know; the engine runs as without it. |
| No `status.json` | atticd is stopped, or runs with `--no-status-file`. |
| `status.json` with an old `updated` | atticd died without removing it (the file is rewritten every `every_s` seconds). |

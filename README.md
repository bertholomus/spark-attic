# spark-attic

**More memory for CUDA on an NVIDIA GB10: lend it the display carveout.**

The NVIDIA driver sets aside about 2 GiB of a GB10's memory for a display. On a box that is not driving a display
that memory lies outside the kernel's System RAM, so MemAvailable never counts it. spark-attic's `atticd`, a small
root daemon, lends it to CUDA programs as ordinary device memory, in 2 MiB slices. An inference engine's KV cache
gets about 2 GiB more without giving up any other memory.

`atticd` lends nothing while a display is in use, and it re-checks the signals every 2 s; a borrow request
consults the verdict without scanning anything. A display is in use when any of these holds: a process other
than `nvidia-persistenced` holds `/dev/nvidia-modeset`; the GPU lists a graphics (type G) process; a DRM
connector reports `connected` (that last one is the KMS systems; stock DGX OS runs with `nvidia-drm modeset=0`
and has no connector state at all, so the first two carry the verdict there). `nvidia-persistenced` holding
the device open is the idle state - it keeps the driver loaded, it does not light anything up. The verdict is
held, not re-derived from each tick's own instant: the signals are read on their own cadence (the modeset
scan walks at most 256 process dirs a tick, the pmon sample runs every 30 s), so once a display is seen in
use, lending stays blocked continuously while it is up, and the block lifts only after several ticks in a
row (about 6 s) whose readings all say idle. `atticd --probe-display` prints one line per signal, read-only,
and exits 1 when any fires. Limit: our test runs against an emulated head (the readings came from a headless
node and a desktop session on one); a real driven monitor has not been proven, and `atticd` errs on the side
of lending nothing.

- **Measured, not assumed.** On stock DGX OS, in vLLM and in TensorFold, the KV cache grows by what atticd lends,
  replies stay identical at temperature 0, and decode speed stays within run-to-run noise. SGLang does the same in
  an earlier run with kernel 7.0.0-1019-nvidia-64k (tables below).
- **Fails closed, per driver.** NVIDIA's driver interface is not a stable ABI, so spark-attic never guesses a
  structure or an offset. The part that talks to the driver is rebuilt from NVIDIA's open-gpu-kernel-modules
  source at the exact tag of each driver, checked on the node, and enabled only if every check passes. On any
  other driver atticd lends nothing and says why.
- **Nothing to flash or reboot.** No firmware setting, no kernel module, no kernel patch. A process borrows, and
  the memory comes back when the process lets go or exits.
- **Works with what you run.** A vLLM plugin and an SGLang hook (installed, not patched in), a pure-Python client,
  and a carveout source for TensorFold (open upstream as TensorFold #2762 and #2763, not in a TensorFold release yet).

**One lender at a time.** Two lenders of the same carveout would fight over RM's heap, so the
installer refuses to run while another one is listening. It checks the sockets named in
`ATTIC_OTHER_SOCKETS` (a list of Unix-socket paths); set it before the install if your node runs a
lender whose socket is not the one the installer knows about. An empty list means nothing else to
check, and the install says so in one line (a warn, not a pass).

> Headless boxes are the tested case: atticd refuses to lend while a display is connected (see Supported).
One limit to know: atticd re-checks the display before every new lend, not mid-lend. Memory lent
before a display starts stays lent until its holder exits; the display waits, or uses what RM still
holds, until then. New engines that start later get nothing while the display is up.

**Where the numbers come from.** Everything marked *measured* below was measured on a GB10 node
(ASUS Ascent GX10, stock DGX OS) with the release binaries. The engine-version claims are wider:
the plugins are exercised on the node for the versions listed, and checked against further upstream
releases by reading those releases' source on the build host (the drift test) - a source check proves
the code shape, not a run. A release note that says only *source-checked* has never run on a node.

A release is three files: `spark-attic-VERSION-aarch64.tar.gz`, its `.sha256`, and a standalone
`install.sh` with the tarball's sha256 stamped in - the last is what a `curl | bash` run gets, so it
can verify the tarball it downloads with nothing else on disk.


## How it works

The carveout is RM's own memory heap. In **heap** mode (the default) each slice is an ordinary RM allocation from
that heap, exported to the client as a file descriptor (`NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD`) and imported
by CUDA (`cuMemImportFromShareableHandle`, POSIX fd), then mapped like any other CUDA memory. RM counts every holder,
so a lent slice stays out of every other allocation, RM's own scanout buffers included, until the last process
holding it lets go, even if atticd dies. RM keeps 2 MiB of the region for itself; atticd lends the rest (2,044 of
2,046 MiB on driver 580.178.04).

Clients speak one small protocol over a Unix socket (`/run/spark-attic/attic.sock`, mode 0660, a key in
`/etc/spark-attic/key`). A keyless status file, `/run/spark-attic/status.json`, tells dashboards what is lent
([STATUS-FILE.md](STATUS-FILE.md)).

A second mode, **list**, lends the region's physical pages directly. It is not released in 0.1.0: it has never
passed a node check (on 580.178.04 RM uses part of the region, so the check refuses list mode there), and atticd
will not run in it until a driver validates it.

## Supported

| | |
|---|---|
| Hardware | NVIDIA GB10. Tested on an ASUS Ascent GX10. NVIDIA's own DGX Spark uses the same chip, driver and DGX OS; not tested on one yet. |
| Display | Headless boxes are the tested case. atticd re-checks three display signals every 2 s - a non-`nvidia-persistenced` holder of `/dev/nvidia-modeset`, a graphics (type G) process, a DRM connector reporting `connected` - and while any of them says a display is in use it lends nothing, says why in its log and status file, and holds that verdict until the readings say idle for a few seconds. Host-tested with fake signal trees, including intermittent ones; no node run has had a real desktop on it yet. |
| OS | Linux aarch64 with systemd. Tested on DGX OS 7.6.0 (Ubuntu 24.04.5) with its kernels 7.0.0-1024-nvidia and 7.0.0-1024-nvidia-64k, and with kernel 7.0.0-1019-nvidia-64k. DGX OS now offers 7.0.0-1025; tested on 1024. |
| NVIDIA driver | The release artifact ships a shim for 580.178.04, 580.196.07, 595.112.07, 615.71.09, 615.71.10, 615.78.08 and 620.70.06; none is trusted until a node checks it. 580.178.04 is validated on a node (heap mode). install.sh validates the driver your node runs before atticd lends anything and lends nothing if that check fails. Any other driver: atticd lends nothing until `attic-update.sh` builds and validates it. |
| vLLM | 0.24.0, 0.31.0 and 0.32.0 (NGC 26.07 ships 0.24.0, 26.10 ships 0.31.0; the plugins' upstream-source check passes on 0.24.0, 0.32.0 and 0.33.0rc3): the V1 and V2 GPU model runners, without sleep mode or a KV connector. On any other vLLM the KV placement turns itself off with one log line and the KV budget stays as vLLM measured it. |
| SGLang | 0.5.14, 0.5.21 and 0.5.23 (NGC `nvcr.io/nvidia/sglang:26.07-py3` ships 0.5.14; `:26.10-py3` ships 0.5.21). The buffers hook reads the buffer layout from the pool's own `_create_buffers` before the KV budget is raised - separate K and V allocations (0.5.14, 0.5.21) or 0.5.22+'s fused (2, padded, H, D) one - and a pattern it cannot read keeps the KV placement off before anything is raised. The standard MHA KV pool (not MLA, sliding window, Mamba, FP4 KV, memory saver, PD disaggregation or speculative decoding). On any other SGLang the KV placement turns itself off with one log line, the same. |

A hook that fails at server start on a shape the plugin knows crashes the server (S2's rule: the budget was
about to be raised, so a silent fallback would allocate the raise in device memory behind the engine's back);
`attic-preflight.py` exercises the same write the hooks do, so this case is caught before the server starts.
Placement errors detected while the plugin registers (a source that fits no shape, a write the runtime context
refuses) turn the placement off with one log line instead: nothing was promised yet, so there is nothing to
fail loudly about. |
| Python | 3.9 or later, no dependencies (ctypes on `libcuda`); `torch` only for `.tensor()`. |

Where a plugin does not know the layout, it stays off and the engine runs exactly as without it, with one log line.

## Measured on stock DGX OS

ASUS Ascent GX10, DGX OS 7.6.0 (Ubuntu 24.04.5) with kernel 7.0.0-1024-nvidia-64k, driver 580.178.04, spark-attic
installed as INSTALL.md says (attic.service, heap mode, validated). The arms of each run went back to back on one
boot, and KV pool sizes come from each engine's own log.

**TensorFold, Qwen3.8 Flash Next** (bf16 KV, text only, a 262,144-token window, up to 5 streams). TensorFold `main`
8db1ee1a with two of our patches, open upstream as TensorFold #2762 and #2763 and in no TensorFold release
yet (the current release, 1.2.7, has neither): the carveout source `TF_CUDA_CARVEOUT_SOURCE=attic`
and Flash Next's KV cache in the carveout. Replies: 8 prompts at temperature 0 and 3 seeded at temperature 1.0,
compared by text, finish reason and token hash. Speeds in tokens per second: decode the median of 5 runs, prefill the
best of 3.

| Arm | KV pool (tokens) | Replies vs off | Decode 1 stream | Decode 5 streams, drafts off | Prefill 8K | Prefill 32K |
|---|---|---|---|---|---|---|
| off | 224,256 | baseline | 45.6 | 78.9 | 1,929.0 | 2,060.8 |
| attic | 293,376 | identical 8/8, 3/3 | 45.1 | 77.7 | 2,097.8 | 2,061.0 |
| off, again | 224,256 | identical 8/8, 3/3 | 44.9 | 77.4 | 2,085.1 | 2,048.0 |

`attic` adds 69,120 tokens (135 pages of 512): TensorFold put 2,032 of the 2,044 MiB atticd lent into the keys and
values of 3 attention layers, the indexer keys of 7 and the pooled keys of 3. Every `attic` speed lies within the two
off runs or above them: decode 0.4 % (1 stream) and 0.6 % (5 streams) below their mean, prefill 4.5 % (8K) and 0.3 %
(32K) above it. Decode with drafts on, as served, is left out: a stream error in TensorFold's drafting
(`ProposalPaddingPick`) ended most of those runs in every arm, with the carveout and without it.

**vLLM, Qwen3-1.7B** (vLLM 0.24.0 from NGC 26.07, the `kv` placement), as in "Placements in vLLM and SGLang" below:
temperature 0, 8 fixed prompts, the same pinned KV budget of 150,000 tokens in every arm.

| Placement | KV pool (tokens) | Replies vs off | Decode, 1 stream (tok/s) | Decode, 8 streams (tok/s) |
|---|---|---|---|---|
| off | 150,000 | baseline | 52.2 | 483.4 |
| kv | 168,688 | identical 8/8 | 51.9 (-0.4%) | 483.1 (-0.1%) |
| off, again | 150,000 | identical 8/8 | 51.9 (-0.4%) | 483.8 (+0.1%) |

`kv` adds 18,688 tokens, all 2,044 MiB atticd lent.

**vLLM 0.31.0, node window 2** (NGC `nvcr.io/nvidia/vllm:26.10-py3`, the `kv` placement, 2026-11-11, an ASUS Ascent GX10 with the same driver, the same pinned 150,000-token budget; every number here is from the window-2 log, `fleet/spark-attic-review-1110/node-1110/WINDOW-2-LOG.md`): kv replies identical 8/8 with the KV pool at 168,688 tokens = base + 18,688 (+2,044.0 MiB, exactly what atticd lent), weights replies identical 8/8 with embed_tokens (594 MiB) in the carveout, off2 identical 8/8, decode within 1.4% of baseline.

**Memory.** Idle MemAvailable on the same node, the median of 5 reads with no engine running: 119,737 MiB on the
4 KiB-page kernel 7.0.0-1024-nvidia (DGX OS 7.5.0). On the 64 KiB-page kernel 7.0.0-1024-nvidia-64k, 108,209 MiB as
DGX OS sets it up (7.5.0; on 7.6.0 the same kernel had 107,689 MiB available at boot) and 121,147 MiB with
INSTALL.md's optional `vm.min_free_kbytes` file from boot (7.6.0) - the installer writes that file
only with `--with-sysctl` (opt-in; section 3 of INSTALL.md).

SGLang has not run on DGX OS yet; its numbers below are from kernel 7.0.0-1019-nvidia-64k.

## Placements in vLLM and SGLang

ASUS Ascent GX10, kernel 7.0.0-1019-nvidia-64k, driver 580.178.04, Qwen3-1.7B, temperature 0, 8
fixed prompts of 200 tokens each. Every arm runs on the same pinned KV budget of 150,000 tokens, so the carveout is the
only difference. KV pool sizes come from each engine's own startup log.

| Engine | Placement | KV pool (tokens) | Replies vs off | Decode, 1 stream (tok/s) | Decode, 8 streams (tok/s) |
|---|---|---|---|---|---|
| vLLM | off | 150,000 | baseline | 50.8 | 476.3 |
| vLLM | kv | 168,688 | identical 8/8 | 50.2 (-1.2%) | 470.2 (-1.3%) |
| vLLM | weights | 150,000 | identical 8/8 | 50.9 (+0.1%) | 477.5 (+0.3%) |
| vLLM | off, again | 150,000 | identical 8/8 | 50.1 (-1.4%) | 472.2 (-0.9%) |
| SGLang | off | 150,000 | baseline | 50.5 | 478.5 |
| SGLang | kv | 168,688 | identical 8/8 | 50.3 (-0.3%) | 479.0 (+0.1%) |
| SGLang | weights | 150,000 | identical 8/8 | 51.0 (+0.9%) | 481.3 (+0.6%) |
| SGLang | off, again | 150,000 | identical 8/8 | 50.4 (-0.3%) | 476.7 (-0.4%) |

`kv` adds 18,688 tokens (2,044 MiB, everything atticd lent). Its decode speed is within 1.3 % of the first off run;
the second off run differs from the first by up to 1.4 %.
`weights` moves the embedding table (594 MiB on this small model) and frees that much device memory. `kv` is the
default because it adds the whole carveout. The script that made this table is `scripts/attic-bench.py`.

## Quick start

One command on the node, as root (INSTALL.md has the details, and the manual path):

```
curl -fsSL https://github.com/bertholomus/spark-attic/releases/download/v0.1.0/install.sh | sudo bash
```

The installer checks the node first (aarch64 GB10, root, systemd, the driver, a shim for that
driver, no display in use, no other lender, disk), fetches the release tarball and verifies it
against its SHA256SUMS, installs the files, makes the key root-only, validates the running
driver and enables the service only if that passed. It is idempotent (a second run changes
nothing), it rolls back if a step fails, and every step is logged to
`/var/log/spark-attic/install.log`. `install.sh --check` runs only the checks; `--from
<tarball>` installs an offline copy.

See it before you run it: in a clone of this repo, `make demo` runs the same install flow against a
simulated node in a throwaway directory and installs nothing (any Linux box with a C compiler; a Y/N
confirm guards the install step; `bash install.sh --demo --fail display` shows a refusal path).

Then:

```
spark-attic status      # state, MiB lent, clients, driver, validated
spark-attic doctor      # every check with its fix; exit code = problems found
```

Full steps, building from source, driver updates: [INSTALL.md](INSTALL.md).

Then start your vLLM or SGLang container with atticd's socket, the key and the Python package
(the installed copy, `/usr/local/share/spark-attic/python`):

```
docker run --gpus all --ipc=host \
  -v /run/spark-attic:/run/spark-attic -v /etc/spark-attic/key:/etc/spark-attic/key:ro \
  -v /usr/local/share/spark-attic/python:/opt/spark-attic/python:ro \
  -e PYTHONPATH=/opt/spark-attic/python -e SPARK_ATTIC_KEY_FILE=/etc/spark-attic/key \
  nvcr.io/nvidia/vllm:26.07-py3 vllm serve <model> ...
```

The plugin loads through the engine's own plugin entry point and adds the carveout to the KV cache.

| Setting | Meaning |
|---|---|
| `SPARK_ATTIC_PLACE` | `kv` (default), `weights`, `weights,kv`, or `off` |
| `SPARK_ATTIC_WEIGHTS` | For `weights`: parameter names or globs, comma-separated (default `embed_tokens`) |
| `SPARK_ATTIC_MIB` | Use at most this many MiB of the carveout (default: all that is free) |
| `SPARK_ATTIC_SOCK` | atticd's socket (default `/run/spark-attic/attic.sock`) |
| `SPARK_ATTIC_KEY`, `SPARK_ATTIC_KEY_FILE` | The key (required; nothing is compiled in: atticd refuses to start without one, the client refuses to borrow without one) |

From Python directly:

```
import spark_attic
m = spark_attic.borrow()        # the largest free run on the current CUDA device, or None (and one log line)
x = m.tensor(torch.bfloat16)    # zero copy; or m.ptr and m.size
```

The client fails open: no atticd, nothing free, or a failed CUDA step gives `None`, never an exception.

## Driver updates

```
scripts/attic-update.sh 580.196.07 <node>     # on a build host, once the node runs that driver
```

It fetches that driver's tag of NVIDIA's open-gpu-kernel-modules, builds the driver shim against it, stages it on
the node, and runs `attic-validate.sh` there. Until that passes, atticd lends nothing on the new driver and its
status says why. It is safe to run on every driver change, because it only enables what passed on the node.

## One lender at a time

Run only one carveout lender on a node. `attic-validate.sh` refuses while another lender's socket answers; stop
the other one first.

atticd speaks the socket protocol of KindlingAI's dispramd, so clients written for dispramd work unchanged.
Point them at atticd's socket and key.

## Security

- atticd runs as root (it asks the driver for the carveout). Its socket is 0660 (`--socket-group` for non-root
  clients) and every request needs the key.
- The status file has no key and no physical address. Reading it never lets anyone borrow.
- A slice goes to one client at a time and comes back only when RM has it back.
- A lent slice arrives with whatever its previous holder left in it: atticd does not clear it (the RM shim
  offers the daemon no write into a lent slice). Clear it before use: `Pool.borrow()` and the module
  `borrow()` zero the mapping before returning it, and the plugins zero theirs; `Pool.map()` and `Pool.lend()`
  do not.

## Provenance

spark-attic was written from NVIDIA's open-gpu-kernel-modules source (MIT) at each driver's tag, NVIDIA's CUDA
documentation, and the public vLLM and SGLang sources.

atticd speaks the socket protocol of KindlingAI's dispramd, so clients written for dispramd work unchanged.
We learned that protocol from KindlingAI's published dispram client.
spark-attic's code was written separately, and its daemon is built from NVIDIA's open headers.

## Credits

spark-attic stands on work that other people did first.

- **Emi Huang** ([coolbho3k](https://github.com/coolbho3k)) first showed publicly that CUDA can use the display
  memory of a headless GB10, in the NVIDIA developer forum:
  <https://forums.developer.nvidia.com/t/deepseek-v4-1-flash-for-2x-dgx-spark-exl3-3bpw-3m-kv-cache-c6-new-2gb-free-ram-unlock-for-all-gb10s/383583>.
- **ezoushen** ([ezoushen](https://github.com/ezoushen)) put a KV cache in that memory in the GLM-5.3-Flash recipe for
  TensorFold (patch 0072, `DISPLAY_KV_MIB`).
- **Paweł Pierścionek** ([urtho](https://github.com/urtho)) wrote TensorFold's display carveout (TensorFold #487).
  **plotarmordev** ([plotarmordev](https://github.com/plotarmordev)) gave it receipts from a second Spark.
- **Matt Mastracci** ([mmastrac](https://github.com/mmastrac)) of KindlingAI wrote dispram, the carveout lender
  whose socket protocol atticd speaks. dispram is part of
  [Kindling Spark OS](https://github.com/kindlingai/kindling-spark-os).
- **Adam Hillson** ([BadAd84](https://github.com/BadAd84)) connected the GLM-5.3-Flash recipe to dispramd (patch
  0087, `DISPLAY_KV_BACKEND=dispram`).
- **Ash Hart** ([ashhart](https://github.com/ashhart)) and the TensorFold contributors wrote
  [TensorFold](https://github.com/ashhart/TensorFold), the engine in the TensorFold measurements above.
- **Mia's AI Lab** ([MiaAI-Lab](https://github.com/MiaAI-Lab)) wrote the GLM-5.3-Flash and Qwen3.8 Flash Next
  recipes for TensorFold. The TensorFold run above uses the checkpoint and settings of the Qwen3.8 Flash Next recipe.
  MiaAI-Lab also wrote sparkDash, the dashboard that the status file is designed for.
- **Kindling Spark OS** is by Matt Mastracci, with contributions from Chuck
  ([adapt-ai-systems](https://github.com/adapt-ai-systems)) and [audreyt](https://github.com/audreyt). The
  measurements with kernel 7.0.0-1019-nvidia-64k ran on Kindling Spark OS 0.9.5, with its own lender stopped.
- **NVIDIA** publishes the open GPU kernel modules that the driver shim is built from.
- The plugins use the public interfaces of **vLLM** and **SGLang**.

## Licence

`MIT OR Apache-2.0`, at your choice: [LICENSE-MIT](LICENSE-MIT) or [LICENSE-APACHE](LICENSE-APACHE).
Copyright 2026 bertholomus.

The driver shim is compiled against NVIDIA's open-gpu-kernel-modules headers (MIT);
a binary release will carry NVIDIA's notice in `THIRD_PARTY_NOTICES`.

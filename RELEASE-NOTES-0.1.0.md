# spark-attic 0.1.0

> Licence: `MIT OR Apache-2.0`.

The first release. spark-attic lends the display carveout of an NVIDIA GB10 (about 2 GiB that CUDA cannot otherwise
use) to CUDA programs, and adds it to vLLM's and SGLang's KV cache.

## In this release

- **atticd**, a small root daemon. It lends the carveout in 2 MiB slices over a Unix socket that needs a key:
  2,044 of 2,046 MiB on driver 580.178.04. Slices are RM allocations from the carveout heap, imported by CUDA
  through a file descriptor. A lent slice stays its holder's until the last process holding it lets go, even if
  atticd dies.
- **Per-driver safety.** The part that talks to the driver is built from NVIDIA's open-gpu-kernel-modules at that
  driver's own tag (`attic-build.sh`), checked on the node (`attic-validate.sh` with `attic-check`), and enabled only
  for the modes that passed. On any other driver atticd lends nothing and says why. `attic-update.sh` does all of
  it in one step when a node moves to a new driver.
- **Engines.** A vLLM plugin (0.24.0, 0.31.0 and 0.32.0 exercised; the source check also passes on 0.33.0rc3)
  and an SGLang hook (0.5.14, 0.5.21 and 0.5.23), both loaded through the engine's own plugin
  entry point, plus a
  pure-Python client. Placements: `kv` (default), `weights`, `weights,kv`, `off`.
- **Status file** for dashboards, `/run/spark-attic/status.json`: no key needed and none in it; schema 1 is
  documented in STATUS-FILE.md and fixed by a test.
- A systemd unit and an install guide, with an optional `vm.min_free_kbytes` file for DGX OS's 64 KiB-page kernel.

## Fixed in this round (review 1112)

- The release binaries carry no build-host path or name: the tools are built with their paths mapped
  away and stripped, the shims compile from the tree itself, and the build refuses to pack a private
  string (it scans the artifact and stops on a hit).
- The kernel memory drop-in (`vm.min_free_kbytes`) is opt-in: the installer writes it only with
  `--with-sysctl`.
- A failed validate keeps its reports: they are copied to `/var/log/spark-attic/failed-<time>/`
  before the rollback removes anything this run created.
- The installer verifies its download even when no `.sha256` sidecar is next to it: the standalone
  installer carries the tarball's sha256 stamped in at build time.
- An upgrade is refused while memory is still lent (`lent_mib` or `draining_mib` not 0), before the
  service is stopped.
- The display check scans all of `/proc` (no entry budget), so a holder far down the pid order is
  still found; the display unit starts after `display-manager.service`.
- The preflight warns, rather than passes, when there is no other-lender socket to check; the socket
  list is `ATTIC_OTHER_SOCKETS` (INSTALL.md, section 4).
- The SGLang hook screens more engine shapes before it raises the KV pool: HND caches, page-major
  pool classes, post-capture pools, MiniMax-sparse and differing head dims turn the placement off
  with one log line instead of risking a wrong placement.

## Tested

- Driver 580.178.04, heap mode, on an ASUS Ascent GX10: stock DGX OS 7.6.0 (kernel 7.0.0-1024-nvidia-64k) with
  spark-attic installed as INSTALL.md says, and earlier with kernel 7.0.0-1019-nvidia-64k.
- vLLM 0.24.0 (NGC 26.07), Qwen3-1.7B, temperature 0: the KV pool grew from 150,000 to 168,688 tokens (+12.5%, all
  2,044 MiB lent) and all 8 replies stayed identical, on stock DGX OS 7.6.0 under attic.service (decode within 0.4 %
  of the run without it) and with kernel 7.0.0-1019-nvidia-64k (within 1.3 %; two runs without it differed by up to
  1.4 %). README, "Measured on stock DGX OS" and "Placements in vLLM and SGLang".
- vLLM 0.31.0 (NGC 26.10) in node window 2 (2026-11-11, an ASUS Ascent GX10 with the same driver; every
  number here is from the window-2 log, `fleet/spark-attic-review-1110/node-1110/WINDOW-2-LOG.md`): kv
  identical 8/8 with the KV pool at base + 18,688 tokens (+2,044.0 MiB, all atticd lent), weights identical
  8/8 with embed_tokens (594 MiB) moved, decode within 1.4% of baseline.
- TensorFold `main` 8db1ee1a with the carveout source and Flash Next's KV cache in the carveout (our two
  patches, open upstream as TensorFold #2762 and #2763, in no TensorFold release yet; the current
  release, 1.2.7, has neither), Qwen3.8 Flash Next, bf16 KV, text only, on stock DGX OS 7.6.0 under
  attic.service: the KV pool grew from 224,256 to 293,376 tokens (+30.8%; 2,032 of the 2,044 MiB lent), the replies
  stayed identical (8 at temperature 0, 3 seeded at 1.0), decode lay between two runs without it and prefill at or
  above both (README, "Measured on stock DGX OS").
- SGLang 0.5.14 (NGC 26.07), Qwen3-1.7B, temperature 0, with kernel 7.0.0-1019-nvidia-64k: the KV pool grew from
  150,000 to 168,688 tokens, all 8 replies identical, and decode speed within 0.3 % of the run without it (README,
  "Placements in vLLM and SGLang").
- The release artifact ships a shim for 580.178.04, 580.196.07, 595.112.07, 615.71.09, 615.71.10, 615.78.08 and 620.70.06. 580.178.04 is validated on a node (heap mode); a shim is trusted only on the node that checked it, so the others stay unvalidated until a node runs them - install.sh validates the node's own driver before anything is lent.

## Known limits

- Headless boxes are the tested case. atticd guards the display itself: every 2 s it checks three display
  signals (a process holding the display device, a graphics process in `nvidia-smi`, and the DRM connectors) and
  lends nothing while any of them shows a display in use. On a GB10 running a desktop on an emulated monitor,
  it refused 20 of 20 borrows in each of two runs. A physical monitor has not been tested.
- One carveout lender per node.
- atticd re-checks the display before each new lend, not mid-lend: memory lent before a display
  starts stays lent until its holder exits.
- Mode **list** is not released in 0.1.0: it has never passed a node check (on 580.178.04 RM uses part of the
  region itself, so the check refuses list mode there).
- The plugins know vLLM 0.24.0 and 0.32.0 (0.31.0 serves with the 0.32 shape; the source check passes on
  0.24.0, 0.32.0 and 0.33.0rc3) and SGLang 0.5.14, 0.5.21 and 0.5.23, with no version pinned: each plugin picks its hooks by what the running engine really has (the buffers hook reads the pool's own buffer
  layout - separate K/V on 0.5.14 and 0.5.21, fused on 0.5.22+ - before the KV budget is raised, so a
  pool it cannot place is left alone, not caught after the raise). On an
  engine of no known shape the KV placement turns itself off with one log line and the engine runs as it would
  without spark-attic; `scripts/attic-engine-drift.sh` fails the daily check when a new engine release breaks one.
- A plugin that fails after the KV budget was raised stops the engine instead of letting it allocate the raised
  budget in device memory (the budget is only raised once the carveout is actually in hand).
- vLLM: not with sleep mode or a KV connector. SGLang: the standard MHA KV pool only (no MLA, sliding window,
  Mamba, FP4 KV, memory saver, PD disaggregation or speculative decoding). Elsewhere the plugin stays off and says so.
- On kernels with 64 KiB pages, vLLM 0.24.0's default lazy weight load can stall, with or without spark-attic: add
  `--safetensors-load-strategy eager`.
- NVIDIA's own DGX Spark has the same chip, driver and DGX OS but has not been tested; every run so far was on an
  ASUS Ascent GX10.

## Thanks

The full list, with links, is in the README under Credits.

- Emi Huang (coolbho3k), who first showed publicly that CUDA can use the display memory of a headless GB10.
- ezoushen and Adam Hillson (BadAd84), for the display KV patches in the GLM-5.3-Flash recipe for TensorFold.
- Paweł Pierścionek (urtho) and plotarmordev, for TensorFold's display carveout.
- Matt Mastracci and KindlingAI, for dispram. atticd speaks the socket protocol of dispramd, and we learned that
  protocol from their published dispram client.
- Chuck (adapt-ai-systems) and audreyt, contributors to Kindling Spark OS. The measurements with kernel
  7.0.0-1019-nvidia-64k ran on Kindling Spark OS.
- Ash Hart and the TensorFold contributors, for TensorFold.
- Mia's AI Lab (MiaAI-Lab), for the recipes and for sparkDash.
- NVIDIA, for publishing the open GPU kernel modules this is built from.

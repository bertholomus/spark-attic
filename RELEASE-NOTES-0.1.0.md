# spark-attic 0.1.0

> Pre-release notes. v0.1.0 is not tagged yet. Licence: `MIT OR Apache-2.0`.

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
- **Engines.** A vLLM plugin (0.24.0) and an SGLang hook (0.5.14), both loaded through the engine's own plugin entry
  point, plus a pure-Python client. Placements: `kv` (default), `weights`, `weights,kv`, `off`.
- **Status file** for dashboards, `/run/spark-attic/status.json`: no key needed and none in it; schema 1 is
  documented in STATUS-FILE.md and fixed by a test.
- A systemd unit and an install guide, with an optional `vm.min_free_kbytes` file for DGX OS's 64 KiB-page kernel.

## Tested

- Driver 580.178.04, heap mode, on an ASUS Ascent GX10: stock DGX OS 7.6.0 (kernel 7.0.0-1024-nvidia-64k) with
  spark-attic installed as INSTALL.md says, and earlier with kernel 7.0.0-1019-nvidia-64k.
- vLLM 0.24.0 (NGC 26.07), Qwen3-1.7B, temperature 0: the KV pool grew from 150,000 to 168,688 tokens (+12.5%, all
  2,044 MiB lent) and all 8 replies stayed identical, on stock DGX OS 7.6.0 under attic.service (decode within 0.4 %
  of the run without it) and with kernel 7.0.0-1019-nvidia-64k (within 1.3 %; two runs without it differed by up to
  1.4 %). README, "Measured on stock DGX OS" and "Placements in vLLM and SGLang".
- TensorFold `main` 8db1ee1a with the carveout source proposed to it and Flash Next's KV cache in the carveout (two
  patches, not in a TensorFold release yet), Qwen3.8 Flash Next, bf16 KV, text only, on stock DGX OS 7.6.0 under
  attic.service: the KV pool grew from 224,256 to 293,376 tokens (+30.8%; 2,032 of the 2,044 MiB lent), the replies
  stayed identical (8 at temperature 0, 3 seeded at 1.0), decode lay between two runs without it and prefill at or
  above both (README, "Measured on stock DGX OS").
- SGLang 0.5.14 (NGC 26.07), Qwen3-1.7B, temperature 0, with kernel 7.0.0-1019-nvidia-64k: the KV pool grew from
  150,000 to 168,688 tokens, all 8 replies identical, and decode speed within 0.3 % of the run without it (README,
  "Placements in vLLM and SGLang").
- Driver 580.196.07 builds cleanly; it is not checked on a node yet.

## Known limits

- Headless only. No run so far had a display connected, and atticd does not check for one yet. Do not run it on a
  box that drives a monitor.
- One carveout lender per node.
- On 580.178.04 the driver uses part of the region itself, so the node check refuses list mode there.
- vLLM 0.24.0 and SGLang 0.5.14 only. On vLLM 0.32.0 and SGLang 0.5.23 the plugins' own source checks fail, so the
  KV placement turns itself off and the engine runs as it would without spark-attic.
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

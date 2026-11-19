# spark-attic 0.1.6

> Licence: `MIT OR Apache-2.0`.

Documentation for coding agents, and a corrected TensorFold hint. The daemon and the installer's checks are the same
as in 0.1.5.

## In this round

- **AGENTS.md now tells a coding agent how to install spark-attic for a user.** It has the procedure (check the node,
  run the read-only preflight, install, verify, attach an engine only when asked) and the rules an agent follows: use
  the release installer, show the preflight result first, never work around a refusal, ask before `--with-sysctl` or
  before restarting a running server, and never print the key. The contributor instructions (Agent Traffic Control)
  are unchanged and follow in the same file. `CLAUDE.md` points to AGENTS.md for agents that read only that file.
  README's Quick start names AGENTS.md.
- **The TensorFold hint names the settings the open TensorFold pull request uses.** The installer's last lines and
  INSTALL.md, section 6, said `TF_CUDA_CARVEOUT_SOURCE=attic` and `TENSORFOLD_ATTIC_*`. Those names are retired. The
  correct settings are `TF_CUDA_CARVEOUT=1 TF_CUDA_CARVEOUT_SOURCE=daemon
  TF_CUDA_CARVEOUT_SOCKET=/run/spark-attic/attic.sock TF_CUDA_CARVEOUT_KEY_FILE=/etc/spark-attic/key
  TF_CUDA_CARVEOUT_MIB=2044`, on a TensorFold built from that branch (no TensorFold release takes the carveout yet).

## Tested

- The host suite (`make check`) passes on the build host, the installer tests included.
- The agent procedure was run on a GB10 on stock DGX OS: `install.sh --check` from the latest-release URL
  exits 0, and `spark-attic status` reads `ready`, 2,044 MiB, validated.

## Known limits

- Unchanged from 0.1.5 (README, "Known limits").

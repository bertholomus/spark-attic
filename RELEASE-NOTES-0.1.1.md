# spark-attic 0.1.1

> Licence: `MIT OR Apache-2.0`.

A bugfix round on 0.1.0: the first public review's smaller findings, no new features.

## Fixed in this round (review 1112)

- The vLLM plugin treats a first `torch.zeros` as the KV cache's backing only when it covers the whole cache: a smaller allocation is left alone instead of taken as the placement.
- atticd frees the client slot of a connection that connects and never speaks: the daemon sweeps silent connections each second and drops them.
- An allocation larger than the whole ledger is refused with the "no free run" sentence: the size is compared before the 32-bit slice count is cut.
- The Python import hook survives a broken meta-path finder on the engine's side: the module it waits for still loads.
- The installer's preflight says so when the host has no `libcuda.so.1`, instead of letting the modes fail one by one.
- The version is written in one place: the Makefile stamps the daemons, build-release.sh stamps the CLI, the installer, the Python package and its dist-info; and the stamped installer carries its single closing brace.
- Doctor prints a warning with no advice as plain text: no empty parenthetical.
- build-release.sh writes SHA256SUMS world-readable itself: the tarball carries a readable sums file whatever umask the build ran with.
- A finished, failed or rolled-back install leaves no temporary directories behind.
- The `nvidia-smi` pmon sample in atticd carries a 5 s time limit; a failed sample holds the last display verdict.
- attic-check's usage names `--key-file`; INSTALL.md sends the key by file, not on a command line.
- The installer picks its colours after the argument parse: `--plain` wins over `ATTIC_COLOR=always`.
- INSTALL.md's display guard matches the three tested signals, not the old connector-only wording.
- The installer removes a stale `/usr/local/bin/attic-check` from the old manual path.
- The host tests ship their CI workflow: `make check` on both runner architectures, no GPU, no zig and no tag cache needed.
- The shipped tree carries none of the review's shorthand: finding tags and private runbook names are reworded, and the release scan now refuses them.
- Plain output stays ASCII end to end: the steps and the preflight use the CLI's words (`ok`, `FAIL`, `->`), the disk sentence names the 64 MiB the check enforces, a piped `--help` prints the usage, and the CLI with no arguments prints help (the docstring lists `demo` too).

## Tested

- The host suite (`make check`) passes on the build host: the units, the server, the validator, the installer and rollback paths, the CLI, the Python client and plugins, the drift, preflight, mem-reserve and memory-trace tests, and both plugin source checks (vLLM and SGLang, cloned at their tags). No GPU, no zig, no tag cache needed.
- No new node runs in 0.1.1; the measurements in 0.1.0's notes stand for this release.

## Known limits

- Unchanged from 0.1.0 (README, "Known limits"): mode **list** is still unreleased, the plugins keep their engine-shape limits, and the display guard still checks before each new lend, not mid-lend.

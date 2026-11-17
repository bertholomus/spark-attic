# spark-attic 0.1.4

> Licence: `MIT OR Apache-2.0`.

An installer fix on 0.1.3, one line.

## In this round

- **The libcuda preflight check no longer races SIGPIPE.** The check ran `ldconfig -p | grep -q` under `set -o pipefail`: `grep -q` exits on the first match, `ldconfig` can get SIGPIPE (rc 141) while it is still writing, and pipefail turned that into "no libcuda.so.1 on this host" on a host that has the library. One of the four GB10 serving nodes failed the 0.1.3 rollout exactly this way and kept its old atticd. The check now captures the cache once and counts the match, so the pipeline cannot lose to its own early exit. Nothing else changed.

## Tested

- The host suite (`make check`) passes on the build host, the installer tests included (the fake node's `ATTIC_LIBCUDA=present|absent` cases are unchanged).
- The failing host shape is reproduced and fixed: with `pipefail` on, the old check returned 141 roughly one run in five while the library was present; the new check passes every run.

## Known limits

- Unchanged from 0.1.3 (README, "Known limits"): mode **list** is still unreleased, the plugins keep their engine-shape limits, and the display guard still checks before each new lend, not mid-lend.

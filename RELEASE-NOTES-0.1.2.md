# spark-attic 0.1.2

> Licence: `MIT OR Apache-2.0`.

A bugfix round on 0.1.1: the display heap comes back by itself after a client dies, no new features.

## Fixed in this round

- atticd recovers the display heap on its own when a client's process dies while RM is still taking the memory back. The mid-release reading is kept honest and measured again (2 s doubling to 64 s, only while no client holds a lend) until the heap is whole again, which is logged once. The same loop covers a daemon that starts while an engine that just exited is still releasing: a short reading at bring-up is not final either.
- "Whole" is defined once per bring-up as the carveout size RM reports minus the bytes RM keeps for itself (that keep is trusted no further than 16 MiB, so a bring-up that ran mid-release cannot push the reference out of reach of a heap that is coming back). A single measurement is never the reference.
- A heap that has read short for ten minutes gets one WARN line with the numbers, and the measuring goes on. The status file is unchanged: schema 1, the same fields the Fleet panel reads.
- The fake driver can hold a released client's slices for a while (`ATTIC_FAKE_LATE`) and open with the heap already short (`ATTIC_FAKE_STARTSHORT`), and the host suite covers both recovery shapes: a client lends nearly the whole heap and dies mid-release, and a daemon starts while the heap is short.

## Tested

- The two recovery cases fail on 0.1.1 (the short reading stands forever and a re-lend is refused with "only 2 MiB free") and pass on this build; measured on the build host against both trees.
- The host suite (`make check`) passes on the build host: the units, the server (the two new recovery cases included), the validator, the installer and rollback paths, the CLI, the Python client and plugins, the drift, preflight, mem-reserve and memory-trace tests, and both plugin source checks (vLLM and SGLang, cloned at their tags). No GPU, no zig, no tag cache needed.

## Known limits

- Unchanged from 0.1.1 (README, "Known limits"): mode **list** is still unreleased, the plugins keep their engine-shape limits, and the display guard still checks before each new lend, not mid-lend.

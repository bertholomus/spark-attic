# spark-attic 0.1.8

> Licence: `MIT OR Apache-2.0`.

A documentation release: KindlingAI's gr8 engine now borrows from atticd natively. The daemon, installer, client and
plugins are the same code as in 0.1.7; only the version stamp changes.

## In this round

- **gr8 borrows from atticd.** spark-attic support is now in gr8 `main` (kindlingai/gr8 #614, commit `4c825327`,
  2026-11-24 Pacific time). gr8's own carveout client, written for Kindling Spark OS's dispramd, now reads
  a key file, so on stock DGX OS it leases from atticd with three settings:
  `GR8_DISPRAM=1 DISPRAM_SOCKET=/run/spark-attic/attic.sock DISPRAM_KEY_FILE=/etc/spark-attic/key`. README has a new
  "gr8" section with what each setting does (quoted from gr8's `docs/ENV.md`); INSTALL.md section 6 and AGENTS.md
  name the settings too.
- **Which gr8.** #614 is in gr8 `main` only: not in a gr8 release (the newest, v1.2.0, came out before it) and not in a
  published gr8 image yet. Until a gr8 release or image carries it, build gr8 from `main`.
- **Credits.** gr8 and #614 are the KindlingAI team's work: Matt Mastracci (@mmastrac) and Chuck (@ck2084; GitHub
  chuck-ads / adapt-ai-systems), the owners, with Steve (@coffeedev) and Albert Lee.

## Tested

- Docs and the version stamp only. The host suite (`make check`) passes on the build host.
- gr8 with #614 against atticd 0.1.7 on two DGX Sparks (GB10, stock DGX OS, driver 580.178.04), 2026-11-25:
  GLM-5.3-Flash, tensor parallel 2, gr8 `main` `77e82fef` through KindlingAI's GLM-5.3-Flash recipe for two Sparks
  (CAP 2,097,152 tokens, 4 sessions, int4 KV), four boots off/on/off/on with only `GR8_DISPRAM` changed. With the
  lease on, atticd lent 2,044 MiB on each Spark and got all of it back at shutdown; MemAvailable at ready rose from
  10.0-10.6 GiB to 12.1-12.8 GiB (about +2 GiB a Spark) at the same KV pool (the recipe fixes CAP, so the pool does
  not grow; a bigger CAP was not tested). Decode the same at 1 chat (66.8 tok/s), 0.9 % lower on the mean at 4 chats
  (82.2 vs 82.9 tok/s; the lease-off runs alone span 82.3 to 84.0); smoke 14/14 on every boot; replies byte-identical except
  2 of 4 batched greedy replies, whose two wordings also appear with the lease off. README's "gr8" section has the
  table.

## Known limits

- Unchanged from 0.1.7.

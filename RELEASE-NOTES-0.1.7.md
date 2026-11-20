# spark-attic 0.1.7

> Licence: `MIT OR Apache-2.0`.

An installer robustness fix and a community platform record. The daemon is the same as in 0.1.6.

## In this round

- **A Ctrl-C during install validation keeps the validation reports.** If a signal (Ctrl-C, a dropped SSH session)
  arrived while `attic-check` was validating, the installer rolled back at once and the reports of the modes it had
  already tried were removed with it. Now the installer first copies them to
  `/var/log/spark-attic/failed-<UTC stamp>/` (mode 0700), says where they are, and then rolls back. This is the same
  thing the installer already did when validation failed on its own.
- **Community validation: Acer Veriton GN100.** Thanks to @1jsamuel (#3, #4), README now records two headless Acer
  Veriton GN100 / GB10 nodes with 4 KiB pages that passed full validation of 2,044 MiB on 0.1.6. INSTALL.md section 4
  explains that a check can be refused when system-wide free memory moves during early boot, even with zero CUDA
  mismatches. The threshold stays as it is.

## Tested

- The host suite (`make check`) passes on the build host. The new installer test (a TERM during validate exits 130,
  rolls back, and the report survives in the kept 0700 folder) fails on 0.1.6 and passes here.
- The rollout to GB10 nodes on stock DGX OS validates 2,044 MiB on each node.

## Known limits

- Unchanged from 0.1.5 (README, "Known limits").

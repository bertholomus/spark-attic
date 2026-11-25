# Instructions for coding agents

This file has two parts. Read the part that matches your task.

- **Installing spark-attic for a user** on their GB10 / DGX Spark: read "Install for a user".
- **Changing this repository** (issues, pull requests, reviews): read "Contributor instructions".

## Install for a user

spark-attic lends the GB10's display carveout (about 2 GiB that the driver keeps for a display) to CUDA programs on a
headless node. The daemon is `atticd`. The full guide is [INSTALL.md](INSTALL.md). This section is the procedure an
agent follows. Run every step on the node itself, as root (`sudo`).

### Rules

1. Use the release installer. Do not build from source, hand-copy binaries or edit the unit unless the user asks.
2. Run the read-only preflight first. Show the user its result before you install anything.
3. Do not work around a refusal. Each refusal line names its cause and the fix: report it to the user. Do not pass
   `--drm-dir`, stop a display or another carveout lender, or change the driver or kernel settings without the
   user's consent.
4. Ask the user before `--with-sysctl` (it changes a kernel setting from the next boot) and before you restart a
   running inference server to attach it.
5. Never print, copy or paste the key in `/etc/spark-attic/key`. Give it to engines as a mounted file.
6. Report what the node says (`spark-attic status`, `/run/spark-attic/status.json`), not what you expect.

### Steps

1. **Check the node.**
   ```
   uname -m                                   # must print aarch64
   head -n1 /proc/driver/nvidia/version       # the NVIDIA driver must be loaded
   ```
2. **Preflight. It is read-only and writes nothing.**
   ```
   curl -fsSL https://github.com/bertholomus/spark-attic/releases/latest/download/install.sh -o install.sh
   sudo bash install.sh --check               # add --format json for machine-readable output
   ```
   Exit 0 means clean. Exit 1 means problems: each one is a sentence with what to do. A line that starts with `~`
   is a warning, not a failure. "no device-tree model readable" is such a warning: the driver and the carveout probe
   still settle whether the box is a GB10. Without the release tarball, `--check` cannot run the shim and display
   checks yet and says so; the install runs them after the download and before it writes anything (or run
   `--check --from <tarball>` to see them first). If the user runs another carveout lender, set
   `ATTIC_OTHER_SOCKETS` to its socket path so the preflight can see it.
3. **Install.**
   ```
   sudo bash install.sh
   ```
   The installer verifies the release tarball against its checksums, installs, creates a root-only key, validates
   the running driver and then starts `attic.service`. If a step after the first write fails, it rolls back what
   that run changed and says so. A second run changes nothing. A newer release upgrades in place. The log is
   `/var/log/spark-attic/install.log`.
   Offline: download `spark-attic-<version>-aarch64.tar.gz` from the release page, then run
   `sudo bash install.sh --from spark-attic-<version>-aarch64.tar.gz`.
4. **Verify.**
   ```
   spark-attic status                         # state, MiB free and lent, clients, driver, validated
   cat /run/spark-attic/status.json           # the same, keyless (STATUS-FILE.md)
   sudo spark-attic doctor                    # every check with a fix line; the exit code is the number of problems
   ```
   Success is `"state":"ready"`, `"validated":true` and a `size_mib` of about 2,044 on a stock GB10. If the state is
   `refusing`, the `why` field says why. Look it up in INSTALL.md, Troubleshooting.
5. **Attach an engine, only if the user asks.** atticd lends nothing until an engine borrows.
   - **vLLM or SGLang:** mount `/run/spark-attic`, the key file (read-only) and the Python package, and set
     `PYTHONPATH` and `SPARK_ATTIC_KEY_FILE`. INSTALL.md, section 6, has the exact `docker run` line. The plugin
     loads by itself and logs one `spark-attic:` line. `SPARK_ATTIC_PLACE=off` turns it off for a comparison run.
   - **TensorFold:** no TensorFold release takes the carveout yet. It needs a build from the open pull request
     branch; INSTALL.md, section 6, has the settings.
   - **gr8:** only gr8 `main` from kindlingai/gr8 #614 on reads atticd's key file; no gr8 release or published gr8
     image has it yet. Set `GR8_DISPRAM=1 DISPRAM_SOCKET=/run/spark-attic/attic.sock
     DISPRAM_KEY_FILE=/etc/spark-attic/key`. README.md, "gr8", has the details.
   - **Your own CUDA program:** `spark_attic.borrow()` (in `python/spark_attic`) maps memory into the calling
     process only. Another process cannot hand that memory to a running server.

### Driver updates and removal

- After a driver update, atticd lends nothing ("no VALIDATED artifact") until a release carries a shim for that
  driver and it passes validation. When one exists, run `sudo spark-attic update --from <newer tarball>`.
- `sudo spark-attic uninstall` removes everything the installer put down. It keeps the key and the logs;
  `--purge` removes those too. Memory that a process still holds comes back when that process exits.

## Contributor instructions

### Agent Traffic Control

Before starting or resuming an issue, or opening, updating or reviewing a pull request, follow [the ATC contributor skill](skills/atc/SKILL.md).
Read `npx github:ashhart/ATC#v0.9.6 board --json` and `npx github:ashhart/ATC#v0.9.6 status --json`, then check the issue and confirm your claim before implementation; use `npx github:ashhart/ATC#v0.9.6 begin <issue> <files...> --json` and require work-ticket guard readiness before edits and publication.

ATC is pinned to version 0.9.6 for this repository; see CONTRIBUTING.md.

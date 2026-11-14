# Contributing

spark-attic is for the whole DGX Spark / GB10 community. Issues and pull requests are welcome.

## Coding agents and Agent Traffic Control

This repository coordinates issues and pull requests with [Agent Traffic Control](https://github.com/ashhart/ATC) (ATC),
pinned to version 0.9.6 so everyone reads claims the same way. Before starting or resuming work:

1. Read `npx github:ashhart/ATC#v0.9.6 board --json` and `npx github:ashhart/ATC#v0.9.6 status --json` to see existing work and proposals.
2. Run `npx github:ashhart/ATC#v0.9.6 check <issue> --json` and require complete coverage. For `free`, claim with `npx github:ashhart/ATC#v0.9.6 claim <issue> --json` and wait for `claimed` before editing; for `yours`, inspect your existing PR before continuing.
3. Run `npx github:ashhart/ATC#v0.9.6 begin <issue> <files...> --json` once the planned files are known, then require `atc guard <file> --json` readiness before edits and `atc guard --publish --json` before publication.
4. Run `npx github:ashhart/ATC#v0.9.6 overlap --json` before opening or updating a PR and resolve conflicts. Link a fixing PR with `Fixes #<issue>`.
5. Follow the skill's review, presence, dependency and handoff instructions throughout the task; core claims use issue numbers, not PR numbers.

Agents follow [the ATC contributor skill](skills/atc/SKILL.md). Coordination comments contain typed `/atc` commands only.
Areas for split work (`.github/atc.json`): 1 daemon (`src/`, `systemd/`, `sysctl/`), 2 scripts, 3 python, 4 tests, 5 build (`Makefile`), 6 docs.

## Before you open a pull request

- `make check` passes (see INSTALL.md for the upstream engine sources the plugin tests read).
- One change per pull request. New files carry the SPDX line `MIT OR Apache-2.0`; contributions are under the same terms.

`skills/atc/SKILL.md` is from ATC by Ash Hart, Apache-2.0, unchanged.

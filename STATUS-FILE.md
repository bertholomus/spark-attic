# spark-attic status file, schema 1

atticd keeps one small JSON file that says what it is doing, for dashboards and anyone else on the node. It needs
no key and no socket, and it carries no key and no physical address: reading it never lets anyone borrow. Readers
today: a Display carveout card written for sparkDash (not in a sparkDash release yet).

This document is the contract. `tests/test_server.py` (`STATUS_SCHEMA`, case `status_file`) fails if atticd writes
anything else.

## Where and when

- Path: `/run/spark-attic/status.json`, beside the socket (`--status-file PATH` moves it, `--no-status-file` turns it
  off). The unit keeps `/run/spark-attic` at mode 0755, so any user can read the file.
- Mode 0644 whatever atticd's umask. atticd writes `status.json.new` and renames it over the old file, so a reader
  never sees half a file. It is one line of JSON and a newline, under 1 KiB.
- Rewritten whenever a value changes, and at least every `every_s` seconds when nothing does (`--status-every`,
  default 10, at least 2).
- Removed when atticd stops. A candidate atticd (the one `attic-validate.sh` runs beside the real socket) writes none.
- After a crash or a `kill -9` the file stays. That is why `updated` exists (see Reading it).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Always `"spark-attic"`. A reader should check it. |
| `version` | string | atticd's version, such as `"0.1.0"`. Not the schema's version. |
| `state` | string | `"ready"`: atticd lends. `"refusing"`: it lends nothing (wrong or unvalidated driver, a failed bring-up check, no driver). |
| `why` | string | When refusing, the reason in words (for people, not for parsing). `""` when ready. |
| `driver` | string | The NVIDIA driver version atticd found in `/proc/driver/nvidia/version`, such as `"580.178.04"`; `"none"` with no driver loaded. |
| `mode` | string | `"heap"` or `"list"` (`--mode`). |
| `validated` | boolean | `true` when the running driver's artifact has a `VALIDATED` record for this mode. atticd only lends when it is `true`. |
| `validated_at` | string | When `attic-validate.sh` validated it, UTC, `"YYYY-MM-DDThh:mm:ssZ"`. `""` when `validated` is false; `"?"` if its record names no time. |
| `size_mib` | integer | MiB atticd lends from (the carveout, less `--keep-mib`). `0` while refusing at bring-up. |
| `free_mib` | integer | MiB not lent and not returning. |
| `largest_mib` | integer | MiB of the largest free run: the most one request can get now. |
| `lent_mib` | integer | MiB lent to clients now. |
| `draining_mib` | integer | MiB clients gave back that the driver has not yet returned (a process still maps it). |
| `clients` | integer | Clients connected now. |
| `every_s` | integer | The rewrite period in seconds (`--status-every`). |
| `pid` | integer | atticd's process id. |
| `updated` | integer | When this file was written, Unix seconds (the node's clock). |

Sizes are whole MiB, rounded down; atticd lends in 2 MiB slices, so they are exact in practice. `free_mib +
lent_mib + draining_mib` is never more than `size_mib`.

Example (an ASUS Ascent GX10, driver 580.178.04, one vLLM server holding the whole carveout as KV cache):

```
{"name":"spark-attic","version":"0.1.0","state":"ready","why":"","driver":"580.178.04","mode":"heap","validated":true,"validated_at":"2026-11-02T23:55:01Z","size_mib":2044,"free_mib":0,"largest_mib":0,"lent_mib":2044,"draining_mib":0,"clients":1,"every_s":10,"pid":2208001,"updated":1793663870}
```

## Reading it

- No file: atticd is not installed, stopped, or runs with `--no-status-file`. Ask systemd (`systemctl show -p
  ActiveState attic.service`) before saying which.
- `updated` older than three `every_s` (by the node's own clock): atticd is not running, so nothing in the file is
  current. Show it as stale, with no numbers. A reader on another host should read the node's clock with the file
  (as the sparkDash probe does: `date +%s` in the same command) rather than compare with its own.
- `updated` in the future by more than three `every_s`: the clocks disagree; treat it as stale too.
- Unreadable or not JSON, or `name` is not `"spark-attic"`: unknown. Never show a lender as up from a file you could
  not read.
- Read at most a few KiB; the file is far smaller.

## Compatibility

Schema 1 is the field list above. Fields are only ever added, at the end; readers must ignore fields they do not
know. A field is never removed, renamed, or given a new type or a new meaning. A change that needs one of those gets a
new file name (`status-2.json`) written beside this one for at least one release, so a reader can move over.

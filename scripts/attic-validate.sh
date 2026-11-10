#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# attic-validate.sh CANDIDATE_DIR [MODES]: on a GB10 node, as root, with nothing borrowed from the display carveout.
# Runs a candidate atticd from CANDIDATE_DIR (rmshim.so + BUILD-INFO from attic-build.sh) on a socket of its own and
# attic-check against it, once per mode (default "heap list"). Only when a mode passes does it install the artifact as
# $ATTIC_LIB/artifacts/<driver>/ with a VALIDATED naming the modes that passed, and tell a running atticd (SIGHUP).
# A build that failed is never enabled; its reports stay under $ATTIC_LIB/candidates/<driver>/.
#   Exit: 0 validated, 1 every mode failed, 2 usage, 3 the node runs another driver, 4 the carveout is in use.
#   ATTIC_BIN (atticd, attic-check; default ../bin beside this script), ATTIC_LIB (/var/lib/attic), ATTIC_RUN
#   (/run/spark-attic), ATTIC_VERSION_FILE, ATTIC_OTHER_SOCKETS (other lenders' sockets that must be stopped; none by default),
#   ATTIC_CHECK_ARGS / ATTIC_DAEMON_ARGS (extra attic-check / candidate atticd flags), ATTIC_HUP (how to tell atticd),
#   ATTIC_KEY or ATTIC_KEY_FILE (the running atticd's key, to read its status; default the unit's /etc/spark-attic/key,
#   else spark-attic-1), ATTIC_TEST=1 (host tests: no root).
set -uo pipefail
CAND=$(cd "${1:?usage: attic-validate.sh CANDIDATE_DIR [MODES]}" && pwd) || exit 2
MODES=${2:-heap list}
HERE=$(dirname "$(readlink -f "$0")")
BIN=${ATTIC_BIN:-$HERE/../bin}
LIB=${ATTIC_LIB:-/var/lib/attic}
RUN=${ATTIC_RUN:-/run/spark-attic}
VFILE=${ATTIC_VERSION_FILE:-/proc/driver/nvidia/version}
OTHERS=${ATTIC_OTHER_SOCKETS:-}
KEYF=${ATTIC_KEY_FILE:-/etc/spark-attic/key}
KEY=attic-check-$(od -An -N8 -tx8 /dev/urandom | tr -d " ")
say() { echo "attic-validate: $*" >&2; }
value() { sed -n "s/^$1=//p" "$2" | head -n1; }

[ "${ATTIC_TEST:-0}" = 1 ] || [ "$(id -u)" = 0 ] || { say "run as root"; exit 2; }
[ -f "$CAND/rmshim.so" ] && [ -f "$CAND/BUILD-INFO" ] || { say "$CAND is not an attic-build.sh artifact"; exit 2; }
DRIVER=$(value driver "$CAND/BUILD-INFO")
RUNNING=$(head -n1 "$VFILE" 2>/dev/null | grep -o -E '[0-9]+(\.[0-9]+){1,3}' | head -n1)
[ -n "$RUNNING" ] || { say "no NVIDIA driver loaded ($VFILE)"; exit 3; }
[ "$RUNNING" = "$DRIVER" ] || { say "this node runs driver $RUNNING; the candidate is for $DRIVER"; exit 3; }

# awk reads all of ss's output: with pipefail, grep -q could exit early and fail the pipe with SIGPIPE.
listening() { ss -H -x -l 2>/dev/null | awk -v p="$1" '$5 == p { found = 1 } END { exit !found }'; }

# Nothing may hold the carveout: our own atticd refuses (as it does on a new driver) and lends nothing, and no other
# lender runs. A ready atticd could lend during the check (and in mode list it holds the whole region), so stop it first.
# Its status needs its key (the unit's is in a key file); an atticd whose status cannot be read counts as in use.
if [ -S "$RUN/attic.sock" ] && listening "$RUN/attic.sock"; then
  K=${ATTIC_KEY:-}
  if [ -z "$K" ] && [ -r "$KEYF" ]; then K=$(head -n1 "$KEYF" | sed 's/[[:space:]]*$//'); fi
  ST=$("$BIN/attic-check" --status --socket "$RUN/attic.sock" --key "${K:-spark-attic-1}" 2>/dev/null || true)
  if [[ $ST != *'"ok":true'* ]]; then
    say "atticd answers on $RUN/attic.sock but gave no status with its key (ATTIC_KEY, or ATTIC_KEY_FILE: $KEYF)"
    say "stop it for the check (systemctl stop attic.service): ${ST:-no reply}"
    exit 4
  fi
  if [[ $ST == *'"state":"ready"'* ]]; then
    say "atticd on $RUN/attic.sock is ready to lend: stop it for the check (systemctl stop attic.service): $ST"
    exit 4
  fi
  if [[ $ST != *'"lent":0,"draining":0'* ]]; then
    say "atticd on $RUN/attic.sock still lends memory: wait for its clients to exit: $ST"
    exit 4
  fi
fi
for s in $OTHERS; do
  if [ -S "$s" ] && listening "$s"; then
    say "another carveout lender answers on $s: stop it for the check"
    exit 4
  fi
done

DEST=$LIB/candidates/$DRIVER
mkdir -p "$LIB/candidates" "$LIB/artifacts" "$RUN"
chmod 755 "$LIB" "$LIB/candidates" "$LIB/artifacts"
mkdir -p "$DEST"
if [ "$CAND" != "$DEST" ]; then  # earlier FAILED-* reports stay
  rm -f "$DEST/rmshim.so" "$DEST/BUILD-INFO" "$DEST"/check-* "$DEST"/atticd-*
  cp "$CAND/rmshim.so" "$CAND/BUILD-INFO" "$DEST/"
fi
chmod 755 "$DEST"
chmod 644 "$DEST/rmshim.so" "$DEST/BUILD-INFO"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PASSED=()
for m in $MODES; do
  SOCK=$RUN/candidate.sock
  rm -f "$DEST/check-$m.json"
  # shellcheck disable=SC2086
  "$BIN/atticd" --candidate "$DEST" --mode "$m" --socket "$SOCK" --key "$KEY" --state "$RUN/candidate-ledger" \
    --version-file "$VFILE" --retry 3600 ${ATTIC_DAEMON_ARGS:-} >"$DEST/atticd-$m.log" 2>&1 &
  PID=$!
  for _ in $(seq 100); do [ -S "$SOCK" ] && break; sleep 0.1; done
  # shellcheck disable=SC2086
  "$BIN/attic-check" --socket "$SOCK" --key "$KEY" --mode "$m" --report "$DEST/check-$m.json" ${ATTIC_CHECK_ARGS:-} \
    >"$DEST/check-$m.out" 2>&1
  RC=$?
  kill -TERM "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  rm -f "$RUN/candidate-ledger"
  if [ $RC = 0 ]; then
    PASSED+=("$m")
    say "mode $m: attic-check passed ($DEST/check-$m.json)"
  else
    cp "$DEST/check-$m.json" "$DEST/FAILED-$STAMP-$m.json" 2>/dev/null ||
      printf '{"pass":false,"rc":%d,"error":"attic-check wrote no report"}\n' "$RC" >"$DEST/FAILED-$STAMP-$m.json"
    say "mode $m: attic-check failed (rc $RC): $(tail -c 600 "$DEST/check-$m.out")"
  fi
done
[ ${#PASSED[@]} -gt 0 ] || { say "driver $DRIVER: no mode passed; nothing enabled"; exit 1; }

MODELIST=$(IFS=,; echo "${PASSED[*]}")
NEW=$LIB/artifacts/.$DRIVER.new
rm -rf "$NEW"
mkdir -p "$NEW"
cp "$DEST/rmshim.so" "$DEST/BUILD-INFO" "$NEW/"
for m in "${PASSED[@]}"; do cp "$DEST/check-$m.json" "$NEW/"; done
SHA=$(sha256sum "$NEW/rmshim.so" | cut -c1-64)
[ "$SHA" = "$(value shim_sha256 "$DEST/BUILD-INFO")" ] || { say "the shim changed while it was checked"; exit 1; }
cat > "$NEW/VALIDATED" <<V
driver=$DRIVER
shim_sha256=$SHA
modes=$MODELIST
checked=$(date -u +%FT%TZ)
host=$(hostname)
report=$(for m in "${PASSED[@]}"; do printf '%s ' "$LIB/artifacts/$DRIVER/check-$m.json"; done)
attic_commit=$(value attic_commit "$DEST/BUILD-INFO")
V
chmod 755 "$NEW"
chmod 644 "$NEW"/*
rm -rf "$LIB/artifacts/.$DRIVER.old"
[ -d "$LIB/artifacts/$DRIVER" ] && mv "$LIB/artifacts/$DRIVER" "$LIB/artifacts/.$DRIVER.old"
mv "$NEW" "$LIB/artifacts/$DRIVER"
say "driver $DRIVER VALIDATED for $MODELIST: $LIB/artifacts/$DRIVER"
if [ -n "${ATTIC_HUP:-}" ]; then
  eval "$ATTIC_HUP"
elif command -v systemctl >/dev/null && systemctl is-active --quiet attic.service; then
  systemctl reload attic.service
else
  pkill -HUP -x atticd || true
fi
exit 0

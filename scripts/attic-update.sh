#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# attic-update.sh VERSION NODE: spark-attic for NVIDIA driver VERSION, checked on NODE. On this build host: fetch
# VERSION's tag of open-gpu-kernel-modules and build the RM shim against it (attic-build.sh), build our tools for the
# node, stage both on NODE; on NODE, as root: attic-validate.sh, which writes VALIDATED only after attic-check passed.
# So an unvalidated build is never enabled. Run it once NODE runs VERSION (atticd then lends nothing until this passes).
#   ATTIC_SSH (default "ssh -o BatchMode=yes"), ATTIC_SCP, ATTIC_SUDO (default "sudo -n"), ATTIC_STAGE (default: a new
#   mktemp -d /tmp/spark-attic-<VERSION>.XXXXXX on NODE), ATTIC_MODES (default "heap list"), ATTIC_NODE_VERSION_FILE.
#   Exit: attic-validate.sh's (0 validated, 1 failed, 3 another driver, 4 carveout in use), 2 usage, 5 build failed.
set -euo pipefail
V=${1:?usage: attic-update.sh DRIVER_VERSION NODE}
NODE=${2:?usage: attic-update.sh DRIVER_VERSION NODE}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SSH=${ATTIC_SSH:-ssh -o BatchMode=yes}
SCP=${ATTIC_SCP:-scp -q -o BatchMode=yes}
SUDO=${ATTIC_SUDO:-sudo -n}
say() { echo "attic-update: $*" >&2; }

RUNNING=$($SSH "$NODE" head -n1 "${ATTIC_NODE_VERSION_FILE:-/proc/driver/nvidia/version}" | grep -o -E '[0-9]+(\.[0-9]+){1,3}' | head -n1 || true)
[ "$RUNNING" = "$V" ] || { say "$NODE runs driver ${RUNNING:-none}, not $V: check $V once $NODE runs it"; exit 3; }

ART=$("$HERE/scripts/attic-build.sh" "$V") || { say "the shim does not build for $V"; exit 5; }
ZIG=${ATTIC_ZIG:-zig}
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-$HERE/cache/zig-global}
export ZIG_LOCAL_CACHE_DIR=${ZIG_LOCAL_CACHE_DIR:-$HERE/cache/zig-local}
make -s -C "$HERE" OUT=build/aarch64 CC="$ZIG cc -target aarch64-linux-gnu.2.28" build/aarch64/atticd \
  build/aarch64/attic-check || { say "the tools do not build"; exit 5; }

B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
mkdir -p "$B/bin" "$B/scripts" "$B/candidate"
cp "$HERE/build/aarch64/atticd" "$HERE/build/aarch64/attic-check" "$B/bin/"
cp "$HERE/scripts/attic-validate.sh" "$B/scripts/"
cp -r "$ART" "$B/candidate/$V"
if [ -n "${ATTIC_STAGE:-}" ]; then
  STAGE=$ATTIC_STAGE
  $SSH "$NODE" "rm -rf '$STAGE' && mkdir -p -m 700 '$STAGE'"
else
  STAGE=$($SSH "$NODE" mktemp -d "/tmp/spark-attic-$V.XXXXXX")
fi
$SCP -r "$B/." "$NODE:$STAGE/"
set +e
$SSH "$NODE" "$SUDO env ATTIC_BIN='$STAGE/bin' '$STAGE/scripts/attic-validate.sh' '$STAGE/candidate/$V' '${ATTIC_MODES:-heap list}'"
RC=$?
say "$NODE, driver $V: attic-validate.sh exit $RC (staged in $NODE:$STAGE)"
exit $RC

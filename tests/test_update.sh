#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# test_update.sh: attic-update.sh on this build host with a pretend node (its ssh and scp run here). It fetches and
# builds the real NVIDIA tag (network on first use). The node's tools are aarch64, so their check cannot pass on this
# host: the update must stage everything and still enable nothing.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
ok() {
  if [ "$1" = 0 ]; then echo "  ok   $2"; else echo "  FAIL $2"; fails=$((fails + 1)); fi
}
driver() { echo "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  $1  Release Build  (test)" >"$T/version"; }
export ATTIC_SSH=$HERE/tests/fake-ssh.sh ATTIC_SCP=$HERE/tests/fake-scp.sh ATTIC_NODE_VERSION_FILE=$T/version
export ATTIC_SUDO="env ATTIC_TEST=1 ATTIC_VERSION_FILE=$T/version ATTIC_LIB=$T/lib ATTIC_RUN=$T/run ATTIC_OTHER_SOCKETS=$T/none ATTIC_HUP=true"
export ATTIC_STAGE=$T/stage

driver 580.178.04
"$HERE/scripts/attic-update.sh" 580.196.07 pretend-node 2>"$T/e1"
rc=$?
[ $rc = 3 ] && grep -q "runs driver 580.178.04, not 580.196.07" "$T/e1" && [ ! -e "$T/stage" ]
ok $? "a node on another driver: exit 3, nothing staged (rc $rc)"

"$HERE/scripts/attic-update.sh" 580.178.04 pretend-node 2>"$T/e2"
rc=$?
[ $rc = 1 ] && [ ! -e "$T/lib/artifacts/580.178.04" ]
ok $? "built and staged; the check cannot pass here: exit 1, nothing enabled (rc $rc)"
for f in bin/atticd bin/attic-check scripts/attic-validate.sh candidate/580.178.04/rmshim.so \
  candidate/580.178.04/BUILD-INFO; do
  [ -f "$T/stage/$f" ]
  ok $? "staged $f"
done
[ "$(od -An -tx1 -j18 -N2 "$T/stage/bin/atticd" | tr -d ' ')" = b700 ]
ok $? "the node's tools are aarch64"
grep -q '^driver=580.178.04$' "$T/stage/candidate/580.178.04/BUILD-INFO"
ok $? "BUILD-INFO names driver 580.178.04"
grep -q '^carveout_types=-DATTIC_HAS_DISPLAY_FRM_RESERVED=1$' "$T/stage/candidate/580.178.04/BUILD-INFO"
ok $? "580.178.04's ctrl2080fb.h declares DISPLAY_FRM_RESERVED (BUILD-INFO)"
ls "$T/lib/candidates/580.178.04/" 2>/dev/null | grep -q '^FAILED-.*-heap.json$'
ok $? "the failed check left a FAILED report"

# 615.78.08 dropped DISPLAY_FRM_RESERVED and renumbered UEFI from 2 to 1 (carveout_kind.h)
rm -rf "$T/stage"
driver 615.78.08
"$HERE/scripts/attic-update.sh" 615.78.08 pretend-node 2>"$T/e4"
rc=$?
[ $rc = 1 ] && [ -f "$T/stage/candidate/615.78.08/rmshim.so" ] && [ ! -e "$T/lib/artifacts/615.78.08" ]
ok $? "615.78.08 builds and stages; the check cannot pass here: exit 1, nothing enabled (rc $rc)"
grep -q '^carveout_types=-DATTIC_HAS_DISPLAY_FRM_RESERVED=0$' "$T/stage/candidate/615.78.08/BUILD-INFO"
ok $? "615.78.08's ctrl2080fb.h has no DISPLAY_FRM_RESERVED (BUILD-INFO)"

rm -rf "$T/stage"
driver 580.999.99
"$HERE/scripts/attic-update.sh" 580.999.99 pretend-node 2>"$T/e3"
rc=$?
[ $rc = 5 ] && [ ! -e "$T/stage" ]
ok $? "a driver NVIDIA has not tagged: exit 5, nothing staged (rc $rc)"
echo "test_update: $fails failures"
[ $fails = 0 ]

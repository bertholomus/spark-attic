#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# test_carveout_kind.sh: region_kind (src/carveout_kind.h) and scripts/attic-carveout-types.sh against fake
# ctrl2080fb.h headers of both carveout type layouts NVIDIA has shipped: 580.178.04 to 615.71.10 (DISPLAY_FRM 0,
# DISPLAY_FRM_RESERVED 1, UEFI 2) and 615.78.08 (DISPLAY_FRM 0, UEFI 1). Host compiler only (CC, default cc).
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
read -ra CC <<<"${CC:-cc}" # CC may be a command with arguments (zig cc)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
ok() {
  if [ "$1" = 0 ]; then echo "  ok   $2"; else echo "  FAIL $2"; fails=$((fails + 1)); fi
}
# fake DIR NAME=VALUE...: a driver include tree whose ctrl2080fb.h declares the carveout type enum with these
# enumerators, the way NVIDIA's does (NvU32 from nvtypes.h)
fake() {
  local d=$T/$1
  shift
  mkdir -p "$d/ctrl/ctrl2080"
  echo 'typedef unsigned int NvU32;' >"$d/nvtypes.h"
  {
    echo '#include <nvtypes.h>'
    echo 'typedef enum NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE {'
    for e in "$@"; do echo "    NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE_${e%%=*} = ${e#*=},"; done
    echo '} NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE;'
  } >"$d/ctrl/ctrl2080/ctrl2080fb.h"
}
types() { "$HERE/scripts/attic-carveout-types.sh" "${CC[@]}" -std=gnu11 -isystem "$T/$1" 2>"$T/types.err"; }
# kinds TREE FLAG: region_kind of RM types 0, 1, 2, 3 and 0xffffffff, built the way rmshim.c includes it
cat >"$T/probe.c" <<'C'
#include <stdio.h>
#include "ctrl/ctrl2080/ctrl2080fb.h"
#include "shim.h"
#include "carveout_kind.h"
int main(void)
{
    const NvU32 types[] = {0, 1, 2, 3, 0xffffffffu};
    for (unsigned i = 0; i < sizeof types / sizeof *types; i++)
        printf("%s%u", i ? " " : "", (unsigned)region_kind(types[i]));
    printf("\n");
    return 0;
}
C
kinds() {
  rm -f "$T/probe"
  "${CC[@]}" -std=gnu11 -Wall -Wextra -Werror -isystem "$T/$1" -I"$HERE/src" ${2:+"$2"} -o "$T/probe" "$T/probe.c" \
    2>"$T/cc.err" && "$T/probe"
}

fake three DISPLAY_FRM=0 DISPLAY_FRM_RESERVED=1 UEFI=2 # 580.178.04 .. 615.71.10
fake two DISPLAY_FRM=0 UEFI=1                          # 615.78.08
fake no-uefi DISPLAY_FRM=0 DISPLAY_FRM_RESERVED=1
fake no-display DISPLAY_FRM_RESERVED=1 UEFI=2
mkdir -p "$T/missing"
fake broken DISPLAY_FRM=0 UEFI=1
echo 'this is not C;' >>"$T/broken/ctrl/ctrl2080/ctrl2080fb.h"

f3=$(types three) && [ "$f3" = -DATTIC_HAS_DISPLAY_FRM_RESERVED=1 ]
ok $? "3-value header: the build is told DISPLAY_FRM_RESERVED is there ($f3)"
f2=$(types two) && [ "$f2" = -DATTIC_HAS_DISPLAY_FRM_RESERVED=0 ]
ok $? "2-value header: the build is told it is not ($f2)"
for t in no-uefi no-display missing broken; do
  out=$(types "$t")
  rc=$?
  [ $rc = 1 ] && [ -z "$out" ] && grep -q "has no carveout type" "$T/types.err"
  ok $? "header $t: attic-carveout-types refuses, prints no flag (rc $rc)"
done

[ "$(kinds three "$f3")" = "0 1 2 255 255" ]
ok $? "3-value: type 0 display, 1 display-reserved, 2 uefi, 3 and ~0 other"
[ "$(kinds two "$f2")" = "0 2 255 255 255" ]
ok $? "2-value: type 0 display, 1 uefi (type 1 changes meaning), 2 and up other"
[ "$(kinds three -DATTIC_HAS_DISPLAY_FRM_RESERVED=0)" = "0 255 2 255 255" ]
ok $? "a reserved type the flag hides comes out as other (refused), never display"
! kinds two -DATTIC_HAS_DISPLAY_FRM_RESERVED=1 >/dev/null
ok $? "the flag on a header without DISPLAY_FRM_RESERVED does not compile"
! kinds three >/dev/null && grep -q "ATTIC_HAS_DISPLAY_FRM_RESERVED is not set" "$T/cc.err"
ok $? "carveout_kind.h without the flag does not compile"
! grep -Eq '^[[:space:]]*case[[:space:]]+[0-9(]' "$HERE/src/carveout_kind.h"
ok $? "region_kind has no numbered case: every type by its header name"
echo "test_carveout_kind: $fails failures"
[ $fails = 0 ]

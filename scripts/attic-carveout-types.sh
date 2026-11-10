#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# attic-carveout-types.sh CC [ARG...]: the -D flag that tells src/carveout_kind.h which carveout region types this
# driver's own ctrl2080fb.h declares, for attic-build.sh. CC [ARG...] is the shim's compiler with its target and the
# driver's include flags. NVIDIA renumbers that enum between drivers (615.78.08 dropped DISPLAY_FRM_RESERVED, so UEFI
# went from 2 to 1), so the compiler looks each type up by name in the header itself; nothing is inferred from a
# version. DISPLAY_FRM and UEFI must be there, and they also prove the probe works: a DISPLAY_FRM_RESERVED it cannot
# see is one the header does not declare. Prints -DATTIC_HAS_DISPLAY_FRM_RESERVED=1 or =0.
# Exit 1: the header lacks DISPLAY_FRM or UEFI, or does not compile.
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: attic-carveout-types.sh CC [ARG...]" >&2; exit 2; }
CC=("$@")
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
# has TYPE: the header declares NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE_<TYPE> as an integer constant
has() {
  printf '#include "ctrl/ctrl2080/ctrl2080fb.h"\nenum { attic_probe = NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE_%s };\n' \
    "$1" >"$T/$1.c"
  "${CC[@]}" -c -o "$T/$1.o" "$T/$1.c" 2>"$T/$1.err" # (zig cc's -fsyntax-only fails on the .o it never wrote)
}
for t in DISPLAY_FRM UEFI; do
  has "$t" || {
    echo "attic-carveout-types: the driver's ctrl2080fb.h has no carveout type $t, or does not compile:" >&2
    head -n 20 "$T/$t.err" >&2
    exit 1
  }
done
if has DISPLAY_FRM_RESERVED; then echo "-DATTIC_HAS_DISPLAY_FRM_RESERVED=1"; else echo "-DATTIC_HAS_DISPLAY_FRM_RESERVED=0"; fi

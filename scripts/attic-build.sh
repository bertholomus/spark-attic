#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# attic-build.sh VERSION [OUTROOT]: the RM shim for NVIDIA driver VERSION, compiled against that driver's own
# open-gpu-kernel-modules headers (attic-fetch.sh) for the GB10 (aarch64, glibc 2.28 and later), into
# OUTROOT/VERSION/ (default build/artifacts): rmshim.so and BUILD-INFO. An RM struct or number that changed between
# drivers fails here, never on a node. It never writes VALIDATED: only attic-validate.sh does, on a node, after
# attic-check passed.
#   ATTIC_ZIG (the pinned zig 0.17.0), ATTIC_TARGET (default aarch64-linux-gnu.2.28; x86_64-linux-gnu.2.28 for a host build)
set -euo pipefail
V=${1:?usage: attic-build.sh DRIVER_VERSION [OUTROOT]}
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${2:-$HERE/build/artifacts}
OUT=$ROOT/$V
NV=$("$HERE/scripts/attic-fetch.sh" "$V")
ZIG=${ATTIC_ZIG:-zig}
TARGET=${ATTIC_TARGET:-aarch64-linux-gnu.2.28}
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-$HERE/cache/zig-global}
export ZIG_LOCAL_CACHE_DIR=${ZIG_LOCAL_CACHE_DIR:-$HERE/cache/zig-local}
COMMIT=$(git -C "$NV" rev-parse HEAD)
TMP=$OUT.part
rm -rf "$TMP"
mkdir -p "$TMP"
INC=(-isystem "$NV/src/common/sdk/nvidia/inc" -isystem "$NV/kernel-open/common/inc"
  -isystem "$NV/src/nvidia/arch/nvalloc/unix/include")
# The carveout region types this driver's header names (the enum is renumbered between drivers: carveout_kind.h).
TYPES=$("$HERE/scripts/attic-carveout-types.sh" "$ZIG" cc -target "$TARGET" -std=gnu11 "${INC[@]}")
# NVIDIA's headers as system headers: their own warnings are theirs; ours fail the build.
"$ZIG" cc -target "$TARGET" -std=gnu11 -O2 -g -Wall -Wextra -Werror -fPIC -shared -fvisibility=hidden \
  -DATTIC_DRIVER="\"$V\"" -DATTIC_SOURCE="\"NVIDIA open-gpu-kernel-modules $V ($COMMIT)\"" "$TYPES" \
  -I"$HERE/src" "${INC[@]}" -o "$TMP/rmshim.so" "$HERE/src/rmshim.c"
grep -q -a "$V" "$TMP/rmshim.so" || { echo "attic-build: the shim does not carry driver $V" >&2; exit 1; }
SHA=$(sha256sum "$TMP/rmshim.so" | cut -c1-64)
cat > "$TMP/BUILD-INFO" <<INFO
driver=$V
nvidia_tag=$V
nvidia_commit=$COMMIT
nvidia_url=${ATTIC_NVIDIA_URL:-https://github.com/NVIDIA/open-gpu-kernel-modules.git}
target=$TARGET
compiler=zig $("$ZIG" version) cc
carveout_types=$TYPES
shim_sha256=$SHA
attic_commit=$(git -C "$HERE" rev-parse --short=12 HEAD 2>/dev/null || echo none)$(git -C "$HERE" diff --quiet -- src 2>/dev/null || echo -dirty)
built=$(date -u +%FT%TZ)
built_on=$(hostname)
INFO
chmod 755 "$TMP"
chmod 644 "$TMP"/*
rm -rf "$OUT"
mv "$TMP" "$OUT"
echo "$OUT"

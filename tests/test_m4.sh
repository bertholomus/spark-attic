#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# test_m4.sh: a real attic-build.sh build ships no host path and no host name. Builds the shim into a temp
# dir (ATTIC_ZIG from the environment or the repo default), runs strings -a over it and over BUILD-INFO,
# and fails on the builder's own paths or its hostname. Fails on the old build script (before the M4 fix),
# which carried the zig toolchain's include directories and the builder's hostname in built_on.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VERSION=${ATTIC_BUILD_VERSION:-580.178.04}
NV=$HERE/cache/nvidia/$VERSION
# The build compiles against the driver's own headers, fetched into cache/nvidia/<driver>. A tree that
# ships without them (an export: the clone is multi-GB and the exported check must not touch the
# network) cannot run this test; it says so and skips, the way the sglang upstream case does - not a
# failure, not a silent pass. A tree with the source (this working tree) always runs it.
if [ ! -d "$NV" ]; then
    echo "  skip: no driver source in cache/nvidia/$VERSION (an exported tree fetches nothing; run this in a tree that has it)"
    exit 0
fi
# The build needs zig (the exported attic-build.sh defaults to zig on the PATH). A tree with the
# driver source but no zig on the PATH and no ATTIC_ZIG cannot run the build either; skip honestly.
if [ -z "${ATTIC_ZIG:-}" ] && ! command -v zig >/dev/null; then
    echo "  skip: no zig on the PATH and no ATTIC_ZIG (the exported build needs a build host with zig; run this where ATTIC_ZIG names one)"
    exit 0
fi
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
ok() { echo "  ok   $1"; }
bad() { echo "  FAIL $1"; fails=$((fails + 1)); }

OUT=$("$HERE/scripts/attic-build.sh" "$VERSION" "$T/artifacts") || {
    echo "  FAIL attic-build.sh did not build (the source is in cache/nvidia/$VERSION):"
    echo "  FAIL $(tail -1 "$T"/build.log 2>/dev/null || true)"
    echo "test_m4: 1 failure"
    exit 1
}
SHIM=$OUT/rmshim.so
INFO=$OUT/BUILD-INFO

# The builder's own paths: this tree, the zig toolchain, the mount roots, and the hostname. The mount
# roots are assembled at run time so this file does not itself carry the strings it hunts.
HOSTNAME=$(hostname)
ROOTS=$(printf '%s/|%s/|%s/|%s/' "/$(printf '\x74')ank" "/$(printf '\x68')ome" "/$(printf '\x72')oot" "/$(printf '\x74')mp")
LEAKS=$(strings -a "$SHIM" | grep -E "$ROOTS|$HERE|$HOSTNAME" | head -4)
[ -z "$LEAKS" ] && ok "rmshim.so carries no host path or host name" || bad "rmshim.so leaks: $LEAKS"
LEAKS=$(grep -E "$ROOTS|$HOSTNAME" "$INFO" | head -3)
[ -z "$LEAKS" ] && ok "BUILD-INFO carries no host path or host name" || bad "BUILD-INFO leaks: $LEAKS"
grep -q "^built_on=" "$INFO" && ! grep -q "^built_on=$" "$INFO" && ok "built_on is set (a name or unknown)" || bad "built_on missing or empty"

# The release tools build the way release/build-release.sh builds them (the Makefile's flags plus
# CFLAGS_PATHMAP over this tree and the zig installation): atticd and attic-check carry no host
# path either (REVIEW-1112 finding 1 measured 20 and 13 /tank strings in the first tarball).
ZIG=${ATTIC_ZIG:-$(command -v zig)}
ZIGROOT=$(cd "$(dirname "$ZIG")/.." && pwd)
if make -s -C "$HERE" OUT="$T/tools" CC="$ZIG cc -target aarch64-linux-gnu.2.28" \
      CFLAGS_PATHMAP="-ffile-prefix-map=$HERE=. -ffile-prefix-map=$ZIGROOT=. -s" \
      "$T/tools/atticd" "$T/tools/attic-check" 2>"$T/tools.err"; then
  LEAKS=$(strings -a "$T/tools/atticd" "$T/tools/attic-check" | grep -E "$ROOTS|$HERE|$HOSTNAME" | head -4)
  [ -z "$LEAKS" ] && ok "atticd and attic-check carry no host path or host name" \
    || bad "the release tools leak: $LEAKS"
else
  bad "the tools did not build with the release path map: $(tail -1 "$T/tools.err")"
fi

echo "test_m4: $fails failures"
[ "$fails" = 0 ]

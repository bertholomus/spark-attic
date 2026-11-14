#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# build-release.sh VERSION [OUTDIR]: the release artifact spark-attic-VERSION-aarch64.tar.gz,
# ready for a node: atticd and attic-check (aarch64, glibc 2.28+), the driver shims for every
# driver in the compatibility list (each with its BUILD-INFO; never a VALIDATED record: only a
# node's attic-validate.sh writes that), the scripts the node needs, the Python package, the
# unit, the sysctl file, the docs, the licences, THIRD_PARTY_NOTICES and SHA256SUMS. The node
# needs no compiler and no zig.
#
# Reproducible: two runs give the same tarball sha256 - in the same directory or in different
# ones. SOURCE_DATE_EPOCH pins every timestamp (default: the committer date of HEAD);
# attic-build.sh maps the shim's toolchain and source paths away (M4), and the tools build maps
# this tree's and zig's paths away the same way (CFLAGS_PATHMAP), so the binaries carry no build
# path; the tar is sorted with fixed mtimes and owners; gzip -n drops its own timestamp.
#
# The art gate: until build/ART-APPROVED exists,
# the header art (share/header-art.txt, share/header-col.txt) is left OUT and one line says so;
# install.sh and the CLI then print the plain text header. ATTIC_ART_GATE names another marker
# file (the tests).
#   ATTIC_DRIVERS  the compatibility list (default: the seven drivers of README's table)
#   ATTIC_ZIG      the zig binary (default attic-build.sh's)
#   Exit: 0 built, 1 a build failed, 2 usage.
set -euo pipefail

NAME=spark-attic
V=${1:?usage: release/build-release.sh VERSION [OUTDIR]}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUTROOT=${2:-$HERE/build/release}
DRIVERS=${ATTIC_DRIVERS:-580.178.04 580.196.07 595.112.07 615.71.09 615.71.10 615.78.08 620.70.06}
SDE=${SOURCE_DATE_EPOCH:-$(git -C "$HERE" log -1 --pretty=%ct 2>/dev/null || date +%s)}
ART_GATE=${ATTIC_ART_GATE:-$HERE/build/ART-APPROVED}
STAGE=$OUTROOT/stage-$V

say() { printf 'build-release: %s\n' "$*" >&2; }
[[ "$V" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { say "not a product version: $V (x.y.z)"; exit 2; }
export SOURCE_DATE_EPOCH=$SDE

# The published artifact is built from an export (release/KIT.md step 2): the private tree's files
# name our hosts, people and projects, and a build here would pack them. The tests build from a
# staged export too; ATTIC_BUILD_PRIVATE_OK=1 is the override for a deliberate private build.
if [ -e "$HERE/release/export.sh" ] && [ "${ATTIC_BUILD_PRIVATE_OK:-0}" != 1 ]; then
  say "this is the private tree (release/export.sh is here): build the artifact from an export (release/export.sh DIR, then release/build-release.sh in it), or set ATTIC_BUILD_PRIVATE_OK=1 to build here anyway"
  exit 2
fi

# ---- 1. the tools for the node (aarch64), the M4 way: zig, every build path mapped away
ZIG=${ATTIC_ZIG:-zig}
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-$HERE/cache/zig-global}
export ZIG_LOCAL_CACHE_DIR=${ZIG_LOCAL_CACHE_DIR:-$HERE/cache/zig-local}
ZIGROOT=$(cd "$(dirname "$ZIG")/.." && pwd)
# -s strips at link: zig's prebuilt libc startup objects (crt1, csu) carry their own build paths in
# their debug info, which no -ffile-prefix-map of ours reaches (REVIEW-1112 finding 1).
say "building the tools for aarch64 (zig cc, the tree's and zig's paths mapped away, stripped)"
make -s -C "$HERE" OUT=build/release-aarch64 CC="$ZIG cc -target aarch64-linux-gnu.2.28" \
  CFLAGS_PATHMAP="-ffile-prefix-map=$HERE=. -ffile-prefix-map=$ZIGROOT=. -s" \
  build/release-aarch64/atticd build/release-aarch64/attic-check

# ---- 2. the shims, one per driver in the compatibility list
rm -rf "$STAGE"
mkdir -p "$STAGE"
for d in $DRIVERS; do
  say "building the shim for driver $d"
  ART=$("$HERE/scripts/attic-build.sh" "$d" "$STAGE/artifacts-build") ||
    { say "the shim does not build for $d"; exit 1; }
  mkdir -p "$STAGE/artifacts/$d"
  cp "$ART/rmshim.so" "$ART/BUILD-INFO" "$STAGE/artifacts/$d/"
done
rm -rf "$STAGE/artifacts-build"

# ---- 3. the tree: binaries, scripts, python, unit, sysctl, docs, licences, notices
mkdir -p "$STAGE/bin" "$STAGE/scripts" "$STAGE/systemd" "$STAGE/sysctl" "$STAGE/share/spark-attic" \
  "$STAGE/python"
# absolute: this script may run with any cwd (the tests run it from the repo root with the
# export as $HERE); a relative path here once staged the calling tree's stale binary
cp "$HERE/build/release-aarch64/atticd" "$HERE/build/release-aarch64/attic-check" "$STAGE/bin/"
sed "s/^APP_VERSION = .*/APP_VERSION = \"$V\"/" "$HERE/scripts/spark-attic" >"$STAGE/bin/spark-attic"
chmod 755 "$STAGE/bin/spark-attic"
printf '%s\n' "$V" >"$STAGE/VERSION"
sed "s/^VERSION=\${INSTALL_VERSION:-0.1.0}/VERSION=\${INSTALL_VERSION:-$V}/" "$HERE/install.sh" \
  >"$STAGE/install.sh"
chmod 755 "$STAGE/install.sh"
cp "$HERE/scripts/attic-validate.sh" "$STAGE/scripts/"
cp "$HERE/systemd/attic.service" "$STAGE/systemd/"
cp "$HERE/sysctl/99-spark-attic-minfree.conf" "$STAGE/sysctl/"
cp -r "$HERE/python/." "$STAGE/python/"
# the docs: the working tree keeps them under release/, the exported tree at its root
for f in README.md INSTALL.md RELEASE-NOTES-0.1.0.md; do
  src=$HERE/release/$f
  [ -f "$src" ] || src=$HERE/$f
  [ -f "$src" ] && cp "$src" "$STAGE/share/spark-attic/$f"
done
cp "$HERE/STATUS-FILE.md" "$STAGE/share/spark-attic/"
cp "$HERE/LICENSE-MIT" "$HERE/LICENSE-APACHE" "$STAGE/share/spark-attic/"
NOTICES="$STAGE/share/spark-attic/THIRD_PARTY_NOTICES"
{
  echo "spark-attic $V third-party notices"
  echo
  echo "== NVIDIA open-gpu-kernel-modules (MIT): the driver shim is compiled against its headers"
  NV=$HERE/cache/nvidia/${DRIVERS%% *}
  if [ -f "$NV/COPYING" ]; then
    cat "$NV/COPYING"
  else
    echo "(the driver source was not in the local cache; the notice ships with the tagged source:"
    echo "  https://github.com/NVIDIA/open-gpu-kernel-modules at each driver tag, MIT)"
  fi
  echo
  echo "== Zig (MIT): the compiler that built the tools and the shims"
  ZLIC=$(cd "$(dirname "$ZIG")/.." 2>/dev/null && find . -maxdepth 3 -iname "LICENSE*" -path "*zig*" | head -n1)
  if [ -n "${ZLIC:-}" ] && [ -f "$(cd "$(dirname "$ZIG")/.." && pwd)/${ZLIC#./}" ]; then
    cat "$(cd "$(dirname "$ZIG")/.." && pwd)/${ZLIC#./}"
  else
    echo "(zig's MIT licence; https://ziglang.org/download/license.html)"
  fi
} >"$NOTICES"
# the art gate: one line, and no art in the artifact until the gate is lifted
if [ -f "$ART_GATE" ]; then
  cp "$HERE/share/spark-attic/header-art.txt" "$HERE/share/spark-attic/header-col.txt" \
    "$STAGE/share/spark-attic/"
else
  say "the header art is left out until build/ART-APPROVED exists (the repo's graphics gate); the installer prints the plain text header"
fi
# no __pycache__ in the artifact (a .pyc carries the build host's paths)
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
# every staged file root-readable, scripts and binaries executable, no group/other write
chmod -R u=rwX,go=rX "$STAGE"

# the tarball is scanned before it exists: a private string must not become the release
# (finding 1). The scanner and its pattern list name our people, hosts and accounts, so they
# stay in the private tree and never ship in an export: the release build points
# ATTIC_PRIVACY_SCAN at the private scan (KIT.md, publish step). An exported build scans when
# that variable names one, and otherwise says so in one line and continues - a missing scan
# must never fail a public user's build.
SCAN=${ATTIC_PRIVACY_SCAN:-$HERE/release/privacy-scan.sh}
if [ -n "${ATTIC_PRIVACY_SCAN:-}" ] || [ -f "$SCAN" ]; then
  if ! bash "$SCAN" "$STAGE"; then
    say "the privacy scan found hits in the staged artifact: refusing to tar it"
    exit 1
  fi
else
  say "no privacy scan here (a public build)"
fi

# ---- 4. SHA256SUMS, then the tarball: sorted, fixed order, no identities
cd "$STAGE"
find . -type f ! -name SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort |
  xargs sha256sum >SHA256SUMS
mkdir -p "$OUTROOT"
# the tarball's top folder is spark-attic-VERSION (install.sh reads VERSION and SHA256SUMS in it)
cd "$OUTROOT"
rm -f "$NAME-$V-aarch64.tar.gz"
mv "$STAGE" "$NAME-$V"
# --sort=name: entries in locale-C name order; --mtime=@0: no clock; --owner/--group 0: no ids;
# --numeric-owner: no names; gzip -n: no timestamp in the gzip header.
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
  -czf "$NAME-$V-aarch64.tar.gz" "$NAME-$V"
sha256sum "$NAME-$V-aarch64.tar.gz" | awk '{print $1}' >"$NAME-$V-aarch64.tar.gz.sha256"

# The standalone installer is published beside the tarball as its own asset, so a curl|bash run
# never sees the .sha256 sidecar: bake the tarball's sha256 into it and let install.sh's fetch
# check the download against that (REVIEW-1112 finding 6). The tarball's own installer verifies
# against the SHA256SUMS it travels with and needs no stamp.
SUM=$(awk '{print $1}' "$NAME-$V-aarch64.tar.gz.sha256")
sed "s|^SYSCTL_NAME=.*|&\nATTIC_SHA256=$SUM|" "$NAME-$V/install.sh" >"$OUTROOT/install.sh"
chmod 755 "$OUTROOT/install.sh"
rm -rf "$NAME-$V"
say "$NAME-$V-aarch64.tar.gz written in $OUTROOT (SOURCE_DATE_EPOCH=$SDE, sha256 in $NAME-$V-aarch64.tar.gz.sha256 and stamped into install.sh)"

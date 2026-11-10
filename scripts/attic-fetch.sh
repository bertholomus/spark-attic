#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# attic-fetch.sh VERSION: NVIDIA's open-gpu-kernel-modules at tag VERSION (a driver's own release tag), shallow, into
# $ATTIC_CACHE/nvidia/VERSION (default cache/ beside this repo). Prints that directory. The RM shim is built from these
# headers only; the clone is checked to be exactly the tag.
set -euo pipefail
V=${1:?usage: attic-fetch.sh DRIVER_VERSION}
[[ $V =~ ^[0-9]+(\.[0-9]+){1,3}$ ]] || { echo "attic-fetch: not a driver version: $V" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")/.." && pwd)
CACHE=${ATTIC_CACHE:-$HERE/cache}/nvidia
URL=${ATTIC_NVIDIA_URL:-https://github.com/NVIDIA/open-gpu-kernel-modules.git}
D=$CACHE/$V
if [ -d "$D/.git" ] && [ "$(git -C "$D" describe --tags --exact-match 2>/dev/null)" = "$V" ]; then
  echo "$D"
  exit 0
fi
mkdir -p "$CACHE"
rm -rf "$D.part"
git clone -q --depth 1 --branch "$V" "$URL" "$D.part" >&2
if [ "$(git -C "$D.part" describe --tags --exact-match 2>/dev/null)" != "$V" ]; then
  echo "attic-fetch: $URL has no tag $V" >&2
  rm -rf "$D.part"
  exit 1
fi
rm -rf "$D"
mv "$D.part" "$D"
echo "$D"

#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# fake-scp.sh [-r] SOURCE HOST:DEST: copies on this host (host tests of attic-update.sh).
[ "$1" = -r ] && shift
exec cp -r "$1" "${2#*:}"

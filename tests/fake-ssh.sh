#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# fake-ssh.sh HOST COMMAND...: runs COMMAND on this host (host tests of attic-update.sh).
shift
exec bash -c "$*"

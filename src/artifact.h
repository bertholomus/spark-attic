// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * artifact.h: one NVIDIA driver's artifact, <root>/<driver>/:
 *   rmshim.so   the RM shim built against that driver's open-gpu-kernel-modules tag (attic-build.sh)
 *   BUILD-INFO  how it was built (key=value lines)
 *   VALIDATED   written by attic-update.sh only after attic-check passed on a node with that driver (key=value):
 *                 driver=<version>  shim_sha256=<64 hex>  modes=<heap,list: the modes attic-check proved>
 *                 checked=<UTC time>  host=<node>  report=<attic-check's JSON report>
 * atticd loads a shim only from an artifact whose VALIDATED names the running driver, the shim's exact digest and
 * the mode asked for. Every path is checked to be owned by root (or by atticd's own user) and writable by no one else:
 * a root daemon dlopens that file.
 */
#ifndef ATTIC_ARTIFACT_H
#define ATTIC_ARTIFACT_H

#include <stddef.h>

struct artifact {
    char dir[1024];
    char shim[1100];
    char sha256[65];  /* the shim's digest, measured */
    char modes[128];  /* VALIDATED's modes (empty for a candidate) */
    char checked[64]; /* VALIDATED's time */
};

/* The artifact for `driver` under `root`, VALIDATED for `mode`; or, with candidate (a directory), that artifact with
 * no VALIDATED needed (attic-check's run of a build not yet validated). 0, or -1 with why. */
int artifact_find(const char *root, const char *driver, const char *mode, const char *candidate, struct artifact *a,
                  char *why, size_t n);

/* The value of `key` in a key=value file (one per line, '#' comments): 0, or -1. */
int artifact_value(const char *path, const char *key, char *out, size_t cap);

/* Whether a comma list ("heap,list") holds `word`. */
int attic_listed(const char *list, const char *word);

#endif

// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* artifact.c: finding and checking a per-driver artifact (artifact.h). */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "artifact.h"
#include "attic.h"
#include "sha256.h"

int attic_listed(const char *list, const char *word)
{
    size_t w = strlen(word);
    for (const char *p = list; *p;) {
        const char *e = strchr(p, ',');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        if (len == w && strncmp(p, word, w) == 0)
            return 1;
        if (e == NULL)
            break;
        p = e + 1;
    }
    return 0;
}

int artifact_value(const char *path, const char *key, char *out, size_t cap)
{
    char buf[8192];
    if (attic_read_file(path, buf, sizeof buf, NULL) != 0)
        return -1;
    size_t k = strlen(key);
    for (char *line = buf; line && *line;) {
        char *next = strchr(line, '\n');
        if (next)
            *next++ = 0;
        while (*line == ' ' || *line == '\t')
            line++;
        if (*line != '#' && strncmp(line, key, k) == 0 && line[k] == '=') {
            char *v = line + k + 1;
            size_t len = strlen(v);
            while (len && (v[len - 1] == ' ' || v[len - 1] == '\t' || v[len - 1] == '\r'))
                len--;
            if (len >= cap)
                return -1;
            memcpy(out, v, len);
            out[len] = 0;
            return 0;
        }
        line = next;
    }
    return -1;
}

/* Owned by root or by us, and writable by no one else. */
static int trusted(const char *path, int dir, char *why, size_t n)
{
    struct stat s;
    if (stat(path, &s) != 0) {
        snprintf(why, n, "%s: %s", path, strerror(errno));
        return 0;
    }
    if (dir ? !S_ISDIR(s.st_mode) : !S_ISREG(s.st_mode)) {
        snprintf(why, n, "%s is not a %s", path, dir ? "directory" : "regular file");
        return 0;
    }
    if (s.st_uid != 0 && s.st_uid != geteuid()) {
        snprintf(why, n, "%s is owned by uid %u, not root", path, (unsigned)s.st_uid);
        return 0;
    }
    if (s.st_mode & (S_IWGRP | S_IWOTH)) {
        snprintf(why, n, "%s is writable by group or others", path);
        return 0;
    }
    return 1;
}

int artifact_find(const char *root, const char *driver, const char *mode, const char *candidate, struct artifact *a,
                  char *why, size_t n)
{
    memset(a, 0, sizeof *a);
    if (candidate) {
        snprintf(a->dir, sizeof a->dir, "%s", candidate);
    } else {
        if (!trusted(root, 1, why, n))
            return -1;
        if (snprintf(a->dir, sizeof a->dir, "%s/%s", root, driver) >= (int)sizeof a->dir) {
            snprintf(why, n, "artifact path too long");
            return -1;
        }
        if (access(a->dir, F_OK) != 0) {
            snprintf(why, n, "driver %s has no VALIDATED artifact (no %s; run attic-update.sh for it)", driver, a->dir);
            return -1;
        }
    }
    snprintf(a->shim, sizeof a->shim, "%s/rmshim.so", a->dir);
    if (!trusted(a->dir, 1, why, n) || !trusted(a->shim, 0, why, n))
        return -1;
    int rc = sha256_file(a->shim, a->sha256);
    if (rc != 0) {
        snprintf(why, n, "cannot read %s: %s", a->shim, strerror(-rc));
        return -1;
    }
    if (candidate)
        return 0;

    char validated[1200], v[256];
    snprintf(validated, sizeof validated, "%s/VALIDATED", a->dir);
    if (access(validated, F_OK) != 0) {
        snprintf(why, n, "driver %s has no VALIDATED artifact (run attic-update.sh for it)", driver);
        return -1;
    }
    if (!trusted(validated, 0, why, n))
        return -1;
    if (artifact_value(validated, "driver", v, sizeof v) != 0 || strcmp(v, driver) != 0) {
        snprintf(why, n, "%s is for driver %s, not %s", validated, v, driver);
        return -1;
    }
    if (artifact_value(validated, "shim_sha256", v, sizeof v) != 0 || strcmp(v, a->sha256) != 0) {
        snprintf(why, n, "%s does not match its VALIDATED digest (rebuilt since attic-check passed?)", a->shim);
        return -1;
    }
    if (artifact_value(validated, "modes", a->modes, sizeof a->modes) != 0 || !attic_listed(a->modes, mode)) {
        snprintf(why, n, "attic-check has not passed mode %s on driver %s (VALIDATED modes: %s)", mode, driver,
                 a->modes[0] ? a->modes : "none");
        return -1;
    }
    if (artifact_value(validated, "checked", a->checked, sizeof a->checked) != 0)
        snprintf(a->checked, sizeof a->checked, "?");
    return 0;
}

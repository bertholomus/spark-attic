// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* lend.c: the slice ledger (lend.h). */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "attic.h"
#include "lend.h"

#define LEDGER_MAX_SLICES (1u << 20) /* 2 TiB of 2 MiB slices: far past any carveout */

int ledger_init(struct ledger *l, uint64_t base, uint64_t size, uint64_t keep)
{
    memset(l, 0, sizeof *l);
    if (size == 0 || size % ATTIC_SLICE != 0 || keep % ATTIC_SLICE != 0 || size / ATTIC_SLICE > LEDGER_MAX_SLICES)
        return -1;
    l->base = base;
    l->count = (uint32_t)(size / ATTIC_SLICE);
    l->state = calloc(l->count, 1);
    l->who = calloc(l->count, sizeof *l->who);
    if (l->state == NULL || l->who == NULL) {
        ledger_free(l);
        return -1;
    }
    uint64_t kept = keep / ATTIC_SLICE < l->count ? keep / ATTIC_SLICE : l->count;
    for (uint64_t i = l->count - kept; i < l->count; i++)
        l->state[i] = SLICE_KEPT;
    return 0;
}

void ledger_free(struct ledger *l)
{
    free(l->state);
    free(l->who);
    memset(l, 0, sizeof *l);
}

long ledger_fit(const struct ledger *l, uint32_t n)
{
    long best = -1;
    uint32_t best_len = 0;
    if (n == 0)
        return -1;
    for (uint32_t i = 0; i < l->count;) {
        if (l->state[i] != SLICE_FREE) {
            i++;
            continue;
        }
        uint32_t j = i;
        while (j < l->count && l->state[j] == SLICE_FREE)
            j++;
        uint32_t len = j - i;
        if (len >= n && (best < 0 || len < best_len)) {
            best = i;
            best_len = len;
        }
        i = j;
    }
    return best;
}

void ledger_mark(struct ledger *l, uint32_t first, uint32_t n, int state, struct holder who)
{
    for (uint32_t i = first; i < first + n && i < l->count; i++) {
        l->state[i] = (uint8_t)state;
        l->who[i] = state == SLICE_FREE ? (struct holder){0, 0} : who;
    }
}

uint32_t ledger_count(const struct ledger *l, int state)
{
    uint32_t c = 0;
    for (uint32_t i = 0; i < l->count; i++)
        c += l->state[i] == state;
    return c;
}

uint32_t ledger_largest(const struct ledger *l)
{
    uint32_t best = 0, run = 0;
    for (uint32_t i = 0; i < l->count; i++) {
        run = l->state[i] == SLICE_FREE ? run + 1 : 0;
        if (run > best)
            best = run;
    }
    return best;
}

uint32_t ledger_reap(struct ledger *l, int (*alive)(struct holder))
{
    uint32_t freed = 0;
    struct holder last = {0, 0};
    int last_alive = 1;
    for (uint32_t i = 0; i < l->count; i++) {
        if (l->state[i] != SLICE_DRAINING)
            continue;
        if (l->who[i].pid != last.pid || l->who[i].start != last.start) {
            last = l->who[i];
            last_alive = alive(last);
        }
        if (!last_alive) {
            l->state[i] = SLICE_FREE;
            l->who[i] = (struct holder){0, 0};
            freed++;
        }
    }
    return freed;
}

int ledger_save(const struct ledger *l, const char *path)
{
    char tmp[4096];
    if (snprintf(tmp, sizeof tmp, "%s.tmp", path) >= (int)sizeof tmp)
        return -ENAMETOOLONG;
    FILE *f = fopen(tmp, "we");
    if (f == NULL)
        return -errno;
    fprintf(f, "spark-attic ledger 1: base size pid start (runs lent or draining)\n");
    for (uint32_t i = 0; i < l->count;) {
        if (l->state[i] != SLICE_LENT && l->state[i] != SLICE_DRAINING) {
            i++;
            continue;
        }
        uint32_t j = i;
        while (j < l->count && (l->state[j] == SLICE_LENT || l->state[j] == SLICE_DRAINING) &&
               l->who[j].pid == l->who[i].pid && l->who[j].start == l->who[i].start)
            j++;
        fprintf(f, "%#" PRIx64 " %#" PRIx64 " %d %" PRIu64 "\n", l->base + (uint64_t)i * ATTIC_SLICE,
                (uint64_t)(j - i) * ATTIC_SLICE, (int)l->who[i].pid, l->who[i].start);
        i = j;
    }
    int bad = ferror(f) | (fflush(f) != 0) | (fsync(fileno(f)) != 0);
    if (fclose(f) != 0 || bad) {
        unlink(tmp);
        return -EIO;
    }
    if (rename(tmp, path) != 0) {
        int e = errno;
        unlink(tmp);
        return -e;
    }
    return 0;
}

int ledger_load(struct ledger *l, const char *path, int (*alive)(struct holder))
{
    FILE *f = fopen(path, "re");
    if (f == NULL)
        return errno == ENOENT ? 0 : -errno;
    char line[256];
    int runs = 0;
    while (fgets(line, sizeof line, f)) {
        uint64_t base, size, start;
        int pid;
        if (sscanf(line, "%" SCNx64 " %" SCNx64 " %d %" SCNu64, &base, &size, &pid, &start) != 4)
            continue;
        if (base < l->base || size == 0 || base % ATTIC_SLICE != 0 || size % ATTIC_SLICE != 0)
            continue;
        uint64_t first = (base - l->base) / ATTIC_SLICE, n = size / ATTIC_SLICE;
        if (first >= l->count || n > l->count - first)
            continue;
        struct holder h = {(pid_t)pid, start};
        if (!alive(h))
            continue;
        ledger_mark(l, (uint32_t)first, (uint32_t)n, SLICE_DRAINING, h);
        runs++;
    }
    fclose(f);
    return runs;
}

/* Field 3 (state) and field 22 (start time) of /proc/<pid>/stat: 0, or -1. */
static int proc_stat(pid_t pid, char *state, uint64_t *start)
{
    char path[64], buf[2048];
    snprintf(path, sizeof path, "/proc/%d/stat", (int)pid);
    if (attic_read_file(path, buf, sizeof buf, NULL) != 0)
        return -1;
    const char *p = strrchr(buf, ')'); /* the command name may hold spaces and parentheses */
    if (p == NULL || p[1] != ' ')
        return -1;
    p += 2;
    *state = *p;
    for (int field = 3; field < 22; field++) {
        p = strchr(p, ' ');
        if (p == NULL)
            return -1;
        p++;
    }
    *start = strtoull(p, NULL, 10);
    return 0;
}

int attic_alive(struct holder h)
{
    char state;
    uint64_t start;
    if (h.pid <= 0 || proc_stat(h.pid, &state, &start) != 0)
        return 0;
    if (state == 'Z' || state == 'X' || state == 'x')
        return 0; /* exited: its descriptors, and so its RM clients, are closed */
    return h.start == 0 || start == h.start;
}

struct holder attic_holder(pid_t pid)
{
    char state;
    uint64_t start = 0;
    if (proc_stat(pid, &state, &start) != 0)
        start = 0;
    return (struct holder){pid, start};
}

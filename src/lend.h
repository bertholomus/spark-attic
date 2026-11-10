// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * lend.h: atticd's ledger of the carveout's 2 MiB slices. Every slice is free, lent (a connected client holds it),
 * draining (its client's socket closed, but the client's process lives and may still hold the memory through CUDA),
 * or kept (never lent: --keep). A draining slice is free again only once its process is gone.
 *
 * Mode list needs the ledger for safety: RM tracks nothing about a memory list's pages, so only the ledger keeps two
 * clients off the same page, and it is saved to a file so a restarted atticd knows what live processes still hold.
 * Mode heap uses it for accounting only (RM refcounts its heap blocks itself).
 */
#ifndef ATTIC_LEND_H
#define ATTIC_LEND_H

#include <stdint.h>
#include <sys/types.h>

enum { SLICE_FREE = 0, SLICE_LENT = 1, SLICE_DRAINING = 2, SLICE_KEPT = 3 };

/* A process, told apart from a later one with the same pid by its start time (field 22 of /proc/<pid>/stat). */
struct holder {
    pid_t pid;
    uint64_t start;
};

struct ledger {
    uint64_t base;      /* the range's first byte (a physical address in mode list; 0 in mode heap) */
    uint32_t count;     /* slices */
    uint8_t *state;     /* one per slice */
    struct holder *who; /* one per slice: who holds a lent or draining slice */
};

/* A ledger over [base, base+size), its last `keep` bytes kept: 0, or -1 (size not whole slices, or too many). */
int ledger_init(struct ledger *l, uint64_t base, uint64_t size, uint64_t keep);
void ledger_free(struct ledger *l);
/* The first slice of the smallest free run of at least n slices (the lowest among equals), or -1. */
long ledger_fit(const struct ledger *l, uint32_t n);
void ledger_mark(struct ledger *l, uint32_t first, uint32_t n, int state, struct holder who);
/* Slices in a state. */
uint32_t ledger_count(const struct ledger *l, int state);
/* The longest free run, in slices. */
uint32_t ledger_largest(const struct ledger *l);
/* Draining slices whose process is gone become free; the number freed. */
uint32_t ledger_reap(struct ledger *l, int (*alive)(struct holder));
/* Writes the lent and draining runs to path (via a temporary file and rename): 0, or -errno. */
int ledger_save(const struct ledger *l, const char *path);
/* Marks the runs a saved ledger lists as draining, where they fall inside this ledger and their process lives. The
 * runs loaded, or -errno (a missing file is 0 runs). */
int ledger_load(struct ledger *l, const char *path, int (*alive)(struct holder));

/* Whether a holder's process still runs (not gone, not a zombie, not a later process with its pid). */
int attic_alive(struct holder h);
/* The holder for pid now (start 0 when /proc says nothing). */
struct holder attic_holder(pid_t pid);

#endif

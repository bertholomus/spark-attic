// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * fakeshim.c: an RM shim with no GPU behind it, for the host tests. It is built and loaded the way rmshim.so is.
 * Its carveout is ATTIC_FAKE_REGION ("base:size", default 0x280000000:0x80000000, one display region), of which the
 * first ATTIC_FAKE_HELD bytes are in use by "RM" (default 2 MiB: 2,046 MiB free, as on GB10). An exported descriptor
 * is a memfd of the block's size. As RM does, a heap block atticd releases stays taken while a process still has its
 * exported memfd open or mapped (found by name in /proc). ATTIC_FAKE_FAIL=open|carveout|heap|list|export fails that call;
 * ATTIC_FAKE_RAM=1 makes the heap hand out memory past the carveout (a heap that is not the carveout).
 * ATTIC_FAKE_EXTRA ("reserved,uefi,...") lists more carveout regions after the display one, 2 MiB each past its end:
 * reserved is display-reserved, uefi is uefi, any other word a type the shim does not know.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "shim.h"

#ifndef ATTIC_DRIVER
#define ATTIC_DRIVER "999.0.1"
#endif
#define SLICE (2ull << 20)
#define MAX_HANDLES 1024
#define NO_MEMORY 0x51 /* NV_ERR_NO_MEMORY */

struct block {
    int used, ram, released; /* released: atticd let go, a process still refers to its memfd */
    uint64_t first, n, size;
    char name[48]; /* the memfd exported for it */
};

struct attic_rm {
    uint64_t base, size;
    uint8_t *taken; /* one per slice */
    struct block h[MAX_HANDLES];
};

static void say(char *err, size_t n, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
}

static int fails(const char *call)
{
    const char *f = getenv("ATTIC_FAKE_FAIL");
    return f && strstr(f, call) != NULL;
}

static int fake_open(struct attic_rm **out, int minor, char *err, size_t n)
{
    (void)minor;
    if (fails("open")) {
        say(err, n, "fake: open refused");
        return -ENODEV;
    }
    struct attic_rm *rm = calloc(1, sizeof *rm);
    if (rm == NULL)
        return -ENOMEM;
    rm->base = 0x280000000ull;
    rm->size = 0x80000000ull;
    const char *r = getenv("ATTIC_FAKE_REGION");
    if (r && sscanf(r, "%lli:%lli", (long long *)&rm->base, (long long *)&rm->size) != 2) {
        free(rm);
        say(err, n, "fake: bad ATTIC_FAKE_REGION");
        return -EINVAL;
    }
    uint64_t held = getenv("ATTIC_FAKE_HELD") ? strtoull(getenv("ATTIC_FAKE_HELD"), NULL, 0) : SLICE;
    rm->taken = calloc(rm->size / SLICE, 1);
    if (rm->taken == NULL) {
        free(rm);
        return -ENOMEM;
    }
    for (uint64_t i = 0; i < held / SLICE && i < rm->size / SLICE; i++)
        rm->taken[i] = 1;
    *out = rm;
    return 0;
}

static void fake_close(struct attic_rm *rm)
{
    free(rm->taken);
    free(rm);
}

static int fake_carveout(struct attic_rm *rm, struct attic_region *out, int max, int *count, char *err, size_t n)
{
    if (fails("carveout")) {
        say(err, n, "fake: carveout refused");
        return -EIO;
    }
    *count = 0;
    if (max < 1)
        return 0;
    out[0] = (struct attic_region){rm->base, rm->size, ATTIC_REGION_DISPLAY};
    *count = 1;
    uint64_t at = rm->base + rm->size;
    for (const char *p = getenv("ATTIC_FAKE_EXTRA"); p && *p && *count < max; at += SLICE) {
        size_t w = strcspn(p, ",");
        uint32_t kind = w == 8 && strncmp(p, "reserved", w) == 0 ? ATTIC_REGION_DISPLAY_RESERVED
                        : w == 4 && strncmp(p, "uefi", w) == 0   ? ATTIC_REGION_UEFI
                                                                 : ATTIC_REGION_OTHER;
        out[(*count)++] = (struct attic_region){at, SLICE, kind};
        p += w + (p[w] == ',');
    }
    return 0;
}

/* Whether any process we may look at has the memfd `name` open or mapped. */
static int referenced(const char *name)
{
    DIR *p = opendir("/proc");
    if (p == NULL)
        return 0;
    int found = 0;
    struct dirent *e;
    while (!found && (e = readdir(p)) != NULL) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9')
            continue;
        char path[300], line[600];
        snprintf(path, sizeof path, "/proc/%s/maps", e->d_name);
        FILE *f = fopen(path, "r");
        if (f) {
            while (!found && fgets(line, sizeof line, f))
                found = strstr(line, name) != NULL;
            fclose(f);
        }
        snprintf(path, sizeof path, "/proc/%s/fd", e->d_name);
        DIR *d = found ? NULL : opendir(path);
        if (d) {
            struct dirent *x;
            while (!found && (x = readdir(d)) != NULL) {
                char link[600];
                snprintf(line, sizeof line, "%s/%s", path, x->d_name);
                ssize_t len = readlink(line, link, sizeof link - 1);
                if (len > 0) {
                    link[len] = 0;
                    found = strstr(link, name) != NULL;
                }
            }
            closedir(d);
        }
    }
    closedir(p);
    return found;
}

/* Frees the released blocks no process refers to any more. */
static void collect(struct attic_rm *rm)
{
    for (int i = 0; i < MAX_HANDLES; i++) {
        struct block *b = &rm->h[i];
        if (b->used && b->released && !referenced(b->name)) {
            memset(rm->taken + b->first, 0, b->n);
            memset(b, 0, sizeof *b);
        }
    }
}

static int new_handle(struct attic_rm *rm, uint32_t *handle, char *err, size_t n)
{
    for (int i = 0; i < MAX_HANDLES; i++)
        if (!rm->h[i].used) {
            *handle = 0xfa000000u + (uint32_t)i;
            return i;
        }
    say(err, n, "fake: out of handles");
    return -1;
}

static int fake_heap(struct attic_rm *rm, uint64_t size, uint32_t *handle, char *err, size_t n)
{
    if (fails("heap")) {
        say(err, n, "fake: heap refused");
        return NO_MEMORY;
    }
    if (size == 0 || size % SLICE) {
        say(err, n, "fake: size not whole slices");
        return -EINVAL;
    }
    collect(rm);
    int k = new_handle(rm, handle, err, n);
    if (k < 0)
        return -ENOSPC;
    uint64_t want = size / SLICE, slices = rm->size / SLICE;
    for (uint64_t i = 0; i + want <= slices; i++) {
        uint64_t j = 0;
        while (j < want && !rm->taken[i + j])
            j++;
        if (j == want) {
            memset(rm->taken + i, 1, want);
            rm->h[k] = (struct block){1, 0, 0, i, want, size, ""};
            return 0;
        }
        i += j;
    }
    if (getenv("ATTIC_FAKE_RAM")) {
        rm->h[k] = (struct block){1, 1, 0, 0, 0, size, ""};
        return 0;
    }
    say(err, n, "fake: NV_ERR_NO_MEMORY");
    return NO_MEMORY;
}

static int fake_list(struct attic_rm *rm, uint64_t base, uint64_t size, uint32_t *handle, char *err, size_t n)
{
    if (fails("list")) {
        say(err, n, "fake: list refused");
        return -EPERM;
    }
    if (size == 0 || size % SLICE || base % SLICE || base < rm->base || base + size > rm->base + rm->size) {
        say(err, n, "fake: range outside the carveout");
        return -EINVAL;
    }
    int k = new_handle(rm, handle, err, n);
    if (k < 0)
        return -ENOSPC;
    rm->h[k] = (struct block){1, 0, 0, 0, 0, size, ""}; /* a list takes no heap room */
    return 0;
}

static struct block *find(struct attic_rm *rm, uint32_t handle)
{
    uint32_t i = handle - 0xfa000000u;
    return i < MAX_HANDLES && rm->h[i].used ? &rm->h[i] : NULL;
}

static int fake_export(struct attic_rm *rm, uint32_t handle, int *fd, char *err, size_t n)
{
    static unsigned serial;
    struct block *b = find(rm, handle);
    if (b == NULL || b->released || fails("export")) {
        say(err, n, "fake: export refused");
        return -EINVAL;
    }
    snprintf(b->name, sizeof b->name, "attic-fake-%d-%08u", (int)getpid(), ++serial);
    int f = memfd_create(b->name, MFD_CLOEXEC);
    if (f < 0 || ftruncate(f, (off_t)b->size) != 0) {
        if (f >= 0)
            close(f);
        say(err, n, "fake: memfd: %s", strerror(errno));
        return -errno;
    }
    *fd = f;
    return 0;
}

static int fake_release(struct attic_rm *rm, uint32_t handle, char *err, size_t n)
{
    struct block *b = find(rm, handle);
    if (b == NULL || b->released) {
        say(err, n, "fake: no handle %#x", handle);
        return -EINVAL;
    }
    if (b->n && b->name[0] && referenced(b->name)) {
        b->released = 1; /* RM keeps it while its memfd is open or mapped somewhere */
        return 0;
    }
    if (b->n)
        memset(rm->taken + b->first, 0, b->n);
    memset(b, 0, sizeof *b);
    return 0;
}

static const char *fake_detail(struct attic_rm *rm)
{
    (void)rm;
    return "fake pages";
}

static const struct attic_shim shim = {
    .abi = ATTIC_SHIM_ABI,
    .driver = ATTIC_DRIVER,
    .source = "spark-attic fakeshim (no GPU)",
    .open = fake_open,
    .close = fake_close,
    .carveout = fake_carveout,
    .heap = fake_heap,
    .list = fake_list,
    .export_fd = fake_export,
    .release = fake_release,
    .detail = fake_detail,
};

__attribute__((visibility("default"))) const struct attic_shim *attic_shim(void)
{
    return &shim;
}

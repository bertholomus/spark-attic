// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * fakecuda.c: a stand-in libcuda.so.1 for the host tests, so attic-check and the Python client run end to end with no
 * GPU. It implements only the driver calls they make: an import maps the lent descriptor (fakeshim's memfd) into this
 * process, cuMemCreate maps anonymous shared memory, and copies and memsets go to and from those mappings (across
 * neighbouring mappings, as in one reserved range). A handle's memory lives until it is released and no longer mapped.
 * ATTIC_FAKECUDA_CORRUPT=1 flips one bit on every read back (attic-check must then fail); ATTIC_FAKECUDA_NOIMPORT=1
 * refuses every import; ATTIC_FAKECUDA_NOCREATE=1 refuses every cuMemCreate.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define FAKE_ERROR 999
#define MAX_MAPS 256

struct imported {
    uint8_t *mem;
    size_t size;
    int maps, released;
};

struct mapping {
    unsigned long long va;
    size_t size;
    struct imported *h;
};

static struct mapping maps[MAX_MAPS];
static unsigned long long next_va = 0x7f0000000000ull;
static void *ctx_stack[16];
static int ctx_depth;

int cuInit(unsigned flags)
{
    (void)flags;
    return 0;
}

int cuDeviceGet(int *dev, int ordinal)
{
    *dev = ordinal;
    return 0;
}

int cuDevicePrimaryCtxRetain(void **ctx, int dev)
{
    *ctx = (void *)(uintptr_t)(0x1000 + dev);
    return 0;
}

int cuDevicePrimaryCtxRelease_v2(int dev)
{
    (void)dev;
    return 0;
}

int cuCtxSetCurrent(void *ctx)
{
    if (ctx_depth == 0)
        ctx_depth = 1;
    ctx_stack[ctx_depth - 1] = ctx;
    return 0;
}

int cuCtxGetCurrent(void **ctx)
{
    *ctx = ctx_depth ? ctx_stack[ctx_depth - 1] : NULL;
    return 0;
}

int cuCtxPushCurrent_v2(void *ctx)
{
    if (ctx_depth == 16)
        return FAKE_ERROR;
    ctx_stack[ctx_depth++] = ctx;
    return 0;
}

int cuCtxPopCurrent_v2(void **ctx)
{
    if (ctx_depth == 0)
        return FAKE_ERROR;
    ctx_depth--;
    if (ctx)
        *ctx = ctx_stack[ctx_depth];
    return 0;
}

int cuCtxSynchronize(void)
{
    return 0;
}

int cuGetErrorName(int err, const char **name)
{
    *name = err == FAKE_ERROR ? "FAKE_ERROR" : "FAKE_UNKNOWN";
    return 0;
}

int cuMemGetAllocationGranularity(size_t *g, const void *prop, int option)
{
    (void)prop;
    (void)option;
    *g = 2u << 20;
    return 0;
}

static unsigned long long handle_of(uint8_t *m, size_t size)
{
    struct imported *h = calloc(1, sizeof *h);
    h->mem = m;
    h->size = size;
    return (unsigned long long)(uintptr_t)h;
}

int cuMemImportFromShareableHandle(unsigned long long *handle, void *os_handle, int type)
{
    int fd = (int)(uintptr_t)os_handle;
    struct stat s;
    if (type != 1 || getenv("ATTIC_FAKECUDA_NOIMPORT") || fstat(fd, &s) != 0 || s.st_size <= 0)
        return FAKE_ERROR;
    void *m = mmap(NULL, (size_t)s.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED)
        return FAKE_ERROR;
    *handle = handle_of(m, (size_t)s.st_size);
    return 0;
}

int cuMemCreate(unsigned long long *handle, size_t size, const void *prop, unsigned long long flags)
{
    (void)prop;
    if (flags != 0 || size == 0 || size % (2u << 20) != 0 || getenv("ATTIC_FAKECUDA_NOCREATE"))
        return FAKE_ERROR;
    void *m = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED)
        return FAKE_ERROR;
    *handle = handle_of(m, size);
    return 0;
}

static void drop(struct imported *h)
{
    if (h->released && h->maps == 0) {
        munmap(h->mem, h->size);
        free(h);
    }
}

int cuMemAddressReserve(unsigned long long *ptr, size_t size, size_t align, unsigned long long addr,
                        unsigned long long flags)
{
    (void)align;
    (void)addr;
    (void)flags;
    *ptr = next_va;
    next_va += (size + (64ull << 20)) & ~((2ull << 20) - 1);
    return 0;
}

int cuMemMap(unsigned long long va, size_t size, size_t offset, unsigned long long handle, unsigned long long flags)
{
    struct imported *h = (struct imported *)(uintptr_t)handle;
    (void)flags;
    if (offset != 0 || size > h->size || h->released)
        return FAKE_ERROR;
    for (int i = 0; i < MAX_MAPS; i++)
        if (maps[i].h == NULL) {
            maps[i] = (struct mapping){va, size, h};
            h->maps++;
            return 0;
        }
    return FAKE_ERROR;
}

int cuMemSetAccess(unsigned long long va, size_t size, const void *desc, size_t count)
{
    (void)va;
    (void)size;
    (void)desc;
    return count == 1 ? 0 : FAKE_ERROR;
}

/* The bytes at va, and how many of the n asked for lie in the same mapping (0: va is not mapped). */
static uint8_t *at(unsigned long long va, size_t n, size_t *run)
{
    for (int i = 0; i < MAX_MAPS; i++)
        if (maps[i].h && va >= maps[i].va && va < maps[i].va + maps[i].size) {
            size_t left = (size_t)(maps[i].va + maps[i].size - va);
            *run = n < left ? n : left;
            return maps[i].h->mem + (va - maps[i].va);
        }
    *run = 0;
    return NULL;
}

int cuMemcpyHtoD_v2(unsigned long long dst, const void *src, size_t n)
{
    const uint8_t *s = src;
    while (n) {
        size_t run;
        uint8_t *p = at(dst, n, &run);
        if (p == NULL)
            return FAKE_ERROR;
        memcpy(p, s, run);
        dst += run, s += run, n -= run;
    }
    return 0;
}

int cuMemcpyDtoH_v2(void *dst, unsigned long long src, size_t n)
{
    uint8_t *d = dst;
    size_t total = n;
    while (n) {
        size_t run;
        uint8_t *p = at(src, n, &run);
        if (p == NULL)
            return FAKE_ERROR;
        memcpy(d, p, run);
        src += run, d += run, n -= run;
    }
    if (getenv("ATTIC_FAKECUDA_CORRUPT") && total)
        ((uint8_t *)dst)[total / 2] ^= 1;
    return 0;
}

int cuMemsetD8_v2(unsigned long long dst, unsigned char v, size_t n)
{
    while (n) {
        size_t run;
        uint8_t *p = at(dst, n, &run);
        if (p == NULL)
            return FAKE_ERROR;
        memset(p, v, run);
        dst += run, n -= run;
    }
    return 0;
}

int cuMemUnmap(unsigned long long va, size_t size)
{
    for (int i = 0; i < MAX_MAPS; i++)
        if (maps[i].h && maps[i].va == va && maps[i].size == size) {
            struct imported *h = maps[i].h;
            maps[i].h = NULL;
            h->maps--;
            drop(h);
            return 0;
        }
    return FAKE_ERROR;
}

int cuMemAddressFree(unsigned long long va, size_t size)
{
    (void)va;
    (void)size;
    return 0;
}

int cuMemRelease(unsigned long long handle)
{
    struct imported *h = (struct imported *)(uintptr_t)handle;
    if (h->released)
        return FAKE_ERROR;
    h->released = 1;
    drop(h);
    return 0;
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * attic-check: proves an atticd (usually a candidate build for a new driver, on its own socket) on a real GB10 before
 * attic-update.sh may mark that driver VALIDATED. Exit 0 only when every phase passes; the JSON report says what each
 * one measured.
 *
 *   status     atticd is ready, in the mode asked for, for the running driver; its carveout regions are all of a type
 *              the shim knows (display, display-reserved, uefi) and its display regions are not System RAM
 *   full       the largest slice it lends, imported through CUDA VMM (cuMemImportFromShareableHandle, POSIX fd),
 *              mapped, then every 32-bit word written and read back with two patterns (an address hash and its
 *              complement); MemFree must not drop by a quarter of the slice while it is held (System RAM would);
 *              mode list: the slice's physical range is not System RAM (/proc/iomem)
 *   double     while that slice is held, a second client sees it gone and cannot get it again
 *   isolation  two clients' slices, imported side by side, hold their own patterns (no overlap)
 *   reclaim    a client killed while holding its slice: everything is free again once the process is gone
 *   drain      a client that closes its socket but keeps the memory mapped: atticd does not lend it again while that
 *              process lives, and does once it is gone
 *   refusal    a wrong key and a malformed request are refused
 *
 * Each CUDA phase runs in a child process of its own (the parent never initializes CUDA, so it may fork). CUDA is
 * loaded at run time (libcuda.so.1), so the tool builds with no CUDA headers.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "attic.h"
#include "proto.h"

#define CHECK_VERSION "0.1.0"
#define CHUNK (64ull << 20)
#define MiB(x) ((unsigned long long)((x) >> 20))

static const char *sock_path = ATTIC_DEFAULT_SOCKET, *key = ATTIC_DEFAULT_KEY, *mode = "heap";
static const char *iomem = "/proc/iomem", *meminfo = "/proc/meminfo", *report = NULL;
static int device_ordinal = 0;

/* ---- a small client ---- */

static int dial(void)
{
    struct sockaddr_un a = {.sun_family = AF_UNIX};
    snprintf(a.sun_path, sizeof a.sun_path, "%s", sock_path);
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0)
        return -1;
    struct timeval tv = {30, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (connect(fd, (struct sockaddr *)&a, sizeof a) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* Sends body, reads one reply into out; an fd that came with it into *rfd (else -1). 0, or -1. */
static int ask(int fd, const char *body, char *out, size_t cap, int *rfd)
{
    if (rfd)
        *rfd = -1;
    if (send(fd, body, strlen(body), MSG_NOSIGNAL) != (ssize_t)strlen(body))
        return -1;
    struct iovec iov = {out, cap - 1};
    union {
        char buf[CMSG_SPACE(4 * sizeof(int))];
        struct cmsghdr align;
    } u;
    struct msghdr m;
    memset(&m, 0, sizeof m);
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = u.buf;
    m.msg_controllen = sizeof u.buf;
    ssize_t r = recvmsg(fd, &m, MSG_CMSG_CLOEXEC);
    if (r <= 0)
        return -1;
    out[r] = 0;
    for (struct cmsghdr *h = CMSG_FIRSTHDR(&m); h; h = CMSG_NXTHDR(&m, h)) {
        if (h->cmsg_level != SOL_SOCKET || h->cmsg_type != SCM_RIGHTS)
            continue;
        int n = (int)((h->cmsg_len - CMSG_LEN(0)) / sizeof(int));
        for (int i = 0; i < n; i++) {
            int x;
            memcpy(&x, CMSG_DATA(h) + i * sizeof(int), sizeof x);
            if (rfd && *rfd < 0)
                *rfd = x;
            else
                close(x);
        }
    }
    return 0;
}

static int request(int fd, const char *op, uint64_t size, char *out, size_t cap, int *rfd)
{
    char body[256];
    if (size)
        snprintf(body, sizeof body, "{\"key\":\"%s\",\"op\":\"%s\",\"size\":%" PRIu64 "}", key, op, size);
    else
        snprintf(body, sizeof body, "{\"key\":\"%s\",\"op\":\"%s\"}", key, op);
    return ask(fd, body, out, cap, rfd);
}

/* A field of one of atticd's replies (top level; occurrences inside escaped strings are skipped). */
static const char *field(const char *body, const char *name)
{
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", name);
    for (const char *p = strstr(body, pat); p; p = strstr(p + 1, pat))
        if (p == body || p[-1] != '\\')
            return p + strlen(pat);
    return NULL;
}

static int jnum(const char *body, const char *name, uint64_t *v)
{
    const char *p = field(body, name);
    if (p == NULL || *p < '0' || *p > '9')
        return -1;
    *v = strtoull(p, NULL, 10);
    return 0;
}

static int jtrue(const char *body, const char *name)
{
    const char *p = field(body, name);
    return p && strncmp(p, "true", 4) == 0;
}

static void jstr(const char *body, const char *name, char *out, size_t cap)
{
    const char *p = field(body, name);
    size_t k = 0;
    if (p && *p == '"')
        for (p++; *p && *p != '"' && k + 1 < cap; p++) {
            if (*p == '\\' && p[1])
                p++;
            out[k++] = *p;
        }
    out[k] = 0;
}

struct info {
    uint64_t size, free, largest, base;
    int has_base;
};

static int get_info(int fd, struct info *in, char *raw, size_t cap)
{
    if (request(fd, "info", 0, raw, cap, NULL) != 0 || !jtrue(raw, "ok"))
        return -1;
    memset(in, 0, sizeof *in);
    if (jnum(raw, "size", &in->size) || jnum(raw, "free", &in->free) || jnum(raw, "largest", &in->largest))
        return -1;
    in->has_base = jnum(raw, "base", &in->base) == 0;
    return 0;
}

/* ---- CUDA, loaded at run time ---- */

typedef int CUresult;
typedef unsigned long long CUdeviceptr, CUhandle;
typedef struct {
    int type, requested_handle_types;
    struct {
        int type, id;
    } location;
    void *win32;
    unsigned char alloc_flags[8];
} MemProp; /* CUmemAllocationProp */
typedef struct {
    struct {
        int type, id;
    } location;
    int flags;
} MemAccess; /* CUmemAccessDesc, 12 bytes */

static struct {
    CUresult (*init)(unsigned);
    CUresult (*device_get)(int *, int);
    CUresult (*ctx_retain)(void **, int);
    CUresult (*ctx_set)(void *);
    CUresult (*import)(CUhandle *, void *, int);
    CUresult (*granularity)(size_t *, const MemProp *, int);
    CUresult (*reserve)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUhandle, unsigned long long);
    CUresult (*access)(CUdeviceptr, size_t, const void *, size_t);
    CUresult (*htod)(CUdeviceptr, const void *, size_t);
    CUresult (*dtoh)(void *, CUdeviceptr, size_t);
    CUresult (*sync)(void);
    CUresult (*unmap)(CUdeviceptr, size_t);
    CUresult (*addr_free)(CUdeviceptr, size_t);
    CUresult (*release)(CUhandle);
    CUresult (*err_name)(CUresult, const char **);
} cu;

static char cu_err[256];

static int cuda_load(void)
{
    void *l = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (l == NULL) {
        snprintf(cu_err, sizeof cu_err, "cannot load libcuda.so.1: %s", dlerror());
        return -1;
    }
#define SYM(f, name)                                                                                                   \
    do {                                                                                                               \
        *(void **)&cu.f = dlsym(l, name);                                                                              \
        if (cu.f == NULL) {                                                                                            \
            snprintf(cu_err, sizeof cu_err, "libcuda has no %s", name);                                                \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)
    SYM(init, "cuInit");
    SYM(device_get, "cuDeviceGet");
    SYM(ctx_retain, "cuDevicePrimaryCtxRetain");
    SYM(ctx_set, "cuCtxSetCurrent");
    SYM(import, "cuMemImportFromShareableHandle");
    SYM(granularity, "cuMemGetAllocationGranularity");
    SYM(reserve, "cuMemAddressReserve");
    SYM(map, "cuMemMap");
    SYM(access, "cuMemSetAccess");
    SYM(htod, "cuMemcpyHtoD_v2");
    SYM(dtoh, "cuMemcpyDtoH_v2");
    SYM(sync, "cuCtxSynchronize");
    SYM(unmap, "cuMemUnmap");
    SYM(addr_free, "cuMemAddressFree");
    SYM(release, "cuMemRelease");
    SYM(err_name, "cuGetErrorName");
#undef SYM
    return 0;
}

static int cu_ok(CUresult r, const char *what)
{
    if (r == 0)
        return 1;
    const char *name = "?";
    if (cu.err_name)
        cu.err_name(r, &name);
    snprintf(cu_err, sizeof cu_err, "%s: %s (%d)", what, name, r);
    return 0;
}

static int cuda_up(int *dev)
{
    void *ctx;
    if (cuda_load() != 0)
        return -1;
    if (!cu_ok(cu.init(0), "cuInit") || !cu_ok(cu.device_get(dev, device_ordinal), "cuDeviceGet") ||
        !cu_ok(cu.ctx_retain(&ctx, *dev), "cuDevicePrimaryCtxRetain") || !cu_ok(cu.ctx_set(ctx), "cuCtxSetCurrent"))
        return -1;
    return 0;
}

struct mapped {
    CUhandle h;
    CUdeviceptr va;
    size_t size;
};

/* Imports a lent descriptor and maps it read-write on the device. */
static int import_map(int dev, int fd, size_t size, struct mapped *mp)
{
    MemProp prop;
    memset(&prop, 0, sizeof prop);
    prop.type = 1;                   /* CU_MEM_ALLOCATION_TYPE_PINNED */
    prop.requested_handle_types = 1; /* CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR */
    prop.location.type = 1;          /* CU_MEM_LOCATION_TYPE_DEVICE */
    prop.location.id = dev;
    size_t g = 0;
    if (!cu_ok(cu.granularity(&g, &prop, 0), "cuMemGetAllocationGranularity"))
        return -1;
    if (g == 0 || ATTIC_SLICE % g != 0) {
        snprintf(cu_err, sizeof cu_err, "granularity %zu does not divide 2 MiB", g);
        return -1;
    }
    memset(mp, 0, sizeof *mp);
    mp->size = size;
    if (!cu_ok(cu.import(&mp->h, (void *)(uintptr_t)fd, 1), "cuMemImportFromShareableHandle"))
        return -1;
    if (!cu_ok(cu.reserve(&mp->va, size, ATTIC_SLICE, 0, 0), "cuMemAddressReserve")) {
        cu.release(mp->h);
        return -1;
    }
    if (!cu_ok(cu.map(mp->va, size, 0, mp->h, 0), "cuMemMap")) {
        cu.addr_free(mp->va, size);
        cu.release(mp->h);
        return -1;
    }
    MemAccess acc;
    memset(&acc, 0, sizeof acc);
    acc.location.type = 1;
    acc.location.id = dev;
    acc.flags = 3; /* CU_MEM_ACCESS_FLAGS_PROT_READWRITE */
    if (!cu_ok(cu.access(mp->va, size, &acc, 1), "cuMemSetAccess")) {
        cu.unmap(mp->va, size);
        cu.addr_free(mp->va, size);
        cu.release(mp->h);
        return -1;
    }
    return 0;
}

static void unmap_release(struct mapped *mp)
{
    cu.unmap(mp->va, mp->size);
    cu.addr_free(mp->va, mp->size);
    cu.release(mp->h);
}

static uint32_t pattern(uint64_t word, uint32_t seed, int invert)
{
    uint32_t v = (uint32_t)(((word + seed) * 0x9E3779B97F4A7C15ull) >> 32);
    return invert ? ~v : v;
}

static void fill_host(uint32_t *buf, uint64_t first_word, size_t words, uint32_t seed, int invert)
{
    for (size_t i = 0; i < words; i++)
        buf[i] = pattern(first_word + i, seed, invert);
}

struct pass {
    uint64_t mismatches, first_bad;
    double write_s, read_s;
};

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* Writes every word of the range, then reads every word back. 0, or -1 on a CUDA error. */
static int write_read(const struct mapped *mp, uint32_t seed, int invert, uint32_t *buf, uint32_t *back, struct pass *p)
{
    memset(p, 0, sizeof *p);
    p->first_bad = UINT64_MAX;
    double t0 = now_s();
    for (size_t off = 0; off < mp->size; off += CHUNK) {
        size_t len = mp->size - off < CHUNK ? mp->size - off : CHUNK;
        fill_host(buf, off / 4, len / 4, seed, invert);
        if (!cu_ok(cu.htod(mp->va + off, buf, len), "cuMemcpyHtoD"))
            return -1;
    }
    if (!cu_ok(cu.sync(), "cuCtxSynchronize"))
        return -1;
    p->write_s = now_s() - t0;
    t0 = now_s();
    for (size_t off = 0; off < mp->size; off += CHUNK) {
        size_t len = mp->size - off < CHUNK ? mp->size - off : CHUNK;
        if (!cu_ok(cu.dtoh(back, mp->va + off, len), "cuMemcpyDtoH"))
            return -1;
        fill_host(buf, off / 4, len / 4, seed, invert);
        if (memcmp(buf, back, len) != 0)
            for (size_t i = 0; i < len / 4; i++)
                if (buf[i] != back[i]) {
                    if (p->first_bad == UINT64_MAX)
                        p->first_bad = off + i * 4;
                    p->mismatches++;
                }
    }
    p->read_s = now_s() - t0;
    return 0;
}

/* ---- the report ---- */

static char out[65536];
static size_t outlen;

static void emit(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void emit(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(out + outlen, sizeof out - outlen, fmt, ap);
    va_end(ap);
    if (k > 0)
        outlen += (size_t)k < sizeof out - outlen ? (size_t)k : sizeof out - outlen - 1;
}

static void emit_text(const char *name, const char *s)
{
    char t[1024];
    attic_json_text(s, t, sizeof t);
    emit("\"%s\":\"%s\"", name, t);
}

/* ---- phases that run in a child ---- */

/* full + double: prints one JSON object, exits 0 when both pass. */
static int phase_full(void)
{
    char raw[ATTIC_REPLY_MAX + 1];
    struct info in, in2;
    int dev, ok = 1;
    if (cuda_up(&dev) != 0) {
        emit("{\"full\":{\"pass\":false,");
        emit_text("why", cu_err);
        emit("}}");
        return 1;
    }
    uint32_t *buf = malloc(CHUNK), *back = malloc(CHUNK);
    int c = dial();
    if (buf == NULL || back == NULL || c < 0 || get_info(c, &in, raw, sizeof raw) != 0 || in.largest == 0) {
        emit("{\"full\":{\"pass\":false,");
        emit_text("why", c < 0 ? "cannot connect" : raw);
        emit("}}");
        return 1;
    }
    uint64_t m0 = attic_mem_free(meminfo);
    int fd;
    if (request(c, "alloc", in.largest, raw, sizeof raw, &fd) != 0 || !jtrue(raw, "ok") || fd < 0) {
        emit("{\"full\":{\"pass\":false,");
        emit_text("why", raw);
        emit("}}");
        return 1;
    }
    struct mapped mp;
    if (import_map(dev, fd, in.largest, &mp) != 0) {
        close(fd);
        emit("{\"full\":{\"pass\":false,\"size\":%" PRIu64 ",", in.largest);
        emit_text("why", cu_err);
        emit("}}");
        return 1;
    }
    close(fd);
    struct pass a, b;
    int rc = write_read(&mp, 0x5eed, 0, buf, back, &a);
    if (rc == 0)
        rc = write_read(&mp, 0x5eed, 1, buf, back, &b);
    uint64_t m1 = attic_mem_free(meminfo);
    int64_t dropped = (int64_t)m0 - (int64_t)m1;
    char ramwhy[256] = "";
    int ram = 0;
    if (in.has_base)
        ram = attic_iomem_touches_ram(iomem, in.base, in.largest, ramwhy, sizeof ramwhy);
    int full_ok = rc == 0 && a.mismatches == 0 && b.mismatches == 0 && m0 && m1 &&
                  dropped < (int64_t)(in.largest / 4) && (!in.has_base || ram == 0);
    ok &= full_ok;
    emit("{\"full\":{\"pass\":%s,\"size\":%" PRIu64 ",\"free_before\":%" PRIu64 ",\"words\":%" PRIu64
         ",\"mismatches\":%" PRIu64 ",\"first_bad\":%" PRId64 ",\"write_gbs\":%.2f,\"read_gbs\":%.2f,"
         "\"memfree_drop_mib\":%" PRId64 ",\"base\":%" PRId64 ",\"system_ram\":%d,",
         full_ok ? "true" : "false", in.largest, in.free, in.largest / 4 * 2, a.mismatches + b.mismatches,
         a.mismatches ? (int64_t)a.first_bad : b.mismatches ? (int64_t)b.first_bad : (int64_t)-1,
         rc == 0 ? 2.0 * (double)in.largest / (a.write_s + b.write_s) / 1e9 : 0.0,
         rc == 0 ? 2.0 * (double)in.largest / (a.read_s + b.read_s) / 1e9 : 0.0, dropped / (1 << 20),
         in.has_base ? (int64_t)in.base : (int64_t)-1, in.has_base ? ram : -2);
    emit_text("why", rc != 0 ? cu_err : ramwhy);
    emit("},");

    /* double: a second client while the slice is held */
    int c2 = dial();
    int dbl_ok = 0, refused = 0;
    char raw2[ATTIC_REPLY_MAX + 1] = "";
    if (c2 >= 0 && get_info(c2, &in2, raw2, sizeof raw2) == 0) {
        int fd2 = -1;
        uint64_t want = in.largest; /* the same bytes again: must not be had while held */
        if (in2.free < want && request(c2, "alloc", want, raw2, sizeof raw2, &fd2) == 0) {
            refused = !jtrue(raw2, "ok") && fd2 < 0;
            if (fd2 >= 0)
                close(fd2);
        }
        dbl_ok = in2.free + in.largest == in.free && refused;
    }
    ok &= dbl_ok;
    emit("\"double\":{\"pass\":%s,\"free_while_held\":%" PRIu64 ",\"refused\":%s,", dbl_ok ? "true" : "false",
         c2 >= 0 ? in2.free : 0, refused ? "true" : "false");
    emit_text("reply", raw2);
    emit("}}");
    if (c2 >= 0)
        close(c2);
    unmap_release(&mp);
    close(c);
    return ok ? 0 : 1;
}

/* isolation: two clients, two slices, side by side. */
static int phase_isolation(void)
{
    char raw[ATTIC_REPLY_MAX + 1] = "";
    int dev;
    if (cuda_up(&dev) != 0) {
        emit("{\"isolation\":{\"pass\":false,");
        emit_text("why", cu_err);
        emit("}}");
        return 1;
    }
    const size_t size = 2 * ATTIC_SLICE;
    uint32_t *buf = malloc(size), *back = malloc(size);
    int ca = dial(), cb = dial(), fa = -1, fb = -1;
    struct info ia, ib;
    int got = buf && back && ca >= 0 && cb >= 0 && request(ca, "alloc", size, raw, sizeof raw, &fa) == 0 &&
              fa >= 0 && get_info(ca, &ia, raw, sizeof raw) == 0 &&
              request(cb, "alloc", size, raw, sizeof raw, &fb) == 0 && fb >= 0 &&
              get_info(cb, &ib, raw, sizeof raw) == 0;
    struct mapped ma, mb;
    if (!got || import_map(dev, fa, size, &ma) != 0 || import_map(dev, fb, size, &mb) != 0) {
        emit("{\"isolation\":{\"pass\":false,");
        emit_text("why", got ? cu_err : raw);
        emit("}}");
        return 1;
    }
    close(fa);
    close(fb);
    fill_host(buf, 0, size / 4, 0xaaaa, 0);
    int rc = cu_ok(cu.htod(ma.va, buf, size), "HtoD a");
    fill_host(buf, 0, size / 4, 0xbbbb, 0);
    rc = rc && cu_ok(cu.htod(mb.va, buf, size), "HtoD b") && cu_ok(cu.sync(), "sync");
    uint64_t bad = 0;
    if (rc && cu_ok(cu.dtoh(back, ma.va, size), "DtoH a")) {
        fill_host(buf, 0, size / 4, 0xaaaa, 0);
        bad += memcmp(buf, back, size) != 0;
    } else {
        rc = 0;
    }
    if (rc && cu_ok(cu.dtoh(back, mb.va, size), "DtoH b")) {
        fill_host(buf, 0, size / 4, 0xbbbb, 0);
        bad += memcmp(buf, back, size) != 0;
    } else {
        rc = 0;
    }
    int pass = rc && bad == 0 && ib.free + size == ia.free;
    emit("{\"isolation\":{\"pass\":%s,\"slices_differ\":%s,\"free_after_a\":%" PRIu64 ",\"free_after_b\":%" PRIu64
         ",",
         pass ? "true" : "false", bad == 0 ? "true" : "false", ia.free, ib.free);
    emit_text("why", rc ? "" : cu_err);
    emit("}}");
    unmap_release(&ma);
    unmap_release(&mb);
    close(ca);
    close(cb);
    return pass ? 0 : 1;
}

/* hold: takes the largest slice, maps it, writes it, says HELD, and waits to be killed. With close_first, closes its
 * socket before saying HELD (the memory stays mapped). */
static int phase_hold(int close_first)
{
    char raw[ATTIC_REPLY_MAX + 1];
    struct info in;
    int dev, fd;
    struct mapped mp;
    int c = dial();
    if (cuda_up(&dev) != 0 || c < 0 || get_info(c, &in, raw, sizeof raw) != 0 ||
        request(c, "alloc", in.largest, raw, sizeof raw, &fd) != 0 || fd < 0 ||
        import_map(dev, fd, in.largest, &mp) != 0) {
        printf("FAILED %s\n", cu_err[0] ? cu_err : raw);
        fflush(stdout);
        return 1;
    }
    close(fd);
    uint32_t *buf = malloc(CHUNK);
    fill_host(buf, 0, CHUNK / 4, 0xcccc, 0);
    cu.htod(mp.va, buf, in.largest < CHUNK ? in.largest : CHUNK);
    cu.sync();
    if (close_first)
        close(c);
    printf("HELD %" PRIu64 "\n", in.largest);
    fflush(stdout);
    for (;;)
        pause();
}

/* ---- the parent ---- */

static int run_child(const char *phase, char *res, size_t cap)
{
    int p[2];
    if (pipe2(p, O_CLOEXEC) != 0)
        return -1;
    pid_t pid = fork();
    if (pid == 0) {
        dup2(p[1], 1);
        char *argv[] = {"attic-check", "--phase", (char *)phase, "--socket", (char *)sock_path, "--key", (char *)key,
                        "--mode", (char *)mode, "--iomem", (char *)iomem, "--meminfo", (char *)meminfo, NULL};
        execv("/proc/self/exe", argv);
        _exit(127);
    }
    close(p[1]);
    size_t k = 0;
    ssize_t r;
    while (k + 1 < cap && (r = read(p[0], res + k, cap - 1 - k)) > 0)
        k += (size_t)r;
    res[k] = 0;
    close(p[0]);
    int st;
    waitpid(pid, &st, 0);
    while (k && (res[k - 1] == '\n' || res[k - 1] == ' '))
        res[--k] = 0;
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128;
}

/* Starts a holder child; returns its pid once it says HELD (its line in line), or -1. */
static pid_t start_holder(int close_first, char *line, size_t cap)
{
    int p[2];
    if (pipe2(p, O_CLOEXEC) != 0)
        return -1;
    pid_t pid = fork();
    if (pid == 0) {
        dup2(p[1], 1);
        char *argv[] = {"attic-check", "--phase", close_first ? "hold-close" : "hold", "--socket", (char *)sock_path,
                        "--key", (char *)key, NULL};
        execv("/proc/self/exe", argv);
        _exit(127);
    }
    close(p[1]);
    size_t k = 0;
    char ch;
    while (k + 1 < cap && read(p[0], &ch, 1) == 1 && ch != '\n')
        line[k++] = ch;
    line[k] = 0;
    close(p[0]);
    if (strncmp(line, "HELD ", 5) != 0) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        return -1;
    }
    return pid;
}

/* The free bytes atticd reports, from a fresh connection: 0 and -1 on failure. */
static int free_now(uint64_t *f)
{
    char raw[ATTIC_REPLY_MAX + 1];
    struct info in;
    int c = dial();
    int rc = c >= 0 && get_info(c, &in, raw, sizeof raw) == 0 ? 0 : -1;
    if (c >= 0)
        close(c);
    if (rc == 0)
        *f = in.free;
    return rc;
}

static int wait_free(uint64_t want, double seconds, uint64_t *last)
{
    double end = now_s() + seconds;
    do {
        if (free_now(last) == 0 && *last == want)
            return 1;
        usleep(250000);
    } while (now_s() < end);
    return 0;
}

static int phase_reclaim(int close_first, uint64_t total)
{
    char line[512];
    const char *name = close_first ? "drain" : "reclaim";
    pid_t pid = start_holder(close_first, line, sizeof line);
    if (pid < 0) {
        emit("\"%s\":{\"pass\":false,", name);
        emit_text("why", line);
        emit("},");
        return 0;
    }
    uint64_t held = strtoull(line + 5, NULL, 10), seen = 0, during = 0;
    int kept = 1;
    if (close_first) { /* socket closed, memory still mapped: must stay lent */
        sleep(3);
        kept = free_now(&during) == 0 && during + held <= total;
    }
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
    int back = wait_free(total, 15.0, &seen);
    int pass = kept && back;
    emit("\"%s\":{\"pass\":%s,\"held\":%" PRIu64 ",\"free_while_holder_lives\":%" PRIu64 ",\"free_after\":%" PRIu64
         ",\"total\":%" PRIu64 "},",
         name, pass ? "true" : "false", held, close_first ? during : 0, seen, total);
    return pass;
}

static int phase_refusal(void)
{
    char raw[ATTIC_REPLY_MAX + 1] = "", raw2[ATTIC_REPLY_MAX + 1] = "";
    int c = dial();
    int wrong = c >= 0 && ask(c, "{\"key\":\"not-the-key\",\"op\":\"info\"}", raw, sizeof raw, NULL) == 0 &&
                !jtrue(raw, "ok");
    int bad = c >= 0 && ask(c, "{\"key\":", raw2, sizeof raw2, NULL) == 0 && !jtrue(raw2, "ok");
    if (c >= 0)
        close(c);
    emit("\"refusal\":{\"pass\":%s,", wrong && bad ? "true" : "false");
    emit_text("wrong_key", raw);
    emit(",");
    emit_text("malformed", raw2);
    emit("},");
    return wrong && bad;
}

static void usage(void)
{
    fprintf(stderr, "usage: attic-check [--socket PATH] [--key KEY] [--mode heap|list] [--report FILE] [--device N]\n"
                    "                   [--iomem PATH] [--meminfo PATH]\n"
                    "       attic-check --status [--socket PATH] [--key KEY]   (print atticd's status)\n");
}

int main(int argc, char **argv)
{
    static const struct option opts[] = {
        {"socket", required_argument, 0, 's'}, {"key", required_argument, 0, 'k'},
        {"mode", required_argument, 0, 'm'},   {"report", required_argument, 0, 'r'},
        {"device", required_argument, 0, 'd'}, {"iomem", required_argument, 0, 'i'},
        {"meminfo", required_argument, 0, 'M'}, {"phase", required_argument, 0, 'p'},
        {"status", no_argument, 0, 'S'},        {0, 0, 0, 0},
    };
    const char *phase = NULL;
    int o, status_only = 0;
    while ((o = getopt_long(argc, argv, "", opts, NULL)) != -1) {
        switch (o) {
        case 's': sock_path = optarg; break;
        case 'k': key = optarg; break;
        case 'm': mode = optarg; break;
        case 'r': report = optarg; break;
        case 'd': device_ordinal = atoi(optarg); break;
        case 'i': iomem = optarg; break;
        case 'M': meminfo = optarg; break;
        case 'p': phase = optarg; break;
        case 'S': status_only = 1; break;
        default: usage(); return 2;
        }
    }
    if (phase) {
        int rc = strcmp(phase, "full") == 0        ? phase_full()
                 : strcmp(phase, "isolation") == 0 ? phase_isolation()
                 : strcmp(phase, "hold") == 0      ? phase_hold(0)
                 : strcmp(phase, "hold-close") == 0 ? phase_hold(1)
                                                    : 2;
        fputs(out, stdout);
        return rc;
    }

    char host[128] = "?", raw[ATTIC_REPLY_MAX + 1] = "", state[32], dmode[16], driver[64];
    if (status_only) { /* the daemon's status reply, as is (attic-update.sh reads it) */
        int s = dial();
        int ok = s >= 0 && request(s, "status", 0, raw, sizeof raw, NULL) == 0 && jtrue(raw, "ok");
        if (s >= 0)
            close(s);
        puts(ok ? raw : "{\"ok\":false,\"error\":\"no atticd answers\"}");
        return ok ? 0 : 1;
    }
    gethostname(host, sizeof host);
    time_t t = time(NULL);
    char when[32];
    strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
    emit("{\"tool\":\"attic-check\",\"version\":\"" CHECK_VERSION "\",");
    emit_text("host", host);
    emit(",\"time\":\"%s\",", when);
    emit_text("socket", sock_path);
    emit(",\"mode\":\"%s\",\"phases\":{", mode);

    /* status */
    int c = dial(), pass = 1;
    if (c < 0 || request(c, "status", 0, raw, sizeof raw, NULL) != 0) {
        emit("\"status\":{\"pass\":false,\"why\":\"no atticd answers\"}},\"pass\":false}");
        puts(out);
        return 1;
    }
    close(c);
    jstr(raw, "state", state, sizeof state);
    jstr(raw, "mode", dmode, sizeof dmode);
    jstr(raw, "driver", driver, sizeof driver);
    uint64_t total = 0;
    jnum(raw, "size", &total);
    /* every region it reports must be clear of System RAM (bases are shown to root) */
    int ram_ok = 1;
    char ramwhy[256] = "";
    for (const char *p = strstr(raw, "\"regions\":["); p && (p = strstr(p, "\"base\":")) != NULL; p++) {
        const char *s = p;
        while (s > raw && *s != '{')
            s--;
        uint64_t base = strtoull(p + 7, NULL, 10), size = 0;
        jnum(s, "size", &size);
        char kind[32];
        jstr(s, "kind", kind, sizeof kind);
        if (strcmp(kind, "display") == 0 && attic_iomem_touches_ram(iomem, base, size, ramwhy, sizeof ramwhy) != 0)
            ram_ok = 0;
    }
    /* and of a type the shim knows: atticd lends nothing beside any other (carveout_kind.h), so one here is a daemon
     * that lends anyway */
    int kinds_ok = 1;
    char kindwhy[128] = "";
    for (const char *p = strstr(raw, "\"regions\":["); p && (p = strstr(p, "{\"kind\":")) != NULL; p++) {
        char kind[32];
        jstr(p, "kind", kind, sizeof kind);
        if (kinds_ok && strcmp(kind, "display") != 0 && strcmp(kind, "display-reserved") != 0 &&
            strcmp(kind, "uefi") != 0) {
            snprintf(kindwhy, sizeof kindwhy, "a carveout region of a type the shim does not know (%s)", kind);
            kinds_ok = 0;
        }
    }
    int st_ok = strcmp(state, "ready") == 0 && strcmp(dmode, mode) == 0 && ram_ok && kinds_ok && total > 0;
    pass &= st_ok;
    emit("\"status\":{\"pass\":%s,\"driver\":\"%s\",\"state\":\"%s\",\"total\":%" PRIu64 ",", st_ok ? "true" : "false",
         driver, state, total);
    emit_text("regions_check", !kinds_ok ? kindwhy : ram_ok ? "not System RAM" : ramwhy);
    emit(",");
    emit_text("raw", raw);
    emit("},");
    if (!st_ok) {
        emit("\"stopped\":true},\"pass\":false}");
        puts(out);
        if (report) {
            FILE *f = fopen(report, "w");
            if (f) {
                fputs(out, f);
                fputc('\n', f);
                fclose(f);
            }
        }
        return 1;
    }

    char res[16384];
    int rc = run_child("full", res, sizeof res);
    pass &= rc == 0;
    /* the child's object is {"full":{...},"double":{...}}: splice its members in */
    if (res[0] == '{' && res[strlen(res) - 1] == '}') {
        res[strlen(res) - 1] = 0;
        emit("%s,", res + 1);
    } else {
        emit("\"full\":{\"pass\":false,");
        emit_text("why", res);
        emit("},");
        pass = 0;
    }
    rc = run_child("isolation", res, sizeof res);
    pass &= rc == 0;
    if (res[0] == '{' && res[strlen(res) - 1] == '}') {
        res[strlen(res) - 1] = 0;
        emit("%s,", res + 1);
    } else {
        emit("\"isolation\":{\"pass\":false,");
        emit_text("why", res);
        emit("},");
        pass = 0;
    }
    pass &= phase_reclaim(0, total);
    pass &= phase_reclaim(1, total);
    pass &= phase_refusal();
    outlen--; /* the last comma */
    emit("},\"pass\":%s}", pass ? "true" : "false");
    puts(out);
    if (report) {
        FILE *f = fopen(report, "w");
        if (f == NULL || fputs(out, f) < 0 || fputc('\n', f) == EOF || fclose(f) != 0) {
            fprintf(stderr, "attic-check: cannot write %s\n", report);
            return 1;
        }
    }
    return pass ? 0 : 1;
}

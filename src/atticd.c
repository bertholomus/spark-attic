// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * atticd: spark-attic's daemon. It lends the GB10's display carveout, the memory RM sets aside for a display these
 * nodes never drive, to CUDA programs: slices of whole 2 MiB handed over as exported RM memory descriptors on an
 * AF_UNIX SOCK_SEQPACKET socket (SCM_RIGHTS), in the protocol our engine's client speaks (proto.h), and taken back
 * when the client's socket closes.
 *
 * Fail closed: until the running NVIDIA driver has an artifact VALIDATED for the mode asked for (artifact.h) and the
 * carveout passes the bring-up checks below, atticd lends nothing. It answers every request with the reason, logs
 * it, and tries again every --retry seconds and on SIGHUP.
 *
 * Mode heap (the default): each slice is a block RM takes from its own scanout carveout heap (shim heap()), so RM
 *   keeps it off every other allocation, its own included, until the last holder lets go. Bring-up proves the heap
 *   is the carveout: RM refuses a block larger than the carveout, gives nothing more once its free room is held, and
 *   holding all of that leaves MemFree where it was.
 * Mode list: each slice is a memory list over the carveout's physical pages (shim list(); root only). Bring-up proves
 *   the region is not System RAM (/proc/iomem) and holds RM's whole heap over it, so RM's own scanout memory cannot
 *   land on a page we lend. The ledger (lend.h), saved to --state, keeps a page from two clients, across restarts too.
 *
 * Beside the socket, status.json (--status-file) says what atticd is doing to anyone on the node, without the key:
 * for dashboards. It carries no key and no address; the keyed socket stays the only way to borrow.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <grp.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "artifact.h"
#include "attic.h"
#include "lend.h"
#include "proto.h"
#include "shim.h"

#define ATTICD_VERSION "0.1.0"
#define TICK_S 2
#define MAX_REGIONS 8
#define MAX_STRIKES 3
#define MiB(x) ((unsigned long long)((x) >> 20))

static struct config {
    const char *socket;
    char key[ATTIC_KEY_MAX + 1];
    const char *artifacts;
    const char *candidate; /* an artifact directory to run without VALIDATED: attic-check only */
    const char *mode;      /* "heap" or "list" */
    const char *version_file, *iomem, *meminfo;
    char state[PATH_MAX];  /* the ledger file (mode list) */
    char status_file[PATH_MAX];
    int no_status, status_every;
    int minor;
    uint64_t keep;
    mode_t socket_mode;
    const char *socket_group;
    int max_clients, max_slices, retry_s;
    int probe; /* bring up, print status, exit */
} cfg = {
    .socket = ATTIC_DEFAULT_SOCKET,
    .key = ATTIC_DEFAULT_KEY,
    .artifacts = ATTIC_DEFAULT_ARTIFACTS,
    .mode = "heap",
    .version_file = ATTIC_DEFAULT_VERSION_FILE,
    .iomem = "/proc/iomem",
    .meminfo = "/proc/meminfo",
    .minor = -1,
    .socket_mode = 0660,
    .max_clients = 64,
    .max_slices = 64,
    .retry_s = 300,
    .status_every = 10,
};

struct lend {
    uint32_t handle;
    uint64_t size;
    long first; /* mode list: the first slice in the ledger */
};

struct client {
    int fd;
    uid_t uid;
    struct holder who;
    struct lend *lends;
    int nlends, strikes;
};

struct drain { /* mode heap: bytes a closed client may still hold through CUDA, until RM has them back */
    struct holder who;
    uint64_t bytes;
};

static struct {
    int up;        /* shim loaded, RM client open */
    int ready;     /* lending */
    int validated; /* the running driver's artifact is VALIDATED for this mode (not a candidate) */
    char why[512];
    void *dl;
    const struct attic_shim *shim;
    struct attic_rm *rm;
    char driver[64];
    struct artifact art;
    struct attic_region regions[MAX_REGIONS];
    int nregions;
    uint64_t display; /* bytes in display regions */
    /* mode heap */
    uint64_t cap, largest, lent;
    struct drain *drains;
    int ndrains, stale;
    double probe_s;     /* how long bring-up's measure took */
    time_t settle_at;   /* the next look for drained memory back in RM's heap */
    int settle_wait;    /* seconds between looks, doubling while it stays out */
    /* mode list */
    struct ledger ledger;
    int reserved;
    uint32_t reserve;
    /* clients */
    struct client **clients;
    int nclients;
    time_t next_retry;
} st;

static int heap_mode(void)
{
    return strcmp(cfg.mode, "heap") == 0;
}

static void say(char *why, size_t n, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void say(char *why, size_t n, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why, n, fmt, ap);
    va_end(ap);
}

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

/* ---- RM, through the shim ---- */

static int hold(uint64_t size, uint32_t *h)
{
    char e[256];
    return st.shim->heap(st.rm, size, h, e, sizeof e) == 0 ? 0 : -1;
}

static void let_go(uint32_t h)
{
    char e[256];
    if (st.shim->release(st.rm, h, e, sizeof e) != 0)
        LOGW("releasing RM handle %#x: %s", h, e);
}

/* The largest block the heap gives now, at most `most` bytes, found by halving (a try is an alloc and a free). */
static uint64_t heap_largest(uint64_t most)
{
    uint64_t lo = 0, hi = most / ATTIC_SLICE;
    uint32_t h;
    if (hi == 0)
        return 0;
    if (hold(hi * ATTIC_SLICE, &h) == 0) { /* the common case: all of it in one block */
        let_go(h);
        return hi * ATTIC_SLICE;
    }
    hi--;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1) / 2;
        if (hold(mid * ATTIC_SLICE, &h) == 0) {
            let_go(h);
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo * ATTIC_SLICE;
}

/* What the heap gives now, up to `most` bytes: every block, taken one after another (and held until let_go_all). */
struct taken {
    uint32_t h[32];
    int n;
    uint64_t total, largest;
};

static void take_all(uint64_t most, struct taken *t)
{
    memset(t, 0, sizeof *t);
    while (t->n < 32 && t->total < most) {
        uint64_t b = heap_largest(most - t->total);
        if (b == 0 || hold(b, &t->h[t->n]) != 0)
            break;
        t->n++;
        t->total += b;
        if (b > t->largest)
            t->largest = b;
    }
}

static void let_go_all(struct taken *t)
{
    while (t->n > 0)
        let_go(t->h[--t->n]);
}

/* ---- bring-up ---- */

static int heap_bring_up(char *why, size_t n)
{
    uint64_t r = st.display;
    uint32_t h;
    double t0 = now_s();
    if (hold(r + ATTIC_SLICE, &h) == 0) {
        let_go(h);
        say(why, n, "RM gave one block larger than the display carveout (%llu MiB): its ISO heap is not the carveout",
            MiB(r));
        return -1;
    }
    uint64_t m0 = attic_mem_free(cfg.meminfo);
    struct taken t;
    take_all(r, &t);
    if (t.total == 0) {
        say(why, n, "RM's display heap gives no free 2 MiB block");
        return -1;
    }
    int more = hold(ATTIC_SLICE, &h) == 0;
    if (more)
        let_go(h);
    uint64_t m1 = attic_mem_free(cfg.meminfo);
    let_go_all(&t);
    if (more) {
        say(why, n, "with all %llu MiB of the display heap held RM still gave a slice: it may come from System RAM",
            MiB(t.total));
        return -1;
    }
    if (m0 == 0 || m1 == 0) {
        say(why, n, "cannot read MemFree from %s", cfg.meminfo);
        return -1;
    }
    if (m0 > m1 && m0 - m1 > t.total / 4) {
        say(why, n, "holding %llu MiB of the display heap took %llu MiB of MemFree: it is not the carveout",
            MiB(t.total), MiB(m0 - m1));
        return -1;
    }
    if (t.total > r) {
        say(why, n, "the display heap gave %llu MiB, more than the carveout's %llu MiB", MiB(t.total), MiB(r));
        return -1;
    }
    st.cap = t.total;
    st.largest = t.largest;
    st.stale = 0;
    st.probe_s = now_s() - t0;
    LOGI("mode heap: %llu MiB free in RM's display heap (largest block %llu MiB, %s), carveout %llu MiB; MemFree "
         "moved %+lld MiB while all of it was held; probe %.2f s",
         MiB(st.cap), MiB(st.largest), st.shim->detail(st.rm), MiB(r), (long long)((int64_t)(m1 - m0) >> 20),
         st.probe_s);
    return 0;
}

static int list_bring_up(char *why, size_t n)
{
    const struct attic_region *g = NULL;
    for (int i = 0; i < st.nregions; i++) {
        if (st.regions[i].kind != ATTIC_REGION_DISPLAY)
            continue;
        if (g) {
            say(why, n, "mode list lends from one display region; RM reports more");
            return -1;
        }
        g = &st.regions[i];
    }
    if (g->base % ATTIC_SLICE || g->size % ATTIC_SLICE) {
        say(why, n, "display region %#" PRIx64 "+%#" PRIx64 " is not 2 MiB aligned", g->base, g->size);
        return -1;
    }
    char w[256];
    if (attic_iomem_touches_ram(cfg.iomem, g->base, g->size, w, sizeof w) != 0) { /* RAM, or no addresses shown */
        say(why, n, "display region %#" PRIx64 "+%#" PRIx64 ": %s", g->base, g->size, w);
        return -1;
    }
    if (hold(g->size, &st.reserve) != 0) {
        uint64_t l = heap_largest(g->size);
        say(why, n, "RM uses part of the display carveout (its largest free block is %llu of %llu MiB): mode list "
            "lends only a carveout RM holds nothing in", MiB(l), MiB(g->size));
        return -1;
    }
    st.reserved = 1;
    if (ledger_init(&st.ledger, g->base, g->size, cfg.keep) != 0) {
        say(why, n, "cannot build the slice ledger");
        return -1;
    }
    int runs = ledger_load(&st.ledger, cfg.state, attic_alive);
    if (runs < 0)
        LOGW("cannot read the ledger %s: %s (nothing assumed lent)", cfg.state, strerror(-runs));
    else if (runs > 0)
        LOGI("ledger %s: %d runs still held by live processes, draining", cfg.state, runs);
    LOGI("mode list: display region %#" PRIx64 "+%llu MiB, not System RAM, RM's heap over it held; %u slices, %u "
         "kept", g->base, MiB(g->size), st.ledger.count, ledger_count(&st.ledger, SLICE_KEPT));
    return 0;
}

static void tear_down(void)
{
    if (st.reserved)
        let_go(st.reserve);
    st.reserved = 0;
    ledger_free(&st.ledger);
    if (st.rm)
        st.shim->close(st.rm);
    st.rm = NULL;
    if (st.dl)
        dlclose(st.dl);
    st.dl = NULL;
    st.shim = NULL;
    st.up = st.ready = 0;
}

static const char *kind_name(uint32_t k)
{
    switch (k) {
    case ATTIC_REGION_DISPLAY: return "display";
    case ATTIC_REGION_DISPLAY_RESERVED: return "display-reserved";
    case ATTIC_REGION_UEFI: return "uefi";
    default: return "other";
    }
}

static void bring_up(void)
{
    char why[512] = "";
    st.next_retry = time(NULL) + cfg.retry_s;
    st.validated = 0;
    if (attic_driver_version(cfg.version_file, st.driver, sizeof st.driver) != 0) {
        snprintf(st.driver, sizeof st.driver, "none");
        say(why, sizeof why, "no NVIDIA driver loaded (%s unreadable)", cfg.version_file);
        goto refuse;
    }
    if (artifact_find(cfg.artifacts, st.driver, cfg.mode, cfg.candidate, &st.art, why, sizeof why) != 0)
        goto refuse;
    st.validated = !cfg.candidate;
    st.dl = dlopen(st.art.shim, RTLD_NOW | RTLD_LOCAL);
    if (st.dl == NULL) {
        say(why, sizeof why, "cannot load %s: %s", st.art.shim, dlerror());
        goto refuse;
    }
    attic_shim_fn fn = (attic_shim_fn)dlsym(st.dl, "attic_shim");
    st.shim = fn ? fn() : NULL;
    if (st.shim == NULL || st.shim->abi != ATTIC_SHIM_ABI) {
        say(why, sizeof why, "%s is not a spark-attic shim of ABI %d", st.art.shim, ATTIC_SHIM_ABI);
        goto refuse;
    }
    if (strcmp(st.shim->driver, st.driver) != 0) {
        say(why, sizeof why, "%s was built for driver %s; %s runs", st.art.shim, st.shim->driver, st.driver);
        goto refuse;
    }
    if (st.shim->open(&st.rm, cfg.minor, why, sizeof why) != 0) {
        st.rm = NULL;
        goto refuse;
    }
    st.up = 1;
    if (st.shim->carveout(st.rm, st.regions, MAX_REGIONS, &st.nregions, why, sizeof why) != 0)
        goto refuse;
    st.display = 0;
    int unknown = -1;
    for (int i = 0; i < st.nregions; i++) {
        uint32_t k = st.regions[i].kind;
        LOGI("carveout region %d: %s %#" PRIx64 "+%llu MiB", i, kind_name(k), st.regions[i].base,
             MiB(st.regions[i].size));
        if (k == ATTIC_REGION_DISPLAY)
            st.display += st.regions[i].size;
        else if (k != ATTIC_REGION_DISPLAY_RESERVED && k != ATTIC_REGION_UEFI && unknown < 0)
            unknown = i;
    }
    /* The shim names every type its driver's header declares (carveout_kind.h). A type it does not know may be display
     * memory under a number that moved (615.78.08 renumbered the enum), so nothing is lent beside it. */
    if (unknown >= 0) {
        say(why, sizeof why, "RM reports carveout region %d (%llu MiB) of a type the shim for driver %s does not know",
            unknown, MiB(st.regions[unknown].size), st.driver);
        goto refuse;
    }
    if (st.display == 0) {
        say(why, sizeof why, "RM reports no display carveout on this GPU");
        goto refuse;
    }
    if ((heap_mode() ? heap_bring_up(why, sizeof why) : list_bring_up(why, sizeof why)) != 0)
        goto refuse;
    st.ready = 1;
    st.why[0] = 0;
    LOGI("lending: driver %s, mode %s, shim %s (sha256 %.12s, %s %s)", st.driver, cfg.mode, st.art.shim,
         st.art.sha256, cfg.candidate ? "candidate" : "validated", cfg.candidate ? "" : st.art.checked);
    return;
refuse:
    tear_down();
    snprintf(st.why, sizeof st.why, "%s", why);
    LOGE("lending nothing: %s (next try in %d s, or on SIGHUP)", st.why, cfg.retry_s);
}

/* SIGHUP: an artifact VALIDATED no longer stops new lends; one that is (again) starts them. */
static void recheck(void)
{
    if (!st.up) {
        bring_up();
        return;
    }
    char why[512] = "";
    struct artifact a;
    if (artifact_find(cfg.artifacts, st.driver, cfg.mode, cfg.candidate, &a, why, sizeof why) == 0) {
        why[0] = 0;
        if (strcmp(a.sha256, st.art.sha256) != 0)
            say(why, sizeof why, "the artifact changed since atticd loaded it: restart atticd to load it");
    }
    st.validated = !why[0] && !cfg.candidate;
    if (why[0]) {
        st.ready = 0;
        snprintf(st.why, sizeof st.why, "%s", why);
        LOGE("lending stopped: %s (slices already lent stay with their clients)", why);
    } else if (!st.ready) {
        st.ready = 1;
        st.why[0] = 0;
        LOGI("lending again");
    }
}

/* ---- accounting ---- */

static uint64_t heap_draining(void)
{
    uint64_t d = 0;
    for (int i = 0; i < st.ndrains; i++)
        d += st.drains[i].bytes;
    return d;
}

static void reap(void)
{
    if (!st.up)
        return;
    if (heap_mode()) {
        int k = 0;
        for (int i = 0; i < st.ndrains; i++)
            if (attic_alive(st.drains[i].who))
                st.drains[k++] = st.drains[i];
        st.ndrains = k;
    } else if (ledger_reap(&st.ledger, attic_alive) > 0) {
        int rc = ledger_save(&st.ledger, cfg.state);
        if (rc != 0)
            LOGW("cannot save the ledger %s: %s", cfg.state, strerror(-rc));
    }
}

static void sizes(uint64_t *size, uint64_t *freeb, uint64_t *largest, uint64_t *lent, uint64_t *draining)
{
    if (!st.up) {
        *size = *freeb = *largest = *lent = *draining = 0;
        return;
    }
    if (heap_mode()) {
        uint64_t cap = st.cap > cfg.keep ? st.cap - cfg.keep : 0, d = heap_draining(), used = st.lent + d;
        *size = cap;
        *freeb = cap > used ? cap - used : 0;
        *largest = st.largest < *freeb ? st.largest : *freeb;
        *lent = st.lent;
        *draining = d;
    } else {
        const struct ledger *l = &st.ledger;
        *size = (uint64_t)(l->count - ledger_count(l, SLICE_KEPT)) * ATTIC_SLICE;
        *freeb = (uint64_t)ledger_count(l, SLICE_FREE) * ATTIC_SLICE;
        *largest = (uint64_t)ledger_largest(l) * ATTIC_SLICE;
        *lent = (uint64_t)ledger_count(l, SLICE_LENT) * ATTIC_SLICE;
        *draining = (uint64_t)ledger_count(l, SLICE_DRAINING) * ATTIC_SLICE;
    }
}

/* Mode heap: drains end when RM has the memory back, not only when their process is gone. RM frees a block once
 * nothing holds it (atticd's handle, the exported descriptor, the client's import), so the heap's free room shows it:
 * take all of it and let go; when that and what is lent make the whole cap, every drain is over. A client that let go
 * before it closed its socket (unmapped, released, descriptor closed) gets its memory counted free at once. While
 * some stays out, look again after 2, 4, ... 64 s. Not when measuring is slow (see heap_remeasure). */
static void heap_settle(int now)
{
    if (!st.up || !heap_mode() || st.ndrains == 0 || st.probe_s > 1.0)
        return;
    if (!now && time(NULL) < st.settle_at)
        return;
    struct taken t;
    uint64_t want = st.cap > st.lent ? st.cap - st.lent : 0;
    take_all(want, &t);
    let_go_all(&t);
    if (t.total >= want) {
        LOGI("%llu MiB drained back into RM's display heap", MiB(heap_draining()));
        st.ndrains = 0;
        st.settle_wait = 2;
        return;
    }
    st.settle_wait = now || st.settle_wait < 2 ? 2 : st.settle_wait >= 64 ? 64 : st.settle_wait * 2;
    st.settle_at = time(NULL) + st.settle_wait;
}

/* Mode heap, idle: measure the heap again (RM's own use of it may have changed since bring-up). Not when measuring
 * is slow: a client's request would wait behind it (our client gives up after 5 s). */
static void heap_remeasure(void)
{
    if (!st.up || !heap_mode() || !st.stale || st.lent || st.ndrains || st.probe_s > 1.0)
        return;
    struct taken t;
    take_all(st.display, &t);
    let_go_all(&t);
    if (t.total != st.cap || t.largest != st.largest)
        LOGI("display heap now %llu MiB free (largest %llu MiB), was %llu (%llu)", MiB(t.total), MiB(t.largest),
             MiB(st.cap), MiB(st.largest));
    st.cap = t.total;
    st.largest = t.largest;
    st.stale = 0;
}

/* ---- replies ---- */

static int send_reply(struct client *c, const char *body, int fd)
{
    return attic_send_reply(c->fd, body, fd);
}

static int refuse(struct client *c, const char *what)
{
    char text[600], body[ATTIC_REPLY_MAX];
    attic_json_text(what, text, sizeof text);
    snprintf(body, sizeof body, "{\"ok\":false,\"error\":\"%s\"}", text);
    return send_reply(c, body, -1);
}

static int op_info(struct client *c)
{
    uint64_t size, freeb, largest, lent, draining;
    char body[ATTIC_REPLY_MAX];
    reap();
    sizes(&size, &freeb, &largest, &lent, &draining);
    int k = snprintf(body, sizeof body, "{\"ok\":true,\"size\":%" PRIu64 ",\"free\":%" PRIu64 ",\"largest\":%" PRIu64,
                     size, freeb, largest);
    if (!heap_mode() && c->uid == 0) /* the physical base only to root */
        k += snprintf(body + k, sizeof body - k, ",\"base\":%" PRIu64, st.ledger.base);
    snprintf(body + k, sizeof body - k, "}");
    return send_reply(c, body, -1);
}

static int op_status(struct client *c)
{
    uint64_t size, freeb, largest, lent, draining;
    char body[ATTIC_REPLY_MAX], why[600], art[600];
    reap();
    sizes(&size, &freeb, &largest, &lent, &draining);
    attic_json_text(st.why, why, sizeof why);
    attic_json_text(st.art.dir, art, sizeof art);
    int k = snprintf(body, sizeof body,
                     "{\"ok\":true,\"name\":\"spark-attic\",\"version\":\"" ATTICD_VERSION "\",\"state\":\"%s\","
                     "\"why\":\"%s\",\"driver\":\"%s\",\"mode\":\"%s\",\"candidate\":%s,\"artifact\":\"%s\","
                     "\"shim_sha256\":\"%s\",\"validated\":\"%s\",\"pages\":\"%s\",\"size\":%" PRIu64 ",\"free\":%" PRIu64
                     ",\"largest\":%" PRIu64 ",\"lent\":%" PRIu64 ",\"draining\":%" PRIu64 ",\"clients\":%d,\"regions\":[",
                     st.ready ? "ready" : "refusing", why, st.driver, cfg.mode, cfg.candidate ? "true" : "false", art,
                     st.art.sha256, st.art.checked, st.up ? st.shim->detail(st.rm) : "", size, freeb, largest, lent,
                     draining, st.nclients);
    for (int i = 0; st.up && i < st.nregions && k < (int)sizeof body - 160; i++) {
        k += snprintf(body + k, sizeof body - k, "%s{\"kind\":\"%s\",\"size\":%" PRIu64, i ? "," : "",
                      kind_name(st.regions[i].kind), st.regions[i].size);
        if (c->uid == 0)
            k += snprintf(body + k, sizeof body - k, ",\"base\":%" PRIu64, st.regions[i].base);
        k += snprintf(body + k, sizeof body - k, "}");
    }
    if (k < (int)sizeof body - 3)
        snprintf(body + k, sizeof body - k, "]}");
    else
        return refuse(c, "status too long");
    return send_reply(c, body, -1);
}

/* ---- the keyless status file ---- */

static struct {
    char last[2048]; /* what the file says, but its "pid" and "updated" */
    time_t written;
    int failing;
} pub;

/* Rewrites --status-file (a new file renamed over it, 0644 whatever the umask) when anything in it changes, and every
 * --status-every seconds even when nothing does (on the TICK_S clock, so that is at least TICK_S), so "updated" (Unix
 * time) tells a reader whether atticd still runs. */
static void publish(void)
{
    if (!cfg.status_file[0])
        return;
    uint64_t size, freeb, largest, lent, draining;
    char body[sizeof pub.last], why[600], checked[sizeof st.art.checked * 2];
    sizes(&size, &freeb, &largest, &lent, &draining);
    attic_json_text(st.why, why, sizeof why);
    attic_json_text(st.validated ? st.art.checked : "", checked, sizeof checked);
    snprintf(body, sizeof body,
             "{\"name\":\"spark-attic\",\"version\":\"" ATTICD_VERSION "\",\"state\":\"%s\",\"why\":\"%s\","
             "\"driver\":\"%s\",\"mode\":\"%s\",\"validated\":%s,\"validated_at\":\"%s\",\"size_mib\":%llu,"
             "\"free_mib\":%llu,\"largest_mib\":%llu,\"lent_mib\":%llu,\"draining_mib\":%llu,\"clients\":%d,"
             "\"every_s\":%d",
             st.ready ? "ready" : "refusing", why, st.driver, cfg.mode, st.validated ? "true" : "false",
             checked, MiB(size), MiB(freeb), MiB(largest), MiB(lent), MiB(draining),
             st.nclients, cfg.status_every);
    time_t now = time(NULL);
    if (strcmp(body, pub.last) == 0 && now - pub.written < cfg.status_every)
        return;
    char text[sizeof body + 64], tmp[PATH_MAX + 8];
    int len = snprintf(text, sizeof text, "%s,\"pid\":%d,\"updated\":%lld}\n", body, (int)getpid(), (long long)now);
    snprintf(tmp, sizeof tmp, "%s.new", cfg.status_file);
    errno = 0;
    int err = 0, fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0 || fchmod(fd, 0644) != 0 || write(fd, text, (size_t)len) != len)
        err = errno ? errno : EIO;
    if (fd >= 0)
        close(fd);
    if (!err && rename(tmp, cfg.status_file) != 0)
        err = errno;
    if (err) {
        if (!pub.failing)
            LOGW("cannot write the status file %s: %s", cfg.status_file, strerror(err));
        pub.failing = 1;
        unlink(tmp);
        return;
    }
    if (pub.failing)
        LOGI("status file %s written again", cfg.status_file);
    pub.failing = 0;
    snprintf(pub.last, sizeof pub.last, "%s", body);
    pub.written = now;
}

static int op_alloc(struct client *c, const struct attic_req *r)
{
    char e[256], w[300];
    if (!r->has_size)
        return refuse(c, "alloc needs a size");
    uint64_t size = r->size;
    if (size == 0 || size % ATTIC_SLICE != 0)
        return refuse(c, "a size is a whole number of 2 MiB slices");
    if (c->nlends >= cfg.max_slices)
        return refuse(c, "this client holds as many slices as one client may");
    reap();
    uint64_t total, freeb, largest, lent, draining;
    sizes(&total, &freeb, &largest, &lent, &draining);
    uint32_t h;
    long first = -1;
    if (heap_mode()) {
        if (size > freeb) {
            say(w, sizeof w, "only %llu MiB free", MiB(freeb));
            return refuse(c, w);
        }
        if (st.shim->heap(st.rm, size, &h, e, sizeof e) != 0) {
            st.stale = 1;
            say(w, sizeof w, "RM's display heap: %s", e);
            return refuse(c, w);
        }
    } else {
        uint32_t n = (uint32_t)(size / ATTIC_SLICE);
        first = ledger_fit(&st.ledger, n);
        if (first < 0) {
            say(w, sizeof w, "no free run of %llu MiB (largest %llu MiB)", MiB(size), MiB(largest));
            return refuse(c, w);
        }
        if (st.shim->list(st.rm, st.ledger.base + (uint64_t)first * ATTIC_SLICE, size, &h, e, sizeof e) != 0) {
            say(w, sizeof w, "RM memory list: %s", e);
            return refuse(c, w);
        }
    }
    int fd;
    if (st.shim->export_fd(st.rm, h, &fd, e, sizeof e) != 0) {
        let_go(h);
        say(w, sizeof w, "RM export: %s", e);
        return refuse(c, w);
    }
    struct lend *more = realloc(c->lends, (size_t)(c->nlends + 1) * sizeof *more);
    if (more == NULL) {
        close(fd);
        let_go(h);
        return refuse(c, "out of memory");
    }
    c->lends = more;
    c->lends[c->nlends++] = (struct lend){h, size, first};
    if (heap_mode()) {
        st.lent += size;
        st.stale = 1;
    } else {
        ledger_mark(&st.ledger, (uint32_t)first, (uint32_t)(size / ATTIC_SLICE), SLICE_LENT, c->who);
        int rc = ledger_save(&st.ledger, cfg.state);
        if (rc != 0)
            LOGW("cannot save the ledger %s: %s", cfg.state, strerror(-rc));
    }
    char body[128];
    snprintf(body, sizeof body, "{\"ok\":true,\"size\":%" PRIu64 "}", size);
    int sent = send_reply(c, body, fd);
    close(fd); /* the message holds its own reference */
    if (sent != 0) {
        /* The descriptor never left: no one can hold the memory, so it is free at once. */
        c->nlends--;
        let_go(h);
        if (heap_mode())
            st.lent -= size;
        else
            ledger_mark(&st.ledger, (uint32_t)first, (uint32_t)(size / ATTIC_SLICE), SLICE_FREE, c->who);
        return -1;
    }
    LOGI("lent %llu MiB to pid %d (uid %u)%s", MiB(size), (int)c->who.pid, (unsigned)c->uid,
         heap_mode() ? "" : " as a memory list");
    return 0;
}

/* ---- clients ---- */

static int epfd = -1;

static void drop(struct client *c, const char *why)
{
    uint64_t back = 0;
    for (int i = 0; i < c->nlends; i++) {
        struct lend *l = &c->lends[i];
        if (st.up)
            let_go(l->handle);
        back += l->size;
        if (heap_mode()) {
            st.lent -= l->size;
            struct drain *more = realloc(st.drains, (size_t)(st.ndrains + 1) * sizeof *more);
            if (more) {
                st.drains = more;
                st.drains[st.ndrains++] = (struct drain){c->who, l->size};
            }
        } else if (st.up) {
            ledger_mark(&st.ledger, (uint32_t)l->first, (uint32_t)(l->size / ATTIC_SLICE), SLICE_DRAINING, c->who);
        }
    }
    if (c->nlends && !heap_mode() && st.up) {
        int rc = ledger_save(&st.ledger, cfg.state);
        if (rc != 0)
            LOGW("cannot save the ledger %s: %s", cfg.state, strerror(-rc));
    }
    if (c->nlends)
        LOGI("pid %d closed (%s): %llu MiB back, free again once RM has it back or that process is gone",
             (int)c->who.pid, why, MiB(back));
    epoll_ctl(epfd, EPOLL_CTL_DEL, c->fd, NULL);
    close(c->fd);
    for (int i = 0; i < st.nclients; i++)
        if (st.clients[i] == c) {
            st.clients[i] = st.clients[--st.nclients];
            break;
        }
    free(c->lends);
    free(c);
    reap();
    heap_settle(1);
}

/* One request: 0 served (there may be more), 1 nothing queued, -1 drop the client. */
static int serve(struct client *c)
{
    char buf[ATTIC_REQUEST_MAX + 1], why[160];
    struct iovec iov = {buf, sizeof buf - 1};
    struct msghdr m;
    memset(&m, 0, sizeof m);
    m.msg_iov = &iov;
    m.msg_iovlen = 1; /* no control buffer: descriptors a client sends are discarded by the kernel */
    ssize_t r = recvmsg(c->fd, &m, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
    if (r < 0)
        return errno == EAGAIN || errno == EINTR ? 1 : -1;
    if (r == 0)
        return -1;
    buf[r] = 0;
    struct attic_req q;
    if (m.msg_flags & MSG_TRUNC) {
        say(why, sizeof why, "request longer than %d bytes", ATTIC_REQUEST_MAX);
    } else if (attic_parse_request(buf, (size_t)r, &q, why, sizeof why) == 0) {
        if (!attic_key_equal(q.key, cfg.key)) {
            say(why, sizeof why, "wrong key");
        } else {
            c->strikes = 0;
            if (strcmp(q.op, "status") == 0)
                return op_status(c);
            if (!st.ready)
                return refuse(c, st.why[0] ? st.why : "not lending");
            if (strcmp(q.op, "info") == 0)
                return op_info(c);
            if (strcmp(q.op, "alloc") == 0)
                return op_alloc(c, &q);
            say(why, sizeof why, "unknown op");
        }
    }
    if (++c->strikes >= MAX_STRIKES) {
        refuse(c, why);
        return -1;
    }
    return refuse(c, why);
}

static void accept_all(int lfd)
{
    for (;;) {
        int fd = accept4(lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0)
            return;
        struct ucred cred;
        socklen_t len = sizeof cred;
        if (st.nclients >= cfg.max_clients || getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) != 0) {
            if (st.nclients >= cfg.max_clients)
                LOGW("refusing a client: %d connected already", st.nclients);
            close(fd);
            continue;
        }
        struct client *c = calloc(1, sizeof *c);
        struct client **more = c ? realloc(st.clients, (size_t)(st.nclients + 1) * sizeof *more) : NULL;
        if (more == NULL) {
            free(c);
            close(fd);
            continue;
        }
        st.clients = more;
        c->fd = fd;
        c->uid = cred.uid;
        c->who = attic_holder(cred.pid);
        struct epoll_event ev = {.events = EPOLLIN | EPOLLRDHUP, .data.ptr = c};
        if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) != 0) {
            free(c);
            close(fd);
            continue;
        }
        st.clients[st.nclients++] = c;
    }
}

/* ---- socket ---- */

static int listen_on(const char *path)
{
    struct sockaddr_un a = {.sun_family = AF_UNIX};
    if (strlen(path) >= sizeof a.sun_path) {
        LOGE("socket path too long: %s", path);
        return -1;
    }
    strcpy(a.sun_path, path);
    char dir[PATH_MAX];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash && slash != dir) {
        *slash = 0;
        if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
            LOGE("cannot make %s: %s", dir, strerror(errno));
            return -1;
        }
    }
    struct stat s;
    if (lstat(path, &s) == 0) {
        if (!S_ISSOCK(s.st_mode)) {
            LOGE("%s exists and is not a socket; not touching it", path);
            return -1;
        }
        int probe = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        int live = probe >= 0 && connect(probe, (struct sockaddr *)&a, sizeof a) == 0;
        if (probe >= 0)
            close(probe);
        if (live) {
            LOGE("another daemon answers on %s", path);
            return -1;
        }
        unlink(path);
    }
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        LOGE("socket: %s", strerror(errno));
        return -1;
    }
    mode_t old = umask(077);
    int rc = bind(fd, (struct sockaddr *)&a, sizeof a);
    umask(old);
    if (rc != 0 || chmod(path, cfg.socket_mode) != 0 || listen(fd, 16) != 0) {
        LOGE("cannot listen on %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (cfg.socket_group) {
        struct group *g = getgrnam(cfg.socket_group);
        if (g == NULL || chown(path, (uid_t)-1, g->gr_gid) != 0) {
            LOGE("cannot give %s to group %s", path, cfg.socket_group);
            close(fd);
            return -1;
        }
    }
    return fd;
}

/* ---- main ---- */

static void usage(FILE *f)
{
    fprintf(f,
            "usage: atticd [options]\n"
            "  --socket PATH        listen here (default " ATTIC_DEFAULT_SOCKET ")\n"
            "  --key KEY | --key-file PATH   the key clients send (default " ATTIC_DEFAULT_KEY ")\n"
            "  --artifacts DIR      per-driver artifacts (default " ATTIC_DEFAULT_ARTIFACTS ")\n"
            "  --candidate DIR      run this artifact unvalidated (attic-check only; not on the default socket)\n"
            "  --mode heap|list     how slices are made (default heap)\n"
            "  --gpu-minor N        the GPU's /dev/nvidiaN (default: the only one)\n"
            "  --keep-mib N         never lend the last N MiB (default 0)\n"
            "  --socket-mode OCTAL  (default 0660)   --socket-group NAME\n"
            "  --max-clients N (64)  --max-slices N (64 a client)  --retry SECONDS (300)\n"
            "  --state PATH         the ledger, mode list (default: beside the socket, \"ledger\")\n"
            "  --status-file PATH   status JSON for anyone on the node, no key in it (default: beside the socket,\n"
            "                       \"status.json\"; none for a candidate)\n"
            "  --no-status-file     --status-every SECONDS (10; at least 2, atticd's clock)\n"
            "  --version-file PATH  --iomem PATH  --meminfo PATH   (tests)\n"
            "  --probe              bring up, print the status JSON, exit (0 ready, 3 refusing)\n");
}

static int read_key_file(const char *path)
{
    char buf[ATTIC_KEY_MAX + 2];
    size_t len;
    if (attic_read_file(path, buf, sizeof buf, &len) != 0)
        return -1;
    while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' '))
        buf[--len] = 0;
    if (len > ATTIC_KEY_MAX)
        return -1;
    memcpy(cfg.key, buf, len + 1);
    return 0;
}

static int parse_args(int argc, char **argv)
{
    static const struct option opts[] = {
        {"socket", required_argument, 0, 's'},       {"key", required_argument, 0, 'k'},
        {"key-file", required_argument, 0, 'K'},     {"artifacts", required_argument, 0, 'a'},
        {"candidate", required_argument, 0, 'c'},    {"mode", required_argument, 0, 'm'},
        {"gpu-minor", required_argument, 0, 'g'},    {"keep-mib", required_argument, 0, 'p'},
        {"socket-mode", required_argument, 0, 'o'},  {"socket-group", required_argument, 0, 'G'},
        {"max-clients", required_argument, 0, 'C'},  {"max-slices", required_argument, 0, 'S'},
        {"retry", required_argument, 0, 'r'},        {"state", required_argument, 0, 'L'},
        {"version-file", required_argument, 0, 'v'}, {"iomem", required_argument, 0, 'i'},
        {"meminfo", required_argument, 0, 'M'},      {"probe", no_argument, 0, 'P'},
        {"status-file", required_argument, 0, 'F'},  {"no-status-file", no_argument, 0, 'N'},
        {"status-every", required_argument, 0, 'E'}, {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };
    int o;
    while ((o = getopt_long(argc, argv, "", opts, NULL)) != -1) {
        char *end = NULL;
        switch (o) {
        case 's': cfg.socket = optarg; break;
        case 'k':
            if (strlen(optarg) > ATTIC_KEY_MAX)
                return -1;
            strcpy(cfg.key, optarg);
            break;
        case 'K':
            if (read_key_file(optarg) != 0) {
                LOGE("cannot read a key from %s", optarg);
                return -1;
            }
            break;
        case 'a': cfg.artifacts = optarg; break;
        case 'c': cfg.candidate = optarg; break;
        case 'm': cfg.mode = optarg; break;
        case 'g': cfg.minor = (int)strtol(optarg, &end, 10); break;
        case 'p': cfg.keep = strtoull(optarg, &end, 10) << 20; break;
        case 'o': cfg.socket_mode = (mode_t)strtoul(optarg, &end, 8); break;
        case 'G': cfg.socket_group = optarg; break;
        case 'C': cfg.max_clients = (int)strtol(optarg, &end, 10); break;
        case 'S': cfg.max_slices = (int)strtol(optarg, &end, 10); break;
        case 'r': cfg.retry_s = (int)strtol(optarg, &end, 10); break;
        case 'L': snprintf(cfg.state, sizeof cfg.state, "%s", optarg); break;
        case 'F': snprintf(cfg.status_file, sizeof cfg.status_file, "%s", optarg); break;
        case 'N': cfg.no_status = 1; break;
        case 'E': cfg.status_every = (int)strtol(optarg, &end, 10); break;
        case 'v': cfg.version_file = optarg; break;
        case 'i': cfg.iomem = optarg; break;
        case 'M': cfg.meminfo = optarg; break;
        case 'P': cfg.probe = 1; break;
        case 'h': usage(stdout); exit(0);
        default: return -1;
        }
        if (end && *end) {
            LOGE("not a number: %s", optarg);
            return -1;
        }
    }
    if (optind != argc || (strcmp(cfg.mode, "heap") != 0 && strcmp(cfg.mode, "list") != 0))
        return -1;
    if (!attic_key_ok(cfg.key)) {
        LOGE("a key is 1 to %d printable characters without quotes or backslashes", ATTIC_KEY_MAX);
        return -1;
    }
    if (cfg.candidate && strcmp(cfg.socket, ATTIC_DEFAULT_SOCKET) == 0) {
        LOGE("a candidate never serves the default socket: give --socket");
        return -1;
    }
    if (cfg.max_clients < 1 || cfg.max_slices < 1 || cfg.retry_s < 1 || cfg.status_every < TICK_S ||
        cfg.socket_mode > 0777)
        return -1;
    const char *slash = strrchr(cfg.socket, '/');
    int dirlen = slash ? (int)(slash - cfg.socket) : 0;
    if (cfg.state[0] == 0)
        snprintf(cfg.state, sizeof cfg.state, "%.*s%sledger", dirlen, cfg.socket, slash ? "/" : "");
    if (cfg.no_status)
        cfg.status_file[0] = 0;
    else if (cfg.status_file[0] == 0 && !cfg.candidate) /* attic-validate.sh runs a candidate beside the real socket */
        snprintf(cfg.status_file, sizeof cfg.status_file, "%.*s%sstatus.json", dirlen, cfg.socket, slash ? "/" : "");
    return 0;
}

static int probe_only(void)
{
    bring_up();
    uint64_t size, freeb, largest, lent, draining;
    sizes(&size, &freeb, &largest, &lent, &draining);
    char why[600];
    attic_json_text(st.why, why, sizeof why);
    printf("{\"ok\":true,\"state\":\"%s\",\"why\":\"%s\",\"driver\":\"%s\",\"mode\":\"%s\",\"pages\":\"%s\","
           "\"size\":%" PRIu64 ",\"free\":%" PRIu64 ",\"largest\":%" PRIu64 ",\"display\":%" PRIu64 "}\n",
           st.ready ? "ready" : "refusing", why, st.driver, cfg.mode, st.up ? st.shim->detail(st.rm) : "", size,
           freeb, largest, st.display);
    int ready = st.ready;
    tear_down();
    return ready ? 0 : 3;
}

int main(int argc, char **argv)
{
    if (parse_args(argc, argv) != 0) {
        usage(stderr);
        return 2;
    }
    if (cfg.probe)
        return probe_only();

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGHUP);
    sigprocmask(SIG_BLOCK, &mask, NULL);
    signal(SIGPIPE, SIG_IGN);
    int sfd = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);
    int tfd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
    struct itimerspec tick = {{TICK_S, 0}, {TICK_S, 0}};
    epfd = epoll_create1(EPOLL_CLOEXEC);
    int lfd = listen_on(cfg.socket);
    if (sfd < 0 || tfd < 0 || epfd < 0 || lfd < 0 || timerfd_settime(tfd, 0, &tick, NULL) != 0)
        return 1;
    static char tag_l, tag_s, tag_t;
    struct epoll_event ev = {.events = EPOLLIN};
    ev.data.ptr = &tag_l;
    epoll_ctl(epfd, EPOLL_CTL_ADD, lfd, &ev);
    ev.data.ptr = &tag_s;
    epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);
    ev.data.ptr = &tag_t;
    epoll_ctl(epfd, EPOLL_CTL_ADD, tfd, &ev);
    LOGI("spark-attic atticd " ATTICD_VERSION " on %s (mode %s%s)", cfg.socket, cfg.mode,
         cfg.candidate ? ", candidate" : "");
    bring_up();
    publish();

    for (int running = 1; running;) {
        struct epoll_event evs[32];
        int n = epoll_wait(epfd, evs, 32, -1);
        if (n < 0 && errno == EINTR)
            continue;
        for (int i = 0; i < n; i++) {
            void *p = evs[i].data.ptr;
            if (p == &tag_l) {
                accept_all(lfd);
            } else if (p == &tag_s) {
                struct signalfd_siginfo si;
                while (read(sfd, &si, sizeof si) == sizeof si) {
                    if (si.ssi_signo == SIGHUP) {
                        LOGI("SIGHUP: checking the artifact again");
                        recheck();
                    } else {
                        running = 0;
                    }
                }
            } else if (p == &tag_t) {
                uint64_t ticks;
                while (read(tfd, &ticks, sizeof ticks) == sizeof ticks)
                    ;
                reap();
                if (!st.up && time(NULL) >= st.next_retry)
                    bring_up();
                heap_settle(0);
                heap_remeasure();
            } else {
                struct client *c = p;
                if (evs[i].events & EPOLLIN) {
                    int rc;
                    while ((rc = serve(c)) == 0)
                        ;
                    if (rc < 0) {
                        drop(c, "socket closed");
                        continue;
                    }
                }
                if (evs[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP))
                    drop(c, "hung up");
            }
        }
        publish();
    }
    LOGI("stopping: %d clients", st.nclients);
    while (st.nclients > 0)
        drop(st.clients[0], "atticd stopping");
    tear_down();
    unlink(cfg.socket);
    if (cfg.status_file[0])
        unlink(cfg.status_file);
    return 0;
}

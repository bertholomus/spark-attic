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
#include <dirent.h>
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
    const char *drm_dir;   /* where the DRM connectors' status files live (tests point it at a fake tree) */
    const char *modeset_dir; /* where the /dev/nvidia-modeset holders' proc dirs live (default /proc; tests) */
    const char *pmon_file;   /* nvidia-smi pmon -c 1 output (default: run nvidia-smi; tests point it at a file) */
    char state[PATH_MAX];  /* the ledger file (mode list) */
    char status_file[PATH_MAX];
    int no_status, status_every;
    int minor;
    uint64_t keep;
    mode_t socket_mode;
    const char *socket_group;
    int max_clients, max_slices, retry_s;
    int probe; /* bring up, print status, exit */
    int probe_display; /* read-only: print each display signal's verdict, exit */
} cfg = {
    .socket = ATTIC_DEFAULT_SOCKET,
    .artifacts = ATTIC_DEFAULT_ARTIFACTS,
    .mode = "heap",
    .version_file = ATTIC_DEFAULT_VERSION_FILE,
    .iomem = "/proc/iomem",
    .meminfo = "/proc/meminfo",
    .drm_dir = "/sys/class/drm",
    .modeset_dir = "/proc",
    .pmon_file = NULL,
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
    int display_block; /* the ready flag is down because a display is in use (st.dwhy says so) */
    char dwhy[384];    /* the display refusal text of the held block (st.why can name a newer reason) */
    int display_idle;  /* complete idle readings in a row while the block is held (B3-R-OSCILLATION) */
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
    int rm_strikes;   /* consecutive RM failures a client saw (N5: a broken RM client is reopened) */
    time_t next_retry;
} st;

static int heap_mode(void)
{
    return strcmp(cfg.mode, "heap") == 0;
}

/* Signal (a): a process other than nvidia-persistenced holds /dev/nvidia-modeset. A holder is a proc
 * dir with an fd whose symlink names the device and whose cmdline starts with something other than
 * nvidia-persistenced: persistenced keeping the device open is the idle state; Xorg, gnome-shell or the
 * modeset console holding it is a display client (the node readings: headless has persistenced only,
 * a desktop adds Xorg, gnome-shell, mutter-x11-frames). A dir we cannot read (a pid that exited
 * mid-scan) is not evidence either way. Tests point --modeset-dir at a fake tree of <pid>/fd/<n>
 * symlinks and <pid>/cmdline files; on the node it is /proc and works as-is. */
static DIR *modeset_scan; /* the scan's place in /proc: 256 entries a tick, the stream stays open */

/* What the scan says: the verdict of the last COMPLETE pass (modeset_last, -1 before one has finished),
 * refreshed the tick a pass finishes. A pass that ran out of its 256-look budget holds that verdict
 * rather than re-deriving from the half it reached: the holder sits in the part a mid-pass tick has not
 * looked at (B3-R-OSCILLATION: on a real GB10 the pass spans two ticks, so per-tick re-derivation made
 * the verdict alternate and borrows through on the idle-reading ticks). */
static int modeset_verdict;
static int modeset_last = -1;
static char modeset_who[128]; /* the holder's name, for the ticks that hold the last complete verdict */

/* complete: read the directory to its end instead of stopping at the tick's 256 looks. The one-shot readers
 * (--probe-display, bring-up and SIGHUP) pass 1: a holder past the budget's edge must not read as "no reading",
 * which is not evidence and would let bring-up go ready over a desktop that is up. The tick passes 0. */
static int modeset_holder(char *who, size_t n, int complete)
{
    if (modeset_scan == NULL) {
        modeset_scan = opendir(cfg.modeset_dir);
        if (modeset_scan == NULL) {
            modeset_verdict = modeset_last; /* no /proc: the last verdict stands; none if never opened */
            return 0;
        }
        rewinddir(modeset_scan);
    }
    struct dirent *e;
    int found = 0;
    int looked = 0;
    while (!found && (complete || looked < 256) && (e = readdir(modeset_scan)) != NULL) {
        looked++;
        if (e->d_name[0] < '0' || e->d_name[0] > '9')
            continue; /* proc dirs are numbers */
        char pdir[PATH_MAX];
        if (snprintf(pdir, sizeof pdir, "%s/%s", cfg.modeset_dir, e->d_name) >= (int)sizeof pdir)
            continue;
        char fddir[PATH_MAX];
        if (snprintf(fddir, sizeof fddir, "%s/fd", pdir) >= (int)sizeof fddir)
            continue;
        DIR *fd = opendir(fddir);
        if (fd == NULL)
            continue; /* no fd dir (a kernel thread): not a holder */
        int holds = 0;
        struct dirent *f;
        while (!holds && (f = readdir(fd)) != NULL) {
            char link[PATH_MAX], target[PATH_MAX];
            if (snprintf(link, sizeof link, "%s/%s", fddir, f->d_name) >= (int)sizeof link)
                continue;
            ssize_t r = readlink(link, target, sizeof target - 1);
            if (r <= 0)
                continue;
            target[r] = 0;
            if (strstr(target, "nvidia-modeset") != NULL)
                holds = 1;
        }
        closedir(fd);
        if (!holds)
            continue;
        char cmd[512];
        size_t len;
        char cmdpath[PATH_MAX];
        if (snprintf(cmdpath, sizeof cmdpath, "%s/cmdline", pdir) >= (int)sizeof cmdpath)
            continue;
        if (attic_read_file(cmdpath, cmd, sizeof cmd, &len) != 0 || len == 0)
            continue; /* a holder we cannot name: skip (a pid that exited between the two reads) */
        for (size_t i = 0; i + 1 < len; i++)
            if (cmd[i] == 0)
                cmd[i] = ' '; /* argv is NUL-separated; flatten it for the name */
        cmd[len] = 0;
        /* argv[0] names the program: nvidia-persistenced keeping the driver loaded is the idle state. */
        char prog[512];
        size_t argv0 = strcspn(cmd, " "); /* the flatten made argv[0] the run up to the first space */
        snprintf(prog, sizeof prog, "%.*s", (int)(argv0 < sizeof prog ? argv0 : sizeof prog - 1), cmd);
        char *slash = strrchr(prog, '/');
        char *base = slash ? slash + 1 : prog;
        if (strcmp(base, "nvidia-persistenced") == 0)
            continue;
        snprintf(who, n, "pid %.24s (%.200s)", e->d_name, prog);
        snprintf(modeset_who, sizeof modeset_who, "%s", who);
        found = 1;
    }
    if (found) {
        modeset_verdict = modeset_last = 1;
        closedir(modeset_scan); /* a holder: the next tick starts a fresh pass */
        modeset_scan = NULL;
    } else if (e == NULL) {
        modeset_verdict = modeset_last = 0;
        rewinddir(modeset_scan); /* the pass is done: the next tick starts over */
    } else {
        modeset_verdict = modeset_last; /* the budget ran out with dirs left: the last complete pass stands */
    }
    return found;
}

/* Signal (b): a graphics (type G) process on the GPU - nvidia-smi pmon's own word for scanout work.
 * The real probe runs nvidia-smi --query-accounted-apps? no: pmon -c 1 -s u. Tests point --pmon-file
 * at a canned output; on the node we run nvidia-smi and read its output. A pmon we cannot run or read
 * is not evidence: the other signals decide. */
static time_t pmon_at; /* the last pmon run: nvidia-smi costs a second, the tick cannot spend it */

/* What pmon says: the verdict of the last FRESH sample (pmon_last, -1 before one has been read), held
 * across the ticks between the 30 s runs - a cached no is the sample's verdict, not a new reading
 * (B3-R-OSCILLATION). A run that fails to read holds the last sample's verdict too; -1 only before the
 * first sample ever read. */
static int pmon_verdict;
static int pmon_last = -1;
static char pmon_who[64]; /* the type-G process' name, for the ticks that hold the last sample */

static int graphics_process(char *who, size_t n)
{
    if (cfg.pmon_file == NULL && pmon_at && time(NULL) - pmon_at < 30) {
        pmon_verdict = pmon_last; /* the cached reading: the sample stands until the next one */
        if (pmon_verdict > 0)
            snprintf(who, n, "%s", pmon_who);
        return pmon_verdict > 0;
    }
    char buf[4096];
    size_t len = 0;
    if (cfg.pmon_file) {
        if (attic_read_file(cfg.pmon_file, buf, sizeof buf - 1, &len) != 0) {
            pmon_verdict = pmon_last;
            return pmon_verdict > 0;
        }
        buf[len] = 0;
    } else {
        FILE *p = popen("nvidia-smi pmon -c 1 -s u 2>/dev/null", "r");
        if (p == NULL) {
            pmon_verdict = pmon_last;
            return pmon_verdict > 0;
        }
        len = fread(buf, 1, sizeof buf - 1, p);
        buf[len] = 0;
        pclose(p);
        pmon_at = time(NULL);
    }
    char *line = buf;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl)
            *nl = 0;
        /* pmon's lines: "gpu pid type sm mem enc dec jpg ofa"; type G is graphics/scanout. */
        char *g = strstr(line, " G ");
        char pid[32] = "";
        if (g) {
            /* the fields before G: gpu, then pid; the pid is the space-delimited field before the type */
            char *p = g;
            while (p > line && p[-1] == ' ')
                p--;
            char *field_end = p;
            while (p > line && p[-1] != ' ')
                p--;
            if (p < field_end && p[0] >= '0' && p[0] <= '9')
                snprintf(pid, sizeof pid, "%.*s", (int)(field_end - p), p);
        }
        if (pid[0]) {
            snprintf(who, n, "pid %s runs graphics (type G)", pid);
            snprintf(pmon_who, sizeof pmon_who, "pid %s runs graphics (type G)", pid);
            pmon_verdict = pmon_last = 1;
            return 1;
        }
        line = nl ? nl + 1 : NULL;
    }
    pmon_verdict = pmon_last = 0; /* a fresh sample read this tick: no graphics process in it */
    return 0;
}

/* A display in use, by any of the three signals (STEER 11's node readings): (a) a process other than
 * nvidia-persistenced holds /dev/nvidia-modeset, (b) the GPU lists a graphics (type G) process, (c) a
 * DRM connector reports "connected" (the KMS systems; on stock DGX OS with nvidia-drm modeset=0 there
 * are no connector dirs at all, so (a) and (b) carry the verdict there). atticd lends RM's free display
 * heap, so a display that may light up and lending do not mix: bring-up refuses, and a display coming
 * into use while lending stops new lends at once. No signal firing (a headless box, an idle desktop
 * that only persistenced keeps loaded) lends. An unreadable probe is not evidence, never a block.
 *
 * The return says what to do about this tick's evidence, and the verdict that gates lending is held
 * across ticks in display_watch: signals (a) and (b) are read on their own cadence (the modeset pass
 * spans ticks, pmon runs every 30 s), and re-deriving per tick from one instant made the refusal
 * oscillate on a real GB10 desktop (B3-R-OSCILLATION). 1 in use (a signal fired now, or its last
 * complete reading found one and has not been replaced), 0 the readable signals read idle, -1 nothing
 * readable at all (no evidence either way; it never blocks). complete: run the modeset scan to the end of the
 * directory (the one-shot callers: --probe-display, bring-up and SIGHUP); the tick keeps its budget. */
static int display_in_use(char *name, size_t n, int complete)
{
    int any_reading = 0; /* at least one signal produced a reading this tick */
    if (modeset_holder(name, n, complete))
        return 1;
    if (modeset_verdict > 0) { /* the last complete pass found a holder and this pass has not overtaken it */
        snprintf(name, n, "%s", modeset_who);
        return 1;
    }
    if (modeset_verdict == 0)
        any_reading = 1;
    if (graphics_process(name, n))
        return 1; /* a type G now, or the last fresh sample's, name filled either way */
    if (pmon_verdict == 0)
        any_reading = 1;
    DIR *d = opendir(cfg.drm_dir);
    if (d == NULL)
        return any_reading ? 0 : -1; /* no DRM: nothing can light up, and that is a reading */
    struct dirent *e;
    int found = 0;
    while (!found && (e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "card", 4) != 0 || strchr(e->d_name, '-') == NULL)
            continue; /* connectors are card<N>-<CONNECTOR>; the card itself has no status */
        char path[PATH_MAX], status[64];
        size_t len;
        if (snprintf(path, sizeof path, "%s/%s/status", cfg.drm_dir, e->d_name) >= (int)sizeof path)
            continue;
        if (attic_read_file(path, status, sizeof status, &len) != 0)
            continue;
        while (len && (status[len - 1] == '\n' || status[len - 1] == '\r' || status[len - 1] == ' '))
            status[--len] = 0;
        if (strcmp(status, "connected") == 0) {
            snprintf(name, n, "%s", e->d_name);
            found = 1;
        }
    }
    closedir(d);
    if (found)
        return 1;
    return 0; /* the connector scan always reads (a missing dir is a reading too) */
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

static uint64_t heap_draining(void);
static void tear_down(void);
static uint64_t rm_out(void);
static void rm_reopen(void);

/* How much memory clients still hold (drains included): reopening the RM client while this is not zero would
 * orphan it. */
static uint64_t rm_out(void)
{
    if (heap_mode())
        return st.lent + heap_draining();
    if (!st.ledger.count)
        return 0;
    return (uint64_t)(ledger_count(&st.ledger, SLICE_LENT) + ledger_count(&st.ledger, SLICE_DRAINING)) * ATTIC_SLICE;
}

/* An RM call a client saw fail (N5): three in a row mean the RM client is broken, not the request. With nothing
 * lent, the client is torn down and reopened (the next tick brings a fresh one up); with slices out, reopening
 * would orphan the memory they hold, so atticd keeps serving refusals until the last client drops, and the tick
 * reopens then. A success clears the strikes. */
static void rm_strike(void)
{
    st.rm_strikes++;
    if (st.rm_strikes < 3)
        return;
    uint64_t out = rm_out();
    if (out) {
        LOGW("the RM client failed %d requests in a row; %llu MiB is still out with clients, so it stays open "
             "until they drop", st.rm_strikes, MiB(out));
        return;
    }
    rm_reopen();
}

/* The strikes said rebuild but slices were still out; once the last one drains, the tick reopens the client. */
static void rm_idle_check(void)
{
    if (st.up && st.rm_strikes >= 3 && rm_out() == 0)
        rm_reopen();
}

static void rm_reopen(void)
{
    snprintf(st.why, sizeof st.why, "the RM client failed %d requests in a row: closing and reopening it",
             st.rm_strikes);
    LOGE("%s", st.why);
    tear_down();
    st.next_retry = time(NULL); /* the next tick brings a fresh client up */
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

static int heap_probe_once(char *why, size_t n, uint64_t r); /* below */

static int heap_bring_up(char *why, size_t n)
{
    uint64_t r = st.display;
    /* A reclaim burst (a parallel build, kswapd) reads exactly like a heap that came from System RAM:
     * MemFree dips hard for a moment. One retry separates them - a burst passes the second time, a
     * dishonest heap loses its bytes every time. */
    if (heap_probe_once(why, n, r) != 0) {
        struct timespec ts = {.tv_sec = 2, .tv_nsec = 0}; /* a burst longer than 200 ms is common */
        nanosleep(&ts, NULL);
        if (heap_probe_once(why, n, r) != 0)
            return -1;
        LOGW("the display-heap probe passed on the second look (the first read a reclaim burst)");
    }
    return 0;
}

static int heap_probe_once(char *why, size_t n, uint64_t r)
{
    uint32_t h;
    double t0 = now_s();
    if (hold(r + ATTIC_SLICE, &h) == 0) {
        let_go(h);
        say(why, n, "RM gave one block larger than the display carveout (%llu MiB): its ISO heap is not the carveout",
            MiB(r));
        return -1;
    }
    uint64_t m0 = 0;
    for (int i = 0; i < 10; i++) { /* the highest pre-hold reading: reclaim dips, the peak is the truth */
        uint64_t m = attic_mem_free(cfg.meminfo);
        if (m > m0)
            m0 = m;
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 10 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    struct taken t;
    take_all(r, &t);
    if (t.total == 0) {
        say(why, n, "RM's display heap gives no free 2 MiB block");
        return -1;
    }
    int more = hold(ATTIC_SLICE, &h) == 0;
    if (more)
        let_go(h);
    uint64_t m1 = 0;
    for (int i = 0; i < 10; i++) { /* the least-churned reading while held: reclaim noise only lowers it */
        uint64_t m = attic_mem_free(cfg.meminfo);
        if (m > m1)
            m1 = m;
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 10 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
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
    if (m0 > m1 && m0 - m1 > t.total * 3 / 4) {
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
    if (st.display_block) {
        /* The held display verdict stands (B3-R-OSCILLATION): a retry does not re-derive it from one
         * instant, because a scan pass that ran out of its 256-look budget or a throttled pmon can read
         * idle while the desktop is still up. Only display_watch's streak of complete idle readings
         * lifts the block. Nothing below runs, so the artifact state the last bring-up found stands. */
        say(why, sizeof why, "%s", st.dwhy[0] ? st.dwhy : "a display is in use");
        goto refuse;
    }
    st.validated = 0;
    /* Mode list is not released (S4): no node ever passed it (580.178.04 refuses memory lists), and its safety
     * rests on the ledger's pid check alone. The candidate path keeps it, so a node can still validate it. */
    if (!heap_mode() && !cfg.candidate) {
        say(why, sizeof why, "mode list is not released in " ATTICD_VERSION ": it has never passed on a node "
            "(580.178.04 refuses memory lists); run it only as a candidate, for the node check");
        goto refuse;
    }
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
    char conn[256];
    if (display_in_use(conn, sizeof conn, 1) > 0) {
        st.display_block = 1;
        say(why, sizeof why, "a display is in use (%s): atticd lends the memory a display needs", conn);
        snprintf(st.dwhy, sizeof st.dwhy, "%s", why);
        goto refuse;
    }
    if ((heap_mode() ? heap_bring_up(why, sizeof why) : list_bring_up(why, sizeof why)) != 0)
        goto refuse;
    st.ready = 1;
    st.display_block = 0;
    st.display_idle = 0;
    st.dwhy[0] = 0;
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
        char conn[256];
        if (st.display_block) {
            /* The held display verdict stands (B3-R-OSCILLATION): SIGHUP does not re-derive it from one
             * instant; display_watch's idle streak lifts it. */
            snprintf(st.why, sizeof st.why, "%s", st.dwhy[0] ? st.dwhy : "a display is in use");
        } else if (display_in_use(conn, sizeof conn, 1) > 0) {
            st.display_block = 1;
            snprintf(st.dwhy, sizeof st.dwhy,
                     "a display is in use (%s): atticd lends the memory a display needs", conn);
            snprintf(st.why, sizeof st.why, "%s", st.dwhy);
            LOGE("lending stopped: %s (slices already lent stay with their clients)", st.why);
        } else {
            st.ready = 1;
            st.display_block = 0;
            st.dwhy[0] = 0;
            st.why[0] = 0;
            LOGI("lending again");
        }
    }
}

/* The display check, every tick. One call to display_in_use reads one instant, and on a real desktop
 * the samples are sparser than this 2 s tick: the modeset pass covers at most 256 proc entries (the
 * holder is found only on the ticks whose pass reaches it) and pmon runs every 30 s, so re-deriving
 * the verdict per tick let borrows through on the idle-reading ticks while the desktop was up all
 * along (B3-R-OSCILLATION: 12 of 20 borrows succeeded on a real GB10). The verdict is therefore held:
 * a display seen in use keeps new lends refused continuously - the idle-reading ticks change nothing -
 * and the block lifts only after DISPLAY_HOLD_TICKS ticks in a row whose readable signals all read
 * idle. One stopped line when the block engages, one again line when it lifts; the ticks between log
 * nothing. */
#define DISPLAY_HOLD_TICKS 3

static void display_watch(void)
{
    char conn[256];
    int in_use = display_in_use(conn, sizeof conn, 0);
    if (in_use > 0) {
        st.display_idle = 0;
        if (st.ready) {
            st.ready = 0;
            st.display_block = 1;
            snprintf(st.dwhy, sizeof st.dwhy,
                     "a display is in use (%s): atticd lends the memory a display needs", conn);
            snprintf(st.why, sizeof st.why, "%s", st.dwhy);
            LOGE("lending stopped: %s (slices already lent stay with their clients)", st.why);
        }
        /* already held, or refusing for another reason: the verdict stands, nothing is logged again */
    } else if (st.display_block && in_use == 0) {
        if (++st.display_idle >= DISPLAY_HOLD_TICKS) {
            st.display_block = 0;
            st.display_idle = 0;
            if (!st.up) {
                bring_up(); /* the block is off: an immediate bring-up try, nobody waits out --retry */
            } else if (st.why[0] == 0 || strcmp(st.why, st.dwhy) == 0) {
                /* the display was the reason lending stopped: resume */
                st.ready = 1;
                st.dwhy[0] = 0;
                st.why[0] = 0;
                LOGI("lending again: the display is gone");
            }
            /* else another refusal stands (an artifact gone or changed): the display block is lifted,
             * that refusal keeps lending stopped */
        }
    }
    /* in_use < 0: nothing readable at all this tick - a held block stands and an idle streak neither
     * grows nor resets. */
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
    /* The display verdict is held by the tick (display_watch): a desktop that starts between ticks
     * stops new lends within one tick, and while it is up the block does not oscillate with the
     * samples (B3-R-OSCILLATION). No scan here - the modeset holder probe walks every proc fd dir,
     * and an alloc must answer in well under the client's timeout. */
    if (st.display_block)
        return refuse(c, st.dwhy[0] ? st.dwhy : "a display is in use: atticd lends the memory a display needs");
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
            rm_strike();
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
            rm_strike();
            return refuse(c, w);
        }
    }
    int fd;
    if (st.shim->export_fd(st.rm, h, &fd, e, sizeof e) != 0) {
        let_go(h);
        say(w, sizeof w, "RM export: %s", e);
        rm_strike();
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
    st.rm_strikes = 0;
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
            "  --key KEY | --key-file PATH   the key clients send (required; no default)\n"
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
            "  --drm-dir PATH       where the DRM connectors' status files live (default /sys/class/drm; tests)\n"
            "  --modeset-dir PATH   where the modeset holders' proc dirs live (default /proc; tests)\n"
            "  --pmon-file PATH     canned nvidia-smi pmon output instead of running nvidia-smi (tests)\n"
            "  --probe-display      read-only: one line per display signal, then exit (0 none, 1 in use)\n"
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
        {"meminfo", required_argument, 0, 'M'},      {"drm-dir", required_argument, 0, 'R'},
        {"modeset-dir", required_argument, 0, 'D'},  {"pmon-file", required_argument, 0, 'U'},
        {"probe", no_argument, 0, 'P'},              {"probe-display", no_argument, 0, 'Q'},
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
        case 'p': {
            uint64_t mib = strtoull(optarg, &end, 10);
            if (end == optarg || mib % 2) {
                LOGE("--keep-mib %s: not a whole number of 2 MiB slices (atticd lends whole 2 MiB slices)", optarg);
                return -1;
            }
            cfg.keep = mib << 20;
            break;
        }
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
        case 'R': cfg.drm_dir = optarg; break;
        case 'D': cfg.modeset_dir = optarg; break;
        case 'U': cfg.pmon_file = optarg; break;
        case 'P': cfg.probe = 1; break;
        case 'Q': cfg.probe_display = 1; break;
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
    if (cfg.key[0] == 0 && !cfg.probe_display) {
        LOGE("no key: give --key or --key-file (as root: umask 077; od -An -N12 -tx1 /dev/urandom | tr -d ' \\n' | "
             "sed s/^/attic_/ > /etc/spark-attic/key)");
        return -1;
    }
    if (!cfg.probe_display && !attic_key_ok(cfg.key)) {
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

/* --probe-display: read-only. One line per signal: the modeset holders, pmon's graphics processes, the
 * DRM connectors. No key, no artifacts, no socket, nothing written. Exit 0 when nothing is in use,
 * 1 when a signal fires; the node's operator and the tests read the lines, not just the code. */
static int probe_display_run(void)
{
    char who[256];
    int a = modeset_holder(who, sizeof who, 1);
    printf("%s modeset-holder: %s\n", a ? "IN-USE" : "idle", a ? who : "only nvidia-persistenced, or nothing readable");
    int b = graphics_process(who, sizeof who);
    printf("%s graphics-process: %s\n", b ? "IN-USE" : "idle", b ? who : "no type G process (or pmon unreadable)");
    int c = 0;
    DIR *d = opendir(cfg.drm_dir);
    if (d) {
        struct dirent *e;
        while (!c && (e = readdir(d)) != NULL) {
            if (strncmp(e->d_name, "card", 4) != 0 || strchr(e->d_name, '-') == NULL)
                continue;
            char path[PATH_MAX], status[64];
            size_t len;
            if (snprintf(path, sizeof path, "%s/%s/status", cfg.drm_dir, e->d_name) >= (int)sizeof path)
                continue;
            if (attic_read_file(path, status, sizeof status, &len) == 0) {
                while (len && (status[len - 1] == '\n' || status[len - 1] == '\r'))
                    status[--len] = 0;
                if (strcmp(status, "connected") == 0) {
                    printf("IN-USE drm-connector: %s reports connected\n", e->d_name);
                    c = 1;
                }
            }
        }
        closedir(d);
        if (!c)
            printf("idle drm-connector: no connector reports connected (none readable counts the same)\n");
    } else {
        printf("idle drm-connector: no DRM directory at %s\n", cfg.drm_dir);
    }
    int any = a || b || c;
    printf("%s: a display %s\n", any ? "IN-USE" : "idle", any ? "is in use; atticd would lend nothing"
           : "is not in use by any signal atticd reads");
    return any ? 1 : 0;
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

    if (cfg.probe_display)
        return probe_display_run();
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
                display_watch();
                rm_idle_check();
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

// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* test_units.c: spark-attic's unit tests (no GPU, no daemon): the request parser, keys, JSON text, fd passing over a
 * socketpair, the slice ledger and its file, artifact files, and the /proc readers. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "artifact.h"
#include "attic.h"
#include "lend.h"
#include "proto.h"

static int failures, checks;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        checks++;                                                                                                      \
        if (!(c)) {                                                                                                    \
            failures++;                                                                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                              \
        }                                                                                                              \
    } while (0)

static char dir[256];

static void write_file(const char *name, const char *text)
{
    char p[512];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "w");
    fputs(text, f);
    fclose(f);
}

static int parse(const char *s, struct attic_req *r)
{
    char why[160];
    return attic_parse_request(s, strlen(s), r, why, sizeof why);
}

static void test_requests(void)
{
    struct attic_req r;
    /* exactly what our TensorFold client sends (its info and alloc) */
    CHECK(parse("{\"key\":\"client_key_1\",\"op\":\"info\"}", &r) == 0);
    CHECK(strcmp(r.key, "client_key_1") == 0 && strcmp(r.op, "info") == 0 && !r.has_size);
    CHECK(parse("{\"key\":\"spark-attic-1\",\"op\":\"alloc\",\"size\":2145386496}", &r) == 0);
    CHECK(r.has_size && r.size == 2145386496ull && strcmp(r.op, "alloc") == 0);
    CHECK(parse(" { \"op\" : \"status\" , \"key\" : \"k\" , \"extra\" : [1] } ", &r) != 0); /* arrays: not ours */
    CHECK(parse("{\"op\":\"status\",\"key\":\"k\",\"note\":\"x\",\"n\":-1.5e3,\"t\":true,\"z\":null}", &r) == 0);
    CHECK(parse("{\"key\":\"a\\u0062c\",\"op\":\"info\"}", &r) == 0 && strcmp(r.key, "abc") == 0);
    CHECK(parse("{\"key\":\"k\",\"op\":\"info\",\"key\":\"j\"}", &r) != 0);  /* a field twice */
    CHECK(parse("{\"key\":\"k\"}", &r) != 0);                                /* no op */
    CHECK(parse("{\"key\":\"k\",\"op\":\"alloc\",\"size\":-2}", &r) != 0);   /* negative */
    CHECK(parse("{\"key\":\"k\",\"op\":\"alloc\",\"size\":2.5}", &r) != 0);  /* not whole */
    CHECK(parse("{\"key\":\"k\",\"op\":\"alloc\",\"size\":012}", &r) != 0);  /* leading zero */
    CHECK(parse("{\"key\":\"k\",\"op\":\"alloc\",\"size\":99999999999999999999}", &r) != 0);
    CHECK(parse("{\"key\":\"k\",\"op\":\"info\"} x", &r) != 0);              /* trailing bytes */
    CHECK(parse("{\"key\":\"k\",\"op\":\"info\",}", &r) != 0);
    CHECK(parse("{\"key\":\"k\\u00e9\",\"op\":\"info\"}", &r) != 0);         /* not ASCII */
    CHECK(parse("{\"key\":\"k\\u0000\",\"op\":\"info\"}", &r) != 0);         /* NUL */
    CHECK(parse("", &r) != 0 && parse("[]", &r) != 0 && parse("{", &r) != 0);
    char big[ATTIC_REQUEST_MAX + 64];
    memset(big, ' ', sizeof big);
    memcpy(big, "{\"key\":\"k\",\"op\":\"info\"}", 23); /* no NUL: spaces follow */
    big[sizeof big - 1] = 0;
    CHECK(parse(big, &r) != 0); /* longer than a request may be */
    char longkey[300] = "{\"key\":\"";
    memset(longkey + 8, 'k', 200);
    strcpy(longkey + 208, "\",\"op\":\"info\"}");
    CHECK(parse(longkey, &r) != 0);
}

static void test_keys_and_text(void)
{
    CHECK(attic_key_equal("spark-attic-1", "spark-attic-1"));
    CHECK(!attic_key_equal("spark-attic-1", "spark-attic-2"));
    CHECK(!attic_key_equal("spark-attic-1", "spark-attic-"));
    CHECK(!attic_key_equal("", "x") && attic_key_equal("", ""));
    CHECK(attic_key_ok("spark-attic-1") && attic_key_ok("client_key_1"));
    CHECK(!attic_key_ok("") && !attic_key_ok("has space") && !attic_key_ok("q\"uote") && !attic_key_ok("b\\s"));
    char out[64];
    attic_json_text("a\"b\\c\nd", out, sizeof out);
    CHECK(strcmp(out, "a\\\"b\\\\c\\u000ad") == 0);
    attic_json_text("0123456789012345678901234567890123456789", out, 16);
    CHECK(strlen(out) < 16);
}

static void test_fd_passing(void)
{
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) == 0);
    int mfd = memfd_create("slice", 0);
    CHECK(mfd >= 0 && ftruncate(mfd, 6 << 20) == 0);
    CHECK(attic_send_reply(sv[0], "{\"ok\":true,\"size\":6291456}", mfd) == 0);
    close(mfd); /* the message holds its own reference */
    char buf[ATTIC_REPLY_MAX + 1];
    union {
        char c[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } u;
    struct iovec iov = {buf, sizeof buf - 1};
    struct msghdr m;
    memset(&m, 0, sizeof m);
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = u.c;
    m.msg_controllen = sizeof u.c;
    ssize_t n = recvmsg(sv[1], &m, 0);
    CHECK(n == 26);
    buf[n > 0 ? n : 0] = 0;
    CHECK(strcmp(buf, "{\"ok\":true,\"size\":6291456}") == 0);
    struct cmsghdr *h = CMSG_FIRSTHDR(&m);
    CHECK(h && h->cmsg_type == SCM_RIGHTS && h->cmsg_len == CMSG_LEN(sizeof(int)));
    int got = -1;
    if (h)
        memcpy(&got, CMSG_DATA(h), sizeof got);
    struct stat s;
    CHECK(got >= 0 && fstat(got, &s) == 0 && s.st_size == (6 << 20));
    close(got);
    /* a reply with no descriptor carries none */
    CHECK(attic_send_reply(sv[0], "{\"ok\":false,\"error\":\"x\"}", -1) == 0);
    memset(&m, 0, sizeof m);
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    m.msg_control = u.c;
    m.msg_controllen = sizeof u.c;
    CHECK(recvmsg(sv[1], &m, 0) > 0 && CMSG_FIRSTHDR(&m) == NULL);
    /* a closed peer is -1, not SIGPIPE */
    close(sv[1]);
    CHECK(attic_send_reply(sv[0], "{}", -1) == -1);
    close(sv[0]);
}

static int dead(struct holder h)
{
    (void)h;
    return 0;
}

static int live(struct holder h)
{
    (void)h;
    return 1;
}

static void test_ledger(void)
{
    struct ledger l;
    const uint64_t S = ATTIC_SLICE;
    CHECK(ledger_init(&l, 0x280000000ull, 3 * S, 0) == 0);
    ledger_free(&l);
    CHECK(ledger_init(&l, 0x280000000ull, S + 1, 0) == -1);
    CHECK(ledger_init(&l, 0x280000000ull, 16 * S, 2 * S) == 0);
    CHECK(ledger_count(&l, SLICE_KEPT) == 2 && ledger_largest(&l) == 14);
    struct holder a = {100, 7}, b = {200, 9};
    long f = ledger_fit(&l, 4);
    CHECK(f == 0);
    ledger_mark(&l, 0, 4, SLICE_LENT, a);
    f = ledger_fit(&l, 2);
    CHECK(f == 4);
    ledger_mark(&l, 4, 2, SLICE_LENT, b);
    ledger_mark(&l, 0, 4, SLICE_FREE, a); /* a 4-slice hole before b, 8 after */
    CHECK(ledger_fit(&l, 3) == 0);         /* the smallest run that fits */
    CHECK(ledger_fit(&l, 5) == 6);
    CHECK(ledger_fit(&l, 9) == -1);
    ledger_mark(&l, 6, 3, SLICE_LENT, a);
    char path[300];
    snprintf(path, sizeof path, "%s/ledger", dir);
    CHECK(ledger_save(&l, path) == 0);
    struct ledger l2;
    CHECK(ledger_init(&l2, 0x280000000ull, 16 * S, 2 * S) == 0);
    CHECK(ledger_load(&l2, path, live) == 2);
    CHECK(ledger_count(&l2, SLICE_DRAINING) == 5 && ledger_count(&l2, SLICE_LENT) == 0);
    CHECK(l2.who[4].pid == 200 && l2.who[6].pid == 100 && l2.who[6].start == 7);
    CHECK(ledger_reap(&l2, dead) == 5 && ledger_count(&l2, SLICE_FREE) == 14);
    ledger_free(&l2);
    CHECK(ledger_init(&l2, 0x290000000ull, 16 * S, 0) == 0); /* another range: nothing of it applies */
    CHECK(ledger_load(&l2, path, live) == 0 && ledger_count(&l2, SLICE_FREE) == 16);
    ledger_free(&l2);
    snprintf(path, sizeof path, "%s/no-such-ledger", dir);
    CHECK(ledger_load(&l, path, live) == 0);
    ledger_free(&l);
    /* our own process is alive; a child that exited is not */
    struct holder me = attic_holder(getpid());
    CHECK(me.start != 0 && attic_alive(me));
    pid_t kid = fork();
    if (kid == 0)
        _exit(0);
    struct holder k = attic_holder(kid);
    waitpid(kid, NULL, 0);
    CHECK(!attic_alive(k));
    CHECK(!attic_alive((struct holder){me.pid, me.start + 1})); /* a later process with our pid */
}

static void test_files(void)
{
    char p[512], v[128];
    write_file("version", "NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.178.04  Release Build  "
                          "(dvs-builder@U22-I3-AF03-09-1)  Thu Jul 31 00:47:52 UTC 2026\nGCC version:  gcc 13\n");
    snprintf(p, sizeof p, "%s/version", dir);
    CHECK(attic_driver_version(p, v, sizeof v) == 0 && strcmp(v, "580.178.04") == 0);
    write_file("version2", "NVRM version: NVIDIA UNIX Kernel Module for aarch64  no.version  here\n");
    snprintf(p, sizeof p, "%s/version2", dir);
    CHECK(attic_driver_version(p, v, sizeof v) != 0);

    write_file("iomem", "00000000-0000ffff : reserved\n80000000-27fffffff : System RAM\n"
                        "280000000-2ffffffff : reserved\n300000000-1fffffffff : System RAM\n");
    snprintf(p, sizeof p, "%s/iomem", dir);
    char why[256];
    CHECK(attic_iomem_touches_ram(p, 0x280000000ull, 0x80000000ull, why, sizeof why) == 0);
    CHECK(attic_iomem_touches_ram(p, 0x27fe00000ull, 0x400000ull, why, sizeof why) == 1);
    CHECK(attic_iomem_touches_ram(p, 0x2ffe00000ull, 0x400000ull, why, sizeof why) == 1);
    write_file("iomem0", "00000000-00000000 : reserved\n00000000-00000000 : System RAM\n"); /* what non-root sees */
    snprintf(p, sizeof p, "%s/iomem0", dir);
    CHECK(attic_iomem_touches_ram(p, 0x280000000ull, 0x80000000ull, why, sizeof why) == -1);

    write_file("meminfo", "MemTotal:       129798784 kB\nMemFree:         1048576 kB\n");
    snprintf(p, sizeof p, "%s/meminfo", dir);
    CHECK(attic_mem_free(p) == 1073741824ull);

    write_file("VALIDATED", "# spark-attic\ndriver=580.178.04\nmodes=heap,list \nshim_sha256=abc\n");
    snprintf(p, sizeof p, "%s/VALIDATED", dir);
    CHECK(artifact_value(p, "modes", v, sizeof v) == 0 && strcmp(v, "heap,list") == 0);
    CHECK(artifact_value(p, "driver", v, sizeof v) == 0 && strcmp(v, "580.178.04") == 0);
    CHECK(artifact_value(p, "nothing", v, sizeof v) != 0);
    CHECK(attic_listed("heap,list", "list") && attic_listed("heap", "heap") && !attic_listed("heaps", "heap") &&
          !attic_listed("", "heap"));
}

int main(void)
{
    snprintf(dir, sizeof dir, "/tmp/attic-units-XXXXXX");
    if (mkdtemp(dir) == NULL)
        return 2;
    test_requests();
    test_keys_and_text();
    test_fd_passing();
    test_ledger();
    test_files();
    char cmd[300];
    snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
    if (system(cmd) != 0)
        fprintf(stderr, "cannot remove %s\n", dir);
    printf("test_units: %d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}

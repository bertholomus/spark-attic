// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* util.c: logging and the few files atticd reads from /proc. */
#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "attic.h"

const char *attic_log_tag = NULL;

void attic_log(const char *level, const char *fmt, ...)
{
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    fprintf(stderr, "atticd%s%s %s: %s\n", attic_log_tag ? " " : "", attic_log_tag ? attic_log_tag : "", level, line);
    fflush(stderr);
}

int attic_read_file(const char *path, char *buf, size_t cap, size_t *len)
{
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -errno;
    size_t got = 0;
    while (got + 1 < cap) {
        ssize_t r = read(fd, buf + got, cap - 1 - got);
        if (r < 0 && errno == EINTR)
            continue;
        if (r < 0) {
            int e = errno;
            close(fd);
            return -e;
        }
        if (r == 0)
            break;
        got += (size_t)r;
    }
    close(fd);
    buf[got] = 0;
    if (len)
        *len = got;
    return 0;
}

static int version_token(const char *t, size_t n)
{
    int dots = 0;
    if (n == 0 || !isdigit((unsigned char)t[0]) || !isdigit((unsigned char)t[n - 1]))
        return 0;
    for (size_t i = 0; i < n; i++) {
        if (t[i] == '.') {
            if (t[i + 1] == '.')
                return 0;
            dots++;
        } else if (!isdigit((unsigned char)t[i])) {
            return 0;
        }
    }
    return dots >= 1 && dots <= 3;
}

int attic_driver_version(const char *path, char *out, size_t cap)
{
    char buf[4096];
    if (attic_read_file(path, buf, sizeof buf, NULL) != 0)
        return -1;
    char *nl = strchr(buf, '\n');
    if (nl)
        *nl = 0;
    const char *p = buf;
    while (*p) {
        while (*p == ' ' || *p == '\t')
            p++;
        const char *s = p;
        while (*p && *p != ' ' && *p != '\t')
            p++;
        size_t n = (size_t)(p - s);
        if (version_token(s, n) && n < cap) {
            memcpy(out, s, n);
            out[n] = 0;
            return 0;
        }
    }
    return -1;
}

int attic_iomem_touches_ram(const char *path, uint64_t base, uint64_t size, char *why, size_t n)
{
    FILE *f = fopen(path, "re");
    if (f == NULL) {
        snprintf(why, n, "cannot read %s: %s", path, strerror(errno));
        return -1;
    }
    char line[256];
    int ranges = 0, nonzero = 0, touches = 0;
    uint64_t end = base + size; /* exclusive */
    while (fgets(line, sizeof line, f)) {
        uint64_t lo, hi;
        char name[160];
        if (sscanf(line, " %" SCNx64 "-%" SCNx64 " : %159[^\n]", &lo, &hi, name) != 3)
            continue;
        ranges++;
        if (lo || hi)
            nonzero = 1;
        if (strcmp(name, "System RAM") != 0)
            continue;
        if (lo < end && base <= hi) {
            snprintf(why, n, "%#" PRIx64 "-%#" PRIx64 " is System RAM", lo, hi);
            touches = 1;
            break;
        }
    }
    fclose(f);
    if (touches)
        return 1;
    if (ranges == 0 || !nonzero) {
        snprintf(why, n, "%s shows no addresses (needs root)", path);
        return -1;
    }
    return 0;
}

uint64_t attic_mem_free(const char *path)
{
    char buf[8192];
    if (attic_read_file(path, buf, sizeof buf, NULL) != 0)
        return 0;
    const char *p = strstr(buf, "MemFree:");
    if (p == NULL)
        return 0;
    return strtoull(p + 8, NULL, 10) * 1024;
}

int attic_key_ok(const char *key)
{
    size_t n = strlen(key);
    if (n == 0 || n > ATTIC_KEY_MAX)
        return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)key[i];
        if (c < 0x21 || c > 0x7e || c == '"' || c == '\\')
            return 0;
    }
    return 1;
}

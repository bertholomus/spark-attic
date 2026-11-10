// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* proto.c: a request's JSON, read strictly, and a reply sent with its descriptor. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

#include "proto.h"

struct cursor {
    const char *p, *end;
};

static void space(struct cursor *c)
{
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r'))
        c->p++;
}

static int hexval(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    return -1;
}

/* A JSON string into out (ASCII only; a \u escape above 0x7f is refused): 0, or -1. */
static int string(struct cursor *c, char *out, size_t cap)
{
    size_t k = 0;
    if (c->p >= c->end || *c->p != '"')
        return -1;
    c->p++;
    while (c->p < c->end) {
        unsigned char ch = (unsigned char)*c->p++;
        if (ch == '"') {
            out[k] = 0;
            return 0;
        }
        if (ch < 0x20)
            return -1;
        if (ch == '\\') {
            if (c->p >= c->end)
                return -1;
            char e = *c->p++;
            switch (e) {
            case '"': ch = '"'; break;
            case '\\': ch = '\\'; break;
            case '/': ch = '/'; break;
            case 'b': ch = '\b'; break;
            case 'f': ch = '\f'; break;
            case 'n': ch = '\n'; break;
            case 'r': ch = '\r'; break;
            case 't': ch = '\t'; break;
            case 'u': {
                if (c->end - c->p < 4)
                    return -1;
                int v = 0;
                for (int i = 0; i < 4; i++) {
                    int h = hexval(c->p[i]);
                    if (h < 0)
                        return -1;
                    v = v * 16 + h;
                }
                c->p += 4;
                if (v == 0 || v > 0x7f)
                    return -1;
                ch = (unsigned char)v;
                break;
            }
            default:
                return -1;
            }
        }
        if (k + 1 >= cap)
            return -1;
        out[k++] = (char)ch;
    }
    return -1;
}

/* A JSON number. With out: a non-negative integer that fits in 63 bits. Without: any number, skipped. */
static int number(struct cursor *c, uint64_t *out)
{
    const char *s = c->p;
    if (c->p < c->end && *c->p == '-')
        c->p++;
    if (c->p >= c->end || *c->p < '0' || *c->p > '9')
        return -1;
    if (*c->p == '0' && c->p + 1 < c->end && c->p[1] >= '0' && c->p[1] <= '9')
        return -1; /* no leading zeros */
    while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
        c->p++;
    int integer = 1;
    if (c->p < c->end && *c->p == '.') {
        integer = 0;
        c->p++;
        if (c->p >= c->end || *c->p < '0' || *c->p > '9')
            return -1;
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
            c->p++;
    }
    if (c->p < c->end && (*c->p == 'e' || *c->p == 'E')) {
        integer = 0;
        c->p++;
        if (c->p < c->end && (*c->p == '+' || *c->p == '-'))
            c->p++;
        if (c->p >= c->end || *c->p < '0' || *c->p > '9')
            return -1;
        while (c->p < c->end && *c->p >= '0' && *c->p <= '9')
            c->p++;
    }
    if (out == NULL)
        return 0;
    if (!integer || *s == '-')
        return -1;
    uint64_t v = 0;
    for (const char *q = s; q < c->p; q++) {
        if (v > (UINT64_C(0x7fffffffffffffff) - (uint64_t)(*q - '0')) / 10)
            return -1;
        v = v * 10 + (uint64_t)(*q - '0');
    }
    *out = v;
    return 0;
}

static int word(struct cursor *c, const char *w)
{
    size_t n = strlen(w);
    if ((size_t)(c->end - c->p) < n || memcmp(c->p, w, n) != 0)
        return -1;
    c->p += n;
    return 0;
}

static int skip_value(struct cursor *c)
{
    char scratch[ATTIC_REQUEST_MAX + 1];
    if (c->p >= c->end)
        return -1;
    switch (*c->p) {
    case '"':
        return string(c, scratch, sizeof scratch);
    case 't':
        return word(c, "true");
    case 'f':
        return word(c, "false");
    case 'n':
        return word(c, "null");
    default:
        return number(c, NULL); /* objects and arrays are not ours to send: refused */
    }
}

int attic_parse_request(const char *buf, size_t len, struct attic_req *r, char *why, size_t n)
{
    struct cursor c = {buf, buf + len};
    int seen_key = 0, seen_op = 0;
    memset(r, 0, sizeof *r);
    if (len > ATTIC_REQUEST_MAX) {
        snprintf(why, n, "request longer than %d bytes", ATTIC_REQUEST_MAX);
        return -1;
    }
    space(&c);
    if (c.p >= c.end || *c.p != '{')
        goto bad;
    c.p++;
    space(&c);
    if (c.p < c.end && *c.p == '}') {
        c.p++;
    } else {
        for (;;) {
            char name[32];
            space(&c);
            if (string(&c, name, sizeof name) != 0)
                goto bad;
            space(&c);
            if (c.p >= c.end || *c.p != ':')
                goto bad;
            c.p++;
            space(&c);
            if (strcmp(name, "key") == 0) {
                if (seen_key++ || string(&c, r->key, sizeof r->key) != 0)
                    goto bad;
            } else if (strcmp(name, "op") == 0) {
                if (seen_op++ || string(&c, r->op, sizeof r->op) != 0)
                    goto bad;
            } else if (strcmp(name, "size") == 0) {
                if (r->has_size++ || number(&c, &r->size) != 0)
                    goto bad;
            } else if (skip_value(&c) != 0) {
                goto bad;
            }
            space(&c);
            if (c.p < c.end && *c.p == ',') {
                c.p++;
                continue;
            }
            if (c.p < c.end && *c.p == '}') {
                c.p++;
                break;
            }
            goto bad;
        }
    }
    space(&c);
    if (c.p != c.end)
        goto bad;
    if (!seen_key || !seen_op) {
        snprintf(why, n, "a request needs \"key\" and \"op\"");
        return -1;
    }
    return 0;
bad:
    snprintf(why, n, "not a request (one flat JSON object of strings and numbers)");
    return -1;
}

int attic_key_equal(const char *a, const char *b)
{
    size_t na = strlen(a), nb = strlen(b);
    unsigned diff = na != nb;
    for (size_t i = 0; i < ATTIC_KEY_MAX + 1; i++) {
        unsigned char x = i < na ? (unsigned char)a[i] : 0, y = i < nb ? (unsigned char)b[i] : 0;
        diff |= x ^ y;
    }
    return diff == 0;
}

void attic_json_text(const char *s, char *out, size_t cap)
{
    size_t k = 0;
    if (cap == 0)
        return;
    for (; *s && k + 7 < cap; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == '"' || ch == '\\') {
            out[k++] = '\\';
            out[k++] = (char)ch;
        } else if (ch < 0x20 || ch > 0x7e) {
            k += (size_t)snprintf(out + k, cap - k, "\\u%04x", ch);
        } else {
            out[k++] = (char)ch;
        }
    }
    out[k] = 0;
}

int attic_send_reply(int sock, const char *body, int fd)
{
    struct iovec iov = {(void *)body, strlen(body)};
    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } u;
    struct msghdr m;
    memset(&m, 0, sizeof m);
    m.msg_iov = &iov;
    m.msg_iovlen = 1;
    if (fd >= 0) {
        memset(&u, 0, sizeof u);
        m.msg_control = u.buf;
        m.msg_controllen = sizeof u.buf;
        struct cmsghdr *h = CMSG_FIRSTHDR(&m);
        h->cmsg_level = SOL_SOCKET;
        h->cmsg_type = SCM_RIGHTS;
        h->cmsg_len = CMSG_LEN(sizeof(int));
        memcpy(CMSG_DATA(h), &fd, sizeof fd);
    }
    for (;;) {
        ssize_t r = sendmsg(sock, &m, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (r < 0 && errno == EINTR)
            continue;
        return r == (ssize_t)iov.iov_len ? 0 : -1;
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * proto.h: the wire format our TensorFold client speaks, server side. One SOCK_SEQPACKET
 * message per request: a flat JSON object with "key" and "op" ("info", "alloc" with "size", or our own "status").
 * One JSON object per reply; an alloc's descriptor travels beside its reply (SCM_RIGHTS).
 */
#ifndef ATTIC_PROTO_H
#define ATTIC_PROTO_H

#include <stddef.h>
#include <stdint.h>

#include "attic.h"

struct attic_req {
    char key[ATTIC_KEY_MAX + 1];
    char op[32];
    int has_size;
    uint64_t size;
};

/* Reads one request. 0, or -1 with the reason in why: not one flat JSON object, a field we read of the wrong type,
 * a field given twice, or a string too long. Unknown fields are skipped (strings, numbers, true, false, null). */
int attic_parse_request(const char *buf, size_t len, struct attic_req *r, char *why, size_t n);

/* Compares two keys in time that does not depend on where they differ. */
int attic_key_equal(const char *a, const char *b);

/* s with quotes, backslashes and control bytes escaped for a JSON string, cut to fit out. */
void attic_json_text(const char *s, char *out, size_t cap);

/* One reply message on a SOCK_SEQPACKET socket, with descriptor fd beside it (SCM_RIGHTS) when fd >= 0. Never
 * blocks and never raises SIGPIPE: 0, or -1 (the client is gone or not reading). */
int attic_send_reply(int sock, const char *body, int fd);

#endif

// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* sha256.h: SHA-256 (FIPS 180-4), for the shim digest a VALIDATED artifact names. */
#ifndef ATTIC_SHA256_H
#define ATTIC_SHA256_H

#include <stddef.h>
#include <stdint.h>

struct sha256 {
    uint32_t h[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
};

void sha256_init(struct sha256 *s);
void sha256_update(struct sha256 *s, const void *data, size_t n);
void sha256_final(struct sha256 *s, uint8_t out[32]);
/* The file's digest as 64 lowercase hex digits: 0, or -errno. */
int sha256_file(const char *path, char hex[65]);

#endif

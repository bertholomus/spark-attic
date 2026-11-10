// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/* attic.h: what atticd's parts share. */
#ifndef ATTIC_H
#define ATTIC_H

#include <stddef.h>
#include <stdint.h>

#define ATTIC_SLICE ((uint64_t)2 << 20) /* lent memory comes in whole 2 MiB slices (our client's granule) */
#define ATTIC_DEFAULT_SOCKET "/run/spark-attic/attic.sock"
#define ATTIC_DEFAULT_KEY "spark-attic-1"
#define ATTIC_DEFAULT_ARTIFACTS "/var/lib/attic/artifacts"
#define ATTIC_DEFAULT_VERSION_FILE "/proc/driver/nvidia/version"
#define ATTIC_REPLY_MAX 4096 /* our client reads a reply of at most this */
#define ATTIC_REQUEST_MAX 1024
#define ATTIC_KEY_MAX 127

/* log.c-ish (util.c): one line on stderr, "atticd <level>: ..." (journald keeps the time). */
extern const char *attic_log_tag;
void attic_log(const char *level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#define LOGI(...) attic_log("info", __VA_ARGS__)
#define LOGW(...) attic_log("warn", __VA_ARGS__)
#define LOGE(...) attic_log("error", __VA_ARGS__)

/* The whole file into buf (NUL-terminated, at most cap-1 bytes): 0, or -errno. */
int attic_read_file(const char *path, char *buf, size_t cap, size_t *len);

/* The driver version in an NVRM version file ("... Kernel Module for aarch64  580.178.04  Release Build ..."): the
 * first token that is digits and dots. 0, or -1 with nothing found. */
int attic_driver_version(const char *path, char *out, size_t cap);

/* Whether [base, base+size) touches a "System RAM" range of an iomem file: 1 it does, 0 it does not, -1 the file is
 * unreadable or shows no addresses (only root sees them). */
int attic_iomem_touches_ram(const char *path, uint64_t base, uint64_t size, char *why, size_t n);

/* MemFree from a meminfo file, in bytes; 0 when unreadable. */
uint64_t attic_mem_free(const char *path);

/* A key: printable ASCII, no quote or backslash (it travels inside a JSON string unescaped), 1..ATTIC_KEY_MAX bytes. */
int attic_key_ok(const char *key);

#endif

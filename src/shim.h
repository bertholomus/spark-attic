// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * shim.h: the one interface between atticd (driver-independent) and an RM shim (rmshim.so, built per NVIDIA driver
 * against that driver's open-gpu-kernel-modules headers). Ours; it changes only together with ATTIC_SHIM_ABI.
 *
 * Every call returns 0 on success, or nonzero (an NV_STATUS, or -errno) with a one-line reason in err.
 */
#ifndef ATTIC_SHIM_H
#define ATTIC_SHIM_H

#include <stddef.h>
#include <stdint.h>

#define ATTIC_SHIM_ABI 1

/* Carveout region kinds, translated by the shim from the driver's own enum. */
enum { ATTIC_REGION_DISPLAY = 0, ATTIC_REGION_DISPLAY_RESERVED = 1, ATTIC_REGION_UEFI = 2, ATTIC_REGION_OTHER = 255 };

struct attic_region {
    uint64_t base, size;
    uint32_t kind;
};

struct attic_rm; /* one RM client holding the GPU's device and subdevice */

struct attic_shim {
    uint32_t abi;       /* ATTIC_SHIM_ABI */
    const char *driver; /* the NVIDIA driver version the shim was built for, e.g. "580.178.04" */
    const char *source; /* the headers it was built from (tag and commit) */
    /* An RM client on the GPU whose device node is /dev/nvidia<minor> (minor < 0: the only GPU). Refuses unless the
     * running driver is exactly `driver` (RM's strict version check). */
    int (*open)(struct attic_rm **rm, int minor, char *err, size_t errlen);
    void (*close)(struct attic_rm *rm);
    /* The carveout regions RM reports, at most `max`, in RM's order. */
    int (*carveout)(struct attic_rm *rm, struct attic_region *out, int max, int *n, char *err, size_t errlen);
    /* Mode heap: `size` contiguous bytes that RM takes from its own display (scanout) carveout heap. */
    int (*heap)(struct attic_rm *rm, uint64_t size, uint32_t *handle, char *err, size_t errlen);
    /* Mode list: the `size` bytes at physical `base`, wrapped as a memory-list object (needs root). */
    int (*list)(struct attic_rm *rm, uint64_t base, uint64_t size, uint32_t *handle, char *err, size_t errlen);
    /* A new descriptor carrying `handle`, for CUDA's POSIX-fd import (cuMemImportFromShareableHandle). */
    int (*export_fd)(struct attic_rm *rm, uint32_t handle, int *fd, char *err, size_t errlen);
    /* Frees our handle. RM keeps the memory while an exported descriptor or an importer still holds it. */
    int (*release)(struct attic_rm *rm, uint32_t handle, char *err, size_t errlen);
    /* What the last heap() asked RM for, for logs ("2MB pages" or "default pages"). */
    const char *(*detail)(struct attic_rm *rm);
};

/* The one symbol a shim exports. */
const struct attic_shim *attic_shim(void);
typedef const struct attic_shim *(*attic_shim_fn)(void);

#endif

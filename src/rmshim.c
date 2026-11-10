// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * rmshim.c: atticd's RM shim for one NVIDIA driver. attic-build.sh compiles it against NVIDIA's open-gpu-kernel-modules
 * headers at the running driver's exact tag (-DATTIC_DRIVER="<version>"), so every struct and number here is that
 * tag's own: an RM interface that changed fails the build, never the node.
 *
 * The calls, all on /dev/nvidiactl:
 *   NV_ESC_CHECK_VERSION_STR (strict: the running driver must be ATTIC_DRIVER), NV_ESC_CARD_INFO
 *   NV_ESC_RM_ALLOC: client NV01_ROOT_CLIENT -> device NV01_DEVICE_0 -> subdevice NV20_SUBDEVICE_0
 *   NV2080_CTRL_CMD_FB_GET_CARVEOUT_REGION_INFO: the carveout regions (ctrl2080fb.h), each type by its name in that
 *     header (carveout_kind.h: the enum is renumbered between drivers)
 *   mode heap: NV01_MEMORY_SYSTEM with NVOS32_ATTR2_ISO_YES. On GB10 RM places ISO system memory in its own scanout
 *     carveout heap (mem_utils.c memUtilsAllocMemDesc, mem_mgr_gb10b_phys.c), and refcounts the heap block across
 *     dups (export, import), so the range stays out of every other allocation until its last holder lets go.
 *   mode list: NV01_MEMORY_LIST_SYSTEM over physical pages (cl84a0.h, mem_list.c; RS_FLAGS_ALLOC_PRIVILEGED: root).
 *   NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD into a fresh /dev/nvidiactl descriptor (ctrl0000unix.h, os.c).
 * RM lets a client allocate a device only while the caller holds the GPU's /dev/nvidiaN open (device.c,
 * nv_is_gpu_accessible), so the shim keeps it open.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "nvtypes.h"
#include "nvmisc.h"
#include "nvstatus.h"
#include "nvos.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl003e.h"
#include "class/cl84a0.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0000/ctrl0000unix.h"
#include "ctrl/ctrl2080/ctrl2080fb.h"
#include "nv-ioctl.h"
#include "nv_escape.h"

#include "shim.h"
#include "carveout_kind.h"

#ifndef ATTIC_DRIVER
#error "build with -DATTIC_DRIVER=\"<driver version>\" (scripts/attic-build.sh)"
#endif
#ifndef ATTIC_SOURCE
#define ATTIC_SOURCE "NVIDIA open-gpu-kernel-modules " ATTIC_DRIVER
#endif

/* RM_PAGE_SHIFT (src/nvidia/inc/kernel/gpu/mem_mgr/rm_page_size.h): a memory list's page numbers are in these units
 * (mem_list.c shifts each one left by it). */
#define ATTIC_RM_PAGE_SHIFT 12
#define ATTIC_ALIGN (2ull << 20)
#define ATTIC_MAX_CARDS 32
#define ATTIC_HANDLE_BASE 0xd0e00000u
#define ATTIC_OWNER 0x41544943u /* "ATIC": the heap owner RM records for our allocations (stdmemValidateParams) */

_Static_assert(sizeof(NvHandle) == sizeof(uint32_t), "RM handles are 32-bit");

struct attic_rm {
    int ctl; /* /dev/nvidiactl: the RM client lives on it */
    int dev; /* /dev/nvidiaN: held open so RM lets this process use the GPU */
    NvHandle client, device, subdevice;
    NvU32 next;
    const char *pages; /* what the last heap() used */
    int no_huge;       /* RM refused 2 MiB pages for this memory once: its default since */
};

static void say(char *err, size_t n, const char *fmt, ...)
{
    if (err == NULL || n == 0)
        return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
}

static const char *status_name(NvU32 s)
{
    switch (s) {
#undef SDK_NVSTATUSCODES_H
#undef NV_STATUS_CODE
#define NV_STATUS_CODE(name, code, string) \
    case code:                             \
        return #name;
#include "nvstatuscodes.h"
#undef NV_STATUS_CODE
    }
    return "unknown status";
}

static int nv_ioctl(int fd, unsigned nr, void *arg, size_t size)
{
    for (;;) {
        if (ioctl(fd, _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, nr, size), arg) == 0)
            return 0;
        if (errno != EINTR && errno != EAGAIN)
            return -errno;
    }
}

/* An RM call's result: the ioctl's errno, else RM's status. */
static int rm_result(int rc, NvU32 status, const char *what, char *err, size_t n)
{
    if (rc != 0) {
        say(err, n, "%s: ioctl: %s", what, strerror(-rc));
        return rc;
    }
    if (status != NV_OK) {
        say(err, n, "%s: %s (0x%x)", what, status_name(status), status);
        return (int)status;
    }
    return 0;
}

static int rm_alloc(struct attic_rm *rm, NvHandle parent, NvHandle object, NvU32 cls, void *params, NvU32 size,
                    const char *what, char *err, size_t n)
{
    NVOS21_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hRoot = rm->client;
    p.hObjectParent = parent;
    p.hObjectNew = object;
    p.hClass = cls;
    p.pAllocParms = NV_PTR_TO_NvP64(params);
    p.paramsSize = size;
    int rc = nv_ioctl(rm->ctl, NV_ESC_RM_ALLOC, &p, sizeof p);
    return rm_result(rc, p.status, what, err, n);
}

static int rm_control(struct attic_rm *rm, NvHandle object, NvU32 cmd, void *params, NvU32 size, const char *what,
                      char *err, size_t n)
{
    NVOS54_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hClient = rm->client;
    p.hObject = object;
    p.cmd = cmd;
    p.params = NV_PTR_TO_NvP64(params);
    p.paramsSize = size;
    int rc = nv_ioctl(rm->ctl, NV_ESC_RM_CONTROL, &p, sizeof p);
    return rm_result(rc, p.status, what, err, n);
}

static int rm_free(struct attic_rm *rm, NvHandle parent, NvHandle object, char *err, size_t n)
{
    NVOS00_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hRoot = rm->client;
    p.hObjectParent = parent;
    p.hObjectOld = object;
    int rc = nv_ioctl(rm->ctl, NV_ESC_RM_FREE, &p, sizeof p);
    return rm_result(rc, p.status, "NV_ESC_RM_FREE", err, n);
}

static void shim_close(struct attic_rm *rm)
{
    if (rm == NULL)
        return;
    if (rm->client != 0) /* freeing the client frees the device, subdevice and every object we still hold */
        rm_free(rm, 0, rm->client, NULL, 0);
    if (rm->dev >= 0)
        close(rm->dev);
    if (rm->ctl >= 0)
        close(rm->ctl);
    free(rm);
}

static int shim_open(struct attic_rm **out, int minor, char *err, size_t n)
{
    struct attic_rm *rm = calloc(1, sizeof *rm);
    int rc = -ENOMEM;
    *out = NULL;
    if (rm == NULL) {
        say(err, n, "out of memory");
        return rc;
    }
    rm->ctl = rm->dev = -1;
    rm->next = ATTIC_HANDLE_BASE;
    rm->pages = "none yet";

    rm->ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (rm->ctl < 0) {
        rc = -errno;
        say(err, n, "open /dev/nvidiactl: %s", strerror(errno));
        goto fail;
    }

    nv_ioctl_rm_api_version_t v;
    memset(&v, 0, sizeof v);
    v.cmd = NV_RM_API_VERSION_CMD_STRICT;
    snprintf(v.versionString, sizeof v.versionString, "%s", ATTIC_DRIVER);
    rc = nv_ioctl(rm->ctl, NV_ESC_CHECK_VERSION_STR, &v, sizeof v);
    if (rc != 0) {
        say(err, n, "RM refused version %s (strict check): the running driver is another one", ATTIC_DRIVER);
        goto fail;
    }

    static nv_ioctl_card_info_t cards[ATTIC_MAX_CARDS];
    memset(cards, 0, sizeof cards);
    rc = nv_ioctl(rm->ctl, NV_ESC_CARD_INFO, cards, sizeof cards);
    if (rc != 0) {
        say(err, n, "NV_ESC_CARD_INFO: %s", strerror(-rc));
        goto fail;
    }
    int pick = -1, valid = 0;
    for (int i = 0; i < ATTIC_MAX_CARDS; i++) {
        if (!cards[i].valid)
            continue;
        valid++;
        if (minor < 0 ? pick < 0 : cards[i].minor_number == (NvU32)minor)
            pick = i;
    }
    if (pick < 0 || (minor < 0 && valid != 1)) {
        rc = -ENODEV;
        say(err, n, "%d GPUs found and none picked (set --gpu-minor)", valid);
        goto fail;
    }
    char path[32];
    snprintf(path, sizeof path, "/dev/nvidia%u", cards[pick].minor_number);
    rm->dev = open(path, O_RDWR | O_CLOEXEC);
    if (rm->dev < 0) {
        rc = -errno;
        say(err, n, "open %s: %s", path, strerror(errno));
        goto fail;
    }

    NV0000_ALLOC_PARAMETERS cp;
    memset(&cp, 0, sizeof cp);
    cp.processID = (NvU32)getpid();
    snprintf(cp.processName, sizeof cp.processName, "atticd");
    NVOS21_PARAMETERS a;
    memset(&a, 0, sizeof a);
    a.hClass = NV01_ROOT_CLIENT;
    a.pAllocParms = NV_PTR_TO_NvP64(&cp);
    a.paramsSize = sizeof cp;
    rc = rm_result(nv_ioctl(rm->ctl, NV_ESC_RM_ALLOC, &a, sizeof a), a.status, "NV01_ROOT_CLIENT", err, n);
    if (rc != 0)
        goto fail;
    rm->client = a.hObjectNew != 0 ? a.hObjectNew : cp.hClient;

    NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS id;
    memset(&id, 0, sizeof id);
    id.gpuId = cards[pick].gpu_id;
    rc = rm_control(rm, rm->client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, &id, sizeof id, "GPU_GET_ID_INFO_V2", err, n);
    if (rc != 0)
        goto fail;

    NV0080_ALLOC_PARAMETERS dp;
    memset(&dp, 0, sizeof dp);
    dp.deviceId = id.deviceInstance;
    rm->device = rm->next++;
    rc = rm_alloc(rm, rm->client, rm->device, NV01_DEVICE_0, &dp, sizeof dp, "NV01_DEVICE_0", err, n);
    if (rc != 0) {
        rm->device = 0;
        goto fail;
    }

    NV2080_ALLOC_PARAMETERS sp;
    memset(&sp, 0, sizeof sp);
    sp.subDeviceId = id.subDeviceInstance;
    rm->subdevice = rm->next++;
    rc = rm_alloc(rm, rm->device, rm->subdevice, NV20_SUBDEVICE_0, &sp, sizeof sp, "NV20_SUBDEVICE_0", err, n);
    if (rc != 0) {
        rm->subdevice = 0;
        goto fail;
    }
    *out = rm;
    return 0;

fail:
    shim_close(rm);
    return rc != 0 ? rc : -EIO;
}

static int shim_carveout(struct attic_rm *rm, struct attic_region *out, int max, int *count, char *err, size_t n)
{
    NV2080_CTRL_FB_GET_CARVEOUT_REGION_INFO_PARAMS p;
    memset(&p, 0, sizeof p);
    *count = 0;
    int rc = rm_control(rm, rm->subdevice, NV2080_CTRL_CMD_FB_GET_CARVEOUT_REGION_INFO, &p, sizeof p,
                        "FB_GET_CARVEOUT_REGION_INFO", err, n);
    if (rc != 0)
        return rc;
    NvU32 k = p.numCarveoutRegions;
    if (k > NV2080_CTRL_FB_GET_CARVEOUT_REGION_INFO_MAX_ENTRIES) {
        say(err, n, "RM reported %u carveout regions, more than its own maximum %u", k,
            (unsigned)NV2080_CTRL_FB_GET_CARVEOUT_REGION_INFO_MAX_ENTRIES);
        return -EPROTO;
    }
    for (NvU32 i = 0; i < k && (int)i < max; i++) {
        out[i].base = p.carveoutRegion[i].base;
        out[i].size = p.carveoutRegion[i].size;
        out[i].kind = region_kind((NvU32)p.carveoutRegion[i].carveoutType);
        (*count)++;
    }
    return 0;
}

static int heap_try(struct attic_rm *rm, uint64_t size, NvU32 page_attr, NvU32 page_attr2, NvHandle h, char *err,
                    size_t n)
{
    NV_MEMORY_ALLOCATION_PARAMS p;
    memset(&p, 0, sizeof p);
    p.owner = ATTIC_OWNER;
    p.type = NVOS32_TYPE_IMAGE;
    p.flags = NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE;
    p.attr = DRF_DEF(OS32, _ATTR, _LOCATION, _PCI) | DRF_DEF(OS32, _ATTR, _PHYSICALITY, _CONTIGUOUS) |
             DRF_DEF(OS32, _ATTR, _COHERENCY, _UNCACHED) | page_attr;
    /* ISO: the display's kind of memory, which RM takes from the scanout carveout. GPU_CACHEABLE must be asked for:
     * RM turns DEFAULT into NO for system memory (mem.c), and a KV pool wants the GPU's L2. */
    p.attr2 = DRF_DEF(OS32, _ATTR2, _ISO, _YES) | DRF_DEF(OS32, _ATTR2, _GPU_CACHEABLE, _YES) | page_attr2;
    p.size = size;
    p.alignment = ATTIC_ALIGN;
    return rm_alloc(rm, rm->device, h, NV01_MEMORY_SYSTEM, &p, sizeof p, "NV01_MEMORY_SYSTEM (ISO)", err, n);
}

static int shim_heap(struct attic_rm *rm, uint64_t size, uint32_t *handle, char *err, size_t n)
{
    if (size == 0 || size % ATTIC_ALIGN != 0) {
        say(err, n, "size %llu is not a whole number of 2 MiB slices", (unsigned long long)size);
        return -EINVAL;
    }
    NvHandle h = rm->next++;
    int rc = -EINVAL;
    /* 2 MiB GPU pages first; RM may refuse them for system memory, then its default from then on. */
    if (!rm->no_huge) {
        rc = heap_try(rm, size, DRF_DEF(OS32, _ATTR, _PAGE_SIZE, _HUGE), DRF_DEF(OS32, _ATTR2, _PAGE_SIZE_HUGE, _2MB),
                      h, err, n);
        if (rc == (int)NV_ERR_INVALID_ARGUMENT || rc == (int)NV_ERR_NOT_SUPPORTED)
            rm->no_huge = 1;
        else if (rc == 0)
            rm->pages = "2MB pages";
    }
    if (rm->no_huge) {
        rc = heap_try(rm, size, DRF_DEF(OS32, _ATTR, _PAGE_SIZE, _DEFAULT), 0, h, err, n);
        if (rc == 0)
            rm->pages = "default pages";
    }
    if (rc == 0)
        *handle = h;
    return rc;
}

static int shim_list(struct attic_rm *rm, uint64_t base, uint64_t size, uint32_t *handle, char *err, size_t n)
{
    if (size == 0 || size % ATTIC_ALIGN != 0 || base % ATTIC_ALIGN != 0) {
        say(err, n, "range %#llx+%#llx is not 2 MiB aligned", (unsigned long long)base, (unsigned long long)size);
        return -EINVAL;
    }
    NvU64 page = base >> ATTIC_RM_PAGE_SHIFT; /* contiguous: one page number, the first */
    NV_MEMORY_LIST_ALLOCATION_PARAMS p;
    memset(&p, 0, sizeof p);
    p.hClient = NV01_NULL_OBJECT;
    p.hParent = NV01_NULL_OBJECT;
    p.hObject = NV01_NULL_OBJECT;
    p.pageCount = 1;
    p.pageNumberList = NV_PTR_TO_NvP64(&page);
    p.limit = size - 1;
    p.heapOwner = ATTIC_OWNER;
    p.flagsOs02 = DRF_DEF(OS02, _FLAGS, _PHYSICALITY, _CONTIGUOUS) | DRF_DEF(OS02, _FLAGS, _LOCATION, _PCI) |
                  DRF_DEF(OS02, _FLAGS, _COHERENCY, _UNCACHED) | DRF_DEF(OS02, _FLAGS, _GPU_CACHEABLE, _YES) |
                  DRF_DEF(OS02, _FLAGS, _MAPPING, _NO_MAP);
    NvHandle h = rm->next++;
    int rc = rm_alloc(rm, rm->device, h, NV01_MEMORY_LIST_SYSTEM, &p, sizeof p, "NV01_MEMORY_LIST_SYSTEM", err, n);
    if (rc == 0)
        *handle = h;
    return rc;
}

static int shim_export(struct attic_rm *rm, uint32_t handle, int *out, char *err, size_t n)
{
    int fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        say(err, n, "open /dev/nvidiactl for export: %s", strerror(errno));
        return -errno;
    }
    NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS p;
    memset(&p, 0, sizeof p);
    p.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
    p.object.data.rmObject.hDevice = rm->device;
    p.object.data.rmObject.hParent = rm->device;
    p.object.data.rmObject.hObject = handle;
    p.fd = fd;
    p.flags = 0;
    int rc = rm_control(rm, rm->client, NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD, &p, sizeof p,
                        "OS_UNIX_EXPORT_OBJECT_TO_FD", err, n);
    if (rc != 0) {
        close(fd);
        return rc;
    }
    *out = fd;
    return 0;
}

static int shim_release(struct attic_rm *rm, uint32_t handle, char *err, size_t n)
{
    return rm_free(rm, rm->device, handle, err, n);
}

static const char *shim_detail(struct attic_rm *rm)
{
    return rm->pages;
}

static const struct attic_shim shim = {
    .abi = ATTIC_SHIM_ABI,
    .driver = ATTIC_DRIVER,
    .source = ATTIC_SOURCE,
    .open = shim_open,
    .close = shim_close,
    .carveout = shim_carveout,
    .heap = shim_heap,
    .list = shim_list,
    .export_fd = shim_export,
    .release = shim_release,
    .detail = shim_detail,
};

__attribute__((visibility("default"))) const struct attic_shim *attic_shim(void)
{
    return &shim;
}

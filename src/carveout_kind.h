// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2026 bertholomus
/*
 * carveout_kind.h: region_kind(), the RM shim's map from the type RM gives a carveout region to spark-attic's region
 * kind (shim.h). Ours. Every type is named from the driver's own ctrl2080fb.h, never by number, because NVIDIA
 * renumbers that enum: 580.178.04 to 615.71.10 have DISPLAY_FRM 0, DISPLAY_FRM_RESERVED 1, UEFI 2; 615.78.08 dropped
 * DISPLAY_FRM_RESERVED, so its UEFI is 1. attic-build.sh sets ATTIC_HAS_DISPLAY_FRM_RESERVED to 1 only when that
 * driver's header declares the enumerator (scripts/attic-carveout-types.sh asks the compiler). A type the header does
 * not name here comes out as ATTIC_REGION_OTHER, and atticd refuses to lend beside a region it cannot place.
 *
 * rmshim.c includes it after ctrl2080fb.h and shim.h; tests/test_carveout_kind.sh builds it against fake headers of
 * both layouts.
 */
#ifndef ATTIC_CARVEOUT_KIND_H
#define ATTIC_CARVEOUT_KIND_H

#ifndef ATTIC_HAS_DISPLAY_FRM_RESERVED
#error "ATTIC_HAS_DISPLAY_FRM_RESERVED is not set: build the shim with scripts/attic-build.sh"
#endif

static uint32_t region_kind(NvU32 type)
{
    switch (type) {
    case NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE_DISPLAY_FRM:
        return ATTIC_REGION_DISPLAY;
#if ATTIC_HAS_DISPLAY_FRM_RESERVED
    case NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE_DISPLAY_FRM_RESERVED:
        return ATTIC_REGION_DISPLAY_RESERVED;
#endif
    case NV2080_CTRL_FB_GET_CARVEOUT_REGION_CARVEOUT_TYPE_UEFI:
        return ATTIC_REGION_UEFI;
    }
    return ATTIC_REGION_OTHER;
}

#endif

# spark-attic 0.1.5

> Licence: `MIT OR Apache-2.0`.

A display-guard fix on 0.1.4 for Ubuntu nodes. Nothing changes on stock DGX OS.

## In this round

- **The firmware console no longer reads as a display.** Ubuntu keeps the UEFI boot framebuffer as a DRM card (simpledrm) with one virtual connector, `card0-Unknown-1`, that always reads `connected`, on a headless box too. Both the installer's display check (`atticd --probe-display`) and the daemon scanned every connector in `/sys/class/drm`, so on Ubuntu the install failed with `FAIL display: a display is in use` and atticd would never lend. Stock DGX OS has no such card, which is why the node runs did not catch it. Reported by a user on two Ubuntu GB10 nodes, driver 580.178.04, who worked around it with `--drm-dir` on an empty folder.
- Now a connector counts unless its card's device or that device's driver is a firmware framebuffer (`simple-framebuffer`, `simpledrm`, `efi-framebuffer`, `efidrm`, `vesa-framebuffer`, `vesadrm`, `ofdrm`). A card whose links cannot be read still counts. The NVIDIA card's own connectors count as before, and the modeset-holder and graphics-process signals are unchanged.
- **Bring-up proves the console cannot write into a lent slice.** When a firmware console is present, atticd lends only after `/proc/iomem` shows its framebuffer (`BOOTFB`) outside every display region. A framebuffer inside the display carveout, or one that cannot be placed (no addresses readable), refuses with the reason. On a GB10 the UEFI region sits right after the display region (for example display `0x280200000+2046 MiB`, UEFI `0x300000000+48 MiB`).
- `atticd --probe-display` prints a `skip drm-connector:` line naming the firmware console it skipped.
- If you used `--drm-dir <empty dir>` as a workaround, you can drop it on 0.1.5: the default `/sys/class/drm` now keeps the NVIDIA connector check live as well.

## Tested

- The host suite (`make check`) passes on the build host.
- New cases fail on 0.1.4 and pass on 0.1.5: `case_firmware_console` (a Ubuntu-shaped tree with simpledrm on `simple-framebuffer.0` and the NVIDIA card: the probe and the daemon skip the console and lend; a connected NVIDIA connector still refuses; unreadable card links still count; a framebuffer inside the carveout refuses; unreadable iomem refuses) and the iomem/name unit checks in `test_units`.
- The GB10 nodes on stock DGX OS take the release with no change in behaviour (no firmware console card there).

## Known limits

- Unchanged from 0.1.4 (README, "Known limits"): mode **list** is still unreleased, the plugins keep their engine-shape limits, and the display guard still checks before each new lend, not mid-lend.

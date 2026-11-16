# spark-attic 0.1.3

> Licence: `MIT OR Apache-2.0`.

A feature round on 0.1.2, six pieces from the November review round, all measured on the build host against a GB10 box's shape.

## In this round

- **The KV carveout can take the pool's head.** With `SPARK_ATTIC_KV_HEAD=1` the daemon lends so that KV block 0 lands in the carveout region instead of the tail, so a paged pool's hottest blocks sit in the fast memory. `attic-check --phase bench` times it: 576-byte random gathers and linear copies on the imported slice beside an ordinary allocation of the same size, so the effect of the carveout on gather-heavy reads is on the table.
- **The granted page size is reported.** The status file, the info reply and `attic-check` name whether the heap was granted 2 MiB pages or the driver's default, and the fake driver models both answers: a driver that refuses 2 MiB pages for system memory and one that grants them.
- **Weights land in the carveout at model construction.** Both loaders hook the framework's model-construction seam, so the carveout storage for the weights is allocated before `load_weights` runs and the loader fills it directly; the post-load move now only handles what a postprocessor swapped since. The Python client and both plugins pre-place under their weights knobs and skip what is already carved, and a carveout cap smaller than the params refuses before anything is loaded.
- **Forward compatibility is checked, not assumed.** `attic-check --phase compat` borrows one slice and runs it through the CUDA compatibility library a host can ship over an older driver: a 16 MiB probe is imported, mapped and copied through, then `cuMemGetAddressRange` and the pointer memory-type attribute are checked against the truth, and the report names the `libcuda.so.1` that was really loaded and the driver API version. The fake CUDA library can lie in both ways (a wrong virtual-address range, a wrong memory type) and the tests walk both failure reports.
- **A plugged-in console cannot corrupt a lent slice, and the docs say why.** Every slice is asked for as isolatable, GPU-cacheable memory, which puts it in RM's refcounted exclusive carveout blocks: the console's scanout surfaces are allocations of the same class and can never alias a block that is lent out. While any display signal is up, the daemon refuses new lends entirely and leaves slices already lent with their clients; giveback works during the block and lending resumes after the console leaves, and a host test walks the whole sequence.
- **The headless node trim is one ops tool.** `node-trim.sh` boots a node to console (the multi-user target, gdm3 off), disables bluetooth, cups, snapd, the DGX dashboard and its admin service and the firmware-update timer, sets `vm.watermark_boost_factor = 0` through a sysctl drop-in, takes `splash` off the kernel command line, and compacts memory before any client borrows. `--check` prints every step and touches nothing; every change records its enable-back line in an undo record before it is made; the steps are idempotent and release-agnostic.

## Tested

- Each piece fails on 0.1.2 (no knob, no phase, no seam, no compat report) or is regression-proofing on it (the scanout case, whose guard predates this round; the analysis is the new content) and passes here. The node-side verdicts - the compat phase over a real compatibility library, and the trim itself - stay the operator's, run on the node.
- The host suite (`make check`) passes on the build host: the units, the server (the new bench, compat and scanout cases included), the validator, the installer and rollback paths, the CLI, the Python client and plugins, the drift, preflight, mem-reserve and memory-trace tests, both plugin source checks (vLLM and SGLang, cloned at their tags), and the new trim suite. No GPU, no zig, no tag cache needed.

## Known limits

- Unchanged from 0.1.2 (README, "Known limits"): mode **list** is still unreleased, the plugins keep their engine-shape limits, and the display guard still checks before each new lend, not mid-lend.

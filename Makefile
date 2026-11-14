# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# spark-attic. Host build and tests: make && make check. Node build (aarch64 GB10, glibc 2.28+, no compiler on the
# node): make OUT=build/aarch64 CC="zig cc -target aarch64-linux-gnu.2.28". The RM shim is built per NVIDIA driver by
# scripts/attic-build.sh, against that driver's own headers.
CC ?= cc
OUT ?= build/host
CFLAGS ?= -std=gnu11 -O2 -g -Wall -Wextra -Wformat=2 -Wno-format-nonliteral -fstack-protector-strong -D_FORTIFY_SOURCE=2
# The release build maps the build tree's and the zig toolchain's paths out of the debug info
# (M4, REVIEW-1112 finding 1): release/build-release.sh fills CFLAGS_PATHMAP.
CFLAGS += $(CFLAGS_PATHMAP)
COMMON := src/util.c src/proto.c src/lend.c src/artifact.c src/sha256.c
HEADERS := $(wildcard src/*.h)

all: $(OUT)/atticd $(OUT)/attic-check $(OUT)/fakeshim.so

# make demo: watch the install on a simulated GB10 from a fresh clone (no node, no root, nothing installed)
demo: all $(OUT)/fakecuda/libcuda.so.1
	bash install.sh --demo

$(OUT)/atticd: src/atticd.c $(COMMON) $(HEADERS) | $(OUT)
	$(CC) $(CFLAGS) -Isrc -o $@ src/atticd.c $(COMMON) -ldl

$(OUT)/attic-check: src/attic-check.c src/util.c src/proto.c $(HEADERS) | $(OUT)
	$(CC) $(CFLAGS) -Isrc -o $@ src/attic-check.c src/util.c src/proto.c -ldl

$(OUT)/fakeshim.so: src/fakeshim.c src/shim.h | $(OUT)
	$(CC) $(CFLAGS) -fPIC -shared -fvisibility=hidden -Isrc -o $@ src/fakeshim.c

$(OUT)/test_units: tests/test_units.c $(COMMON) $(HEADERS) | $(OUT)
	$(CC) $(CFLAGS) -Isrc -o $@ tests/test_units.c $(COMMON)

$(OUT)/fakecuda/libcuda.so.1: tests/fakecuda.c | $(OUT)
	mkdir -p $(OUT)/fakecuda
	$(CC) $(CFLAGS) -fPIC -shared -o $@ tests/fakecuda.c

# Every upstream release the plugins support, when its clone is in cache/upstream/ (scripts/attic-engine-drift.sh
# fetches the newest tags on top; a clone there that no shape matches fails the upstream case: drift). Each test
# takes only its own engine's sources: the other test refuses foreign --*-src arguments.
VLLM_UPSTREAM := $(foreach v,vllm-0.24.0 vllm-0.32.0 vllm-0.33.0,$(if $(wildcard cache/upstream/$(v)),--vllm-src cache/upstream/$(v),))
SGLANG_UPSTREAM := $(foreach v,sglang-0.5.14 sglang-0.5.21 sglang-0.5.23,$(if $(wildcard cache/upstream/$(v)),--sglang-src cache/upstream/$(v),))

check: all $(OUT)/test_units $(OUT)/fakecuda/libcuda.so.1
	$(OUT)/test_units
	python3 -I tests/test_server.py --bin $(OUT)
	python3 -I tests/test_validate.py --bin $(OUT)
	python3 -I tests/test_python.py --bin $(OUT)
	python3 -I tests/test_vllm_plugin.py --bin $(OUT) $(VLLM_UPSTREAM)
	python3 -I tests/test_sglang_plugin.py --bin $(OUT) $(SGLANG_UPSTREAM)
	python3 -I tests/test_bench.py
	CC="$(CC)" tests/test_carveout_kind.sh
	tests/test_m4.sh
	python3 -I tests/test_preflight.py
	tests/test_install.sh --bin $(OUT)
	python3 -I tests/test_cli.py

$(OUT):
	mkdir -p $@

clean:
	rm -rf build/host build/aarch64

.PHONY: all demo check check-release check-update clean

# Needs network on first use (the NVIDIA tag) and the zig toolchain; slower than check.
check-update:
	tests/test_update.sh

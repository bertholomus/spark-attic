# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# spark-attic. Host build and tests: make && make check. Node build (aarch64 GB10, glibc 2.28+, no compiler on the
# node): make OUT=build/aarch64 CC="zig cc -target aarch64-linux-gnu.2.28". The RM shim is built per NVIDIA driver by
# scripts/attic-build.sh, against that driver's own headers.
CC ?= cc
OUT ?= build/host
CFLAGS ?= -std=gnu11 -O2 -g -Wall -Wextra -Wformat=2 -Wno-format-nonliteral -fstack-protector-strong -D_FORTIFY_SOURCE=2
COMMON := src/util.c src/proto.c src/lend.c src/artifact.c src/sha256.c
HEADERS := $(wildcard src/*.h)

all: $(OUT)/atticd $(OUT)/attic-check $(OUT)/fakeshim.so

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

check: all $(OUT)/test_units $(OUT)/fakecuda/libcuda.so.1
	$(OUT)/test_units
	python3 -I tests/test_server.py --bin $(OUT)
	python3 -I tests/test_validate.py --bin $(OUT)
	python3 -I tests/test_python.py --bin $(OUT)
	python3 -I tests/test_vllm_plugin.py --bin $(OUT) --vllm-src cache/upstream/vllm-0.24.0
	python3 -I tests/test_sglang_plugin.py --bin $(OUT) --sglang-src cache/upstream/sglang-0.5.14
	python3 -I tests/test_bench.py
	CC="$(CC)" tests/test_carveout_kind.sh

$(OUT):
	mkdir -p $@

clean:
	rm -rf build/host build/aarch64

.PHONY: all check check-update clean

# Needs network on first use (the NVIDIA tag) and the zig toolchain; slower than check.
check-update:
	tests/test_update.sh

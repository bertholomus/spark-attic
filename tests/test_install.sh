#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# test_install.sh: install.sh and its rollback on the host, against a fake GB10 node: a fake root
# (--root), a fake /proc/driver/nvidia/version, a fake systemctl, a fake ss and the fake shim
# (build/host's fakeshim.so + fakecuda, the same pair test_validate.py uses). Covers the preflight
# refusals, the checksum refusal, the idempotent re-run, the upgrade, the downgrade refusal, the
# rollback after a failed validate, the key that is never overwritten, the uninstall that leaves
# nothing, and plain versus colour output.
#   usage: tests/test_install.sh --bin build/host
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
BIN="build/host"
while [ $# -gt 0 ]; do
  case $1 in
    --bin) shift; BIN=$1 ;;
    *) echo "usage: tests/test_install.sh --bin build/host" >&2; exit 2 ;;
  esac
  shift
done
[ -x "$BIN/atticd" ] && [ -x "$BIN/attic-check" ] && [ -f "$BIN/fakeshim.so" ] ||
  { echo "test_install: $BIN is not a host build (make first)" >&2; exit 2; }

T=$(mktemp -d /tmp/test-install.XXXXXX)
[ -n "${TEST_INSTALL_KEEP:-}" ] || trap 'rm -rf "$T"' EXIT
fails=0
ok() { if [ "$1" = 0 ]; then echo "  ok   $2"; else echo "  FAIL $2"; fails=$((fails + 1)); fi; }
HAVE=$(command -v sha256sum diff find) || { echo "test_install: need GNU coreutils" >&2; exit 2; }

DRIVER=999.0.1
KEY_EXPECT=attic_

# ---------------------------------------------------------------- the fake node and its bins
fakebin() { # $1 = dir: uname (aarch64), systemctl (stateful), ss (silent), optional df
  local d=$1
  mkdir -p "$d"
  printf '#!/bin/sh\necho aarch64\n' >"$d/uname"
  cat >"$d/systemctl" <<EOF
#!/bin/bash
# fake systemctl: state in \$ATTIC_FAKE_STATE (a directory), calls appended to \$ATTIC_FAKE_LOG
S=\${ATTIC_FAKE_STATE:?}; L=\${ATTIC_FAKE_LOG:?}
echo "\$*" >>"\$L"
active() { [ -f "\$S/active" ]; }
enabled() { [ -f "\$S/enabled" ]; }
fake_status() {
  if [ -n "\${ATTIC_FAKE_STATUS_FILE:-}" ]; then
    mkdir -p "\$(dirname "\$ATTIC_FAKE_STATUS_FILE")"
    printf '{"name":"spark-attic","version":"%s","state":"ready","why":"","driver":"%s","mode":"heap","validated":true,"validated_at":"2026-11-12T00:00:00Z","size_mib":2044,"free_mib":2044,"largest_mib":2044,"lent_mib":0,"draining_mib":0,"clients":0,"every_s":10,"pid":1,"updated":%s}\n' \
      "\${ATTIC_FAKE_VERSION:-0.1.0}" "$DRIVER" "\$(date +%s)" >"\$ATTIC_FAKE_STATUS_FILE"
  fi
}
case "\$1 \$2" in
  "is-active --quiet") if active; then exit 0; else exit 1; fi ;;
  "is-active attic.service") if active; then echo active; exit 0; else echo inactive; exit 1; fi ;;
  "is-enabled attic.service") if enabled; then echo enabled; exit 0; else echo disabled; exit 1; fi ;;
  "daemon-reload"*)
    if [ -n "\${ATTIC_FAKE_PAUSE:-}" ] && [ ! -f "\$ATTIC_FAKE_PAUSE" ]; then
      : >"\$ATTIC_FAKE_PAUSE"; sleep "\${ATTIC_FAKE_PAUSE_S:-3}"
    fi
    exit 0 ;;
  "stop attic.service") rm -f "\$S/active"; exit 0 ;;
  "disable attic.service"|"disable --now") rm -f "\$S/enabled" "\$S/active"; exit 0 ;;
  "start attic.service"|"enable --now"*)
    echo enabled >"\$S/enabled"; echo active >"\$S/active"; fake_status; exit 0 ;;
  "enable attic.service") echo enabled >"\$S/enabled"; exit 0 ;;
  *) echo "fake systemctl: does not handle: \$*" >&2; exit 1 ;;
esac
EOF
  printf '#!/bin/sh\n[ -n "$ATTIC_SS_LISTEN" ] && echo "u_str LISTEN 0 1 $ATTIC_SS_LISTEN 1 * 0"\nexit 0\n' >"$d/ss"
  chmod 755 "$d"/*
  [ -n "${1:-}" ] || true
}

mknode() { # $1 = fake root: the dirs and files a GB10 has
  local R=$1
  mkdir -p "$R/run/systemd/system" "$R/proc/driver/nvidia" "$R/usr/local" "$R/etc/sysctl.d"
  printf 'NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  %s  Release Build  (test)\n' "$DRIVER" \
    >"$R/proc/driver/nvidia/version"
  printf 'MemTotal: 129798784 kB\nMemFree: 64000000 kB\n' >"$R/meminfo"
  printf '80000000-27fffffff : System RAM\n280000000-2ffffffff : reserved\n' >"$R/iomem"
  printf 'test-fake-gb10\n' >"$R/devicetree-model"
  mkdir -p "$R/proc/device-tree"
  cp "$R/devicetree-model" "$R/proc/device-tree/model"
}

mkartifact() { # $1 = artifact dir, $2 = version: what build-release.sh lays down (host binaries)
  local A=$1 V=$2
  mkdir -p "$A/bin" "$A/artifacts/$DRIVER" "$A/scripts" "$A/systemd" "$A/sysctl" "$A/share"
  cp "$BIN/atticd" "$BIN/attic-check" "$A/bin/"
  cp "$HERE/scripts/spark-attic" "$A/bin/spark-attic"
  sed "s/^APP_VERSION = .*/APP_VERSION = \"$V\"/" "$HERE/scripts/spark-attic" >"$A/bin/spark-attic.new"
  mv "$A/bin/spark-attic.new" "$A/bin/spark-attic" && chmod 755 "$A/bin/spark-attic"
  cp "$BIN/fakeshim.so" "$A/artifacts/$DRIVER/rmshim.so"
  {
    echo "driver=$DRIVER"
    echo "shim_sha256=$(sha256sum "$A/artifacts/$DRIVER/rmshim.so" | cut -c1-64)"
    echo "attic_commit=test"
  } >"$A/artifacts/$DRIVER/BUILD-INFO"
  cp "$HERE/scripts/attic-validate.sh" "$A/scripts/"
  cp "$HERE/install.sh" "$A/install.sh"
  sed "s/^VERSION=\${INSTALL_VERSION:-0.1.0}/VERSION=\${INSTALL_VERSION:-$V}/" "$A/install.sh" >"$A/install.sh.new"
  mv "$A/install.sh.new" "$A/install.sh" && chmod 755 "$A/install.sh"
  cp "$HERE/systemd/attic.service" "$A/systemd/"
  cp "$HERE/sysctl/99-spark-attic-minfree.conf" "$A/sysctl/"
  cp "$HERE/share/spark-attic/header-art.txt" "$HERE/share/spark-attic/header-col.txt" "$A/share/"
  # the working tree keeps the docs under release/; the exported tree keeps them at its root
  DOC=$HERE/release/README.md
  [ -f "$DOC" ] || DOC=$HERE/README.md
  DOCI=$HERE/release/INSTALL.md
  [ -f "$DOCI" ] || DOCI=$HERE/INSTALL.md
  cp "$DOC" "$DOCI" "$HERE/STATUS-FILE.md" "$A/share/"
  RELEASE_NOTES=$HERE/release/RELEASE-NOTES-0.1.0.md
  [ -f "$RELEASE_NOTES" ] || RELEASE_NOTES=$HERE/RELEASE-NOTES-0.1.0.md
  cp "$RELEASE_NOTES" "$A/share/"
  cp "$HERE/LICENSE-MIT" "$HERE/LICENSE-APACHE" "$A/share/"
  printf 'third-party notices (test)\n' >"$A/share/THIRD_PARTY_NOTICES"
  cp -r "$HERE/python" "$A/python"
  echo "$V" >"$A/VERSION"
  (cd "$A" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum >SHA256SUMS)
}

# the env a fake node needs for attic-validate to pass with the fake shim
validate_env() {
  local R=$1
  echo "ATTIC_CHECK_ARGS=--meminfo $R/meminfo --iomem $R/iomem"
  echo "ATTIC_DAEMON_ARGS=--meminfo $R/meminfo --iomem $R/iomem"
  echo "LD_LIBRARY_PATH=$BIN/fakecuda"
  echo "ATTIC_FAKE_REGION=0x280000000:0x8000000"
  echo "ATTIC_FAKE_HELD=0"
  echo "ATTIC_HUP=true"
  echo "ATTIC_PROBE_ARGS=--modeset-dir $R/proc --drm-dir $R/sys/class/drm --pmon-file $R/absent-pmon"
}

instenv() { # $1 = fake root: the environment one install run needs
  local R=$1
  export ATTIC_FAKE_STATE="$R/.systemctl" ATTIC_FAKE_LOG="$R/.systemctl-log"
  export ATTIC_SHARE_DIR="$R/.share/spark-attic"
  export ATTIC_FAKE_STATUS_FILE="$R/run/spark-attic/status.json"
  export ATTIC_VERSION_FILE="$R/proc/driver/nvidia/version"
  export ATTIC_OTHER_SOCKETS="$R/other-lender.sock"
  export ATTIC_PROBE_ARGS="--modeset-dir $R/proc --drm-dir $R/sys/class/drm --pmon-file $R/absent-pmon"
  export ATTIC_CHECK_ARGS="--meminfo $R/meminfo --iomem $R/iomem"
  export ATTIC_DAEMON_ARGS="--meminfo $R/meminfo --iomem $R/iomem"
  export LD_LIBRARY_PATH="$BIN/fakecuda"
  export ATTIC_FAKE_REGION="0x280000000:0x8000000" ATTIC_FAKE_HELD="0" ATTIC_HUP="true"
  export ATTIC_MODES=heap
  export ATTIC_TEST=1
  export PATH="$R/.fakebin:$PATH_ORIG"
  mkdir -p "$R/.fakebin" "$R/.systemctl" "$R/.share"
  fakebin "$R/.fakebin"
  ln -sfn "$INSTANT_ART/share" "$R/.share/spark-attic" 2>/dev/null || true
}
INSTANT_ART=""

unset_environ() {
  unset ATTIC_FAKE_STATE ATTIC_FAKE_LOG ATTIC_SHARE_DIR ATTIC_FAKE_STATUS_FILE ATTIC_VERSION_FILE \
    ATTIC_OTHER_SOCKETS ATTIC_PROBE_ARGS ATTIC_CHECK_ARGS ATTIC_DAEMON_ARGS LD_LIBRARY_PATH \
    ATTIC_FAKE_REGION ATTIC_FAKE_HELD ATTIC_HUP ATTIC_SS_LISTEN NO_COLOR ATTIC_COLOR ATTIC_INSTALL
  export PATH=${PATH_ORIG:-$PATH}
}
PATH_ORIG=$PATH

# ---------------------------------------------------------------- 1. preflight refusals
echo "test_install: preflight refusals"
A=$T/artifact
mkartifact "$A" 0.1.0
INSTANT_ART=$A

R=$T/pf-arch; mknode "$R"; instenv "$R"
printf '#!/bin/sh\necho x86_64\n' >"$R/.fakebin/uname"
bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "not aarch64" "$R/out" && [ ! -e "$R/usr/local/sbin/atticd" ]
ok $? "a non-aarch64 machine: exit 1, the sentence, nothing installed (rc $rc)"

R=$T/pf-driver; mknode "$R"; instenv "$R"
rm "$R/proc/driver/nvidia/version"
bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "no NVIDIA driver is loaded" "$R/out"
ok $? "no driver: exit 1 with the sentence (rc $rc)"

R=$T/pf-systemd; mknode "$R"; instenv "$R"
rm -rf "$R/run/systemd/system"
bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
[ $? = 1 ] && grep -q "systemd is not running here" "$R/out"
ok $? "no systemd: exit 1 with the sentence"

R=$T/pf-display; mknode "$R"; instenv "$R"
mkdir -p "$R/sys/class/drm/card0-DP-1"
echo connected >"$R/sys/class/drm/card0-DP-1/status"
bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
[ $? = 1 ] && grep -q "a display is in use" "$R/out"
ok $? "a connected display: exit 1 with the sentence"

R=$T/pf-lender; mknode "$R"; instenv "$R"
python3 -c "import socket; s = socket.socket(socket.AF_UNIX); s.bind('$R/other-lender.sock')"
ATTIC_SS_LISTEN="$R/other-lender.sock" bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
[ $? = 1 ] && grep -q "another carveout lender answers on $R/other-lender.sock" "$R/out"
ok $? "another lender on its socket: exit 1 with the sentence"

# --check --from a release TARBALL: the unpacked copy must live until the preflight has read it (it
# was removed first, so every check said "the artifact has no shim ... (it carries: none)").
R=$T/pf-tgz; mknode "$R"; instenv "$R"
tar -C "$T" -czf "$T/artifact-check.tar.gz" --transform "s|^artifact|spark-attic-0.1.0|" artifact
bash "$A/install.sh" --check --from "$T/artifact-check.tar.gz" --root "$R" >"$R/out" 2>&1
grep -q "the artifact carries a shim for driver" "$R/out" && ! grep -q "it carries: none" "$R/out"
ok $? "--check --from a tarball: the preflight reads the unpacked artifact"
unset_environ

R=$T/pf-disk; mknode "$R"; instenv "$R"
cat >"$R/.fakebin/df" <<'EOF'
#!/bin/sh
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "fake 1000000 999000 100 100% $2"
EOF
chmod 755 "$R/.fakebin/df"
bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
[ $? = 1 ] && grep -q "MiB free at .*: spark-attic needs about 10 MiB" "$R/out"
ok $? "no disk space: exit 1 with the sentence"

R=$T/pf-good; mknode "$R"; instenv "$R"
bash "$A/install.sh" --check --from "$A" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 0 ] && grep -q "everything checks out" "$R/out" && grep -q "shim for driver $DRIVER" "$R/out"
ok $? "a good node: --check exit 0, the shim named (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 1b. no other-lender socket configured
# The public tree's default list is empty (no other vendor's socket is named). With an empty list the
# loop never ran and the preflight printed "no other carveout lender is listening": a pass for a check
# that checked nothing. It is a warn now.
echo "test_install: the other-lender check with no socket list"
RNL=$T/pf-nolenders; mknode "$RNL"; instenv "$RNL"
ATTIC_OTHER_SOCKETS=" " bash "$A/install.sh" --check --from "$A" --root "$RNL" >"$RNL/out" 2>&1
rc=$?
[ $rc = 0 ] && grep -q "no other-lender socket is configured" "$RNL/out" &&
  ! grep -q "no other carveout lender is listening" "$RNL/out"
ok $? "an empty socket list: a warn, not a pass (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 2. checksum refusal
echo "test_install: the checksum refusal"
R=$T/sum; mknode "$R"; instenv "$R"
A2=$T/artifact-tampered
cp -a "$A" "$A2"
echo tampered >>"$A2/share/README.md"
bash "$A2/install.sh" --from "$A2" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "does not match its SHA256SUMS" "$R/out" && [ ! -e "$R/usr/local/sbin/atticd" ] &&
  [ ! -e "$R/etc/spark-attic" ]
ok $? "a tampered artifact: refused before anything is written (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 2b. the download path's .sha256
# The .sha256 fetch was optional and its failure silent: with no .sha256 the only check was the tarball's
# own SHA256SUMS, which detects a broken download but not a replaced tarball. It fails closed now. The
# host test serves the tarball over file:// (curl accepts it), so no network is needed.
echo "test_install: the download path fails closed without a published .sha256"
W=$T/www; mkdir -p "$W/v0.1.0"
tar -C "$T" -czf "$W/v0.1.0/spark-attic-0.1.0-aarch64.tar.gz" \
  --transform "s|^artifact|spark-attic-0.1.0|" artifact
(cd "$W/v0.1.0" && sha256sum spark-attic-0.1.0-aarch64.tar.gz >spark-attic-0.1.0-aarch64.tar.gz.sha256)
RDL=$T/dl; mknode "$RDL"; instenv "$RDL"
bash "$A/install.sh" --url "file://$W" --root "$RDL" >"$RDL/out" 2>&1
rc=$?
[ $rc = 0 ] && [ "$(cat "$RDL/var/lib/attic/version" 2>/dev/null)" = 0.1.0 ]
ok $? "a download with its .sha256: installed (rc $rc)"
unset_environ
RDN=$T/dl-nosha; mknode "$RDN"; instenv "$RDN"
rm -f "$W/v0.1.0/spark-attic-0.1.0-aarch64.tar.gz.sha256"
bash "$A/install.sh" --url "file://$W" --root "$RDN" >"$RDN/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "publishes no .sha256" "$RDN/out" && [ ! -e "$RDN/usr/local/sbin/atticd" ]
ok $? "a download without its .sha256: refused, nothing written (rc $rc)"
unset_environ
# The standalone release installer carries ATTIC_SHA256 (build-release.sh stamps it): with the stamp the
# download installs without a sidecar, and a download that does not match the stamp is refused.
GOOD=$(sha256sum "$W/v0.1.0/spark-attic-0.1.0-aarch64.tar.gz" | awk '{print $1}')
sed "s|^SYSCTL_NAME=.*|&\nATTIC_SHA256=$GOOD|" "$A/install.sh" >"$T/install-stamped.sh"
RDS=$T/dl-stamped; mknode "$RDS"; instenv "$RDS"
bash "$T/install-stamped.sh" --url "file://$W" --root "$RDS" >"$RDS/out" 2>&1
rc=$?
[ $rc = 0 ] && [ "$(cat "$RDS/var/lib/attic/version" 2>/dev/null)" = 0.1.0 ]
ok $? "the stamped installer, no .sha256 published: installed (rc $rc)"
unset_environ
sed "s|^SYSCTL_NAME=.*|&\nATTIC_SHA256=$(printf '0%.0s' $(seq 64))|" "$A/install.sh" >"$T/install-badstamp.sh"
RDB=$T/dl-badstamp; mknode "$RDB"; instenv "$RDB"
bash "$T/install-badstamp.sh" --url "file://$W" --root "$RDB" >"$RDB/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "does not match the sha256 this installer carries" "$RDB/out" && [ ! -e "$RDB/usr/local/sbin/atticd" ]
ok $? "the stamped installer, a download that differs from the stamp: refused, nothing written (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 3. the happy install
echo "test_install: the install"
R=$T/in; mknode "$R"; instenv "$R"
bash "$A/install.sh" --from "$A" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 0 ] || { sed -n 1,40p "$R/out"; }
[ $rc = 0 ]
ok $? "install exit 0 (rc $rc)"
for f in usr/local/sbin/atticd usr/local/sbin/attic-check usr/local/bin/spark-attic \
         usr/local/share/spark-attic/header-art.txt usr/local/share/spark-attic/install.sh \
         etc/systemd/system/attic.service \
         var/lib/attic/version var/lib/attic/artifacts/$DRIVER/VALIDATED; do
  [ -e "$R/$f" ] || { echo "     missing $f"; }
  [ -e "$R/$f" ]
done
ok $? "the files, the unit and the VALIDATED record are in place"
[ ! -e "$R/etc/sysctl.d/99-spark-attic-minfree.conf" ]
ok $? "the default install leaves the kernel memory setting alone (no drop-in)"
[ "$(cat "$R/var/lib/attic/version")" = 0.1.0 ]
ok $? "the installed version is recorded"
K=$(cat "$R/etc/spark-attic/key")
case $K in attic_*) [ "$(stat -c %a "$R/etc/spark-attic/key")" = 600 ] ;; *) false ;; esac
ok $? "the key exists, root-only, attic_-prefixed"
grep -q '^modes=heap$' "$R/var/lib/attic/artifacts/$DRIVER/VALIDATED"
ok $? "VALIDATED names the modes that passed"
grep -q "+2,044 MiB lent to CUDA" "$R/out"
ok $? "the closing line is the big number"
grep -q "spark-attic 0.1.0" "$R/out"
ok $? "the header names the product and the version"
[ -f "$R/var/log/spark-attic/install.log" ] && grep -q "validate exit 0" "$R/var/log/spark-attic/install.log"
ok $? "the log records the steps"

# ---------------------------------------------------------------- 3e. doctor uses the installed artifact
# doctor called install.sh --check with no artifact, so a healthy node printed two "no artifact yet"
# warnings for the shim and the display. With the installed atticd and /var/lib/attic/artifacts/<driver>
# it hands install.sh a view of them.
echo "test_install: doctor uses the installed artifact"
python3 "$A/bin/spark-attic" doctor --root "$R" >"$R/doc" 2>&1
rc=$?
! grep -q "no artifact yet" "$R/doc"
ok $? "doctor: no 'no artifact yet' warnings on an installed node"
grep -q "the artifact carries a shim for driver $DRIVER" "$R/doc"
ok $? "doctor: the installed artifact's shim check passes (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 3b. the release tarball's layout
# build-release.sh puts the docs and the art under share/spark-attic/, not a flat share/ (node window 3: the
# real tarball was refused with "missing share/README.md", because this test only ever used the flat layout).
echo "test_install: the release tarball layout (share/spark-attic/)"
AR=$T/art-release; rm -rf "$AR"; cp -a "$A" "$AR"
mkdir -p "$AR/share/spark-attic" && (cd "$AR/share" && for f in *; do [ "$f" = spark-attic ] || mv "$f" spark-attic/; done)
(cd "$AR" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
tar -C "$T" -czf "$T/spark-attic-0.1.0-aarch64.tar.gz" --transform "s|^art-release|spark-attic-0.1.0|" art-release
R=$T/inrel; mknode "$R"; instenv "$R"
bash "$A/install.sh" --from "$T/spark-attic-0.1.0-aarch64.tar.gz" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 0 ] || { sed -n 1,40p "$R/out"; }
[ $rc = 0 ] && [ -f "$R/usr/local/share/spark-attic/README.md" ] && [ -f "$R/usr/local/share/spark-attic/header-art.txt" ]
ok $? "a tarball with share/spark-attic/ installs, docs and art in place (rc $rc)"
bash "$A/install.sh" --from "$T/spark-attic-0.1.0-aarch64.tar.gz" --root "$R" >"$R/out2" 2>&1
grep -q "already installed: nothing to do" "$R/out2"
ok $? "the same tarball again: nothing to do"
unset_environ

# ---------------------------------------------------------------- 3c. --no-sysctl keeps the version record
# install.sh returned from install_files before it wrote var/lib/attic/version when the flag was set: a
# --no-sysctl install left no version record, so every later run reinstalled and stopped the service.
echo "test_install: --no-sysctl keeps the version record"
RNS=$T/nosys; mknode "$RNS"; instenv "$RNS"
bash "$A/install.sh" --from "$A" --root "$RNS" --no-sysctl >"$RNS/out" 2>&1
rc=$?
[ $rc = 0 ] || { sed -n 1,30p "$RNS/out"; }
[ $rc = 0 ] && [ "$(cat "$RNS/var/lib/attic/version" 2>/dev/null)" = 0.1.0 ] &&
  [ ! -e "$RNS/etc/sysctl.d/99-spark-attic-minfree.conf" ]
ok $? "--no-sysctl: exit 0, the version is recorded, no drop-in (rc $rc)"
: >"$RNS/.systemctl-log"
bash "$A/install.sh" --from "$A" --root "$RNS" --no-sysctl >"$RNS/out2" 2>&1
rc=$?
[ $rc = 0 ] && grep -q "already installed: nothing to do" "$RNS/out2" &&
  ! grep -qE "stop|start|enable" "$RNS/.systemctl-log"
ok $? "--no-sysctl again: nothing to do, the service untouched (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 3d. --with-sysctl installs the drop-in
# The drop-in sets vm.min_free_kbytes on every node and the product does not need it: it is opt-in now,
# and this is the flag that installs it.
echo "test_install: --with-sysctl installs the drop-in"
RWS=$T/withsys; mknode "$RWS"; instenv "$RWS"
bash "$A/install.sh" --from "$A" --root "$RWS" --with-sysctl >"$RWS/out" 2>&1
rc=$?
[ $rc = 0 ] && [ -f "$RWS/etc/sysctl.d/99-spark-attic-minfree.conf" ] &&
  [ "$(cat "$RWS/var/lib/attic/version" 2>/dev/null)" = 0.1.0 ]
ok $? "--with-sysctl: exit 0, the drop-in and the version record are there (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 4. idempotent re-run
echo "test_install: the idempotent re-run"
instenv "$R"
: >"$R/.systemctl-log"
bash "$A/install.sh" --from "$A" --root "$R" >"$R/out2" 2>&1
rc=$?
grep -q "already installed: nothing to do" "$R/out2" || { tail -3 "$R/out2"; }
[ $rc = 0 ] && grep -q "already installed: nothing to do" "$R/out2"
ok $? "re-run: nothing to do (rc $rc)"
! grep -qE "stop|start|enable" "$R/.systemctl-log"
ok $? "re-run: the service was not touched"
unset_environ

# ---------------------------------------------------------------- 5. upgrade
echo "test_install: the upgrade"
A3=$T/artifact-0.2.0
mkartifact "$A3" 0.2.0
echo "upgrade test line" >>"$A3/share/README.md"
(cd "$A3" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum >SHA256SUMS)
instenv "$R"
export ATTIC_FAKE_VERSION=0.2.0
bash "$A3/install.sh" --from "$A3" --root "$R" >"$R/out3" 2>&1
rc=$?
[ $rc = 0 ] && grep -q "upgrading" "$R/out3" && [ "$(cat "$R/var/lib/attic/version")" = 0.2.0 ]
ok $? "upgrade: the newer artifact installs (rc $rc)"
grep -q "stopped attic.service" "$R/out3" && grep -q "enabled attic.service" "$R/out3"
ok $? "upgrade: the service was stopped for the check and started again"
grep -q "validate exit 0" "$R/var/log/spark-attic/install.log"
ok $? "upgrade: validated again before enable"
grep -q "the existing key is kept" "$R/out3"
ok $? "upgrade: the key is kept"
unset_environ

# ---------------------------------------------------------------- 6. the downgrade refusal
echo "test_install: the downgrade refusal"
A4=$T/artifact-0.0.9
mkartifact "$A4" 0.0.9
instenv "$R"
: >"$R/.systemctl-log"
bash "$A4/install.sh" --from "$A4" --root "$R" >"$R/out4" 2>&1
rc=$?
[ $rc = 0 ] && grep -q "installed 0.2.0 is newer than this 0.0.9: nothing changed" "$R/out4" &&
  [ "$(cat "$R/var/lib/attic/version")" = 0.2.0 ]
ok $? "an older artifact: refused, nothing changed (rc $rc)"
[ ! -s "$R/.systemctl-log" ]
ok $? "downgrade: the service was not touched"
unset_environ

# ---------------------------------------------------------------- 7. rollback after a failed validate
echo "test_install: the rollback"
R=$T/roll; mknode "$R"; instenv "$R"
printf 'alloc\nopen\n' >"$R/fail-calls"
ATTIC_FAKE_FAIL_FILE="$R/fail-calls" bash "$A/install.sh" --from "$A" --root "$R" >"$R/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "attic-check failed on every mode" "$R/out" && grep -q "rolled back" "$R/out"
ok $? "a failed validate: exit 1, rolled back (rc $rc)"
for f in usr/local/sbin/atticd usr/local/sbin/attic-check usr/local/bin/spark-attic \
         usr/local/share/spark-attic etc/systemd/system/attic.service \
         etc/sysctl.d/99-spark-attic-minfree.conf var/lib/attic etc/spark-attic/key; do
  [ -e "$R/$f" ] && { echo "     still there: $f"; false; }
  [ ! -e "$R/$f" ]
done
ok $? "nothing this run created is left"
reports=$(ls -d "$R"/var/log/spark-attic/failed-* 2>/dev/null | head -n1)
[ -n "$reports" ] && [ "$(stat -c %a "$reports")" = 700 ] && ls "$reports"/FAILED-*.json >/dev/null 2>&1 &&
  grep -q "$reports" "$R/out"
ok $? "the failed reports survive the rollback in $reports (0700) and the message names it"
[ -f "$R/var/log/spark-attic/install.log" ] && grep -q "rolled back" "$R/var/log/spark-attic/install.log"
ok $? "the log stays and says what happened"
unset_environ

# ---------------------------------------------------------------- 7b. an interrupted install rolls back
# install.sh had no signal handling: Ctrl-C (or a dropped SSH session) between install_files and enable
# left the new files down with the service stopped. From the first write on, a signal rolls back and
# exits 130. The fake systemctl pauses in daemon-reload so the test can send the signal mid-run; it sends
# TERM, not INT, because a shell started in the background inherits INT as ignored and cannot trap it.
echo "test_install: an interrupted install rolls back"
RS=$T/signal; mknode "$RS"; instenv "$RS"
export ATTIC_FAKE_PAUSE="$RS/paused" ATTIC_FAKE_PAUSE_S=3
bash "$A/install.sh" --from "$A" --root "$RS" >"$RS/out" 2>&1 &
pid=$!
for i in $(seq 200); do [ -f "$RS/paused" ] && break; sleep 0.1; done
kill -TERM "$pid" 2>/dev/null
wait "$pid"; rc=$?
[ $rc = 130 ] && grep -q "rolling back" "$RS/out"
ok $? "an interrupted install: exit 130, rolled back (rc $rc)"
for f in usr/local/sbin/atticd usr/local/sbin/attic-check usr/local/bin/spark-attic \
         usr/local/share/spark-attic etc/systemd/system/attic.service var/lib/attic etc/spark-attic/key; do
  [ -e "$RS/$f" ] && { echo "     still there: $f"; false; }
  [ ! -e "$RS/$f" ]
done
ok $? "the interrupted install left nothing behind"
unset ATTIC_FAKE_PAUSE ATTIC_FAKE_PAUSE_S
unset_environ

# ---------------------------------------------------------------- 7c. an upgrade while memory is lent
# The service was stopped before validate, so the validate's own exit-4 sentence ("still lends memory:
# wait for its clients to exit") could no longer fire: the user got "attic-check failed on every mode".
# The status file is read before anything is written and the run refuses with that sentence.
echo "test_install: an upgrade while the carveout is lent"
lent_status() { # $1 = fake root, $2 = lent_mib, $3 = draining_mib
  mkdir -p "$1/run/spark-attic"
  printf '{"name":"spark-attic","version":"0.1.0","state":"ready","why":"","driver":"%s","mode":"heap","validated":true,"validated_at":"2026-11-12T00:00:00Z","size_mib":2044,"free_mib":0,"largest_mib":0,"lent_mib":%s,"draining_mib":%s,"clients":1,"every_s":10,"pid":1,"updated":0}\n' \
    "$DRIVER" "$2" "$3" >"$1/run/spark-attic/status.json"
}
RL=$T/lent; mknode "$RL"; instenv "$RL"
touch "$RL/.systemctl/active"; lent_status "$RL" 2044 0
bash "$A3/install.sh" --from "$A3" --root "$RL" >"$RL/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "still lends memory: wait for its clients to exit" "$RL/out" &&
  [ ! -e "$RL/usr/local/sbin/atticd" ]
ok $? "a lent carveout: exit 1 with the wait sentence, nothing installed (rc $rc)"
! grep -q "stop attic.service" "$RL/.systemctl-log"
ok $? "a lent carveout: the service was not stopped"
unset_environ
RD=$T/draining; mknode "$RD"; instenv "$RD"
touch "$RD/.systemctl/active"; lent_status "$RD" 0 2044
bash "$A3/install.sh" --from "$A3" --root "$RD" >"$RD/out" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "still lends memory: wait for its clients to exit" "$RD/out" &&
  [ ! -e "$RD/usr/local/sbin/atticd" ]
ok $? "a draining carveout: refused too, nothing installed (rc $rc)"
unset_environ

# ---------------------------------------------------------------- 8. a failed upgrade restores the old files
echo "test_install: the failed upgrade"
instenv "$R"
bash "$A/install.sh" --from "$A" --root "$R" >"$R/out5" 2>&1
[ $? = 0 ]
ok $? "0.1.0 installed again on the rollback node"
SHA_BEFORE=$(sha256sum "$R/usr/local/sbin/atticd" | cut -c1-64)
export ATTIC_FAKE_VERSION=0.2.0
printf 'alloc\nopen\n' >"$R/fail-calls"
ATTIC_FAKE_FAIL_FILE="$R/fail-calls" bash "$A3/install.sh" --from "$A3" --root "$R" >"$R/out6" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "rolled back" "$R/out6"
ok $? "a failing 0.2.0 upgrade: exit 1, rolled back (rc $rc)"
[ "$(cat "$R/var/lib/attic/version")" = 0.1.0 ] &&
  [ "$(sha256sum "$R/usr/local/sbin/atticd" | cut -c1-64)" = "$SHA_BEFORE" ]
ok $? "the old binaries and version record are back"
grep -q "attic.service is running again" "$R/out6"
ok $? "the service was started again (as before the run)"
unset_environ

# ---------------------------------------------------------------- 9. uninstall
echo "test_install: uninstall"
instenv "$R"
export ATTIC_INSTALL=""
"$R/usr/local/bin/spark-attic" uninstall --yes --root "$R" >"$R/out7" 2>&1
rc=$?
[ $rc = 0 ] || { sed -n 1,30p "$R/out7"; }
[ $rc = 0 ]
ok $? "uninstall exit 0 (rc $rc)"
for f in etc/systemd/system/attic.service usr/local/sbin/atticd usr/local/sbin/attic-check \
         usr/local/bin/spark-attic usr/local/share/spark-attic var/lib/attic \
         run/spark-attic etc/sysctl.d/99-spark-attic-minfree.conf; do
  [ -e "$R/$f" ] && { echo "     still there: $f"; false; }
  [ ! -e "$R/$f" ]
done
ok $? "no unit, socket, binary, artifact or sysctl file is left"
[ -f "$R/etc/spark-attic/key" ] && [ -f "$R/var/log/spark-attic/install.log" ]
ok $? "the key and the logs stay"
grep -q "disable --now attic.service" "$R/.systemctl-log" && grep -q "daemon-reload" "$R/.systemctl-log"
ok $? "the unit was disabled and stopped, and systemd reloaded"
unset_environ

echo "test_install: uninstall --purge"
R=$T/purge; mknode "$R"; instenv "$R"
bash "$A/install.sh" --from "$A" --root "$R" >/dev/null 2>&1
python3 "$A/bin/spark-attic" uninstall --purge --yes --root "$R" >/dev/null 2>&1
[ ! -e "$R/etc/spark-attic" ] && [ ! -e "$R/var/log/spark-attic" ]
ok $? "--purge: the key and the logs are gone too"
unset_environ

echo "test_install: uninstall on a node with nothing installed"
R=$T/empty; mknode "$R"; instenv "$R"
bash "$A/install.sh" --from "$A" --root "$R" >/dev/null 2>&1
python3 "$A/bin/spark-attic" uninstall --yes --root "$R" >/dev/null 2>&1
python3 "$A/bin/spark-attic" uninstall --yes --root "$R" >"$R/out9" 2>&1
[ $? = 0 ] && grep -q "nothing to remove" "$R/out9"
ok $? "a second uninstall: nothing to remove, exit 0"
unset_environ

# ---------------------------------------------------------------- 10. plain versus colour
echo "test_install: plain versus colour output"
R=$T/plain; mknode "$R"; instenv "$R"
NO_COLOR=1 bash "$A/install.sh" --from "$A" --root "$R" >"$R/outp" 2>&1
! grep -q $'\033' "$R/outp"
ok $? "NO_COLOR: no escapes in the output"
[ "$(head -n1 "$R/outp")" = "spark-attic 0.1.0" ]
ok $? "NO_COLOR: the header is the plain text line"
unset_environ
R=$T/plain2; mknode "$R"; instenv "$R"
bash "$A/install.sh" --from "$A" --root "$R" --plain >"$R/outpp" 2>&1
! grep -q $'\033' "$R/outpp"
ok $? "--plain: no escapes either"
unset_environ
R=$T/colour; mknode "$R"; instenv "$R"
env -u NO_COLOR ATTIC_COLOR=always TERM=xterm-256color bash "$A/install.sh" --from "$A" --root "$R" \
  >"$R/outc" 2>&1
grep -q $'\033\[38;5;255m' "$R/outc" && grep -q $'\033\[38;5;179m' "$R/outc"
ok $? "ATTIC_COLOR=always: paper and gold escapes"
# the braille wordmark only when this tree carries the art (the exported tree may not: the art gate)
if [ -f "$A/share/header-art.txt" ]; then
  python3 -c "import sys; data = open(sys.argv[1], encoding='utf-8').read(); sys.exit(0 if any(0x2800 <= ord(c) <= 0x28FF for c in data) else 1)" \
    "$R/outc"
  ok $? "ATTIC_COLOR=always: the braille wordmark is printed"
  ! grep -q $'\033\[38;5;202m' "$R/outc"
  ok $? "ATTIC_COLOR=always: the header uses the Bertholomus colours only"
else
  echo "  ok   ATTIC_COLOR=always: no art in this tree, the header falls back to text (the art gate)"
fi
unset_environ

# ---------------------------------------------------------------- 11. the demo
echo "test_install: the demo (DEMO_PACE=0)"
R=$T/demo; mknode "$R" 2>/dev/null || mkdir -p "$R"
DEMO_PACE=0 PATH_ORIG=$PATH bash "$HERE/install.sh" --demo >"$R/outd" 2>&1
rc=$?
[ $rc = 0 ] || { tail -12 "$R/outd"; }
[ $rc = 0 ]
ok $? "the demo: exit 0 (rc $rc)"
grep -q "^DEMO: nothing is installed" "$R/outd"
ok $? "the demo: the first line says it is a demo"
grep -q "+2,044 MiB" "$R/outd" && grep -q "state     ready" "$R/outd"
ok $? "the demo: the closing number and the status screen"
rootdir=$(grep -o "/tmp/spark-attic-demo\.[A-Za-z0-9]*" "$R/outd" | tail -1)
[ -n "$rootdir" ] && [ ! -e "$rootdir" ]
ok $? "the demo: its fake root is removed ($rootdir)"

echo "test_install: the demo --fail display"
R=$T/demofail; mkdir -p "$R"
DEMO_PACE=0 PATH_ORIG=$PATH bash "$HERE/install.sh" --demo --fail display >"$R/outf" 2>&1
rc=$?
[ $rc = 1 ] && grep -q "a display is in use" "$R/outf" && grep -q "the preflight refused" "$R/outf"
ok $? "the demo refusal: exit 1, the plain reason (rc $rc)"

echo "test_install: $fails failures"
[ "$fails" = 0 ]

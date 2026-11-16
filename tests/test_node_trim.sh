#!/bin/bash
# test_node_trim.sh: scripts/node-trim.sh against stubs and fake trees. No root, no systemd: PATH gets a systemctl
# whose state is files (a units list, an enabled list, a default target, a log of every write call), a sysctl, and
# update-grub; /etc and /proc/sys are fake trees under a temp dir. Proves: apply refuses as non-root; --check prints
# every step and every enable-back line it would record while changing nothing; apply boots to console, disables the
# services, writes the watermark drop-in, takes splash off grub, runs the grub update, compacts memory, and records
# every enable-back line before the change; a second apply does nothing new but compacts again; a release without
# gdm3/snapd/splash skips them with notes and stays green; where the release ships grub2-mkconfig instead, it is used.
set -u
H=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d)
cleanup() { rm -rf "$T"; }
trap cleanup EXIT
fails=0 n=0
ok() { n=$((n + 1)); echo "ok   $1"; }
bad() { n=$((n + 1)); fails=$((fails + 1)); echo "FAIL $1"; }
TRIM=$H/scripts/node-trim.sh
bash -n "$TRIM" && ok "node-trim.sh parses" || bad "node-trim.sh parses"
mkdir -p "$T/fakebin"

ALL_UNITS="gdm3.service bluetooth.service cups.service snapd.service snapd.socket snapd.seeded.service dgx-dashboard.service dgx-dashboard-admin.service fwupd-refresh.timer"

# Full desktop release: every unit present and enabled, graphical default, splash on the command line, factor 40.
fresh_full() {
  rm -rf "$T/state" "$T/etc" "$T/proc"
  mkdir -p "$T/state" "$T/etc/sysctl.d" "$T/etc/default" "$T/proc/vm" "$T/fakebin"
  for u in $ALL_UNITS; do echo "$u" >> "$T/state/units"; echo "$u" >> "$T/state/enabled"; done
  echo graphical.target > "$T/state/default"
  cat > "$T/etc/default/grub" <<'EOF'
GRUB_TIMEOUT=1
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX="splash vt.handoff=7"
EOF
  echo 40 > "$T/proc/vm/watermark_boost_factor"
  echo 0 > "$T/proc/vm/drop_caches"
  echo 0 > "$T/proc/vm/compact_memory"
}

cat > "$T/fakebin/systemctl" <<'EOF'
#!/bin/bash
set -u
S=$STATE_DIR
cmd=$1; shift
case $cmd in
  cat) [ $# -ge 1 ] && grep -qx "$1" "$S/units" && exit 0; exit 4;;
  is-enabled) grep -qx "$1" "$S/enabled" && { echo enabled; exit 0; } || { echo disabled; exit 1; };;
  disable) echo "disable $*" >> "$S/systemctl.log"
           for u in "$@"; do [ "$u" = "--now" ] && continue; sed -i "/^$u\$/d" "$S/enabled"; done; exit 0;;
  enable)  echo "enable $*" >> "$S/systemctl.log"
           for u in "$@"; do grep -qx "$u" "$S/enabled" || echo "$u" >> "$S/enabled"; done; exit 0;;
  get-default) cat "$S/default"; exit 0;;
  set-default) echo "set-default $1" >> "$S/systemctl.log"; echo "$1" > "$S/default"; exit 0;;
esac
echo "systemctl: unknown call $cmd $*" >&2; exit 1
EOF
cat > "$T/fakebin/sysctl" <<'EOF'
#!/bin/bash
if [ "${1:-}" = "-w" ] && [ "$2" = "vm.watermark_boost_factor=0" ]; then
  echo "-w vm.watermark_boost_factor=0" >> "$STATE_DIR/sysctl.log"; echo 0 > "$PROC_DIR/vm/watermark_boost_factor"; exit 0
fi
echo "sysctl: unknown call $*" >&2; exit 1
EOF
for g in update-grub grub2-mkconfig; do
  printf '#!/bin/bash\necho "%s" >> "$STATE_DIR/grub.log"\nexit 0\n' "$g" > "$T/fakebin/$g"
done
chmod +x "$T/fakebin/"*

run() {  # run [--check] with the fake PATH and trees; stdout+rc in out/rc globals. PATH is fakebin plus only /usr/bin
  # and /bin: no sbin, so the host's own update-grub can never answer when a release does not ship one.
  PATH="$T/fakebin:/usr/bin:/bin" STATE_DIR="$T/state" PROC_DIR="$T/proc" \
    ATTIC_TRIM_ETC="$T/etc" ATTIC_TRIM_PROC="$T/proc" ATTIC_TRIM_RECORD="$T/undo.log" \
    ATTIC_TRIM_NO_ROOT=1 bash "$TRIM" "$@" > "$T/out" 2>&1
  rc=$?
}

# (a) apply as non-root refuses before touching anything
fresh_full
rc0=$(PATH="$T/fakebin:$PATH" STATE_DIR="$T/state" ATTIC_TRIM_ETC="$T/etc" ATTIC_TRIM_PROC="$T/proc" \
  ATTIC_TRIM_RECORD="$T/undo.log" bash "$TRIM" > "$T/out" 2>&1; echo $?)
[ "$rc0" = 1 ] && grep -q "needs root" "$T/out" && [ ! -e "$T/undo.log" ] && [ "$(cat "$T/state/default")" = graphical.target ] \
  && ok "apply as non-root refuses and touches nothing" || bad "apply as non-root refuses (rc $rc0: $(cat "$T/out"))"

# (b) --check: prints every step and every enable-back, changes nothing
fresh_full
md5_grub=$(md5sum "$T/etc/default/grub" | cut -d' ' -f1)
run --check
[ "$rc" = 0 ] || bad "--check rc 0 (rc $rc)"
for want in "would: systemctl set-default multi-user.target" "would: systemctl disable --now gdm3.service" \
            "would: systemctl disable --now snapd.socket" "would: systemctl disable --now dgx-dashboard-admin.service" \
            "would: systemctl disable --now fwupd-refresh.timer" \
            "would: write $T/etc/sysctl.d/99-spark-attic-headless.conf" \
            "would: sysctl -w vm.watermark_boost_factor=0" \
            "would: remove splash from $T/etc/default/grub" "would: update-grub" \
            "enable-back: systemctl enable --now gdm3.service" \
            "enable-back: systemctl set-default graphical.target" \
            "enable-back: rm $T/etc/sysctl.d/99-spark-attic-headless.conf && sysctl -w vm.watermark_boost_factor=40" \
            "would: sync; echo 3 > $T/proc/vm/drop_caches; echo 1 > $T/proc/vm/compact_memory" \
            "check only: nothing was changed"; do
  grep -qF "$want" "$T/out" && ok "--check says: $want" || bad "--check missing: $want (got $(cat "$T/out"))"
done
[ "$(cat "$T/state/default")" = graphical.target ] && [ "$(wc -l < "$T/state/enabled")" = 9 ] \
  && [ ! -e "$T/undo.log" ] && [ ! -e "$T/etc/sysctl.d/99-spark-attic-headless.conf" ] \
  && [ "$(md5sum "$T/etc/default/grub" | cut -d' ' -f1)" = "$md5_grub" ] \
  && [ "$(cat "$T/proc/vm/watermark_boost_factor")" = 40 ] \
  && [ "$(cat "$T/proc/vm/drop_caches")" = 0 ] && [ "$(cat "$T/proc/vm/compact_memory")" = 0 ] \
  && [ ! -e "$T/state/systemctl.log" ] && ok "--check changed nothing on the fake release" \
  || bad "--check changed something"

# (c) apply: the whole trim, with every enable-back recorded first
fresh_full
run
[ "$rc" = 0 ] || bad "apply rc 0 (rc $rc: $(cat "$T/out"))"
grep -q "^set-default multi-user.target$" "$T/state/systemctl.log" && ok "apply boots to console" \
  || bad "apply set-default (log: $(cat "$T/state/systemctl.log"))"
for u in $ALL_UNITS; do
  grep -q "^disable --now $u$" "$T/state/systemctl.log" && ok "apply disables $u" || bad "apply disables $u"
done
[ ! -s "$T/state/enabled" ] && ok "apply leaves nothing enabled" || bad "units still enabled: $(cat "$T/state/enabled")"
[ "$(cat "$T/state/default")" = multi-user.target ] && ok "the default target file moved" || bad "default target unchanged"
[ "$(cat "$T/etc/sysctl.d/99-spark-attic-headless.conf")" = "vm.watermark_boost_factor = 0" ] \
  && ok "the watermark drop-in is written" || bad "drop-in missing/wrong: $(cat "$T/etc/sysctl.d"/* 2>/dev/null)"
[ "$(cat "$T/proc/vm/watermark_boost_factor")" = 0 ] && ok "sysctl -w made the value live now" \
  || bad "watermark_boost_factor still $(cat "$T/proc/vm/watermark_boost_factor")"
[ "$(cat "$T/proc/vm/drop_caches")" = 3 ] && [ "$(cat "$T/proc/vm/compact_memory")" = 1 ] \
  && ok "apply compacts memory" || bad "defrag did not run ($(cat "$T/proc/vm/drop_caches")/$(cat "$T/proc/vm/compact_memory"))"
[ "$(cat "$T/state/grub.log")" = update-grub ] && ok "apply updates grub" || bad "grub update: $(cat "$T/state/grub.log" 2>/dev/null)"
grep -q 'GRUB_CMDLINE_LINUX_DEFAULT="quiet"' "$T/etc/default/grub" \
  && grep -q 'GRUB_CMDLINE_LINUX="vt.handoff=7"' "$T/etc/default/grub" \
  && ! grep -qw splash "$T/etc/default/grub" && ok "splash is gone, the rest of the command line stands" \
  || bad "grub edit wrong: $(cat "$T/etc/default/grub")"
for want in "enable-back: systemctl set-default graphical.target" "enable-back: systemctl enable --now gdm3.service" \
            "enable-back: systemctl enable --now bluetooth.service" "enable-back: systemctl enable --now snapd.socket" \
            "enable-back: systemctl enable --now dgx-dashboard.service" \
            "enable-back: systemctl enable --now fwupd-refresh.timer" \
            "enable-back: rm $T/etc/sysctl.d/99-spark-attic-headless.conf && sysctl -w vm.watermark_boost_factor=40" \
            "GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash\"" "GRUB_CMDLINE_LINUX=\"splash vt.handoff=7\""; do
  grep -qF "$want" "$T/undo.log" && ok "recorded: $want" || bad "record missing: $want (undo: $(cat "$T/undo.log"))"
done
[ "$(grep -c '^enable-back:' "$T/undo.log")" = 12 ] && ok "one enable-back line per change (12)" \
  || bad "enable-back lines: $(grep -c '^enable-back:' "$T/undo.log")"

# (d) a second apply does nothing new but compacts again
sl_before=$(wc -l < "$T/state/systemctl.log"); gr_before=$(cat "$T/state/grub.log")
md5_undo=$(md5sum "$T/undo.log" | cut -d' ' -f1)
run
[ "$rc" = 0 ] && [ "$(wc -l < "$T/state/systemctl.log")" = "$sl_before" ] \
  && [ "$(cat "$T/state/grub.log")" = "$gr_before" ] \
  && [ "$(md5sum "$T/undo.log" | cut -d' ' -f1)" = "$md5_undo" ] \
  && [ "$(cat "$T/proc/vm/drop_caches")" = 3 ] \
  && ok "a second apply changes nothing new but compacts again" \
  || bad "second apply changed state (rc $rc, systemctl $(wc -l < "$T/state/systemctl.log") vs $sl_before)"
grep -q "^note:  splash is not in" "$T/out" && ok "the second apply skips the finished grub edit" \
  || bad "second apply did not note the finished grub edit"

# (e) a minimal release: no gdm3, no snapd (splash stays on grub so the grub2-mkconfig route is exercised)
fresh_full
rm -f "$T/undo.log"
rm -f "$T/fakebin/update-grub"; printf '#!/bin/bash\necho grub2-mkconfig >> "$STATE_DIR/grub.log"\nexit 0\n' > "$T/fakebin/grub2-mkconfig"
printf '%s\n' bluetooth.service cups.service > "$T/state/units"
printf '%s\n' bluetooth.service cups.service > "$T/state/enabled"
run
[ "$rc" = 0 ] || bad "minimal release rc 0 (rc $rc: $(cat "$T/out"))"
grep -q "^note:  gdm3 is not on this release" "$T/out" && grep -q "^note:  snapd.service is not on this release" "$T/out" \
  && ok "a minimal release skips what it lacks, with notes" \
  || bad "minimal-release notes missing ($(grep '^note' "$T/out"))"
grep -q "^did:   systemctl disable --now bluetooth.service$" "$T/out" \
  && [ "$(grep -c '^disable --now' "$T/state/systemctl.log")" = 2 ] \
  && ok "a minimal release disables exactly what exists" || bad "minimal release disables wrong set"
! grep -q "enable --now gdm3" "$T/undo.log" && ! grep -q "enable --now snapd" "$T/undo.log" \
  && ok "no enable-back lines for units the release never had" || bad "undo lines for absent units"
grep -q "^did:   grub2-mkconfig$" "$T/out" && ok "grub2-mkconfig is used where the release ships it" \
  || bad "grub2-mkconfig not used (out: $(grep grub "$T/out"))"
grep -q "FAIL" "$T/out" && bad "no FAIL lines on a clean minimal run" || ok "no FAIL lines on a clean minimal run"

echo "test_node_trim: $n checks, $fails failures"
[ "$fails" = 0 ]

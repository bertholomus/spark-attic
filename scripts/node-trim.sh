#!/bin/bash
# node-trim.sh [--check]: headless-trim a node that will serve the attic. One tool, run by the operator on the node
# before any client borrows (never from a client). Steps: boot to console (multi-user.target, gdm3 off), disable
# bluetooth, cups, snapd, dgx-dashboard, dgx-dashboard-admin and fwupd-refresh.timer, set vm.watermark_boost_factor=0
# through a sysctl.d drop-in, drop `splash` from the kernel command line, then compact memory (sync, drop_caches,
# compact_memory) so the first borrow starts on an unfragmented heap. Every change is recorded - before it is made -
# as the exact enable-back line in the undo record, so a node can be returned to its desktop state line by line.
# --check prints every step it would take and touches nothing.
# Release-agnostic: every unit is checked for existence first; a release without gdm3 or snapd skips them with a
# note. Update-grub is picked up wherever the release puts it (update-grub or grub2-mkconfig).
# Test knobs (never set on a node): ATTIC_TRIM_ETC (default /etc) for /etc/default/grub and the sysctl.d drop-in,
# ATTIC_TRIM_PROC (default /proc/sys) for the sysctl values and the defrag writes, ATTIC_TRIM_RECORD (default
# /var/lib/spark-attic/trim-undo.log) for the undo record, ATTIC_TRIM_NO_ROOT=1 to let apply mode run as non-root in
# tests. systemctl and update-grub come from PATH.
set -u

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1
[ "${1:-}" = "--help" ] && { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
[ $# -gt 1 ] && { echo "usage: node-trim.sh [--check]" >&2; exit 2; }
if [ "$CHECK" = 0 ] && [ "${ATTIC_TRIM_NO_ROOT:-0}" != 1 ] && [ "$(id -u)" != 0 ]; then
  echo "node-trim.sh: apply mode needs root (sudo node-trim.sh; --check works as any user)" >&2
  exit 1
fi

ETC=${ATTIC_TRIM_ETC:-/etc}
PROC=${ATTIC_TRIM_PROC:-/proc/sys}
RECORD=${ATTIC_TRIM_RECORD:-/var/lib/spark-attic/trim-undo.log}

rc=0
say() { if [ "$CHECK" = 1 ]; then echo "would: $*"; else echo "did:   $*"; fi; }
note() { echo "note:  $*"; }
fail() { echo "FAIL:  $*" >&2; rc=1; }

# Record the exact enable-back line before making the change: a run killed mid-way still leaves the undo.
undo() {
  if [ "$CHECK" = 0 ]; then
    mkdir -p "$(dirname "$RECORD")"
    printf 'enable-back: %s\n' "$1" >> "$RECORD"
  fi
}

unit_exists() { systemctl cat "$1" >/dev/null 2>&1; }

# disable_unit UNIT: skip when absent or not enabled; otherwise record and disable --now.
disable_unit() {
  local u=$1
  if ! unit_exists "$u"; then
    note "$u is not on this release: nothing to do"
    return
  fi
  local st
  st=$(systemctl is-enabled "$u" 2>/dev/null || true)
  if [ "$st" != "enabled" ] && [ "$st" != "static" ] && [ "$st" != "alias" ]; then
    note "$u is already $st: nothing to do"
    return
  fi
  if [ "$CHECK" = 1 ]; then
    say "systemctl disable --now $u   (enable-back: systemctl enable --now $u)"
  else
    undo "systemctl enable --now $u"
    systemctl disable --now "$u" >/dev/null && say "systemctl disable --now $u" \
      || fail "systemctl disable --now $u"
  fi
}

# 1. boot to console -------------------------------------------------------
def=$("systemctl" get-default 2>/dev/null || echo unknown)
if [ "$def" = "multi-user.target" ]; then
  note "default target is already multi-user.target"
else
  if [ "$CHECK" = 1 ]; then
    say "systemctl set-default multi-user.target   (enable-back: systemctl set-default $def)"
  else
    undo "systemctl set-default $def"
    systemctl set-default multi-user.target >/dev/null && say "systemctl set-default multi-user.target" \
      || fail "systemctl set-default multi-user.target"
  fi
fi
if unit_exists gdm3.service; then
  disable_unit gdm3.service
else
  note "gdm3 is not on this release: nothing to do"
fi

# 2. the display's helpers ------------------------------------------------
for u in bluetooth.service cups.service snapd.service snapd.socket snapd.seeded.service \
         dgx-dashboard.service dgx-dashboard-admin.service; do
  disable_unit "$u"
done
disable_unit fwupd-refresh.timer

# 3. vm.watermark_boost_factor=0 through a drop-in ------------------------
drop="$ETC/sysctl.d/99-spark-attic-headless.conf"
cur=0
[ -r "$PROC/vm/watermark_boost_factor" ] && cur=$(cat "$PROC/vm/watermark_boost_factor" 2>/dev/null)
cur=${cur:-0}
if [ -f "$drop" ] && grep -q '^vm.watermark_boost_factor = 0$' "$drop" 2>/dev/null; then
  note "the watermark drop-in is already in place"
elif [ "$CHECK" = 1 ]; then
  say "write $drop (vm.watermark_boost_factor = 0)   (enable-back: rm $drop && sysctl -w vm.watermark_boost_factor=$cur)"
  say "sysctl -w vm.watermark_boost_factor=0"
else
  undo "rm $drop && sysctl -w vm.watermark_boost_factor=$cur"
  printf 'vm.watermark_boost_factor = 0\n' > "$drop" && say "write $drop (vm.watermark_boost_factor = 0)" \
    || fail "write $drop"
  sysctl -w vm.watermark_boost_factor=0 >/dev/null && say "sysctl -w vm.watermark_boost_factor=0" \
    || fail "sysctl -w vm.watermark_boost_factor=0"
fi

# 4. splash off the kernel command line -----------------------------------
grub="$ETC/default/grub"
if [ ! -f "$grub" ]; then
  note "no $grub on this release: nothing to do"
elif grep -qw splash "$grub"; then
  if [ "$CHECK" = 1 ]; then
    say "remove splash from $grub and run the grub update   (enable-back: restore the original GRUB_CMDLINE lines, saved in the record)"
  else
    undo "restore the original kernel command line in $grub:"
    grep -n 'GRUB_CMDLINE' "$grub" >> "$RECORD"
    sed -i 's/[[:space:]]\+splash//g; s/splash[[:space:]]\+//g; s/"splash"/""/g' "$grub"
    if grep -qw splash "$grub"; then
      fail "splash is still in $grub after the edit"
    else
      say "remove splash from $grub"
    fi
  fi
  for g in update-grub grub2-mkconfig; do
    if command -v "$g" >/dev/null 2>&1; then
      if [ "$CHECK" = 1 ]; then
        say "$g"
      else
        "$g" >/dev/null 2>&1 && say "$g" || fail "$g"
      fi
      break
    fi
  done
else
  note "splash is not in $grub: nothing to do"
fi

# 5. compact memory before any client borrows -----------------------------
if [ "$CHECK" = 1 ]; then
  say "sync; echo 3 > $PROC/vm/drop_caches; echo 1 > $PROC/vm/compact_memory"
else
  sync
  echo 3 > "$PROC/vm/drop_caches" 2>/dev/null && say "drop_caches" || fail "write $PROC/vm/drop_caches"
  echo 1 > "$PROC/vm/compact_memory" 2>/dev/null && say "compact_memory" || fail "write $PROC/vm/compact_memory"
fi

if [ "$CHECK" = 1 ]; then
  echo "check only: nothing was changed"
elif [ "$rc" = 0 ]; then
  echo "trim done; enable-back lines are in $RECORD; the node is ready for clients"
else
  echo "trim finished with failures: fix them and run again" >&2
fi
exit $rc

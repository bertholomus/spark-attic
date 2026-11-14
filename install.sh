#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 bertholomus
# install.sh: spark-attic on a GB10 node, one command. Also safe as
#   curl -fsSL <release url>/install.sh | sudo bash
# Preflight (read-only, each failure one sentence with what to do), fetch the artifact for the
# version (--from DIR|TARBALL for an offline copy) and verify it against SHA256SUMS, install the
# files, make the key root-only when none exists (never overwrite a key), install the unit, run
# attic-validate.sh for the running driver, enable and start attic.service only if validate
# passed, then read the status file and print the result. A second run on an installed node
# changes nothing, or upgrades in place when the artifact is newer. Any failure after the first
# write rolls back what this run changed and says what it rolled back. Every step is logged to
# /var/log/spark-attic/install.log (--check writes nothing anywhere).
#   --check            run only the preflight and change nothing (exit 0 clean, 1 problems)
#   --format json      with --check: one JSON object per check, for spark-attic doctor
#   --from DIR|TGZ     install from an unpacked artifact or a release tarball (default: fetch
#                      the release for --version from --url)
#   --version V        the artifact version (default the stamped INSTALL_VERSION)
#   --url BASE         where releases live (default the bertholomus release bucket below)
#   --with-sysctl      install the optional vm.min_free_kbytes drop-in (the default leaves it out)
#   --no-sysctl        accepted and ignored: no drop-in is the default
#   --root DIR         install under DIR instead of / (the host tests)
#   --plain            no colour and no header art
#   ATTIC_VERSION_FILE, ATTIC_OTHER_SOCKETS, ATTIC_PROBE_ARGS, ATTIC_MODES, ATTIC_COLOR:
#   the same seams attic-validate.sh and the CLI honour.
#   Exit: 0 done (or nothing to do), 1 a check or a step failed, 2 usage.
set -uo pipefail

NAME=spark-attic
HERE=$(cd "$(dirname "$0")" && pwd)
VERSION=${INSTALL_VERSION:-0.1.0}   # build-release.sh stamps the release version here
URL_BASE=${ATTIC_URL_BASE:-https://github.com/bertholomus/spark-attic/releases/download}
SYSCTL_NAME=99-spark-attic-minfree.conf
CHECK=0; JSON=0; PLAIN=0; WITH_SYSCTL=0; NO_SYSCTL=0; DEMO=0; FAIL_STEP=""
FROM=""; WANT_VERSION=""; ROOT=""; ART=""
ESC=$'\033'

say() { printf 'install.sh: %s\n' "$*" >&2; }
die_usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

# ---------------------------------------------------------------- colour and the header
# The Bertholomus colours (the website icon kit): the frames take the paper colour, the L nose
# the gold-on-dark one, on the terminal's own background. Colour only on a colour TTY.
COLOUR=0; TRUECOLOR=0
if [ "$PLAIN" = 0 ] && [ -z "${NO_COLOR:-}" ] && [ "${ATTIC_COLOR:-auto}" != never ]; then
  case "${ATTIC_COLOR:-auto}" in
    always) COLOUR=1 ;;
    auto) [ -t 1 ] && [ -n "${TERM:-}" ] && [ "$TERM" != dumb ] && COLOUR=1 ;;
  esac
fi
if [ "$COLOUR" = 1 ]; then
  case "${TERM:-}" in *direct*) TRUECOLOR=1 ;; esac
  [ -n "${COLORTERM:-}" ] && TRUECOLOR=1
fi
if [ "$COLOUR" = 1 ]; then
  # The icon kit's terminal colours: paper #f3f2f2, the L's gold #e1ad66.
  if [ "$TRUECOLOR" = 1 ]; then
    C_PAPER="$ESC[38;2;243;242;242m"; C_GOLD="$ESC[38;2;225;173;102m"
    C_PAPER_BOLD="$ESC[1;38;2;243;242;242m"; C_GOLD_BOLD="$ESC[1;38;2;225;173;102m"
  else
    C_PAPER="$ESC[38;5;255m"; C_GOLD="$ESC[38;5;179m"
    C_PAPER_BOLD="$ESC[1;38;5;255m"; C_GOLD_BOLD="$ESC[1;38;5;179m"
  fi
  C_RESET="$ESC[0m"
else
  C_PAPER=""; C_GOLD=""; C_PAPER_BOLD=""; C_GOLD_BOLD=""; C_RESET=""
fi

share_dir() {
  if [ -n "${ATTIC_SHARE_DIR:-}" ]; then printf '%s' "$ATTIC_SHARE_DIR"; return; fi
  local here; here=$(cd "$(dirname "$0")" && pwd)
  if [ -d "$here/share/$NAME" ]; then printf '%s' "$here/share/$NAME"
  else printf '%s' "/usr/local/share/$NAME"; fi
}
SHARE=$(share_dir)

render_header() {
  # The Bertholomus wordmark: header-art.txt is the braille,
  # header-col.txt the colour key per cell (p paper, g the L's gold, . the terminal's own background).
  local art="$SHARE/header-art.txt" cols="$SHARE/header-col.txt"
  if [ ! -r "$art" ] || [ ! -r "$cols" ] || [ "$COLOUR" != 1 ]; then
    printf '%s %s\n' "$NAME" "$VERSION"
    return
  fi
  local line kline i ch key out run rk width title pad
  width=0
  while IFS= read -r line && IFS= read -r kline <&3; do
    out=""; run=""; rk=""
    for ((i = 0; i < ${#line}; i++)); do
      ch=${line:i:1}
      key=${kline:i:1}
      case $key in
        g) ;;
        *) key=p ;;
      esac
      if [ "$key" = "$rk" ]; then
        run="$run$ch"
      else
        if [ -n "$run" ]; then
          case $rk in
            g) out="$out$C_GOLD$run" ;;
            *) out="$out$C_PAPER$run" ;;
          esac
        fi
        rk=$key; run=$ch
      fi
    done
    case $rk in
      g) out="$out$C_GOLD$run" ;;
      *) out="$out$C_PAPER$run" ;;
    esac
    printf '%s%s\n' "$out" "$C_RESET"
    width=${#line}
  done <"$art" 3<"$cols"
  title="$NAME $VERSION"
  pad=$(( (width - ${#title}) / 2 )); [ "$pad" -lt 0 ] && pad=0
  printf '%*s%s%s%s\n' "$pad" "" "$C_PAPER_BOLD" "$title" "$C_RESET"
}

step_start() {
  printf '  %s·%s %s\n' "$C_PAPER" "$C_RESET" "$1"
  [ "$DEMO" = 1 ] && sleep "$DEMO_PACE"
  return 0
}
step_done() {
  printf '    %s✓%s %s%s\n' "$C_GOLD" "$C_RESET" "$1" "${2:+ ($2)}"
  [ "$DEMO" = 1 ] && sleep "$DEMO_PACE"
  return 0
}
step_fail() {
  printf '    %s✗%s %s — %s\n' "$C_PAPER_BOLD" "$C_RESET" "$1" "$2"
  [ -n "${3:-}" ] && printf '        %s\n' "$3"
  return 0
}

# ---------------------------------------------------------------- the log (install mode only)
LOGFILE=""
log_open() {
  local d="$ROOT/var/log/$NAME"
  if mkdir -p "$d" 2>/dev/null; then
    LOGFILE="$d/install.log"
    printf '%s install.sh: version %s, from %s\n' "$(date -u +%FT%TZ)" "$VERSION" "${FROM:-the release url}" >>"$LOGFILE"
  else
    say "cannot write the log directory $d: continuing without a log"
  fi
}
log() { [ -n "$LOGFILE" ] && printf '%s install.sh: %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOGFILE"; return 0; }

# ---------------------------------------------------------------- preflight
# One record per check: the step list, or JSON lines for spark-attic doctor.
PRE_N=0; PRE_PROBLEMS=0
pre_record() { # id state what [fix]
  PRE_N=$((PRE_N + 1))
  eval "PRE_ID_$PRE_N=\$1; PRE_ST_$PRE_N=\$2; PRE_WHAT_$PRE_N=\$3; PRE_FIX_$PRE_N=\${4:-}"
  [ "$DEMO" = 1 ] && sleep "$DEMO_PACE"
  if [ "$JSON" = 1 ]; then
    printf '{"id":"%s","state":"%s","what":"%s","fix":"%s"}\n' "$1" "$2" "$3" "${4:-}"
  elif [ "$2" = pass ]; then
    printf '  %s✓%s %s\n' "$C_GOLD" "$C_RESET" "$3"
  elif [ "$2" = warn ]; then
    printf '  ~ %s (fix when you can: %s)\n' "$3" "${4:-see INSTALL.md}"
  else
    printf '  %s✗%s %s\n        %s\n' "$C_PAPER_BOLD" "$C_RESET" "$3" "${4:-}"
  fi
  return 0
}
listening() { # PATH-socket: is anything listening on it? (awk reads all of ss's output: pipefail)
  ss -H -x -l 2>/dev/null | awk -v p="$1" '$5 == p { found = 1 } END { exit !found }'
}
running_driver() { # the NVIDIA driver the node runs, from the version file
  local VFILE=${ATTIC_VERSION_FILE:-$ROOT/proc/driver/nvidia/version}
  [ -r "$VFILE" ] || return 0
  head -n1 "$VFILE" | grep -o -E '[0-9]+(\.[0-9]+){1,3}' | head -n1
  return 0
}

preflight_run() {
  local problems=0
  local mach; mach=$(uname -m 2>/dev/null || echo unknown)
  if [ "$mach" = aarch64 ]; then
    pre_record arch pass "this machine is aarch64"
  else
    pre_record arch fail "this machine is $mach, not aarch64: spark-attic $VERSION is built for GB10 nodes" \
      "run install.sh on the GB10 node"
    problems=$((problems + 1))
  fi
  local model="" modelf
  for modelf in "$ROOT/proc/device-tree/model" "$ROOT/sys/firmware/devicetree/base/model"; do
    if [ -r "$modelf" ]; then model=$(tr -d '\0' <"$modelf" | head -n1); break; fi
  done
  if [ -n "$model" ]; then
    pre_record gb10 pass "the device tree says: $model"
  else
    pre_record gb10 warn "no device-tree model readable: the driver and the carveout probe settle whether this is a GB10" \
      "if this is not a GB10, atticd refuses to lend (it reports no carveout)"
  fi
  if [ "$(id -u)" = 0 ]; then
    pre_record root pass "running as root"
  elif [ "${ATTIC_TEST:-0}" = 1 ]; then
    pre_record root pass "ATTIC_TEST=1: the host test's fake node, no root"
  elif [ "$DEMO" = 1 ]; then
    pre_record root pass "no root needed: the demo writes only its throwaway fake root"
  elif [ "$CHECK" = 1 ]; then
    pre_record root warn "not root: the checks that need root are weaker" "run it as root for the full check"
  else
    pre_record root fail "install.sh must run as root (atticd needs root for the driver)" \
      "re-run with sudo, or curl -fsSL <url>/install.sh | sudo bash"
    problems=$((problems + 1))
  fi
  if [ -d "$ROOT/run/systemd/system" ]; then
    pre_record systemd pass "systemd is running"
  else
    pre_record systemd fail "systemd is not running here (no $ROOT/run/systemd/system): spark-attic installs as a systemd service" \
      "run it on the node with systemd"
    problems=$((problems + 1))
  fi
  local driver; driver=$(running_driver)
  if [ -n "$driver" ]; then
    pre_record driver pass "the NVIDIA driver is loaded: $driver"
  else
    pre_record driver fail "no NVIDIA driver is loaded (nothing to read at ${ATTIC_VERSION_FILE:-$ROOT/proc/driver/nvidia/version})" \
      "install the NVIDIA driver for the GB10 first, then re-run"
    problems=$((problems + 1))
  fi
  if [ -n "$ART" ]; then
    if [ -n "$driver" ] && [ -f "$ART/artifacts/$driver/rmshim.so" ] && [ -f "$ART/artifacts/$driver/BUILD-INFO" ]; then
      pre_record shim pass "the artifact carries a shim for driver $driver"
    else
      local have; have=$(ls "$ART/artifacts" 2>/dev/null | tr '\n' ' ')
      pre_record shim fail "the artifact has no shim for driver ${driver:-?} (it carries: ${have:-none})" \
        "build one on a machine with zig: scripts/attic-build.sh <driver>  (INSTALL.md, Build)"
      problems=$((problems + 1))
    fi
    if [ -x "$ART/bin/atticd" ]; then
      # ATTIC_PROBE_BIN: another atticd for the probe (the host tests: the artifact's is aarch64)
      # shellcheck disable=SC2086
      if "${ATTIC_PROBE_BIN:-$ART/bin/atticd}" --probe-display ${ATTIC_PROBE_ARGS:-} >/dev/null 2>&1; then
        pre_record display pass "no display is in use (atticd --probe-display)"
      else
        pre_record display fail "a display is in use: atticd lends nothing while one is" \
          "log out or stop the desktop session on the node, then re-run"
        problems=$((problems + 1))
      fi
    else
      pre_record display warn "no atticd in the artifact yet: the display check runs once it is here"
    fi
  else
    pre_record shim warn "no artifact yet (--from, or a download): the shim check runs once it is here"
    pre_record display warn "no artifact yet: the display check runs once it is here"
  fi
  local others=${ATTIC_OTHER_SOCKETS:-} s bad=0 n=0
  for s in $others; do
    n=$((n + 1))
    if [ -S "$s" ] && listening "$s"; then
      pre_record lenders fail "another carveout lender answers on $s: only one lender may hold the carveout" \
        "stop that lender first, then re-run"
      bad=1
    fi
  done
  if [ "$n" = 0 ]; then
    pre_record lenders warn "no other-lender socket is configured (ATTIC_OTHER_SOCKETS): this check cannot see a second lender" \
      "set ATTIC_OTHER_SOCKETS to that lender's socket if one runs here (INSTALL.md, one lender at a time)"
  elif [ "$bad" = 0 ]; then
    pre_record lenders pass "no other carveout lender is listening"
  fi
  [ "$bad" = 1 ] && problems=$((problems + 1))
  local df_target="$ROOT/usr/local"
  [ -d "$df_target" ] || df_target="$ROOT"
  [ -d "$df_target" ] || df_target="."
  local avail_kb; avail_kb=$(df -Pk "$df_target" 2>/dev/null | awk 'NR == 2 { print $4 }')
  case "${avail_kb:-}" in
    ''|*[!0-9]*) pre_record disk warn "could not read free space at $df_target" ;;
    *)
      if [ "$avail_kb" -lt 65536 ]; then
        pre_record disk fail "only $((avail_kb / 1024)) MiB free at $df_target: spark-attic needs about 10 MiB" \
          "free some space, then re-run"
        problems=$((problems + 1))
      else
        pre_record disk pass "$((avail_kb / 1024)) MiB free at $df_target"
      fi ;;
  esac
  PRE_PROBLEMS=$problems
  [ "$JSON" = 1 ] && printf '{"id":"summary","problems":%d}\n' "$problems"
  return 0
}

# ---------------------------------------------------------------- fetch and verify
tarball_root() { # the root directory name inside a release tarball (its first member's head)
  tar -tzf "$1" 2>/dev/null | head -n1 | cut -d/ -f1
}
fetch_into() { # $1 = temp dir; sets ART; writes only inside $1
  local tmp=$1 url sums rootname
  if [ -n "$FROM" ]; then
    if [ -d "$FROM" ]; then
      ART=$FROM
      if [ -f "$ART/SHA256SUMS" ]; then
        (cd "$ART" && sha256sum -c --quiet SHA256SUMS) || {
          say "a file in $FROM does not match its SHA256SUMS: refusing (a broken artifact must not install)"
          return 1; }
      else
        say "no SHA256SUMS in $FROM: the checksum verify is skipped"
      fi
    elif [ -f "$FROM" ]; then
      if [ -f "$FROM.sha256" ]; then
        sums=$(awk '{print $1}' "$FROM.sha256" | head -n1)
        [ "$(sha256sum "$FROM" | awk '{print $1}')" = "$sums" ] || {
          say "$FROM does not match its .sha256: refusing"; return 1; }
      fi
      tar -xzf "$FROM" -C "$tmp" || { say "cannot unpack $FROM"; return 1; }
      rootname=$(tarball_root "$FROM")
      ART="$tmp/${rootname:-.}"
      [ -f "$ART/SHA256SUMS" ] || { say "the tarball carries no SHA256SUMS: refusing"; return 1; }
      (cd "$ART" && sha256sum -c --quiet SHA256SUMS) || {
        say "a file in $FROM does not match its SHA256SUMS: refusing"; return 1; }
    else
      say "--from $FROM is neither a directory nor a file"; return 1
    fi
    return 0
  fi
  local v=${WANT_VERSION:-$VERSION}
  url="$URL_BASE/v$v/$NAME-$v-aarch64.tar.gz"
  command -v curl >/dev/null || { say "curl is missing: it fetches the release, or use --from"; return 1; }
  say "fetching $url"
  curl -fsSL --retry 3 -o "$tmp/a.tar.gz" "$url" || { say "the fetch failed: $url"; return 1; }
  # The standalone installer of a release carries its tarball's sha256 (build-release.sh stamps
  # ATTIC_SHA256). That is the trust root for its own version; the .sha256 sidecar is checked too when
  # the release publishes one. Another version (--version) has no stamp here and needs the sidecar.
  local stamped=""
  [ -n "${ATTIC_SHA256:-}" ] && [ "$v" = "$VERSION" ] && stamped=$ATTIC_SHA256
  if curl -fsSL --retry 3 -o "$tmp/a.tar.gz.sha256" "$url.sha256"; then
    sums=$(awk '{print $1}' "$tmp/a.tar.gz.sha256" | head -n1)
  elif [ -n "$stamped" ]; then
    sums=""
  else
    say "the release publishes no .sha256 at $url.sha256: refusing (a checksum inside the tarball cannot prove the tarball is the one that was published)"
    return 1
  fi
  got=$(sha256sum "$tmp/a.tar.gz" | awk '{print $1}')
  if [ -n "$stamped" ] && [ "$got" != "$stamped" ]; then
    say "the download does not match the sha256 this installer carries: refusing"; return 1
  fi
  if [ -n "$sums" ] && [ "$got" != "$sums" ]; then
    say "the download does not match its published sha256: refusing"; return 1
  fi
  tar -xzf "$tmp/a.tar.gz" -C "$tmp" || { say "cannot unpack the download"; return 1; }
  rootname=$(tarball_root "$tmp/a.tar.gz")
  ART="$tmp/${rootname:-.}"
  [ -f "$ART/SHA256SUMS" ] || { say "the artifact carries no SHA256SUMS: refusing"; return 1; }
  (cd "$ART" && sha256sum -c --quiet SHA256SUMS) || {
    say "a file in the artifact does not match its SHA256SUMS: refusing"; return 1; }
  return 0
}

art_share() { # the artifact's share files: share/spark-attic/ in a release tarball (build-release.sh), or a flat share/
  if [ -d "$ART/share/$NAME" ]; then printf '%s' "share/$NAME"; else printf '%s' "share"; fi
}

artifact_check() { # every file install.sh puts down must be in the artifact
  local f AS; AS=$(art_share)
  # the header art files are optional: the release's art gate may leave them out (plain header)
  for f in VERSION bin/atticd bin/attic-check bin/spark-attic scripts/attic-validate.sh install.sh \
           systemd/attic.service sysctl/$SYSCTL_NAME $AS/README.md python/pyproject.toml; do
    [ -e "$ART/$f" ] || { say "the artifact is missing $f: refusing (fetch the release tarball again)"; return 1; }
  done
  return 0
}
artifact_preflight() { # the shim and display checks, once the artifact is here (they warned before)
  local driver; driver=$(running_driver)
  if [ -n "$driver" ] && [ -f "$ART/artifacts/$driver/rmshim.so" ] && [ -f "$ART/artifacts/$driver/BUILD-INFO" ]; then
    step_done "the artifact carries a shim for driver $driver"
  else
    step_fail "shim" "the artifact has no shim for driver ${driver:-?}" \
      "build one on a machine with zig: scripts/attic-build.sh <driver>  (INSTALL.md, Build)"
    return 1
  fi
  if "${ATTIC_PROBE_BIN:-$ART/bin/atticd}" --probe-display ${ATTIC_PROBE_ARGS:-} >/dev/null 2>&1; then
    step_done "no display is in use"
  else
    step_fail "display" "a display is in use: atticd lends nothing while one is" \
      "log out or stop the desktop session on the node, then re-run"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- install with rollback
BK=""
jrnl() { printf '%s\n' "$*" >>"$BK/journal"; }
BACKUP_TARGETS="/usr/local/sbin/atticd /usr/local/sbin/attic-check /usr/local/bin/$NAME
/usr/local/share/$NAME
/etc/systemd/system/attic.service
/var/lib/attic
/etc/sysctl.d/$SYSCTL_NAME"

backup_all() { # snapshot what exists now; the journal drives the rollback
  BK=$(mktemp -d)
  : >"$BK/journal"
  local was; was=$(systemctl is-active attic.service 2>/dev/null || true)
  jrnl "state ${was:-none}"
  local t dst
  for t in $BACKUP_TARGETS; do
    t="$ROOT$t"
    if [ -e "$t" ]; then
      dst="$BK/files$(printf %s "$t" | tr / _)"
      mkdir -p "$(dirname "$dst")"
      cp -a "$t" "$dst" || { say "cannot back up $t before installing: refusing"; return 1; }
      jrnl "replaced $t"
    else
      jrnl "created $t"
    fi
  done
  [ -e "$ROOT/etc/spark-attic/key" ] && jrnl "key kept" || jrnl "key absent"
  return 0
}

rollback() { # undo, in reverse, what this run changed; say what it undid
  [ -n "${ROLLED_BACK:-}" ] && return 0   # a signal during the rollback itself must not undo twice
  ROLLED_BACK=1
  say "rolling back what this run changed"
  local was="" line path backup
  [ -f "$BK/journal" ] && was=$(sed -n 's/^state //p' "$BK/journal" | head -n1)
  if [ -f "$BK/journal" ]; then
    awk '{lines[NR] = $0} END { for (i = NR; i >= 1; i--) print lines[i] }' "$BK/journal" |
    while IFS= read -r line; do
      case $line in
        replaced\ *)
          path=${line#replaced }
          backup="$BK/files$(printf %s "$path" | tr / _)"
          if [ -e "$backup" ]; then
            rm -rf -- "$path" && cp -a "$backup" "$path" && echo "  restored $path"
          fi ;;
        created\ *)
          path=${line#created }
          if [ -e "$path" ]; then rm -rf -- "$path"; echo "  removed $path (was not there before)"; fi ;;
        key\ absent)
          if [ -e "$ROOT/etc/spark-attic/key" ]; then
            rm -f "$ROOT/etc/spark-attic/key"; echo "  removed the key this run created"
          fi ;;
      esac
    done
  fi
  systemctl daemon-reload 2>/dev/null
  if [ "$was" = active ]; then
    systemctl start attic.service 2>/dev/null &&
      echo "  attic.service is running again (as before this run)"
  else
    systemctl stop attic.service 2>/dev/null
    systemctl disable attic.service 2>/dev/null
  fi
  log "rolled back (a failure after the first write)"
  say "rolled back: the system is as it was before this run (the log stays in /var/log/$NAME)"
}

failed_reports() { # $1 = driver: copy the candidate reports where the rollback cannot remove them
  # A failed validate writes its reports under /var/lib/attic/candidates/<driver>, and /var/lib/attic is one
  # of the backup targets, so the rollback removes them: the message's "send them with a bug report" then
  # pointed at files that were gone. Copy them to /var/log/spark-attic/failed-<UTC stamp>/ first (0700): the
  # log directory is not a backup target, so the reports outlive the rollback. Prints the copy's path.
  local src="$ROOT/var/lib/attic/candidates/$1" dst
  [ -d "$src" ] || return 0
  dst="$ROOT/var/log/$NAME/failed-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dst" 2>/dev/null || return 0
  cp -a "$src/." "$dst/" 2>/dev/null || return 0
  chmod 700 "$dst" 2>/dev/null   # after cp: cp -a would carry the candidate dir's own mode onto it
  printf '%s' "$dst"
}

install_files() {
  local R="$ROOT" d f
  step_start "installing the files"
  for d in usr/local/sbin usr/local/bin "/usr/local/share/$NAME/scripts" etc/systemd/system \
           etc/sysctl.d var/lib/attic etc/spark-attic; do
    mkdir -p "$R/$d" || return 1
  done
  install -m 755 "$ART/bin/atticd" "$R/usr/local/sbin/atticd" || return 1
  install -m 755 "$ART/bin/attic-check" "$R/usr/local/sbin/attic-check" || return 1
  install -m 755 "$ART/bin/spark-attic" "$R/usr/local/bin/$NAME" || return 1
  for f in header-art.txt header-col.txt README.md INSTALL.md STATUS-FILE.md \
           RELEASE-NOTES-0.1.0.md THIRD_PARTY_NOTICES LICENSE-MIT LICENSE-APACHE; do
    # the art files are optional (the release's art gate may leave them out); the rest are not
    [ -f "$ART/$(art_share)/$f" ] && install -m 644 "$ART/$(art_share)/$f" "$R/usr/local/share/$NAME/$f"
  done
  for f in README.md INSTALL.md STATUS-FILE.md RELEASE-NOTES-0.1.0.md \
           THIRD_PARTY_NOTICES LICENSE-MIT LICENSE-APACHE; do
    [ -f "$R/usr/local/share/$NAME/$f" ] ||
      { say "the artifact is missing share/$f: refusing"; return 1; }
  done
  install -m 755 "$ART/scripts/attic-validate.sh" "$R/usr/local/share/$NAME/scripts/attic-validate.sh" || return 1
  install -m 755 "$ART/install.sh" "$R/usr/local/share/$NAME/install.sh" || return 1
  rm -rf "$R/usr/local/share/$NAME/python"
  cp -a "$ART/python" "$R/usr/local/share/$NAME/python" || return 1
  install -m 644 "$ART/systemd/attic.service" "$R/etc/systemd/system/attic.service" || return 1
  if [ "$WITH_SYSCTL" = 1 ]; then
    install -m 644 "$ART/sysctl/$SYSCTL_NAME" "$R/etc/sysctl.d/$SYSCTL_NAME" || return 1
  fi
  install -m 644 "$ART/VERSION" "$R/var/lib/attic/version" || return 1
  if [ "$WITH_SYSCTL" = 1 ]; then
    step_done "installed" "binaries, share files, python, the unit, the sysctl drop-in"
  else
    step_done "installed" "binaries, share files, python, the unit"
  fi
  log "files installed from $ART"
  return 0
}

ensure_key() {
  if [ -e "$ROOT/etc/spark-attic/key" ]; then
    step_done "key" "the existing key is kept"
    log "the existing key is kept"
    return 0
  fi
  local k="$ROOT/etc/spark-attic/key"
  (umask 077; od -An -N12 -tx1 /dev/urandom | tr -d ' \n' | sed 's/^/attic_/' >"$k") || return 1
  chmod 600 "$k"
  step_done "key" "created, root-only (0600)"
  log "created a key at $k"
  return 0
}

wait_status() { # wait for a fresh status file; nothing printed, 0 when one is there
  local sf="$ROOT/run/spark-attic/status.json" i
  for i in $(seq 60); do
    if [ -r "$sf" ] && grep -q "\"name\":\"$NAME\"" "$sf" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

next_steps() {
  local py="/usr/local/share/$NAME/python"
  printf '      next: vLLM — put %s on PYTHONPATH, mount /run/spark-attic and the key (INSTALL.md, Engines)\n' "$py"
  printf '      next: SGLang — the same mount and PYTHONPATH (INSTALL.md, Engines)\n'
  printf '      next: TensorFold — TF_CUDA_CARVEOUT_SOURCE=attic and the key (INSTALL.md, Engines)\n'
}

# ---------------------------------------------------------------- demo mode
# --demo runs this same install flow against a simulated GB10 in a throwaway fake root: a fake
# driver file (580.178.04), fake systemctl and ss, the host build's fakeshim + fakecuda. Nothing
# outside the temp dir is touched and nothing needs root. --fail STEP shows a refusal path:
# display (a connector reports connected), driver (no driver loaded), lender (another lender
# answers), validate (the carveout check fails; the rollback path).
DEMO_ROOT=""
DEMO_PACE=${DEMO_PACE:-0.7}   # 0: no pacing (the host tests)
demo_prepare() {
  DEMO_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/spark-attic-demo.XXXXXX") || return 1
  local r=$DEMO_ROOT d
  for d in run/systemd/system proc/driver/nvidia proc/device-tree usr/local etc/sysctl.d \
           fakebin sys/class/drm .systemctl; do
    mkdir -p "$r/$d"
  done
  printf 'NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.178.04  Release Build  (demo)\n' \
    >"$r/proc/driver/nvidia/version"
  printf 'ASUS Ascent GX10 (demo root; nothing is installed)\n' >"$r/proc/device-tree/model"
  printf 'MemTotal: 129798784 kB\nMemFree: 64000000 kB\n' >"$r/meminfo"
  printf '80000000-27fffffff : System RAM\n280000000-2ffffffff : reserved\n' >"$r/iomem"
  case $FAIL_STEP in
    driver) : >"$r/proc/driver/nvidia/version" ;;
    display) mkdir -p "$r/sys/class/drm/card0-DP-1"; echo connected >"$r/sys/class/drm/card0-DP-1/status" ;;
  esac
  # the fake bins: uname says aarch64, systemctl keeps its state in the root, ss is silent
  printf '#!/bin/sh\necho aarch64\n' >"$r/fakebin/uname"
  cat >"$r/fakebin/systemctl" <<FAKE
#!/bin/bash
S=$r/.systemctl
echo "\$*" >>"\$S-log"
case "\$1 \$2" in
  "is-active --quiet"|"is-active attic.service") [ -f "\$S/active" ]; exit \$((1 - \$?)) ;;
  "is-enabled attic.service") [ -f "\$S/enabled" ]; exit \$((1 - \$?)) ;;
  "daemon-reload"*) exit 0 ;;
  "stop attic.service") rm -f "\$S/active"; exit 0 ;;
  "disable attic.service") rm -f "\$S/enabled"; exit 0 ;;
  "start attic.service"|"enable --now"*)
    echo enabled >"\$S/enabled"; echo active >"\$S/active"
    mkdir -p "$r/run/spark-attic"
    printf '{"name":"spark-attic","version":"0.1.0","state":"ready","why":"","driver":"580.178.04","mode":"heap","validated":true,"validated_at":"%s","size_mib":2044,"free_mib":0,"largest_mib":0,"lent_mib":2044,"draining_mib":0,"clients":1,"every_s":10,"pid":1,"updated":%s}\n' \
      "\$(date -u +%FT%TZ)" "\$(date +%s)" >"$r/run/spark-attic/status.json"
    exit 0 ;;
  "enable attic.service") echo enabled >"\$S/enabled"; exit 0 ;;
  *) exit 0 ;;
esac
FAKE
  printf '#!/bin/sh\nexit 0\n' >"$r/fakebin/ss"
  chmod 755 "$r/fakebin"/*
  case $FAIL_STEP in
    lender)
      touch "$r/other-lender.sock"
      printf '#!/bin/sh\necho "u_str LISTEN 0 1 $ATTIC_SS_LISTEN 1 * 0"\nexit 0\n' >"$r/fakebin/ss"
      chmod 755 "$r/fakebin/ss" ;;
  esac
  # the demo artifact: this tree's own files; the binaries from a host build (make) or an artifact
  local binsrc=""
  [ -x "$HERE/build/host/atticd" ] && binsrc=$HERE/build/host
  [ -z "$binsrc" ] && [ -x "$HERE/bin/atticd" ] && binsrc=$HERE/bin
  [ -n "$binsrc" ] || { say "the demo needs a host build (make) or an artifact with bin/atticd"; return 1; }
  local A=$r/artifact
  mkdir -p "$A/bin" "$A/artifacts/580.178.04" "$A/scripts" "$A/systemd" "$A/sysctl" "$A/share"
  cp "$binsrc/atticd" "$binsrc/attic-check" "$A/bin/"
  cp "$HERE/scripts/spark-attic" "$A/bin/spark-attic" 2>/dev/null || cp "$HERE/scripts/spark-attic" "$A/bin/spark-attic"
  # the demo shim: the fake shim compiled for the demo driver (580.178.04), so the validate runs
  # its real checks and passes them; without a compiler the host build's own fakeshim is used and
  # the validate fails honestly (the demo shows its refusal path).
  if command -v cc >/dev/null && [ -f "$HERE/src/fakeshim.c" ]; then
    cc -std=gnu11 -O2 -fPIC -shared -fvisibility=hidden -I"$HERE/src" \
      -DATTIC_DRIVER="\"580.178.04\"" -o "$A/artifacts/580.178.04/rmshim.so" "$HERE/src/fakeshim.c" ||
      cp "$HERE/build/host/fakeshim.so" "$A/artifacts/580.178.04/rmshim.so"
  elif [ -f "$HERE/build/host/fakeshim.so" ]; then
    cp "$HERE/build/host/fakeshim.so" "$A/artifacts/580.178.04/rmshim.so"
  else
    say "the demo needs src/fakeshim.c and a compiler, or build/host/fakeshim.so"; return 1
  fi
  {
    echo "driver=580.178.04"
    echo "shim_sha256=$(sha256sum "$A/artifacts/580.178.04/rmshim.so" | cut -c1-64)"
    echo "attic_commit=demo"
  } >"$A/artifacts/580.178.04/BUILD-INFO"
  cp "$HERE/scripts/attic-validate.sh" "$A/scripts/"
  cp "$HERE/install.sh" "$A/install.sh"
  cp "$HERE/systemd/attic.service" "$A/systemd/"
  cp "$HERE/sysctl/$SYSCTL_NAME" "$A/sysctl/"
  [ -f "$HERE/share/$NAME/header-art.txt" ] &&
    cp "$HERE/share/$NAME/header-art.txt" "$HERE/share/$NAME/header-col.txt" "$A/share/"
  # the docs: the working tree keeps them under release/, the exported tree at its root
  for doc in README.md INSTALL.md RELEASE-NOTES-0.1.0.md; do
    [ -f "$HERE/release/$doc" ] && cp "$HERE/release/$doc" "$A/share/"
    [ -f "$HERE/$doc" ] && cp "$HERE/$doc" "$A/share/"
  done
  cp "$HERE/STATUS-FILE.md" "$A/share/"
  cp "$HERE/LICENSE-MIT" "$HERE/LICENSE-APACHE" "$A/share/"
  printf 'third-party notices (demo)\n' >"$A/share/THIRD_PARTY_NOTICES"
  cp -r "$HERE/python" "$A/python"
  echo "${WANT_VERSION:-$VERSION}" >"$A/VERSION"
  (cd "$A" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum >SHA256SUMS)
  ART=$A
  ROOT=$r
  # the env the demo node needs; the validate runs for real against the fake shim
  export PATH="$r/fakebin:$PATH"
  export ATTIC_VERSION_FILE="$r/proc/driver/nvidia/version"
  export ATTIC_OTHER_SOCKETS="$r/other-lender.sock"
  export ATTIC_TEST=1
  export ATTIC_CHECK_ARGS="--meminfo $r/meminfo --iomem $r/iomem"
  export ATTIC_DAEMON_ARGS="--meminfo $r/meminfo --iomem $r/iomem"
  export LD_LIBRARY_PATH="$HERE/build/host/fakecuda"
  export ATTIC_FAKE_REGION="0x280000000:0x8000000" ATTIC_FAKE_HELD="0" ATTIC_HUP="true"
  export ATTIC_MODES=heap
  export ATTIC_PROBE_ARGS="--modeset-dir $r/proc --drm-dir $r/sys/class/drm --pmon-file $r/absent-pmon"
  if [ "$FAIL_STEP" = validate ]; then
    # the fake shim fails the calls the fail file names (re-read per call): the carveout check
    # fails on every mode, nothing is enabled, and the demo shows the rollback path
    printf 'alloc\nopen\n' >"$r/fail-calls"
    export ATTIC_FAKE_FAIL_FILE="$r/fail-calls"
  fi
  [ "$FAIL_STEP" = lender ] && export ATTIC_SS_LISTEN="$r/other-lender.sock"
  return 0
}

demo_run() {
  COLOUR=1; PLAIN=0
  # the demo's colours are always on (it is made to be watched); --plain turns them off
  [ "$PLAIN" = 0 ] && [ -z "${NO_COLOR:-}" ] && { COLOUR=1; TRUECOLOR=0; }
  if [ "$COLOUR" = 1 ] && [ -z "$C_RESET" ]; then
    C_PAPER="$ESC[38;5;255m"; C_GOLD="$ESC[38;5;179m"
    C_PAPER_BOLD="$ESC[1;38;5;255m"; C_GOLD_BOLD="$ESC[1;38;5;179m"; C_RESET="$ESC[0m"
  fi
  printf 'DEMO: nothing is installed; this runs the real install flow in a throwaway fake root.\n\n'
  demo_prepare || exit 1
  VERSION=$(cat "$ART/VERSION" 2>/dev/null || echo "$VERSION")
  # re-derive the colours with VERSION now set (the header prints the version)
  render_header
  printf '  artifact: ready in the demo root (the fetch and checksum verify steps run there too:\n'
  printf '  the artifact carries its SHA256SUMS and is checked before anything is installed)\n\n'
  : # the confirm
  if [ -t 0 ]; then
    printf '  the next step writes into the fake root as root would on a real node. continue? [y/N] '
    read -r answer
    case $answer in
      y|Y|yes|Yes) ;;
      *) printf '\n  demo cancelled: nothing was written\n'; rm -rf "$DEMO_ROOT"; exit 0 ;;
    esac
  else
    printf '  (not a terminal: the confirm is skipped for the recording)\n'
  fi
  printf '\n'
  preflight_run
  if [ "$PRE_PROBLEMS" != 0 ]; then
    printf '\n  DEMO: the preflight refused, so nothing was touched. The fake root %s is removed.\n' "$DEMO_ROOT"
    rm -rf "$DEMO_ROOT"
    exit 1
  fi
  log_open
  step_done "artifact" "checked against its SHA256SUMS"
  local failed=""
  install_files || failed="install the files"
  [ -z "$failed" ] && { ensure_key || failed="create the key"; }
  if [ -z "$failed" ]; then
    step_start "reloading systemd"
    systemctl daemon-reload && step_done "reloaded systemd" || failed="daemon-reload"
  fi
  if [ -z "$failed" ]; then
    local driver; driver=$(running_driver)
    step_start "validating driver $driver (attic-check writes and reads the carveout)"
    local vrc=0
    env ATTIC_BIN="$ROOT/usr/local/sbin" ATTIC_LIB="$ROOT/var/lib/attic" \
        ATTIC_RUN="$ROOT/run/spark-attic" ATTIC_VERSION_FILE="$ROOT/proc/driver/nvidia/version" \
        ATTIC_TEST=1 ATTIC_CHECK_ARGS="$ATTIC_CHECK_ARGS" ATTIC_DAEMON_ARGS="$ATTIC_DAEMON_ARGS" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" ATTIC_FAKE_REGION="$ATTIC_FAKE_REGION" \
        ATTIC_FAKE_HELD="$ATTIC_FAKE_HELD" ATTIC_HUP=true ATTIC_MODES=heap \
        bash "$ROOT/usr/local/share/$NAME/scripts/attic-validate.sh" "$ART/artifacts/$driver" heap || vrc=$?
    if [ $vrc = 0 ]; then
      step_done "validated" "driver $driver"
    else
      local reports; reports=$(failed_reports "$driver")
      [ -n "$reports" ] || reports="/var/lib/attic/candidates/$driver"
      step_fail "validate" "attic-check failed on every mode: nothing was enabled" \
        "the reports are in $reports: send them with a bug report"
      failed="validate"
    fi
  fi
  if [ -z "$failed" ]; then
    step_start "enabling and starting attic.service"
    systemctl enable --now attic.service && step_done "enabled attic.service" || failed="enable attic.service"
  fi
  if [ -z "$failed" ]; then
    step_done "atticd is ready"
    printf '\n  %s+2,044 MiB%s lent to CUDA\n\n' "$C_GOLD_BOLD" "$C_RESET"
    next_steps
  fi
  if [ -n "$failed" ]; then
    rollback
    printf '\n  DEMO: failed at "%s" — this is the refusal path; nothing outside the fake root was touched.\n' "$failed"
    rm -rf "$DEMO_ROOT"
    exit 1
  fi
  sleep 0.8
  printf '\n  the same node, from the installed CLI:\n\n'
  "$ROOT/usr/local/bin/$NAME" status --root "$ROOT" 2>&1 | sed 's/^/  /'
  printf '\n  DEMO finished: the fake root %s is removed, nothing is installed.\n' "$DEMO_ROOT"
  rm -rf "$DEMO_ROOT"
  exit 0
}

files_identical() { # is what is installed byte-equal to this artifact?
  local t a AS; AS=$(art_share)
  for pair in \
    "usr/local/sbin/atticd:bin/atticd" \
    "usr/local/sbin/attic-check:bin/attic-check" \
    "usr/local/bin/$NAME:bin/spark-attic" \
    "usr/local/share/$NAME/header-art.txt:$AS/header-art.txt" \
    "usr/local/share/$NAME/header-col.txt:$AS/header-col.txt" \
    "usr/local/share/$NAME/RELEASE-NOTES-0.1.0.md:$AS/RELEASE-NOTES-0.1.0.md" \
    "usr/local/share/$NAME/README.md:$AS/README.md" \
    "usr/local/share/$NAME/INSTALL.md:$AS/INSTALL.md" \
    "usr/local/share/$NAME/STATUS-FILE.md:$AS/STATUS-FILE.md" \
    "usr/local/share/$NAME/scripts/attic-validate.sh:scripts/attic-validate.sh" \
    "usr/local/share/$NAME/install.sh:install.sh" \
    "etc/systemd/system/attic.service:systemd/attic.service" \
    "etc/sysctl.d/$SYSCTL_NAME:sysctl/$SYSCTL_NAME" \
    "var/lib/attic/version:VERSION"; do
    t="$ROOT/${pair%%:*}"; a="$ART/${pair##*:}"
    [ "$WITH_SYSCTL" = 1 ] || case $pair in etc/sysctl.d/*) continue ;; esac
    case $pair in usr/local/share/$NAME/header-*) [ -e "$t" ] || [ -e "$a" ] || continue ;; esac
    cmp -s "$t" "$a" || return 1
  done
  diff -r "$ART/python" "$ROOT/usr/local/share/$NAME/python" >/dev/null 2>&1 || return 1
  return 0
}

# ---------------------------------------------------------------- main
main() {
  while [ $# -gt 0 ]; do
    case $1 in
      --check) CHECK=1 ;;
      --format) shift
        if [ "${1:-}" = json ]; then JSON=1; else die_usage; fi ;;
      --from) shift; FROM=${1:?--from needs a directory or a tarball} ;;
      --from=*) FROM=${1#--from=} ;;
      --version) shift; WANT_VERSION=${1:?--version needs a version} ;;
      --version=*) WANT_VERSION=${1#--version=} ;;
      --url) shift; URL_BASE=${1:?--url needs a base} ;;
      --url=*) URL_BASE=${1#--url=} ;;
      --with-sysctl) WITH_SYSCTL=1 ;;
      --no-sysctl) NO_SYSCTL=1 ;;  # the default: accepted, no drop-in
      --demo) DEMO=1 ;;
      --fail) shift; FAIL_STEP=${1:?--fail needs a step name: display, driver, lender, validate} ;;
      --root) shift; ROOT=${1:?--root needs a directory} ;;
      --root=*) ROOT=${1#--root=} ;;
      --plain) PLAIN=1 ;;
      -h|--help) die_usage ;;
      *) say "unknown argument: $1"; die_usage ;;
    esac
    shift
  done
  if [ "$JSON" = 1 ] && [ "$CHECK" = 0 ]; then
    say "--format json works with --check only"; exit 2
  fi
  if [ "$DEMO" = 1 ]; then
    if [ "$CHECK" = 1 ] || [ -n "$FROM" ] || [ -n "$ROOT" ]; then
      say "--demo takes no --check, --from or --root: it makes its own fake root"; exit 2
    fi
    demo_run
  fi

  local tmpfetch=""
  if [ "$CHECK" = 1 ]; then
    if [ -n "$FROM" ]; then
      tmpfetch=$(mktemp -d)
      fetch_into "$tmpfetch" || { rm -rf "$tmpfetch"; exit 1; }
    fi
    preflight_run
    # the unpacked artifact is read by the preflight above, so it is removed only after it
    [ -n "$tmpfetch" ] && rm -rf "$tmpfetch"; tmpfetch=""
    if [ "$JSON" != 1 ]; then
      if [ "$PRE_PROBLEMS" = 0 ]; then
        printf '  preflight: everything checks out\n'
      else
        printf '  preflight: %d problem%s\n' "$PRE_PROBLEMS" "$([ "$PRE_PROBLEMS" = 1 ] && echo "" || echo s)"
      fi
    fi
    [ "$PRE_PROBLEMS" = 0 ] || exit 1
    exit 0
  fi

  render_header

  if [ -n "$FROM" ]; then
    tmpfetch=$(mktemp -d)
    fetch_into "$tmpfetch" || {
      step_fail "fetch" "the artifact at $FROM did not arrive or failed its checksums" "check the path"; exit 1; }
  fi
  preflight_run
  [ "$PRE_PROBLEMS" = 0 ] || exit 1

  log_open

  if [ -z "$FROM" ]; then
    tmpfetch=$(mktemp -d)
    step_start "fetching the artifact"
    fetch_into "$tmpfetch" || {
      step_fail "fetch" "the artifact for $VERSION did not arrive or failed its checksums" \
        "check the url, or use --from DIR|TARBALL for an offline copy"; log "fetch failed"; exit 1; }
    step_done "fetched" "$NAME-$VERSION-aarch64"
  fi
  artifact_check || { step_fail "artifact" "the artifact is incomplete" "fetch the release tarball again"; exit 1; }
  artifact_preflight || { log "artifact preflight failed"; exit 1; }
  VERSION=$(cat "$ART/VERSION" 2>/dev/null || echo "$VERSION")

  # idempotency: a second run changes nothing; an older artifact never downgrades
  local rec="$ROOT/var/lib/attic/version" installed=""
  [ -r "$rec" ] && installed=$(cat "$rec")
  if [ -n "$installed" ]; then
    local older; older=$(printf '%s\n%s\n' "$VERSION" "$installed" | sort -V | head -n1)
    if [ "$older" = "$VERSION" ] && [ "$installed" != "$VERSION" ]; then
      say "installed $installed is newer than this $VERSION: nothing changed"
      log "refused a downgrade: installed $installed, artifact $VERSION"
      exit 0
    fi
    if [ "$installed" = "$VERSION" ] && files_identical; then
      say "spark-attic $VERSION is already installed: nothing to do"
      log "already installed, nothing to do"
      exit 0
    fi
    step_done "upgrading" "installed $installed -> $VERSION"
  fi

  backup_all || exit 1
  # F7: a Ctrl-C or a dropped SSH session between the first write and the enable must not leave a
  # half install with the service stopped: from here every signal rolls back and exits 130.
  trap 'rollback; exit 130' INT TERM HUP
  # F8: an upgrade while an engine holds the carveout would stop atticd, find no free block and fail
  # with "attic-check failed on every mode", while the reports the message names were about to be
  # rolled away. Read the status file first: a lent or draining carveout refuses with the validate
  # script's own exit-4 sentence and this run writes nothing.
  local was; was=$(systemctl is-active attic.service 2>/dev/null || true)
  if [ "$was" = active ]; then
    local sf="$ROOT/run/spark-attic/status.json" lent draining
    lent=$(sed -n 's/.*"lent_mib":\([0-9]*\).*/\1/p' "$sf" 2>/dev/null | head -n1)
    draining=$(sed -n 's/.*"draining_mib":\([0-9]*\).*/\1/p' "$sf" 2>/dev/null | head -n1)
    if [ -r "$sf" ] && { [ "${lent:-0}" != 0 ] || [ "${draining:-0}" != 0 ]; }; then
      step_fail "atticd" "still lends memory: wait for its clients to exit" \
        "the engines hold the carveout: stop them, then re-run (nothing was changed)"
      log "refused: the carveout is still lent (lent_mib=${lent:-?} draining_mib=${draining:-?})"
      say "nothing changed"
      exit 1
    fi
  fi
  local failed=""
  install_files || failed="install the files"
  [ -z "$failed" ] && { ensure_key || failed="create the key"; }
  if [ -z "$failed" ]; then
    step_start "reloading systemd"
    if systemctl daemon-reload; then step_done "reloaded systemd"; else failed="daemon-reload"; fi
  fi
  if [ -z "$failed" ]; then
    local driver; driver=$(running_driver)
    if [ "$was" = active ]; then
      step_start "stopping attic.service for the check"
      if systemctl stop attic.service; then step_done "stopped attic.service"; else failed="stop attic.service"; fi
    fi
    if [ -z "$failed" ]; then
      step_start "validating driver $driver (attic-check writes and reads the carveout)"
      local vrc=0
      env ATTIC_BIN="$ROOT/usr/local/sbin" ATTIC_LIB="$ROOT/var/lib/attic" \
          ATTIC_RUN="$ROOT/run/spark-attic" \
          ATTIC_VERSION_FILE="${ATTIC_VERSION_FILE:-$ROOT/proc/driver/nvidia/version}" \
          ${ROOT:+ATTIC_TEST=1} \
          bash "$ROOT/usr/local/share/$NAME/scripts/attic-validate.sh" "$ART/artifacts/$driver" \
          "${ATTIC_MODES:-heap list}" || vrc=$?
      case $vrc in
        0) step_done "validated" "driver $driver" ;;
        1) local reports; reports=$(failed_reports "$driver")
           [ -n "$reports" ] || reports="/var/lib/attic/candidates/$driver"
           step_fail "validate" "attic-check failed on every mode: nothing was enabled" \
           "the reports are in $reports: send them with a bug report"
           failed="validate" ;;
        3) step_fail "validate" "the node runs another driver than the artifact expects" \
           "build the shim for the running driver: scripts/attic-build.sh <driver>"; failed="validate" ;;
        4) step_fail "validate" "the carveout is in use: a lender or an engine still holds it" \
           "wait for the engines to exit (or stop the other lender), then re-run"; failed="validate" ;;
        *) step_fail "validate" "attic-validate.sh failed (exit $vrc)" "see its output above"; failed="validate" ;;
      esac
      log "validate exit $vrc"
    fi
  fi
  if [ -z "$failed" ]; then
    step_start "enabling and starting attic.service"
    if systemctl enable --now attic.service; then
      step_done "enabled attic.service"
      log "attic.service enabled and started"
    else
      failed="enable attic.service"
    fi
  fi
  if [ -z "$failed" ]; then
    if wait_status; then
      local sf="$ROOT/run/spark-attic/status.json" state why size num
      state=$(sed -n 's/.*"state":"\([a-z]*\)".*/\1/p' "$sf" | head -n1)
      why=$(sed -n 's/.*"why":"\([^"]*\)".*/\1/p' "$sf" | head -n1)
      size=$(sed -n 's/.*"size_mib":\([0-9]*\).*/\1/p' "$sf" | head -n1)
      if [ "$state" = ready ]; then
        step_done "atticd is ready"
        if [ -n "${size:-}" ] && [ "$size" -gt 0 ] 2>/dev/null; then
          num=$(printf '%d' "$size" | sed -e ':a;s/\B[0-9]\{3\}\>/,&/;ta')
          printf '\n  %s+%s MiB%s lent to CUDA\n\n' "$C_GOLD_BOLD" "$num" "$C_RESET"
        fi
        next_steps
        log "install finished: atticd ready, size_mib=${size:-?}"
      else
        step_fail "atticd" "it is up but refuses to lend: ${why:-unknown}" \
          "spark-attic doctor says which check to fix"
        log "install finished: atticd refusing (${why:-?})"
      fi
    else
      step_fail "atticd" "no status file appeared: atticd may have failed to start" \
        "journalctl -u attic.service -n 50"
      log "no status file after start"
      failed="atticd wrote no status file"
    fi
  fi
  if [ -n "$failed" ]; then
    rollback
    printf 'install.sh: failed at "%s": rolled back, the system is as it was before this run\n' "$failed"
    exit 1
  fi
  trap - INT TERM HUP
  exit 0
}

main "$@"

#!/usr/bin/env bash
# Push a firmware image to RuView ESP32 nodes over WiFi (ADR-050 OTA).
#
#   scripts/ota_update_nodes.sh <firmware.bin> <node-ip> [<node-ip> ...]
#
# The pre-shared key is read from $RUVIEW_OTA_PSK_FILE (default
# ~/.config/ruview/ota_psk, as provisioned with `provision.py --ota-psk`) and
# passed to curl via a private header file, never on the command line. Each
# node reboots into the other OTA slot and must pass its health check
# (ota_health.c) before it is confirmed; otherwise the bootloader rolls back.
# Nodes are updated one at a time, and the script stops at the first failure.
set -euo pipefail
image=${1:?usage: $0 <firmware.bin> <node-ip>...}; shift
[ $# -gt 0 ] || { echo "usage: $0 <firmware.bin> <node-ip>..." >&2; exit 2; }
psk_file=${RUVIEW_OTA_PSK_FILE:-$HOME/.config/ruview/ota_psk}
[ -r "$psk_file" ] || { echo "no OTA key at $psk_file" >&2; exit 2; }

hdr=$(mktemp); chmod 600 "$hdr"; trap 'rm -f "$hdr"' EXIT
printf 'Authorization: Bearer %s\n' "$(tr -d '[:space:]' < "$psk_file")" > "$hdr"

for ip in "$@"; do
  echo "== $ip"
  before=$(curl -fsS -m5 "http://$ip:8032/ota/status")
  echo "   before: $before"
  curl -fsS -m300 -H @"$hdr" -H 'Content-Type: application/octet-stream' \
       --data-binary @"$image" "http://$ip:8032/ota"; echo
  # Wait for the reboot, then for the health check to confirm the image.
  for _ in $(seq 1 60); do
    sleep 3
    status=$(curl -fsS -m3 "http://$ip:8032/ota/status" 2>/dev/null || true)
    case "$status" in *'"ota_state":"valid"'*) echo "   after:  $status"; continue 2 ;; esac
  done
  echo "   $ip did not confirm the new image within 180 s (it will roll back on failure)" >&2
  exit 1
done
echo "all nodes updated and confirmed"

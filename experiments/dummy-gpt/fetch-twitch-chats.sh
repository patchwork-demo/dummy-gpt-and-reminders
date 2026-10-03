#!/usr/bin/env bash
#
# Download Twitch chat for a list of VODs and merge everything into one chat.json.
#
# Usage:
#   ./fetch-twitch-chats.sh                      # uses vod-urls.txt
#   ./fetch-twitch-chats.sh urls1.txt urls2.txt  # several files / globs
#   cat urls.txt | ./fetch-twitch-chats.sh -     # read from stdin
#
# Each input line may be a full URL (".../videos/123") or a bare id; anything else
# on the line is ignored, so raw scraped output works as-is.
# Already-downloaded VODs are skipped, so the script is safe to re-run.
#
# Config via environment variables:
#   TWITCH_DOWNLOADER  path to TwitchDownloaderCLI  (default: the Downloads folder)
#   RAW_DIR            folder for per-VOD files     (default: chats)
#   OUT                merged output file           (default: chat.json)

set -euo pipefail

CLI="${TWITCH_DOWNLOADER:-/home/wowzers/Downloads/TwitchDownloaderCLI-1.56.5-Linux-x64/TwitchDownloaderCLI}"
RAW_DIR="${RAW_DIR:-chats}"
OUT="${OUT:-chat.json}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

inputs=("$@")
if [[ ${#inputs[@]} -eq 0 ]]; then
  inputs=(vod-urls.txt)
fi

if [[ ! -x "$CLI" ]]; then
  echo "TwitchDownloaderCLI not found or not executable: $CLI" >&2
  echo "Set TWITCH_DOWNLOADER=/path/to/TwitchDownloaderCLI" >&2
  exit 1
fi

for f in "${inputs[@]}"; do
  if [[ "$f" != "-" && ! -f "$f" ]]; then
    echo "Input file not found: $f" >&2
    exit 1
  fi
done

mkdir -p "$RAW_DIR"

# Collect unique numeric video ids from URLs (".../videos/123") or bare ids.
text=""
for f in "${inputs[@]}"; do
  if [[ "$f" == "-" ]]; then
    text+="$(cat)"
  else
    text+="$(cat "$f")"
  fi
  text+=$'\n'
done
ids="$(printf '%s\n' "$text" | sed -E 's#/videos/# #g' | grep -oE '[0-9]{6,}' | sort -u || true)"

if [[ -z "$ids" ]]; then
  echo "No video ids found in: ${inputs[*]}" >&2
  exit 1
fi

total="$(printf '%s\n' "$ids" | wc -l)"
echo "Found $total video id(s)."

i=0
for id in $ids; do
  i=$((i + 1))
  out="$RAW_DIR/$id.json"
  if [[ -s "$out" ]]; then
    echo "[$i/$total] $id: already downloaded, skipping"
    continue
  fi
  echo "[$i/$total] $id: downloading chat..."
  if ! "$CLI" chatdownload --id "$id" -o "$out" --collision Overwrite; then
    echo "[$i/$total] $id: FAILED (continuing with the rest)" >&2
    rm -f "$out"
  fi
done

shopt -s nullglob
downloaded=("$RAW_DIR"/*.json)
if [[ ${#downloaded[@]} -eq 0 ]]; then
  echo "Nothing downloaded, nothing to merge." >&2
  exit 1
fi

python3 "$SCRIPT_DIR/merge-chats.py" "${downloaded[@]}" -o "$OUT"
echo "Done: $OUT"

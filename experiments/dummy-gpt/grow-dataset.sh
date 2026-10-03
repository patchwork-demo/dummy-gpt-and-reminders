#!/usr/bin/env bash
#
# Grow the dataset end-to-end, in one command:
#   1. append any new VOD links to the master list (vod-urls.txt)
#   2. download the chat of every VOD that is not downloaded yet
#   3. rebuild chat.json from chats/*.json
#   4. rebuild messages.txt / messages_deduped.txt and print dataset stats
#
# Usage:
#   ./grow-dataset.sh new-links.txt           # add a batch, then rebuild everything
#   ./grow-dataset.sh links/*.txt             # several batches at once
#   cat new-links.txt | ./grow-dataset.sh -   # from stdin
#   ./grow-dataset.sh                         # no new links: just rebuild
#
# Already-downloaded VODs are skipped, so this is safe to re-run at any time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -gt 0 ]]; then
  python3 add-vods.py "$@"
fi

./fetch-twitch-chats.sh vod-urls.txt
python3 chat-message-body-to-txt.py
python3 dedupe-messages.py
python3 dataset-stats.py

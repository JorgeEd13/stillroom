#!/usr/bin/env bash
#
# Stop the assistant. Your documents, your index and your settings are all kept
# — this only stops the program.

set -u
cd "$(dirname "$0")" || exit 1

echo "Stopping your document assistant…"
docker compose down
echo ""
echo "Stopped. Run start.sh whenever you want it back."
echo ""
read -r -p "Press Enter to close." _ || true

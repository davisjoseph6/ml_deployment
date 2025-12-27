#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:-pcbqc-endpt}"
DEPLOYMENT="${2:-blue}"
IMAGE="${3:-test_images/board_01.jpg}"
MIN_CONF="${4:-0.95}"

python3 scripts/invoke_and_visualize.py \
  --endpoint "$ENDPOINT" \
  --deployment "$DEPLOYMENT" \
  --image "$IMAGE" \
  --out-json "out/$(basename "${IMAGE%.*}").json" \
  --out-vis "out/$(basename "${IMAGE%.*}").overlay.png" \
  --min-conf "$MIN_CONF"


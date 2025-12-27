#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?endpoint name required}"
DEPLOYMENT="${2:?deployment name required}"
REQUEST_FILE="${3:?request json file required}"
RAW_OUT="${4:-response.raw.json}"
PRETTY_OUT="${5:-response.json}"

az ml online-endpoint invoke -n "$ENDPOINT" --deployment-name "$DEPLOYMENT" --request-file "$REQUEST_FILE" \
  --only-show-errors 2>/dev/null \
| tee "$RAW_OUT" \
| ./scripts/pretty_az_json.py > "$PRETTY_OUT"

echo "Saved raw:    $RAW_OUT"
echo "Saved pretty: $PRETTY_OUT"


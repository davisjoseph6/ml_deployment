#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?endpoint name required}"
REQUEST_FILE="${2:?request json file required}"
RAW_OUT="${3:-response.curl.raw.json}"
PRETTY_OUT="${4:-response.curl.json}"

./scripts/invoke_curl.sh "$ENDPOINT" "$REQUEST_FILE" \
| tee "$RAW_OUT" \
| ./scripts/pretty_az_json.py > "$PRETTY_OUT"

echo "Saved raw:    $RAW_OUT"
echo "Saved pretty: $PRETTY_OUT"


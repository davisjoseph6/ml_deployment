#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?endpoint name required}"
REQUEST_FILE="${2:?request json file required}"

./scripts/invoke_curl.sh "$ENDPOINT" "$REQUEST_FILE" \
| ./scripts/pretty_az_json.py


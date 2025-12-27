#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?endpoint name required}"
DEPLOYMENT="${2:?deployment name required}"
REQUEST_FILE="${3:?request json file required}"

az ml online-endpoint invoke -n "$ENDPOINT" --deployment-name "$DEPLOYMENT" --request-file "$REQUEST_FILE" \
  --only-show-errors 2>/dev/null \
| ./scripts/pretty_az_json.py


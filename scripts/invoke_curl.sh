#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?endpoint name required}"
REQUEST_FILE="${2:?request json file required}"

KEY="$(az ml online-endpoint get-credentials -n "$ENDPOINT" --only-show-errors -o tsv --query primaryKey)"
URI="$(az ml online-endpoint show -n "$ENDPOINT" --only-show-errors -o tsv --query scoring_uri)"

curl -sS -X POST "$URI" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  --data @"$REQUEST_FILE"


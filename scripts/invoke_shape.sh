#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${1:?endpoint name required}"
DEPLOYMENT="${2:?deployment name required}"
REQUEST_FILE="${3:?request json file required}"

az ml online-endpoint invoke -n "$ENDPOINT" --deployment-name "$DEPLOYMENT" --request-file "$REQUEST_FILE" \
  --only-show-errors 2>/dev/null \
| python3 -c 'import sys; s=sys.stdin.read().lstrip(); print("first_char =", repr(s[:1]));'


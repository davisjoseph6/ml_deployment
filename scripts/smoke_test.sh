#!/usr/bin/env bash
set -euo pipefail

: "${AML_ENDPT:?Missing AML_ENDPT}"
: "${SCORING_URI:?Missing SCORING_URI}"
: "${AML_KEY:?Missing AML_KEY}"

echo "[1/2] Health:"
BASE_URI="${SCORING_URI%/score}/"
curl -fsS "$BASE_URI" >/dev/null
echo "OK"

echo "[2/2] Score:"
curl -fsS -X POST "$SCORING_URI" \
  -H "Authorization: Bearer $AML_KEY" \
  -H "Content-Type: application/json" \
  --data @request.json >/dev/null
echo "OK"


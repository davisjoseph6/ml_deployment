#!/usr/bin/env python3
"""
Invoke the Azure ML online endpoint via plain HTTP.

- Reads SCORING_URI and AML_KEY from environment variables.
- Sends a JSON payload (request.json) and prints parsed JSON result.
"""
import json
import os
import sys
import urllib.request


def main() -> int:
    scoring_uri = os.environ.get("SCORING_URI")
    aml_key = os.environ.get("AML_KEY")
    if not scoring_uri or not aml_key:
        print("Missing SCORING_URI or AML_KEY in environment", file=sys.stderr)
        return 2

    if len(sys.argv) != 2:
        print("Usage: scripts/invoke_aml.py request.json", file=sys.stderr)
        return 2

    payload_path = sys.argv[1]
    with open(payload_path, "rb") as f:
        payload_bytes = f.read()

    req = urllib.request.Request(
        scoring_uri,
        data=payload_bytes,
        headers={
            "Authorization": f"Bearer {aml_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    # Your service returns a JSON string containing JSON
    try:
        inner = json.loads(raw)     # turn outer JSON string into inner JSON text
        result = json.loads(inner)  # parse the inner JSON text
    except Exception:
        # fallback: maybe the service starts returning JSON properly later
        result = json.loads(raw)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


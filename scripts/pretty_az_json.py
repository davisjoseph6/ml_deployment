#!/usr/bin/env python3
"""
Pretty-print Azure ML invoke output.

Handles both:
- normal JSON (starts with { or [)
- JSON-string-of-JSON (starts with " ... ") by decoding repeatedly.

Also handles SIGPIPE cleanly (e.g. when piping into `head`), so it won't
print a BrokenPipeError stack trace.
"""
import json
import sys


def main() -> int:
    s = sys.stdin.read().strip()
    if not s:
        return 0

    x = s
    # Unwrap up to a few layers in case of double/triple encoding.
    for _ in range(5):
        try:
            x2 = json.loads(x) if isinstance(x, str) else x
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[pretty_az_json] JSON decode failed: {e}\n")
            sys.stderr.write(
                "[pretty_az_json] Raw output begins with: "
                f"{repr(s[:80])}\n"
            )
            return 2

        x = x2
        if not isinstance(x, str):
            break

    try:
        sys.stdout.write(json.dumps(x, indent=2, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        # Downstream consumer (e.g. `head`) closed early; exit quietly.
        return 0

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)


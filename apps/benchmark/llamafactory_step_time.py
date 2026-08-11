#!/usr/bin/env python3
"""Extract average seconds per completed step from a trainer JSONL log."""

import argparse
import json
from pathlib import Path


def elapsed_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    completed_steps = 0
    with open(args.log, encoding="utf-8") as log:
        for line_number, line in enumerate(log, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                elapsed = elapsed_seconds(record["elapsed_time"])
                steps = int(record["current_steps"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid trainer record at {args.log}:{line_number}"
                ) from error
            if elapsed > args.duration:
                break
            completed_steps = max(completed_steps, steps)

    if completed_steps == 0:
        raise RuntimeError(
            f"No completed training step at or before {args.duration} seconds"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{args.duration / completed_steps:.6f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

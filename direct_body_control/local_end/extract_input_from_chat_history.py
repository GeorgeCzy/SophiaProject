#!/usr/bin/env python3
"""
Extract the latest robot/AI utterance from chat_history.jsonl into input.txt.

The JSONL file may contain the whole conversation. This script keeps only the
newest message whose role is ai, assistant, or robot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_ROLES = {"ai", "assistant", "robot"}


def latest_robot_text(path: Path, roles: set[str] = DEFAULT_ROLES) -> tuple[str, tuple]:
    latest_item: dict[str, Any] | None = None
    latest_line_no = 0

    with path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = str(item.get("role", "")).strip().lower()
            text = str(item.get("text", "")).strip()
            if role in roles and text:
                latest_item = item
                latest_line_no = line_no

    if not latest_item:
        return "", ("empty",)

    text = str(latest_item.get("text", "")).strip()
    signature = (
        latest_item.get("time", ""),
        latest_item.get("sequence", ""),
        latest_line_no,
        text,
    )
    return text, signature


def write_if_changed(output_path: Path, text: str) -> bool:
    try:
        old_text = output_path.read_text(encoding="utf-8").strip()
    except Exception:
        old_text = ""

    if old_text == text.strip():
        return False
    output_path.write_text(text.strip() + "\n", encoding="utf-8")
    return True


def extract_once(history_path: Path, output_path: Path, roles: set[str]) -> tuple[str, tuple, bool]:
    text, signature = latest_robot_text(history_path, roles)
    if not text:
        raise RuntimeError(f"No robot/AI text found in {history_path}")
    changed = write_if_changed(output_path, text)
    return text, signature, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latest AI/robot utterance from chat_history.jsonl to input.txt."
    )
    parser.add_argument("--history", default="chat_history.jsonl", help="Path to chat_history.jsonl.")
    parser.add_argument("--output", default="input.txt", help="Path to write extracted text.")
    parser.add_argument(
        "--roles",
        default="ai,assistant,robot",
        help="Comma-separated roles treated as robot speech.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep watching and update output.")
    parser.add_argument("--interval", type=float, default=0.5, help="Watch polling interval seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = Path(args.history)
    output_path = Path(args.output)
    roles = {role.strip().lower() for role in args.roles.split(",") if role.strip()}

    if not args.watch:
        text, _, changed = extract_once(history_path, output_path, roles)
        status = "updated" if changed else "unchanged"
        print(f"{status}: {output_path}")
        print(text)
        return 0

    print(f"Watching {history_path} -> {output_path}")
    last_signature: tuple | None = None
    while True:
        try:
            text, signature, changed = extract_once(history_path, output_path, roles)
            if signature != last_signature:
                status = "updated" if changed else "unchanged"
                print(f"{status}: {output_path}")
                print(text, flush=True)
                last_signature = signature
        except FileNotFoundError:
            print(f"Waiting for {history_path}", flush=True)
        except Exception as exc:
            print(f"extract failed: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

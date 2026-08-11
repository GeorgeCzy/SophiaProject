#!/usr/bin/env python3
"""
Extract the latest robot/AI utterance from memory_sessions.jsonl or chat_history
into input.txt.

The chat-history file may contain the whole conversation as a JSON array/object
or as JSON-lines. This script keeps only the newest message whose role is ai,
assistant, or robot. JSONL records may be full nested memory-session objects.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_ROLES = {"ai", "assistant", "robot"}


def text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [text_from_content(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in (
            "text",
            "message",
            "content",
            "answer",
            "response",
            "utterance",
            "output_text",
            "final_answer",
            "value",
            "parts",
        ):
            text = text_from_content(value.get(key))
            if text:
                return text
    return ""


def role_from_item(item: dict[str, Any]) -> str:
    for key in ("role", "speaker", "sender", "author", "from", "type"):
        value = item.get(key)
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, dict):
            for nested_key in ("role", "name", "type"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested.strip().lower()
    return ""


def walk_history_json(data: Any):
    if isinstance(data, list):
        for item in data:
            yield from walk_history_json(item)
        return

    if not isinstance(data, dict):
        return

    if role_from_item(data) and text_from_content(data):
        yield data

    known_keys = (
        "messages",
        "history",
        "conversation",
        "conversations",
        "chat_history",
        "items",
        "data",
        "records",
        "turns",
    )
    walked_known = False
    for key in known_keys:
        if key in data:
            walked_known = True
            yield from walk_history_json(data[key])

    if not walked_known:
        for value in data.values():
            if isinstance(value, (dict, list)):
                yield from walk_history_json(value)


def read_history_items(path: Path) -> list[dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        items: list[dict[str, Any]] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                walked = [nested for nested in walk_history_json(item) if isinstance(nested, dict)]
                items.extend(walked or [item])
        return items

    return [item for item in walk_history_json(parsed) if isinstance(item, dict)]


def latest_robot_text(path: Path, roles: set[str] = DEFAULT_ROLES) -> tuple[str, tuple]:
    latest_item: dict[str, Any] | None = None
    latest_index = 0

    for index, item in enumerate(read_history_items(path), start=1):
        role = role_from_item(item)
        text = text_from_content(item)
        if role in roles and text:
            latest_item = item
            latest_index = index

    if not latest_item:
        return "", ("empty",)

    text = text_from_content(latest_item)
    signature = (
        str(path),
        latest_item.get("time", ""),
        latest_item.get("sequence", ""),
        latest_index,
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
        description="Extract latest AI/robot utterance from chat_history.json or .jsonl to input.txt."
    )
    parser.add_argument(
        "--history",
        default="",
        help=(
            "Path to chat-history file. Default: ../memory_supervisor/memory_sessions.jsonl, "
            "falling back to ../chat_history.json and ../chat_history.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Path to write extracted text. Default: ./input.txt beside this script.",
    )
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
    script_dir = Path(__file__).resolve().parent
    if args.history:
        history_path = Path(args.history)
    else:
        memory_path = script_dir.parent / "memory_supervisor" / "memory_sessions.jsonl"
        json_path = script_dir.parent / "chat_history.json"
        jsonl_path = script_dir.parent / "chat_history.jsonl"
        if memory_path.exists():
            history_path = memory_path
        elif json_path.exists():
            history_path = json_path
        else:
            history_path = jsonl_path
    output_path = Path(args.output) if args.output else script_dir / "input.txt"
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

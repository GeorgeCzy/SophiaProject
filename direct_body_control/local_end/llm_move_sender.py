#!/usr/bin/env python3
"""
Read action-duration pairs and send motion_repo keyframes with Sophia_control.

This file keeps the local TCP client unchanged:

    Sophia_control.call_remote(index=<motor_index>, value=<radians>)

Each index is one robot motor and each value is one scalar radian target.
There is no SMPL-X mapping and no axis-angle/vector conversion here.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import Sophia_control
from motion_repo import ALL_JOINTS, MOTIONS, get_motion


MOTOR_INDEX_TO_ACTUATOR: Dict[int, str] = {
    index: actuator for index, actuator in enumerate(ALL_JOINTS)
}
ACTUATOR_TO_MOTOR_INDEX: Dict[str, int] = {
    actuator: index for index, actuator in MOTOR_INDEX_TO_ACTUATOR.items()
}
ALL_INDICES = sorted(MOTOR_INDEX_TO_ACTUATOR)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


DEFAULT_DURATION_SCALE = env_float("SOPHIA_MOTION_DURATION_SCALE", 1.25)


def deg2rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def motion_names(action_name: str) -> List[str]:
    return [name.strip() for name in action_name.split("+") if name.strip()]


def standby_angles_deg() -> Dict[str, float]:
    return {actuator: 0.0 for actuator in MOTOR_INDEX_TO_ACTUATOR.values()}


def get_merged_motion(action_name: str) -> Dict[str, float]:
    names = motion_names(action_name)
    if not names:
        raise KeyError("Empty motion name")

    merged: Dict[str, float] = {}
    for name in names:
        if name == "stay":
            continue
        if name == "standby":
            merged.update(standby_angles_deg())
            continue
        if name not in MOTIONS:
            raise KeyError(f"Unknown motion: {name}. Available: {list(MOTIONS.keys())}")
        for actuator, value in get_motion(name).items():
            if actuator in merged and merged[actuator] != value:
                print(
                    f"[WARN] compound action {action_name!r} overwrites actuator {actuator}",
                    flush=True,
                )
            merged[actuator] = value
    return merged


def motion_to_robot_commands(angles_deg: Dict[str, float]) -> List[Tuple[int, float]]:
    """Convert motion_repo degrees to scalar (motor_index, radians) commands."""
    commands: List[Tuple[int, float]] = []
    ignored: List[str] = []

    for actuator, degrees in angles_deg.items():
        if actuator not in ACTUATOR_TO_MOTOR_INDEX:
            ignored.append(actuator)
            continue
        commands.append((ACTUATOR_TO_MOTOR_INDEX[actuator], deg2rad(degrees)))

    if ignored:
        print(f"[WARN] ignored unsupported actuators: {sorted(ignored)}", flush=True)
    return sorted(commands, key=lambda item: item[0])


def reset_commands() -> List[Tuple[int, float]]:
    return [(index, 0.0) for index in ALL_INDICES]


def parse_action_pairs(text: str) -> List[Tuple[str, float]]:
    """
    Parse action pairs from text. Supports:
    - JSON: [["thumbup", 2.0], ["wave", 1.5]]
    - Line-based: "thumbup 2.0" or "leftA+rightB 0.8" per line
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty input for action pairs.")

    try:
        data = json.loads(text)
        if isinstance(data, list):
            pairs = []
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    name = str(item[0]).strip()
                    duration = float(item[1])
                elif isinstance(item, dict):
                    name = str(item.get("action") or item.get("name") or item.get("motion") or "").strip()
                    duration = float(item.get("duration") or item.get("time") or item.get("seconds"))
                else:
                    raise ValueError(f"Invalid action pair: {item!r}")
                if duration < 0:
                    raise ValueError(f"Duration must be >= 0: {item!r}")
                pairs.append((name, duration))
            return _validate_pairs(pairs)
    except json.JSONDecodeError:
        pass

    action_token = r"[A-Za-z][A-Za-z0-9_]*"
    action_combo = rf"{action_token}(?:\+{action_token})*"
    line_pattern = re.compile(
        rf'^(?:[-*]|\d+[.)])?\s*"?({action_combo})"?\s*[:,]?\s*"?([0-9]+(?:\.[0-9]+)?)"?\s*(?:s|sec|seconds)?$'
    )

    pairs = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_pattern.match(line)
        if not match:
            raise ValueError(f"Invalid action line (expected 'name duration'): {line}")
        pairs.append((match.group(1), float(match.group(2))))

    return _validate_pairs(pairs)


def _validate_pairs(pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    pairs = [(name, duration) for name, duration in pairs if name]
    if not pairs:
        raise ValueError("No action pairs found.")
    for name, duration in pairs:
        if duration < 0:
            raise ValueError(f"Duration must be >= 0: {name} {duration}")
    return pairs


def send_commands(
    commands: List[Tuple[int, float]],
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
) -> None:
    for index, value in commands:
        actuator = MOTOR_INDEX_TO_ACTUATOR.get(index, "<unknown>")
        if dry_run:
            print(f"  [DRY] index={index} actuator={actuator} value_rad={value:.6f}", flush=True)
        else:
            Sophia_control.call_remote(index=index, value=value, host=host, port=port, timeout=timeout)


def run_actions(
    pairs: List[Tuple[str, float]],
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
    reset_first: bool = True,
    reset_last: bool = False,
    duration_scale: float = 1.0,
) -> None:
    if duration_scale <= 0:
        raise ValueError("duration_scale must be > 0")

    if reset_first:
        print("[INIT] reset all controlled motors to zero", flush=True)
        send_commands(reset_commands(), host, port, timeout, dry_run)
        time.sleep(0.1)

    for order, (action_name, duration_s) in enumerate(pairs, start=1):
        hold_s = duration_s * duration_scale
        if duration_scale == 1.0:
            print(f"[{order}/{len(pairs)}] {action_name}: hold {duration_s:.3f}s", flush=True)
        else:
            print(
                f"[{order}/{len(pairs)}] {action_name}: hold {duration_s:.3f}s "
                f"-> {hold_s:.3f}s",
                flush=True,
            )
        if action_name == "stay":
            commands = []
        elif action_name == "standby":
            commands = reset_commands()
        else:
            commands = motion_to_robot_commands(get_merged_motion(action_name))

        send_commands(commands, host, port, timeout, dry_run)
        if hold_s > 0:
            time.sleep(hold_s)

    if reset_last and (not pairs or pairs[-1][0] != "standby"):
        print("[DONE] reset all controlled motors to zero", flush=True)
        send_commands(reset_commands(), host, port, timeout, dry_run)


def read_text(input_file: str) -> str:
    if input_file:
        with open(input_file, "r", encoding="utf-8") as file:
            return file.read()
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send motion_repo keyframes as scalar motor-index commands via Sophia_control.py."
    )
    parser.add_argument("--host", default="10.0.0.10", help="Robot TCP host.")
    parser.add_argument("--port", type=int, default=5005, help="Robot TCP port.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds.")
    parser.add_argument("--input-file", default="", help="Action text file. If omitted, read stdin.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print without sending.")
    parser.add_argument("--no-reset-first", action="store_true", help="Do not reset before the sequence.")
    parser.add_argument("--reset-last", action="store_true", help="Force reset after the sequence.")
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=DEFAULT_DURATION_SCALE,
        help="Multiply every action hold duration. Default: SOPHIA_MOTION_DURATION_SCALE or 1.25.",
    )
    args = parser.parse_args()

    pairs = parse_action_pairs(read_text(args.input_file))
    print(f"Actions: {pairs}", flush=True)
    run_actions(
        pairs=pairs,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        dry_run=args.dry_run,
        reset_first=not args.no_reset_first,
        reset_last=args.reset_last,
        duration_scale=args.duration_scale,
    )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

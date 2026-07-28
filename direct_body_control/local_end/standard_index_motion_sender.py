#!/usr/bin/env python3
"""
Send motion_repo keyframes to bodycontrol_tcp_standard.py without SMPL-X math.

This keeps the original robot-end protocol:

    {"index": <int>, "value": [x, y, z]}

but fills [x, y, z] directly with actuator target radians in the exact slots
that bodycontrol_tcp_standard.py extracts. There is no axis-angle conversion,
no sign coupling, and no SMPL/SMPL-X interpretation.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

from direct_robot_protocol import (
    filter_supported_angles_deg,
    parse_action_pairs,
    standby_angles_deg,
)
from motion_repo import MOTIONS, get_motion


# Must match bodycontrol_tcp_standard.py INDEX_MAP exactly.
ACTUATOR_TO_INDEX_SLOT: dict[str, tuple[int, int]] = {
    "LeftShoulderPitch": (16, 0),
    "LeftShoulderRoll": (16, 2),
    "RightShoulderPitch": (17, 0),
    "RightShoulderRoll": (17, 2),
    "LeftShoulderYaw": (18, 0),
    "LeftElbowPitch": (18, 1),
    "RightShoulderYaw": (19, 0),
    "RightElbowPitch": (19, 1),
    "LeftElbowYaw": (20, 0),
    "RightElbowYaw": (21, 0),
    "LeftIndexFinger": (25, 2),
    "LeftMiddleFinger": (28, 2),
    "LeftPinkyFinger": (31, 2),
    "LeftRingFinger": (34, 2),
    "LeftThumbRoll": (37, 0),
    "LeftThumbFinger": (37, 2),
    "RightIndexFinger": (40, 2),
    "RightMiddleFinger": (43, 2),
    "RightPinkyFinger": (46, 2),
    "RightRingFinger": (49, 2),
    "RightThumbRoll": (52, 0),
    "RightThumbFinger": (52, 2),
    "NeckRotation": (60, 1),
}

RESET_INDICES = (16, 17, 18, 19, 20, 21, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 60)


def deg_to_rad(value: float) -> float:
    return value * 3.141592653589793 / 180.0


def motion_names(action_name: str) -> list[str]:
    return [name.strip() for name in action_name.split("+") if name.strip()]


def get_merged_motion(action_name: str) -> dict[str, float]:
    names = motion_names(action_name)
    if not names:
        raise KeyError("Empty motion name")
    if names == ["standby"]:
        return standby_angles_deg()

    merged: dict[str, float] = {}
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


def angles_to_index_commands(
    angles_deg: dict[str, float],
    *,
    strict_actuators: bool = False,
) -> list[tuple[int, list[float]]]:
    supported, ignored = filter_supported_angles_deg(angles_deg, strict=strict_actuators)
    unsupported_by_standard = {
        name: value
        for name, value in supported.items()
        if name not in ACTUATOR_TO_INDEX_SLOT
    }
    for name in unsupported_by_standard:
        supported.pop(name, None)
    ignored.update(unsupported_by_standard)
    if ignored:
        print(f"[WARN] ignored unsupported actuators: {sorted(ignored)}", flush=True)

    index_values: dict[int, list[float]] = {}
    for actuator, value_deg in supported.items():
        index, slot = ACTUATOR_TO_INDEX_SLOT[actuator]
        if index not in index_values:
            index_values[index] = [0.0, 0.0, 0.0]
        index_values[index][slot] = deg_to_rad(value_deg)

    return [(index, index_values[index]) for index in sorted(index_values)]


def call_standard_remote(
    index: int,
    value: list[float],
    *,
    host: str,
    port: int,
    timeout: float,
) -> dict:
    req = {"index": index, "value": value}
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(json.dumps(req).encode("utf-8"))
        data = sock.recv(4096).decode("utf-8")
    resp = json.loads(data)
    if not isinstance(resp, dict):
        raise TypeError(f"Response is not a JSON object: {resp!r}")
    if resp.get("code") != 0:
        raise RuntimeError(f"Server error: {resp.get('error')}")
    return resp["result"]


def send_index_commands(
    commands: list[tuple[int, list[float]]],
    *,
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
) -> None:
    for index, value in commands:
        if dry_run:
            print(f"  [DRY][INDEX] index={index} value={value}", flush=True)
            continue
        result = call_standard_remote(index, value, host=host, port=port, timeout=timeout)
        print(f"  [INDEX] sent {result}", flush=True)


def reset_commands() -> list[tuple[int, list[float]]]:
    return [(index, [0.0, 0.0, 0.0]) for index in RESET_INDICES]


def run_actions(
    pairs: list[tuple[str, float]],
    *,
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
    reset_first: bool,
    reset_last: bool,
    strict_actuators: bool,
) -> None:
    if reset_first:
        print("[INIT] standard-index reset", flush=True)
        send_index_commands(
            reset_commands(),
            host=host,
            port=port,
            timeout=timeout,
            dry_run=dry_run,
        )
        time.sleep(0.1)

    for order, (action_name, duration_s) in enumerate(pairs, start=1):
        print(f"[{order}/{len(pairs)}] {action_name}: hold {duration_s:.3f}s", flush=True)
        if action_name == "standby":
            commands = reset_commands()
            send_index_commands(
                commands,
                host=host,
                port=port,
                timeout=timeout,
                dry_run=dry_run,
            )
        elif action_name != "stay":
            angles = get_merged_motion(action_name)
            commands = angles_to_index_commands(angles, strict_actuators=strict_actuators)
            send_index_commands(
                commands,
                host=host,
                port=port,
                timeout=timeout,
                dry_run=dry_run,
            )
        if duration_s > 0:
            time.sleep(duration_s)

    if reset_last and (not pairs or pairs[-1][0] != "standby"):
        print("[DONE] standard-index reset", flush=True)
        send_index_commands(
            reset_commands(),
            host=host,
            port=port,
            timeout=timeout,
            dry_run=dry_run,
        )


def read_text(input_file: str) -> str:
    if input_file:
        with open(input_file, "r", encoding="utf-8") as file:
            return file.read()
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send motion_repo keyframes to bodycontrol_tcp_standard.py without axis-angle conversion."
    )
    parser.add_argument("--host", default="10.0.0.10", help="Robot TCP host.")
    parser.add_argument("--port", type=int, default=5005, help="bodycontrol_tcp_standard.py TCP port.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds.")
    parser.add_argument("--input-file", default="", help="Action text file. If omitted, read stdin.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print without sending.")
    parser.add_argument("--no-reset-first", action="store_true", help="Do not reset before the sequence.")
    parser.add_argument("--reset-last", action="store_true", help="Force reset after the sequence.")
    parser.add_argument(
        "--strict-actuators",
        action="store_true",
        help="Fail on unsupported actuators instead of warning and skipping them.",
    )
    args = parser.parse_args()

    pairs = parse_action_pairs(read_text(args.input_file))
    print(f"Actions: {pairs}", flush=True)
    run_actions(
        pairs,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        dry_run=args.dry_run,
        reset_first=not args.no_reset_first,
        reset_last=args.reset_last,
        strict_actuators=args.strict_actuators,
    )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

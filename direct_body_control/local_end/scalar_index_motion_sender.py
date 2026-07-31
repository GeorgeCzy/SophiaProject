#!/usr/bin/env python3
"""
Send motion_repo keyframes through the scalar-index robot protocol.

The local command shape is one scalar per motor:

    {"commands": [{"index": 0, "value": -1.22}], "unit": "rad"}

Each keyframe is sent as one batch request, so multiple actuators can start at
the same time on the robot end.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

from direct_robot_protocol import parse_action_pairs, standby_angles_deg
from motion_repo import MOTIONS, get_motion
from scalar_index_protocol import (
    angles_deg_to_scalar_commands,
    load_motor_index_map,
    scalar_payload,
    standby_scalar_commands,
)


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


def call_scalar_remote(
    commands: list[tuple[int, float]],
    *,
    host: str,
    port: int,
    timeout: float,
) -> dict:
    request = scalar_payload(commands)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(json.dumps(request).encode("utf-8"))
        data = sock.recv(65536).decode("utf-8")

    response = json.loads(data)
    if not isinstance(response, dict):
        raise TypeError(f"Response is not a JSON object: {response!r}")
    if response.get("code") != 0:
        raise RuntimeError(f"Server error: {response.get('error')}")
    return response["result"]


def send_scalar_commands(
    commands: list[tuple[int, float]],
    *,
    motor_map: dict[int, str],
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
) -> None:
    if not commands:
        print("[WARN] no scalar commands to send", flush=True)
        return

    if dry_run:
        print("  [DRY][SCALAR]", flush=True)
        for index, value in commands:
            name = motor_map.get(index, "<unknown>")
            print(f"    index={index} actuator={name} value_rad={value:.6f}", flush=True)
        return

    result = call_scalar_remote(commands, host=host, port=port, timeout=timeout)
    print(f"  [SCALAR] sent {result}", flush=True)


def motion_to_scalar_commands(
    angles_deg: dict[str, float],
    *,
    motor_map: dict[int, str],
    strict_actuators: bool,
) -> list[tuple[int, float]]:
    commands, ignored = angles_deg_to_scalar_commands(
        angles_deg,
        motor_map,
        strict_actuators=strict_actuators,
    )
    if ignored:
        print(f"[WARN] ignored unsupported actuators: {sorted(ignored)}", flush=True)
    return commands


def run_actions(
    pairs: list[tuple[str, float]],
    *,
    motor_map: dict[int, str],
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
    reset_first: bool,
    reset_last: bool,
    strict_actuators: bool,
) -> None:
    if reset_first:
        print("[INIT] scalar-index reset", flush=True)
        send_scalar_commands(
            standby_scalar_commands(motor_map),
            motor_map=motor_map,
            host=host,
            port=port,
            timeout=timeout,
            dry_run=dry_run,
        )
        time.sleep(0.1)

    for order, (action_name, duration_s) in enumerate(pairs, start=1):
        print(f"[{order}/{len(pairs)}] {action_name}: hold {duration_s:.3f}s", flush=True)
        if action_name == "stay":
            commands = []
        elif action_name == "standby":
            commands = standby_scalar_commands(motor_map)
        else:
            angles = get_merged_motion(action_name)
            commands = motion_to_scalar_commands(
                angles,
                motor_map=motor_map,
                strict_actuators=strict_actuators,
            )

        if commands:
            send_scalar_commands(
                commands,
                motor_map=motor_map,
                host=host,
                port=port,
                timeout=timeout,
                dry_run=dry_run,
            )
        if duration_s > 0:
            time.sleep(duration_s)

    if reset_last and (not pairs or pairs[-1][0] != "standby"):
        print("[DONE] scalar-index reset", flush=True)
        send_scalar_commands(
            standby_scalar_commands(motor_map),
            motor_map=motor_map,
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
        description="Send motion_repo keyframes to the scalar-index body bridge."
    )
    parser.add_argument("--host", default="10.0.0.10", help="Robot TCP host.")
    parser.add_argument("--port", type=int, default=5007, help="Scalar-index robot TCP port.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds.")
    parser.add_argument("--input-file", default="", help="Action text file. If omitted, read stdin.")
    parser.add_argument("--motor-map", default="", help="Optional JSON motor map shared with robot end.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print without sending.")
    parser.add_argument("--no-reset-first", action="store_true", help="Do not reset before the sequence.")
    parser.add_argument("--reset-last", action="store_true", help="Force reset after the sequence.")
    parser.add_argument(
        "--strict-actuators",
        action="store_true",
        help="Fail on unsupported actuators instead of warning and skipping them.",
    )
    args = parser.parse_args()

    motor_map = load_motor_index_map(args.motor_map or None)
    pairs = parse_action_pairs(read_text(args.input_file))
    print(f"Actions: {pairs}", flush=True)
    run_actions(
        pairs,
        motor_map=motor_map,
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

#!/usr/bin/env python3
"""
Read action-duration pairs and send motion_repo keyframes directly to robot actuators.

This is the simplified local-end replacement for llm_move_sender.py. It sends
actuator names and angles directly, so there is no SMPL-X index or axis-angle
mapping in the path.
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


def motion_names(action_name: str) -> list[str]:
    return [name.strip() for name in action_name.split("+") if name.strip()]


def get_direct_motion(action_name: str) -> dict[str, float]:
    """Return one merged actuator pose for a single keyframe or A+B compound."""
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


def call_direct_remote(
    angles_deg: dict[str, float],
    *,
    host: str,
    port: int,
    timeout: float,
) -> dict:
    req = {"actuators": angles_deg, "unit": "deg"}
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(json.dumps(req).encode("utf-8"))
        data = sock.recv(65536).decode("utf-8")
    resp = json.loads(data)
    if not isinstance(resp, dict):
        raise TypeError(f"Response is not a JSON object: {resp!r}")
    if resp.get("code") != 0:
        raise RuntimeError(f"Server error: {resp.get('error')}")
    return resp["result"]


def send_angles(
    angles_deg: dict[str, float],
    *,
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
    strict_actuators: bool,
) -> None:
    supported, ignored = filter_supported_angles_deg(angles_deg, strict=strict_actuators)
    if ignored:
        print(f"[WARN] ignored unsupported actuators: {sorted(ignored)}", flush=True)
    if not supported:
        print("[WARN] no supported actuators to send", flush=True)
        return
    if dry_run:
        print(f"  [DRY][DIRECT] {supported}", flush=True)
        return
    result = call_direct_remote(supported, host=host, port=port, timeout=timeout)
    print(f"  [DIRECT] sent {result}", flush=True)


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
        print("[INIT] direct standby reset", flush=True)
        send_angles(
            standby_angles_deg(),
            host=host,
            port=port,
            timeout=timeout,
            dry_run=dry_run,
            strict_actuators=strict_actuators,
        )
        time.sleep(0.1)

    for index, (action_name, duration_s) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {action_name}: hold {duration_s:.3f}s", flush=True)
        if action_name == "stay":
            angles = {}
        else:
            angles = get_direct_motion(action_name)

        if angles:
            send_angles(
                angles,
                host=host,
                port=port,
                timeout=timeout,
                dry_run=dry_run,
                strict_actuators=strict_actuators,
            )
        if duration_s > 0:
            time.sleep(duration_s)

    if reset_last and (not pairs or pairs[-1][0] != "standby"):
        print("[DONE] direct standby reset", flush=True)
        send_angles(
            standby_angles_deg(),
            host=host,
            port=port,
            timeout=timeout,
            dry_run=dry_run,
            strict_actuators=strict_actuators,
        )


def read_text(input_file: str) -> str:
    if input_file:
        with open(input_file, "r", encoding="utf-8") as file:
            return file.read()
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send motion_repo keyframes directly to robot actuators."
    )
    parser.add_argument("--host", default="10.0.0.10", help="Robot TCP host.")
    parser.add_argument("--port", type=int, default=5006, help="Direct robot TCP port.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds.")
    parser.add_argument("--input-file", default="", help="Action text file. If omitted, read stdin.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print without sending.")
    parser.add_argument(
        "--no-reset-first",
        action="store_true",
        help="Do not send standby before the sequence.",
    )
    parser.add_argument(
        "--reset-last",
        action="store_true",
        help="Force standby after the sequence even if it does not end with standby.",
    )
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

#!/usr/bin/env python3
"""
Read action pairs ["action name", "time"] from text and send to robot via Sophia_control.
Uses motion_repo for preset poses and Sophia_control.call_remote (same as smpl_visualizer).
"""

import argparse
import json
import math
import re
import sys
import time
from typing import Dict, List, Tuple

import Sophia_control

try:
    from motion_repo import get_motion, MOTIONS
except ImportError:
    get_motion = None
    MOTIONS = {}


def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def to_axisangle(val, index: int):
    """Convert motion_repo value to axis-angle [x,y,z] for Sophia_control. Same logic as smpl_visualizer."""
    if isinstance(val, (int, float)):
        v = float(val)
        if index in (25, 26, 27, 28, 29, 30):
            return [0.0, 0.0, v]
        if index in (31, 32, 33):
            return [v, 0.0, v]
        if index in (34, 35, 36):
            return [v * 0.3, 0.0, v]
        if index in (20, 21):
            return [v, 0.0, 0.0]
        if index in (38, 39):
            return [v, 0.2 * v, -v]
        if index in (40, 41, 42, 43, 44, 45):
            return [0.0, 0.0, -v]
        if index in (46, 47, 48):
            return [v, 0.0, -v]
        if index in (49, 50, 51):
            return [v * 0.3, 0.0, -v]
        if index in (53, 54):
            return [v, 0.2 * v, -v]
        return [0.0, v, 0.0] # neckRotation falls for this case

    if len(val) == 2:
        x, y = float(val[0]), float(val[1])
        if index == 16:
            return [x, 0.2 * x, y]
        if index == 17:
            return [x, -0.2 * x, y]
        if index == 18:
            return [0.1 * x, y, -x]
        if index == 19:
            return [x, y, x]
        if index == 37:
            return [x, y, y]
        if index == 52:
            return [x, y, y]
    return list(val)


# Mapping: motion_repo actuator name -> (SMPL index, component in composite index)
# For composite indices (16,17,18,19,37,52), we collect all components and build value together.
ACTUATOR_TO_INDEX = {
    "LeftShoulderPitch": 16,
    "LeftShoulderRoll": 16,
    "RightShoulderPitch": 17,
    "RightShoulderRoll": 17,
    "LeftShoulderYaw": 18,
    "LeftElbowPitch": 18,
    "RightShoulderYaw": 19,
    "RightElbowPitch": 19,
    "LeftElbowYaw": 20,
    "RightElbowYaw": 21,
    "LeftIndexFinger": 25,
    "LeftMiddleFinger": 28,
    "LeftPinkyFinger": 31,
    "LeftRingFinger": 34,
    "LeftThumbRoll": 37,
    "LeftThumbFinger": 37,
    "RightIndexFinger": 40,
    "RightMiddleFinger": 43,
    "RightPinkyFinger": 46,
    "RightRingFinger": 49,
    "RightThumbRoll": 52,
    "RightThumbFinger": 52,
    "NeckRotation": 60,
}

# Composite indices: (index, order) -> actuator name for building (v1, v2) tuple
COMPOSITE_INDEX_PARTS = {
    16: ("LeftShoulderPitch", "LeftShoulderRoll"),
    17: ("RightShoulderPitch", "RightShoulderRoll"),
    18: ("LeftShoulderYaw", "LeftElbowPitch"),
    19: ("RightShoulderYaw", "RightElbowPitch"),
    37: ("LeftThumbRoll", "LeftThumbFinger"),
    52: ("RightThumbRoll", "RightThumbFinger"),
}

# All robot indices we send (deterministic order). Unspecified joints default to 0.
ALL_INDICES = [16, 17, 18, 19, 20, 21, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 60]

# A-pose startup bias. Reset to this pose after motions complete. Remove/comment entries for motor=0.
def deg(d: float) -> float:
    return d * math.pi / 180.0


A_POSE_OFFSET: Dict[str, float] = {
    # "RightShoulderRoll": deg(45.0),
    # "RightElbowPitch": deg(-127.0),
    # "LeftShoulderRoll": deg(-45.0),
    # "LeftElbowPitch": deg(127.0),
}


def motion_to_robot_commands(angles_deg: dict) -> List[Tuple[int, list]]:
    """Convert motion_repo angles (degrees) to (index, value) for Sophia_control.call_remote.
    Unspecified joints are set to 0 (per motion_repo spec: 'all other motors to 0').
    """
    index_values: dict = {}
    for idx in ALL_INDICES:
        if idx in COMPOSITE_INDEX_PARTS:
            index_values[idx] = [0.0, 0.0]
        else:
            index_values[idx] = 0.0

    for actuator, deg in angles_deg.items():
        if actuator not in ACTUATOR_TO_INDEX:
            continue
        idx = ACTUATOR_TO_INDEX[actuator]
        rad = deg2rad(deg)

        if idx in COMPOSITE_INDEX_PARTS:
            parts = COMPOSITE_INDEX_PARTS[idx]
            arr = index_values[idx]
            if actuator == parts[0]:
                arr[0] = rad
            elif actuator == parts[1]:
                arr[1] = rad
        else:
            index_values[idx] = rad

    commands = []
    for idx in ALL_INDICES:
        val = index_values[idx]
        if isinstance(val, list):
            value = to_axisangle(tuple(val), idx)
        else:
            value = to_axisangle(val, idx)
        commands.append((idx, value))
    return commands



def parse_action_pairs(text: str) -> List[Tuple[str, float]]:
    """
    Parse action pairs from text. Supports:
    - JSON: [["thumbup", 2.0], ["wave", 1.5]]
    - Line-based: "thumbup 2.0" or "thumbup\t2.0" per line
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
                    dur = float(item[1])
                    if dur < 0:
                        raise ValueError(f"Duration must be >= 0: {item}")
                    pairs.append((name, dur))
                else:
                    raise ValueError(f"Invalid action pair: {item}")
            if pairs:
                return pairs
    except json.JSONDecodeError:
        pass

    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\S+)\s+([\d.]+)", line)
        if m:
            name = m.group(1).strip()
            dur = float(m.group(2))
            if dur < 0:
                raise ValueError(f"Duration must be >= 0: {line}")
            pairs.append((name, dur))
        else:
            raise ValueError(f"Invalid action line (expected 'name duration'): {line}")

    if not pairs:
        raise ValueError("No action pairs found.")
    return pairs


def run_actions(
    pairs: List[Tuple[str, float]],
    host: str,
    port: int,
    timeout: float,
    dry_run: bool,
) -> None:
    """Execute action presets: send to robot and hold for duration."""
    if not get_motion:
        raise RuntimeError("motion_repo not available. Ensure motion_repo.py is in the same directory.")

    for i, (action_name, duration_s) in enumerate(pairs, start=1):
        if action_name not in MOTIONS:
            raise KeyError(f"Unknown motion: {action_name}. Available: {list(MOTIONS.keys())}")

        angles_deg = get_motion(action_name)
        commands = motion_to_robot_commands(angles_deg)

        print(f"[{i}/{len(pairs)}] {action_name}: hold {duration_s:.3f}s")

        for idx, value in commands:
            if dry_run:
                print(f"  [DRY] index={idx} value={value}")
            else:
                Sophia_control.call_remote(index=idx, value=value, host=host, port=port, timeout=timeout)

        if duration_s > 0:
            time.sleep(duration_s)


def read_text(input_file: str) -> str:
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read action pairs from text and send to robot via Sophia_control."
    )
    parser.add_argument("--host", default="10.0.0.10", help="Robot TCP host (default: 10.0.0.10).")
    parser.add_argument("--port", type=int, default=5005, help="Robot TCP port.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Socket timeout seconds.")
    parser.add_argument(
        "--input-file",
        default="",
        help="Path to text file. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print without sending to robot.",
    )
    args = parser.parse_args()

    text = read_text(args.input_file)
    pairs = parse_action_pairs(text)
    print(f"Actions: {pairs}")

    run_actions(
        pairs=pairs,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

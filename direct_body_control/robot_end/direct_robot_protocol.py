from __future__ import annotations

import json
import math
import re
from typing import Any


DIRECT_ACTUATOR_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "RightShoulderPitch": (-145.0, 35.0),
    "RightShoulderRoll": (-4.0, 101.0),
    "RightShoulderYaw": (-66.0, 83.0),
    "RightElbowPitch": (-127.0, 119.0),
    "RightElbowYaw": (-123.0, 123.0),
    "RightWristPitch": (-35.0, 35.0),
    "RightWristRoll": (-35.0, 35.0),
    "RightThumbRoll": (-31.0, 22.0),
    "RightThumbFinger": (-75.0, 44.0),
    "RightIndexFinger": (-123.0, 18.0),
    "RightMiddleFinger": (-18.0, 132.0),
    "RightRingFinger": (-18.0, 136.0),
    "RightPinkyFinger": (-75.0, 4.0),
    "LeftShoulderPitch": (-35.0, 145.0),
    "LeftShoulderRoll": (-101.0, 4.0),
    "LeftShoulderYaw": (-83.0, 66.0),
    "LeftElbowPitch": (-119.0, 127.0),
    "LeftElbowYaw": (-123.0, 123.0),
    "LeftWristPitch": (-35.0, 35.0),
    "LeftWristRoll": (-35.0, 35.0),
    "LeftThumbRoll": (-31.0, 22.0),
    "LeftThumbFinger": (-44.0, 75.0),
    "LeftIndexFinger": (-18.0, 123.0),
    "LeftMiddleFinger": (-18.0, 132.0),
    "LeftRingFinger": (-18.0, 136.0),
    "LeftPinkyFinger": (-4.0, 75.0),
    "NeckRotation": (-40.0, 40.0),
}

DIRECT_ACTUATORS = tuple(DIRECT_ACTUATOR_LIMITS_DEG)


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def rad_to_deg(value: float) -> float:
    return value * 180.0 / math.pi


def clamp_deg(actuator: str, value: float) -> float:
    lo, hi = DIRECT_ACTUATOR_LIMITS_DEG[actuator]
    return max(lo, min(hi, value))


def clamp_rad(actuator: str, value: float) -> float:
    return deg_to_rad(clamp_deg(actuator, rad_to_deg(value)))


def filter_supported_angles_deg(
    angles_deg: dict[str, float],
    *,
    strict: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    supported: dict[str, float] = {}
    ignored: dict[str, float] = {}
    for name, raw_value in angles_deg.items():
        value = float(raw_value)
        if name not in DIRECT_ACTUATOR_LIMITS_DEG:
            ignored[name] = value
            continue
        supported[name] = clamp_deg(name, value)
    if strict and ignored:
        names = ", ".join(sorted(ignored))
        raise KeyError(f"Unsupported direct actuator(s): {names}")
    return supported, ignored


def standby_angles_deg() -> dict[str, float]:
    return {name: 0.0 for name in DIRECT_ACTUATORS}


def parse_action_pairs(text: str) -> list[tuple[str, float]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty input for action pairs.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        pairs: list[tuple[str, float]] = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]).strip(), _positive_duration(item[1])))
            elif isinstance(item, dict):
                name = str(item.get("action") or item.get("name") or item.get("motion") or "").strip()
                duration = _positive_duration(item.get("duration") or item.get("time") or item.get("seconds"))
                pairs.append((name, duration))
            else:
                raise ValueError(f"Invalid action pair: {item!r}")
        return _validate_pairs(pairs)

    action_token = r"[A-Za-z][A-Za-z0-9_]*"
    action_combo = rf"{action_token}(?:\+{action_token})*"
    line_pattern = re.compile(
        rf'^(?:[-*]|\d+[.)])?\s*"?({action_combo})"?\s*[:,]?\s*"?([0-9]+(?:\.[0-9]+)?)"?\s*(?:s|sec|seconds)?$'
    )
    pairs = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_pattern.match(line)
        if not match:
            raise ValueError(f"Invalid action line (expected 'name duration'): {line}")
        pairs.append((match.group(1), _positive_duration(match.group(2))))
    return _validate_pairs(pairs)


def _positive_duration(value: Any) -> float:
    duration = float(value)
    if duration < 0:
        raise ValueError(f"Duration must be >= 0: {value!r}")
    return duration


def _validate_pairs(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    pairs = [(name, duration) for name, duration in pairs if name]
    if not pairs:
        raise ValueError("No action pairs found.")
    return pairs

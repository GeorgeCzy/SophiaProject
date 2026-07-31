from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from direct_robot_protocol import (
    DIRECT_ACTUATORS,
    clamp_rad,
    deg_to_rad,
    filter_supported_angles_deg,
)


# Default scalar motor table used by the new TCP protocol.
#
# The protocol meaning is intentionally simple:
#     one integer index -> one robot actuator -> one scalar radian value
#
# If the robot already has an official motor index table, put that table in a
# JSON file and pass it with --motor-map on both local and robot ends.
DEFAULT_MOTOR_INDEX_TO_ACTUATOR: dict[int, str] = {
    index: actuator for index, actuator in enumerate(DIRECT_ACTUATORS)
}


def load_motor_index_map(path: str | None = None) -> dict[int, str]:
    if not path:
        return dict(DEFAULT_MOTOR_INDEX_TO_ACTUATOR)

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        mapping = {index: str(actuator) for index, actuator in enumerate(data)}
    elif isinstance(data, dict):
        mapping = {int(index): str(actuator) for index, actuator in data.items()}
    else:
        raise ValueError("motor map must be a JSON object or list")

    validate_motor_index_map(mapping)
    return mapping


def validate_motor_index_map(mapping: dict[int, str]) -> None:
    if not mapping:
        raise ValueError("motor index map is empty")

    unknown = sorted({name for name in mapping.values() if name not in DIRECT_ACTUATORS})
    if unknown:
        raise ValueError(f"motor map contains unknown actuators: {unknown}")

    duplicate_actuators = sorted(
        {
            name
            for name in mapping.values()
            if list(mapping.values()).count(name) > 1
        }
    )
    if duplicate_actuators:
        raise ValueError(f"motor map contains duplicate actuators: {duplicate_actuators}")


def actuator_to_motor_index(mapping: dict[int, str]) -> dict[str, int]:
    return {actuator: index for index, actuator in mapping.items()}


def normalize_unit(unit: Any, *, default: str = "rad") -> str:
    raw = str(default if unit is None else unit).strip().lower()
    if raw in {"rad", "radian", "radians"}:
        return "rad"
    if raw in {"deg", "degree", "degrees"}:
        return "deg"
    raise ValueError("unit must be 'rad' or 'deg'")


def value_to_radians(actuator: str, value: Any, unit: str) -> float:
    scalar = float(value)
    radians = deg_to_rad(scalar) if unit == "deg" else scalar
    return clamp_rad(actuator, radians)


def request_to_scalar_commands(
    request: Any,
    motor_map: dict[int, str],
) -> list[tuple[int, float]]:
    """Parse a TCP request into [(motor_index, target_radians), ...]."""
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    if request.get("command") == "reset":
        return [(index, 0.0) for index in sorted(motor_map)]

    default_unit = normalize_unit(request.get("unit"), default="rad")
    commands: list[tuple[int, float]] = []

    if isinstance(request.get("commands"), list):
        for item in request["commands"]:
            commands.append(_parse_command_item(item, default_unit, motor_map))
    elif isinstance(request.get("indices"), list) and isinstance(request.get("values"), list):
        indices = request["indices"]
        values = request["values"]
        if len(indices) != len(values):
            raise ValueError("indices and values must have the same length")
        for index, value in zip(indices, values):
            commands.append(_parse_index_value(index, value, default_unit, motor_map))
    elif "index" in request and "value" in request:
        commands.append(_parse_index_value(request["index"], request["value"], default_unit, motor_map))
    elif isinstance(request.get("actuators"), dict):
        commands.extend(_parse_actuator_values(request["actuators"], request.get("unit"), motor_map))
    else:
        raise ValueError(
            "request must include index/value, commands, indices/values, actuators, or command='reset'"
        )

    if not commands:
        raise ValueError("no scalar motor commands in request")
    return commands


def _parse_command_item(
    item: Any,
    default_unit: str,
    motor_map: dict[int, str],
) -> tuple[int, float]:
    if isinstance(item, dict):
        if "index" not in item:
            raise ValueError(f"command missing index: {item!r}")
        unit = normalize_unit(item.get("unit"), default=default_unit)
        if "value" in item:
            return _parse_index_value(item["index"], item["value"], unit, motor_map)
        if "rad" in item:
            return _parse_index_value(item["index"], item["rad"], "rad", motor_map)
        if "radians" in item:
            return _parse_index_value(item["index"], item["radians"], "rad", motor_map)
        if "deg" in item:
            return _parse_index_value(item["index"], item["deg"], "deg", motor_map)
        if "degrees" in item:
            return _parse_index_value(item["index"], item["degrees"], "deg", motor_map)
        raise ValueError(f"command missing scalar value: {item!r}")

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return _parse_index_value(item[0], item[1], default_unit, motor_map)

    raise ValueError(f"invalid scalar command item: {item!r}")


def _parse_index_value(
    raw_index: Any,
    raw_value: Any,
    unit: str,
    motor_map: dict[int, str],
) -> tuple[int, float]:
    index = int(raw_index)
    if index not in motor_map:
        raise ValueError(f"index {index} is not in the motor map")
    actuator = motor_map[index]
    return index, value_to_radians(actuator, raw_value, unit)


def _parse_actuator_values(
    actuators: dict[str, Any],
    unit: Any,
    motor_map: dict[int, str],
) -> list[tuple[int, float]]:
    # This compatibility form mirrors bodycontrol_tcp_direct.py. Actuator-name
    # requests usually come from motion_repo, where values are degrees.
    resolved_unit = normalize_unit(unit, default="deg")
    actuator_map = actuator_to_motor_index(motor_map)
    commands: list[tuple[int, float]] = []
    for actuator, value in actuators.items():
        actuator_name = str(actuator)
        if actuator_name not in actuator_map:
            raise ValueError(f"actuator {actuator_name!r} is not in the motor map")
        index = actuator_map[actuator_name]
        commands.append((index, value_to_radians(actuator_name, value, resolved_unit)))
    return commands


def coalesce_scalar_commands(commands: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Keep one value per motor index; later duplicate commands win."""
    merged: dict[int, float] = {}
    for index, value in commands:
        merged[index] = value
    return list(merged.items())


def scalar_commands_to_names_values(
    commands: list[tuple[int, float]],
    motor_map: dict[int, str],
) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for index, value in coalesce_scalar_commands(commands):
        actuator = motor_map[index]
        names.append(actuator)
        values.append(clamp_rad(actuator, value))
    return names, values


def angles_deg_to_scalar_commands(
    angles_deg: dict[str, float],
    motor_map: dict[int, str],
    *,
    strict_actuators: bool = False,
) -> tuple[list[tuple[int, float]], dict[str, float]]:
    supported, ignored = filter_supported_angles_deg(angles_deg, strict=strict_actuators)
    actuator_map = actuator_to_motor_index(motor_map)
    commands: list[tuple[int, float]] = []
    missing_from_map: dict[str, float] = {}

    for actuator, degrees in supported.items():
        if actuator not in actuator_map:
            missing_from_map[actuator] = degrees
            continue
        commands.append((actuator_map[actuator], deg_to_rad(degrees)))

    ignored.update(missing_from_map)
    if strict_actuators and ignored:
        names = ", ".join(sorted(ignored))
        raise KeyError(f"Unsupported scalar actuator(s): {names}")

    return sorted(commands, key=lambda item: item[0]), ignored


def standby_scalar_commands(motor_map: dict[int, str]) -> list[tuple[int, float]]:
    return [(index, 0.0) for index in sorted(motor_map)]


def scalar_payload(commands: list[tuple[int, float]]) -> dict[str, Any]:
    return {
        "unit": "rad",
        "commands": [
            {"index": index, "value": value}
            for index, value in commands
        ],
    }

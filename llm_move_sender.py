#!/usr/bin/env python3
import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Map actuator -> (server index, vector component in [x, y, z]).
ACTUATOR_TO_INDEX_COMPONENT: Dict[str, Tuple[int, int]] = {
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
}


@dataclass
class MoveCmd:
    actuator: str
    delta_rad: float
    duration_s: float
    raw_line: str


def parse_llm_output(text: str) -> Tuple[str, List[MoveCmd]]:
    summary = ""
    moves: List[MoveCmd] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("SUMMARY|"):
            summary = line.split("|", 1)[1].strip()
            continue
        if not line.startswith("MOVE|"):
            continue

        parts = line.split("|")
        if len(parts) != 4:
            raise ValueError(f"Invalid MOVE format: {line}")

        actuator = parts[1].strip()
        if actuator not in ACTUATOR_TO_INDEX_COMPONENT:
            raise ValueError(
                f"Unsupported actuator in MOVE line: {actuator}. "
                "This TCP mapping does not expose every actuator name."
            )

        try:
            delta_rad = float(parts[2].strip())
            duration_s = float(parts[3].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in MOVE line: {line}") from exc

        if duration_s < 0:
            raise ValueError(f"duration_s must be >= 0 in MOVE line: {line}")

        moves.append(
            MoveCmd(
                actuator=actuator,
                delta_rad=delta_rad,
                duration_s=duration_s,
                raw_line=line,
            )
        )

    if not moves:
        raise ValueError("No MOVE lines found.")

    return summary, moves


def send_one(host: str, port: int, index: int, value_xyz: List[float], timeout_s: float) -> Dict:
    payload = {"index": index, "value": value_xyz}
    data = json.dumps(payload).encode("utf-8")

    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall(data)
        raw = sock.recv(4096)

    if not raw:
        raise RuntimeError("Empty response from robot TCP server.")

    resp = json.loads(raw.decode("utf-8"))
    if not isinstance(resp, dict):
        raise RuntimeError(f"Invalid response payload: {resp}")
    if resp.get("code") != 0:
        raise RuntimeError(f"Robot server error: {resp}")
    return resp


def run_moves(
    host: str,
    port: int,
    timeout_s: float,
    moves: List[MoveCmd],
    dry_run: bool,
) -> None:
    actuator_state: Dict[str, float] = {}
    index_state: Dict[int, List[float]] = {}

    for i, move in enumerate(moves, start=1):
        idx, comp = ACTUATOR_TO_INDEX_COMPONENT[move.actuator]
        actuator_state[move.actuator] = actuator_state.get(move.actuator, 0.0) + move.delta_rad

        vec = index_state.get(idx, [0.0, 0.0, 0.0])
        vec[comp] = actuator_state[move.actuator]
        index_state[idx] = vec

        print(
            f"[{i}/{len(moves)}] {move.actuator}: delta={move.delta_rad:+.4f} rad "
            f"-> index={idx}, value={vec}, hold={move.duration_s:.3f}s"
        )

        if not dry_run:
            resp = send_one(host, port, idx, vec, timeout_s)
            print(f"  server_ack: {resp.get('result')}")

        if move.duration_s > 0:
            time.sleep(move.duration_s)


def read_text(input_file: str) -> str:
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse LLM SUMMARY/MOVE output and send TCP requests to Sophia body bridge."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Robot TCP host.")
    parser.add_argument("--port", type=int, default=5005, help="Robot TCP port.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Socket timeout seconds.")
    parser.add_argument(
        "--input-file",
        default="",
        help="Path to a text file containing LLM output. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print outgoing requests without sending to TCP server.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = read_text(args.input_file)
    summary, moves = parse_llm_output(text)

    if summary:
        print(f"Summary: {summary}")
    else:
        print("Summary: (none)")

    run_moves(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        moves=moves,
        dry_run=args.dry_run,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

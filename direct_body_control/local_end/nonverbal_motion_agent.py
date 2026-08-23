from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from motion_repo import MOTIONS, motion_catalog_text


DEFAULT_CANDIDATE_COUNT = int(os.getenv("SOPHIA_NONVERBAL_CANDIDATES", "3"))
MIN_ACTION_DURATION_SEC = 0.2
MAX_ACTION_DURATION_SEC = 1.6
DEFAULT_STANDBY_SEC = 0.6


@dataclass(frozen=True, slots=True)
class MotionRequest:
    spoken_text: str
    speech_duration_sec: float | None = None


def parse_motion_request(raw_text: str) -> MotionRequest:
    """
    Parse a text+time request.

    Supported forms:
    - Plain text. Duration is estimated.
    - JSON: {"text": "...", "duration": 3.5}
    - Text plus final line: duration: 3.5
    - Text plus suffix: ... || 3.5
    """
    text = raw_text.strip()
    if not text:
        return MotionRequest("")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        spoken = str(
            parsed.get("text")
            or parsed.get("spoken_text")
            or parsed.get("answer")
            or parsed.get("utterance")
            or ""
        ).strip()
        duration = _safe_float(parsed.get("duration") or parsed.get("time") or parsed.get("seconds"))
        return MotionRequest(spoken, duration)

    pipe_match = re.match(
        r"(?s)^(.*?)\s*\|\|\s*(?:duration\s*[:=]\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?\s*$",
        text,
    )
    if pipe_match:
        return MotionRequest(pipe_match.group(1).strip(), float(pipe_match.group(2)))

    duration_line = re.search(
        r"(?im)^\s*(?:duration|time|seconds)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?\s*$",
        text,
    )
    if duration_line:
        duration = float(duration_line.group(1))
        text = (text[: duration_line.start()] + text[duration_line.end() :]).strip()
        return MotionRequest(text, duration)

    return MotionRequest(text, None)


def estimate_speech_duration(text: str) -> float:
    """Rough duration estimate when the audio pipeline has not provided one."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))
    pauses = len(re.findall(r"[,.!?;:，。！？；：]", text))
    chinese_seconds = chinese_chars / 4.5 if chinese_chars else 0.0
    english_seconds = english_words / 2.7 if english_words else 0.0
    estimated = max(chinese_seconds, english_seconds, 1.2) + min(pauses * 0.12, 1.2)
    return max(1.0, min(estimated, 20.0))


def effective_speech_duration(request: MotionRequest) -> float:
    if request.speech_duration_sec and request.speech_duration_sec > 0:
        return request.speech_duration_sec
    return estimate_speech_duration(request.spoken_text)


def target_motion_duration(speech_duration_sec: float | None) -> float | None:
    """Prefer gestures that support speech without moving for the entire answer."""
    if speech_duration_sec is None or speech_duration_sec <= 0:
        return None
    if speech_duration_sec <= 2.0:
        return max(1.0, speech_duration_sec * 0.85)
    if speech_duration_sec <= 7.0:
        return speech_duration_sec * 0.8
    return min(7.0, speech_duration_sec * 0.65)


def build_candidate_prompt(
    spoken_text: str,
    speech_duration_sec: float,
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
) -> str:
    motion_target = target_motion_duration(speech_duration_sec)
    return f"""You are Planner LLM in a two-stage nonverbal-motion agent.
Your task is to propose several feasible robot upper-body motion sequences for Sophia.

Spoken text Sophia is about to say:
{spoken_text}

Estimated or measured speech duration: {speech_duration_sec:.2f} seconds
Target total motion duration: about {motion_target:.2f} seconds

Allowed atomic keyframes with physical pose descriptions and usage hints:
{motion_catalog_text()}

Generate {candidate_count} distinct candidate sequences. Each candidate should be safe, natural, and semantically matched to the spoken text.

Planning rules:
- Use only keyframes from the allowed catalog.
- Read each catalog description as a physical robot pose. Infer higher-level
  meaning by composing these atomic poses, especially with A+B for simultaneous
  left/right body parts.
- You may combine compatible listed keyframes with + to make simultaneous compound poses.
  Example: leftHandRaise+rightHandRaise means raise both hands at the same time.
- Use + when the meaning needs both sides together, such as surrender, two-handed
  presenting, balanced emphasis, or strong agreement.
- Digit/figure keyframes only move the fingers; combine them with the matching
  raised-hand atom so the number is visible, such as rightHandRaise+rightFigureTwo
  or leftHandRaise+leftFigureFour.
- rightThumbUp already includes the raised right arm and should not be treated
  as a hand-only digit gesture or combined with rightHandRaise.
- Do not invent new combined names. Compose from listed primitive keyframes instead.
- Prefer one clear communicative idea per sequence: greeting, positive feedback, thinking, presenting, or subtle speaking beats.
- For short speech, use 2-4 action lines. For longer explanations, use 4-7 action lines.
- Do not make the robot move constantly for a long answer. It is okay for motion to cover only the most meaningful part.
- Always end with standby unless holding the current pose is intentionally better.
- Use stay only when the spoken text implies holding or pausing.
- Durations should usually be 0.2 to 1.6 seconds per action.
- Avoid random fine-arm motions unless they support longer explanatory speech.
- Do not invent keyframe names.

Output JSON only in this exact shape:
{{
  "candidates": [
    {{
      "id": "A",
      "intent": "short reason for the gesture concept",
      "sequence": [
        {{"action": "leftHandReachOut+rightHandReachOut", "duration": 0.8}},
        {{"action": "standby", "duration": 0.6}}
      ]
    }}
  ]
}}
"""


def build_judge_prompt(
    spoken_text: str,
    speech_duration_sec: float,
    candidate_output: str,
) -> str:
    motion_target = target_motion_duration(speech_duration_sec)
    return f"""You are Judge LLM in a two-stage nonverbal-motion agent.
Choose or repair the single best robot motion sequence for Sophia.

Spoken text Sophia is about to say:
{spoken_text}

Estimated or measured speech duration: {speech_duration_sec:.2f} seconds
Target total motion duration: about {motion_target:.2f} seconds

Allowed atomic keyframes with physical pose descriptions and usage hints:
{motion_catalog_text()}

Planner candidates:
{candidate_output}

Selection criteria, in priority order:
1. Semantic fit to the spoken text.
2. Safe and natural upper-body motion.
3. Correct sequencing and composition: digit/figure keyframes should be combined with the matching hand raise; rightThumbUp already includes the raised right arm.
4. Valid keyframe names only.
5. Appropriate duration relative to the speech duration.
6. Clean ending: usually standby.
7. Use A+B composition when the speech implies simultaneous two-sided motion.

If the best candidate has small errors, repair it. If all candidates are poor, create a better valid sequence.

Final output format:
Plain text only. No JSON, no markdown, no explanation.
One action per line:
<action_name> <duration>
"""


def normalize_action_output(output_text: str, speech_duration_sec: float | None = None) -> str:
    pairs = parse_action_pairs(output_text)
    if not pairs:
        pairs = _default_action_combo(output_text)

    cleaned: list[tuple[str, float]] = []
    for name, duration in pairs:
        if not _is_valid_action_name(name):
            continue
        duration = max(MIN_ACTION_DURATION_SEC, min(float(duration), MAX_ACTION_DURATION_SEC))
        cleaned.append((name, duration))

    if not cleaned:
        cleaned = _default_action_combo(output_text)

    if len(cleaned) == 1:
        cleaned.append(("standby", DEFAULT_STANDBY_SEC))

    if cleaned[-1][0] not in {"standby", "stay"}:
        cleaned.append(("standby", DEFAULT_STANDBY_SEC))

    max_lines = _max_lines_for_duration(speech_duration_sec)
    if len(cleaned) > max_lines:
        tail = cleaned[-1]
        cleaned = cleaned[: max_lines - 1] + [tail]

    cleaned = _scale_to_target_duration(cleaned, target_motion_duration(speech_duration_sec))
    return "\n".join(f"{name} {duration:.2f}" for name, duration in cleaned)


def parse_action_pairs(output_text: str) -> list[tuple[str, float]]:
    stripped = _strip_code_fence(output_text.strip())
    if not stripped:
        return []

    json_pairs = _parse_json_action_pairs(stripped)
    if json_pairs:
        return json_pairs

    line_pairs: list[tuple[str, float]] = []
    line_pattern = re.compile(
        r'^(?:[-*]|\d+[.)])?\s*"?([A-Za-z][A-Za-z0-9_]*(?:\+[A-Za-z][A-Za-z0-9_]*)*)"?\s*[:,]?\s*"?([0-9]+(?:\.[0-9]+)?)"?\s*(?:s|sec|seconds)?$'
    )
    for raw in stripped.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = line_pattern.match(line)
        if not match:
            continue
        name = match.group(1)
        duration = _safe_float(match.group(2))
        if duration is None or duration <= 0:
            continue
        line_pairs.append((name, duration))
    return line_pairs


def _parse_json_action_pairs(text: str) -> list[tuple[str, float]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        for key in (
            "selected_sequence",
            "sequence",
            "actions",
            "motion_sequence",
            "final_sequence",
        ):
            pairs = _pairs_from_json_value(parsed.get(key))
            if pairs:
                return pairs
        candidates = parsed.get("candidates")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                pairs = _pairs_from_json_value(first.get("sequence") or first.get("actions"))
                if pairs:
                    return pairs
    return _pairs_from_json_value(parsed)


def _pairs_from_json_value(value: Any) -> list[tuple[str, float]]:
    if not isinstance(value, list):
        return []
    pairs: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("action") or item.get("name") or item.get("motion") or "").strip()
            duration = _safe_float(item.get("duration") or item.get("time") or item.get("seconds"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name = str(item[0]).strip()
            duration = _safe_float(item[1])
        else:
            continue
        if name and duration and duration > 0:
            pairs.append((name, duration))
    return pairs


def _default_action_combo(context: str) -> list[tuple[str, float]]:
    lowered = context.lower()
    if any(word in lowered for word in ("hello", "hi", "welcome", "goodbye", "bye", "你好", "欢迎", "再见")):
        return [
            ("rightHandRaise", 0.50),
            ("rightHandWaveLeft", 0.40),
            ("rightHandWaveRight", 0.45),
            ("standby", DEFAULT_STANDBY_SEC),
        ]
    if any(word in lowered for word in ("great", "good", "nice", "yes", "agree", "棒", "好", "赞", "同意")):
        return [("rightThumbUp", 0.80), ("standby", DEFAULT_STANDBY_SEC)]
    if any(word in lowered for word in ("think", "idea", "maybe", "check", "想", "思考", "让我看看")):
        return [("idea", 0.80), ("standby", DEFAULT_STANDBY_SEC)]
    return [("leftHandReachOut+rightHandReachOut", 0.75), ("standby", DEFAULT_STANDBY_SEC)]


def _is_valid_action_name(name: str) -> bool:
    if name in MOTIONS:
        return True
    parts = [part.strip() for part in name.split("+") if part.strip()]
    return bool(parts) and all(part in MOTIONS for part in parts)


def _scale_to_target_duration(
    pairs: list[tuple[str, float]],
    target_sec: float | None,
) -> list[tuple[str, float]]:
    if not target_sec:
        return pairs
    total = sum(duration for _, duration in pairs)
    if total <= 0 or total <= target_sec * 1.2:
        return pairs
    scale = target_sec / total
    return [
        (name, max(MIN_ACTION_DURATION_SEC, min(duration * scale, MAX_ACTION_DURATION_SEC)))
        for name, duration in pairs
    ]


def _max_lines_for_duration(speech_duration_sec: float | None) -> int:
    if speech_duration_sec is None:
        return 5
    if speech_duration_sec <= 2.0:
        return 4
    if speech_duration_sec <= 7.0:
        return 6
    return 7


def _strip_code_fence(text: str) -> str:
    match = re.match(r"^```(?:json|text)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

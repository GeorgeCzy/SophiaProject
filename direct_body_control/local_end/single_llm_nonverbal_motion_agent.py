from __future__ import annotations

from motion_repo import motion_catalog_text
from nonverbal_motion_agent import target_motion_duration


def build_single_motion_prompt(
    spoken_text: str,
    speech_duration_sec: float,
) -> str:
    motion_target = target_motion_duration(speech_duration_sec)
    target_text = f"about {motion_target:.2f} seconds" if motion_target else "not specified"
    return f"""You are Sophia's single-pass nonverbal-motion planning LLM.
Your task is to generate one final robot upper-body motion sequence for Sophia.

Spoken text Sophia is about to say:
{spoken_text}

Estimated or measured speech duration: {speech_duration_sec:.2f} seconds
Target total motion duration: {target_text}

Allowed keyframes with semantic descriptions:
{motion_catalog_text()}

Generate exactly one safe, natural, semantically matched sequence. Do not generate alternatives, rankings, candidate lists, or self-review text.

Planning rules:
- Use only keyframes from the allowed catalog.
- Follow catalog descriptions and cautions as sequencing rules.
- Digit/figure hand-shape keyframes only move fingers; first raise the matching hand with rightHandRaise or leftHandRaise so the figure is visible.
- rightThumbUp already includes the raised right arm and is not a hand-only digit gesture.
- Prefer one clear communicative idea per sequence: greeting, positive feedback, thinking, presenting, or subtle speaking beats.
- For short speech, use 2-4 action lines. For longer explanations, use 4-7 action lines.
- Do not make the robot move constantly for a long answer. It is okay for motion to cover only the most meaningful part.
- Always end with standby unless holding the current pose is intentionally better.
- Use stay only when the spoken text implies holding or pausing.
- Durations should usually be 0.2 to 1.6 seconds per action.
- Avoid random fine-arm motions unless they support longer explanatory speech.
- Do not invent keyframe names.

Plain text only. No JSON, no markdown, no explanation.
One action per line:
<action_name> <duration>
"""

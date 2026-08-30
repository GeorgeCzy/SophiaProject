from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import deepseek_chat_nonverbal_from_txt as deepseek_agent
from nonverbal_motion_agent import MotionRequest, effective_speech_duration, parse_motion_request
from single_llm_nonverbal_motion_agent import build_single_motion_prompt


BASE_DIR = deepseek_agent.BASE_DIR
PROMPT_PATH = Path(
    os.getenv(
        "SOPHIA_SINGLE_LLM_PROMPT_FILE",
        str(BASE_DIR / "system_prompt_single_llm.txt"),
    )
)
DEEPSEEK_SINGLE_MAX_TOKENS = deepseek_agent.env_int("DEEPSEEK_SINGLE_MAX_TOKENS", 512)
LATENCY_LOG_PATH = Path(
    os.getenv(
        "SOPHIA_DEEPSEEK_SINGLE_NONVERBAL_LATENCY_LOG",
        str(BASE_DIR / "motion_latency_deepseek_single_llm_log.json"),
    )
)
LATENCY_REPORT_PATH = Path(
    os.getenv(
        "SOPHIA_DEEPSEEK_SINGLE_NONVERBAL_LATENCY_REPORT",
        str(BASE_DIR / "motion_latency_deepseek_single_llm_report.txt"),
    )
)

SYS_PROMPT = deepseek_agent.load_prompt(PROMPT_PATH)
deepseek_agent.SYS_PROMPT = SYS_PROMPT


def single_latency_metrics() -> dict[str, float]:
    pairs = [
        ("extract_latest_conversation", "source_read_start", "source_read_done"),
        ("single_prompt_build", "single_prompt_build_start", "single_prompt_built"),
        ("single_deepseek_complete", "single_request_start", "single_response_done"),
        ("normalize_output", "normalize_start", "normalize_done"),
        ("write_actions_file", "write_actions_start", "write_actions_done"),
        ("motion_sender_subprocess", "motion_sender_start", "motion_sender_done"),
        ("total_until_actions_ready", "turn_start", "write_actions_done"),
        ("total_until_sender_done", "turn_start", "motion_sender_done"),
    ]
    metrics: dict[str, float] = {}
    for name, start_name, end_name in pairs:
        elapsed = deepseek_agent.latency_elapsed(start_name, end_name)
        if elapsed is not None:
            metrics[name] = elapsed
    return metrics


def write_single_latency_record() -> None:
    if not deepseek_agent.LATENCY_PROFILING or not deepseek_agent.current_latency:
        return

    current = deepseek_agent.current_latency
    record = {
        "time": datetime.now().isoformat(timespec="milliseconds"),
        "provider": "deepseek",
        "ablation": "single_llm",
        "model": deepseek_agent.DEEPSEEK_MODEL,
        "thinking_type": deepseek_agent.DEEPSEEK_THINKING_TYPE,
        "turn_id": current.get("turn_id"),
        "spoken_text": current.get("spoken_text"),
        "spoken_chars": current.get("spoken_chars"),
        "speech_duration_sec": current.get("speech_duration_sec"),
        "motion_source": current.get("motion_source"),
        "motion_source_path": current.get("motion_source_path"),
        "single_prompt_chars": current.get("single_prompt_chars"),
        "single_output_chars": current.get("single_output_chars"),
        "metrics_sec": single_latency_metrics(),
    }
    try:
        LATENCY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        append_latency_json(record)
        append_latency_report(record)
    except Exception as exc:
        print(f"[Latency] failed to write {LATENCY_LOG_PATH}: {exc}", flush=True)


def append_latency_json(record: dict[str, Any]) -> None:
    if LATENCY_LOG_PATH.exists():
        try:
            existing = json.loads(LATENCY_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
        if not isinstance(existing, list):
            existing = [existing]
    else:
        existing = []

    existing.append(record)
    LATENCY_LOG_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_latency_report(record: dict[str, Any]) -> None:
    metrics = record.get("metrics_sec") or {}
    LATENCY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "",
        "=" * 72,
        f"Turn {record.get('turn_id')} | {record.get('time')}",
        "-" * 72,
        f"provider: {record.get('provider')}",
        f"ablation: {record.get('ablation')}",
        f"model: {record.get('model')}",
        f"thinking_type: {record.get('thinking_type')}",
        f"source: {record.get('motion_source') or ''}",
        f"source_path: {record.get('motion_source_path') or ''}",
        f"spoken_chars: {record.get('spoken_chars')}",
        f"speech_duration_sec: {record.get('speech_duration_sec')}",
        f"single_prompt_chars: {record.get('single_prompt_chars')}",
        f"single_output_chars: {record.get('single_output_chars')}",
        "",
        "spoken_text:",
        str(record.get("spoken_text") or ""),
        "",
        "latency_sec:",
    ]
    if metrics:
        for name, seconds in metrics.items():
            lines.append(f"  {name}: {float(seconds):.3f}")
    else:
        lines.append("  <no latency metrics recorded>")

    with LATENCY_REPORT_PATH.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_motion_turn(
    raw_text: str,
    *,
    api_key: str,
    duration_hint: float | None = None,
    source_latency: dict[str, Any] | None = None,
    no_send: bool = False,
) -> None:
    request = parse_motion_request(raw_text)
    if not request.spoken_text:
        print("Empty motion request; ignoring.", flush=True)
        return

    duration = request.speech_duration_sec or duration_hint
    request = MotionRequest(request.spoken_text, duration)
    duration = effective_speech_duration(request)

    deepseek_agent.current_latency = {
        "turn_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "spoken_text": request.spoken_text,
        "spoken_chars": len(request.spoken_text),
        "speech_duration_sec": duration,
    }
    if source_latency:
        deepseek_agent.current_latency.update(source_latency)
    deepseek_agent.latency_mark("turn_start")

    deepseek_agent.latency_mark("single_prompt_build_start")
    single_prompt = build_single_motion_prompt(request.spoken_text, duration)
    deepseek_agent.current_latency["single_prompt_chars"] = len(single_prompt)
    deepseek_agent.latency_mark("single_prompt_built")

    print(f"[Agent:DeepSeek:Single] generating one motion sequence for {duration:.2f}s spoken text", flush=True)
    deepseek_agent.latency_mark("single_request_start")
    single_output = deepseek_agent.deepseek_chat_completion(
        single_prompt,
        api_key=api_key,
        max_tokens=DEEPSEEK_SINGLE_MAX_TOKENS,
    )
    deepseek_agent.latency_mark("single_response_done")
    single_output = deepseek_agent.finalize_response_text(single_output)
    deepseek_agent.current_latency["single_output_chars"] = len(single_output)
    print("[Agent:DeepSeek:Single] generated motion:", flush=True)
    print(single_output, flush=True)

    deepseek_agent.handle_output(single_output, duration, no_send=no_send)
    deepseek_agent.latency_mark("turn_done")
    write_single_latency_record()


def dry_run_once(text: str | None = None, duration_hint: float | None = None) -> int:
    if text is None:
        try:
            content, signature = deepseek_agent.read_motion_source()
        except FileNotFoundError as exc:
            print(str(exc), flush=True)
            return 1
    else:
        content = text
        signature = ("cli", len(text), duration_hint)

    if not content:
        print("No latest ai/assistant/robot text was extracted yet.", flush=True)
        return 0

    request = parse_motion_request(content)
    duration = request.speech_duration_sec or duration_hint
    request = MotionRequest(request.spoken_text, duration)
    duration = effective_speech_duration(request)
    prompt = build_single_motion_prompt(request.spoken_text, duration)
    print("Dry run only. No DeepSeek request and no robot command were sent.", flush=True)
    print(f"source_signature = {signature}", flush=True)
    print(f"spoken_chars = {len(request.spoken_text)}", flush=True)
    print(f"speech_duration_sec = {duration:.2f}", flush=True)
    print(f"single_prompt_chars = {len(prompt)}", flush=True)
    return 0


def print_settings() -> None:
    print("starting DeepSeek single-LLM nonverbal motion agent...")
    print(f"BASE_URL = {deepseek_agent.DEEPSEEK_BASE_URL}")
    print(f"MODEL = {deepseek_agent.DEEPSEEK_MODEL}")
    print(f"THINKING = {deepseek_agent.DEEPSEEK_THINKING_TYPE}")
    print(f"PROMPT = {PROMPT_PATH}")
    print(f"ACTIONS = {deepseek_agent.ACTIONS_PATH}")
    print(f"INPUT = {deepseek_agent.INPUT_PATH}")
    print(f"CHAT_HISTORY = {', '.join(str(path) for path in deepseek_agent.CHAT_HISTORY_PATHS)}")
    print(f"DURATION = {deepseek_agent.DURATION_PATH}")
    print(f"MOTION_SENDER = {deepseek_agent.MOTION_SENDER}")
    print(f"MOTION_DURATION_SCALE = {deepseek_agent.MOTION_DURATION_SCALE}")
    print(f"LATENCY_LOG = {LATENCY_LOG_PATH}")
    print(f"LATENCY_REPORT = {LATENCY_REPORT_PATH}")
    if deepseek_agent.MOTION_SENDER in {"scalar", "scalar_index", "motor_index", "index"}:
        print(f"SCALAR_ROBOT = {deepseek_agent.SCALAR_ROBOT_HOST}:{deepseek_agent.SCALAR_ROBOT_PORT}")
    elif deepseek_agent.MOTION_SENDER in {"standard", "standard_index"}:
        print(f"STANDARD_ROBOT = {deepseek_agent.STANDARD_ROBOT_HOST}:{deepseek_agent.STANDARD_ROBOT_PORT}")
    elif deepseek_agent.MOTION_SENDER == "direct":
        print(f"DIRECT_ROBOT = {deepseek_agent.DIRECT_ROBOT_HOST}:{deepseek_agent.DIRECT_ROBOT_PORT}")
    else:
        print(f"LEGACY_ROBOT = {deepseek_agent.LEGACY_ROBOT_HOST}:{deepseek_agent.LEGACY_ROBOT_PORT}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Sophia nonverbal motions with one DeepSeek Chat Completions call."
    )
    parser.add_argument("--text", help="Generate motion for one spoken answer instead of watching input files.")
    parser.add_argument("--duration", type=float, help="Optional speech duration for --text.")
    parser.add_argument("--once", action="store_true", help="Process the current input once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Read input and build the prompt without calling DeepSeek.")
    parser.add_argument("--no-send", action="store_true", help="Write actions.txt but do not send commands to the robot.")
    args = parser.parse_args()

    print_settings()

    if args.dry_run:
        return dry_run_once(text=args.text, duration_hint=args.duration)

    api_key = deepseek_agent.load_deepseek_api_key()

    if args.text:
        source_latency = {"motion_source": "cli", "motion_source_path": ""}
        run_motion_turn(
            args.text,
            api_key=api_key,
            duration_hint=args.duration,
            source_latency=source_latency,
            no_send=args.no_send,
        )
        return 0

    last_sent_signature: tuple | None = None
    last_empty_signature: tuple | None = None
    missing_reported = False

    while True:
        try:
            content, signature, source_latency = deepseek_agent.read_source_with_latency()
            missing_reported = False
        except FileNotFoundError as exc:
            if not missing_reported:
                print(str(exc), flush=True)
                missing_reported = True
            if args.once:
                return 1
            time.sleep(deepseek_agent.POLL_INTERVAL_SEC)
            continue
        except KeyboardInterrupt:
            print("\nmanual exit...")
            return 0
        except Exception as exc:
            print(f"Failed to read motion input source: {exc}", flush=True)
            if args.once:
                return 1
            time.sleep(deepseek_agent.POLL_INTERVAL_SEC)
            continue

        if not content:
            if signature != last_empty_signature:
                print(
                    "Chat history is present, but no latest ai/assistant/robot text "
                    "was extracted yet; waiting for robot reply.",
                    flush=True,
                )
                last_empty_signature = signature
            if args.once:
                return 0
            time.sleep(deepseek_agent.POLL_INTERVAL_SEC)
            continue

        if signature == last_sent_signature:
            if args.once:
                return 0
            time.sleep(deepseek_agent.POLL_INTERVAL_SEC)
            continue

        duration_hint = deepseek_agent.read_duration_hint()
        if duration_hint:
            print(f"\n[Spoken Answer] duration={duration_hint:.2f}s\n{content}\n", flush=True)
        else:
            print(f"\n[Spoken Answer] duration=estimated\n{content}\n", flush=True)

        try:
            run_motion_turn(
                content,
                api_key=api_key,
                duration_hint=duration_hint,
                source_latency=source_latency,
                no_send=args.no_send,
            )
            last_sent_signature = signature
        except KeyboardInterrupt:
            print("\nmanual exit...")
            return 0
        except Exception as exc:
            print(f"\nDeepSeek single-LLM motion generation failed: {exc}\n", flush=True)
            if args.once:
                return 1

        if args.once:
            return 0
        time.sleep(deepseek_agent.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

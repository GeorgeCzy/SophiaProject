from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nonverbal_motion_agent import (
    MotionRequest,
    build_candidate_prompt,
    build_judge_prompt,
    effective_speech_duration,
    normalize_action_output,
    parse_motion_request,
)


# ===== Settings ==========================================================
BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "system_prompt.txt"
ACTIONS_PATH = BASE_DIR / "actions.txt"
SYNC_DIR = Path(os.getenv("SOPHIA_ROBOT_SYNC_DIR", "/tmp/robot_sync"))
DURATION_PATH = Path(os.getenv("SOPHIA_NONVERBAL_DURATION_FILE", str(SYNC_DIR / "audio_response.duration")))

# Fill this with your DeepSeek key if you do not want to use an env var or .env.
# Keep this file private when the key is filled.
DEFAULT_DEEPSEEK_API_KEY = ""
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


INPUT_PATH = Path(
    os.getenv(
        "SOPHIA_NONVERBAL_INPUT_FILE",
        str(BASE_DIR / "input.txt"),
    )
)
CHAT_HISTORY_ENV = os.getenv("SOPHIA_CHAT_HISTORY_FILE")
if CHAT_HISTORY_ENV:
    CHAT_HISTORY_PATHS = [Path(CHAT_HISTORY_ENV)]
else:
    CHAT_HISTORY_PATHS = [
        BASE_DIR.parent / "memory_supervisor" / "memory_sessions.jsonl",
        BASE_DIR.parent / "chat_history.json",
        BASE_DIR.parent / "chat_history.jsonl",
    ]
CHAT_HISTORY_PATH = CHAT_HISTORY_PATHS[0]
APPEND_RANDOM_GESTURE = os.getenv("SOPHIA_NONVERBAL_APPEND_RANDOM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MOTION_SENDER = os.getenv("SOPHIA_MOTION_SENDER", "scalar_index").strip().lower()
MOTION_DURATION_SCALE = env_float("SOPHIA_MOTION_DURATION_SCALE", 1.25)
SCALAR_ROBOT_HOST = os.getenv("SOPHIA_SCALAR_ROBOT_HOST", "10.0.0.10")
SCALAR_ROBOT_PORT = int(os.getenv("SOPHIA_SCALAR_ROBOT_PORT", "5005"))
STANDARD_ROBOT_HOST = os.getenv("SOPHIA_STANDARD_ROBOT_HOST", "10.0.0.10")
STANDARD_ROBOT_PORT = int(os.getenv("SOPHIA_STANDARD_ROBOT_PORT", "5005"))
DIRECT_ROBOT_HOST = os.getenv("SOPHIA_DIRECT_ROBOT_HOST", "10.0.0.10")
DIRECT_ROBOT_PORT = int(os.getenv("SOPHIA_DIRECT_ROBOT_PORT", "5006"))
LEGACY_ROBOT_HOST = os.getenv("SOPHIA_LEGACY_ROBOT_HOST", "10.0.0.10")
LEGACY_ROBOT_PORT = int(os.getenv("SOPHIA_LEGACY_ROBOT_PORT", "5005"))
POLL_INTERVAL_SEC = env_float("SOPHIA_NONVERBAL_POLL_SEC", 0.5)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
DEEPSEEK_TIMEOUT_SEC = env_float("DEEPSEEK_TIMEOUT_SEC", 30.0)
DEEPSEEK_TEMPERATURE = env_float("DEEPSEEK_TEMPERATURE", 0.2)
DEEPSEEK_CANDIDATE_MAX_TOKENS = env_int("DEEPSEEK_CANDIDATE_MAX_TOKENS", 900)
DEEPSEEK_JUDGE_MAX_TOKENS = env_int("DEEPSEEK_JUDGE_MAX_TOKENS", 350)
DEEPSEEK_THINKING_TYPE = os.getenv("DEEPSEEK_THINKING_TYPE", "disabled").strip().lower()

LATENCY_PROFILING = os.getenv("SOPHIA_NONVERBAL_LATENCY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LATENCY_LOG_PATH = Path(
    os.getenv(
        "SOPHIA_DEEPSEEK_NONVERBAL_LATENCY_LOG",
        str(BASE_DIR / "motion_latency_deepseek_log.json"),
    )
)
LATENCY_REPORT_PATH = Path(
    os.getenv(
        "SOPHIA_DEEPSEEK_NONVERBAL_LATENCY_REPORT",
        str(BASE_DIR / "motion_latency_deepseek_report.txt"),
    )
)
# ======================================================================

current_latency: dict[str, Any] = {}


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


SYS_PROMPT = load_prompt(PROMPT_PATH)


def _load_api_key_from_dotenv() -> str:
    for dotenv_path in (BASE_DIR / ".env", BASE_DIR / "Sophia_Face_HCI" / ".env"):
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"DEEPSEEK_API_KEY", "DeepSeek_API_KEY"}:
                return value.strip().strip("'\"")
    return ""


def load_deepseek_api_key() -> str:
    api_key = (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("DeepSeek_API_KEY", "").strip()
        or DEFAULT_DEEPSEEK_API_KEY.strip()
        or _load_api_key_from_dotenv()
    )
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required. Export DEEPSEEK_API_KEY, add it to "
            f"{BASE_DIR / '.env'}, or fill DEFAULT_DEEPSEEK_API_KEY in this file."
        )
    return api_key


def latency_mark(name: str) -> None:
    if LATENCY_PROFILING:
        current_latency[name] = time.perf_counter()


def latency_elapsed(start_name: str, end_name: str) -> float | None:
    start = current_latency.get(start_name)
    end = current_latency.get(end_name)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return max(0.0, end - start)
    return None


def latency_metrics() -> dict[str, float]:
    pairs = [
        ("extract_latest_conversation", "source_read_start", "source_read_done"),
        ("candidate_prompt_build", "candidate_prompt_build_start", "candidate_prompt_built"),
        ("candidate_complete", "candidate_request_start", "candidate_response_done"),
        ("judge_prompt_build", "judge_prompt_build_start", "judge_prompt_built"),
        ("judge_complete", "judge_request_start", "judge_response_done"),
        ("normalize_output", "normalize_start", "normalize_done"),
        ("write_actions_file", "write_actions_start", "write_actions_done"),
        ("motion_sender_subprocess", "motion_sender_start", "motion_sender_done"),
        ("total_until_actions_ready", "turn_start", "write_actions_done"),
        ("total_until_sender_done", "turn_start", "motion_sender_done"),
    ]
    metrics: dict[str, float] = {}
    for name, start_name, end_name in pairs:
        elapsed = latency_elapsed(start_name, end_name)
        if elapsed is not None:
            metrics[name] = elapsed
    return metrics


def write_latency_record() -> None:
    if not LATENCY_PROFILING or not current_latency:
        return

    record = {
        "time": datetime.now().isoformat(timespec="milliseconds"),
        "provider": "deepseek",
        "model": DEEPSEEK_MODEL,
        "thinking_type": DEEPSEEK_THINKING_TYPE,
        "turn_id": current_latency.get("turn_id"),
        "spoken_text": current_latency.get("spoken_text"),
        "spoken_chars": current_latency.get("spoken_chars"),
        "speech_duration_sec": current_latency.get("speech_duration_sec"),
        "motion_source": current_latency.get("motion_source"),
        "motion_source_path": current_latency.get("motion_source_path"),
        "candidate_prompt_chars": current_latency.get("candidate_prompt_chars"),
        "judge_prompt_chars": current_latency.get("judge_prompt_chars"),
        "candidate_output_chars": current_latency.get("candidate_output_chars"),
        "judge_output_chars": current_latency.get("judge_output_chars"),
        "metrics_sec": latency_metrics(),
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
        f"model: {record.get('model')}",
        f"thinking_type: {record.get('thinking_type')}",
        f"source: {record.get('motion_source') or ''}",
        f"source_path: {record.get('motion_source_path') or ''}",
        f"spoken_chars: {record.get('spoken_chars')}",
        f"speech_duration_sec: {record.get('speech_duration_sec')}",
        f"candidate_prompt_chars: {record.get('candidate_prompt_chars')}",
        f"candidate_output_chars: {record.get('candidate_output_chars')}",
        f"judge_prompt_chars: {record.get('judge_prompt_chars')}",
        f"judge_output_chars: {record.get('judge_output_chars')}",
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


def deepseek_chat_completion(
    prompt: str,
    *,
    api_key: str,
    max_tokens: int,
    response_format: dict[str, str] | None = None,
) -> str:
    url = DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": DEEPSEEK_TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if DEEPSEEK_THINKING_TYPE:
        payload["thinking"] = {"type": DEEPSEEK_THINKING_TYPE}
    if response_format:
        payload["response_format"] = response_format

    try:
        return _post_deepseek_payload(url, payload, api_key)
    except RuntimeError as exc:
        if "DeepSeek HTTP 400" not in str(exc):
            raise
        optional_keys = {"thinking", "response_format"} & set(payload)
        if not optional_keys:
            raise
        for key in optional_keys:
            payload.pop(key, None)
        print(
            "DeepSeek rejected an optional request field; retrying once without optional fields.",
            flush=True,
        )
        return _post_deepseek_payload(url, payload, api_key)


def _post_deepseek_payload(url: str, payload: dict[str, Any], api_key: str) -> str:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=DEEPSEEK_TIMEOUT_SEC) as response:
            raw_body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

    data = json.loads(raw_body)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"DeepSeek response has no choices: {raw_body}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"DeepSeek response has no text content: {raw_body}")
    return content.strip()


def finalize_response_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def read_duration_hint() -> float | None:
    try:
        raw = DURATION_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"Failed to read duration hint {DURATION_PATH}: {exc}", flush=True)
        return None
    try:
        duration = float(raw)
    except ValueError:
        return None
    return duration if duration > 0 else None


def write_actions_file(output_text: str) -> None:
    print(f"Writing actions ({len(output_text)} chars) to {ACTIONS_PATH}", flush=True)
    with ACTIONS_PATH.open("w", encoding="utf-8") as file:
        file.write(output_text)
        if output_text and not output_text.endswith("\n"):
            file.write("\n")

    print(f"Saved actions to {ACTIONS_PATH}", flush=True)


def run_move_sender(no_send: bool = False) -> None:
    if no_send:
        print("Skipping motion sender because --no-send was set.", flush=True)
        return

    if APPEND_RANDOM_GESTURE:
        try:
            subprocess.run(
                [sys.executable, "generating_random_gesture.py"],
                cwd=BASE_DIR,
                text=True,
                check=False,
            )
        except Exception as exc:
            print(f"Failed to run generating_random_gesture.py: {exc}", flush=True)

    print("Sending motions immediately.", flush=True)

    if MOTION_SENDER in {"scalar", "scalar_index", "motor_index", "index"}:
        sender_cmd = [
            sys.executable,
            "llm_move_sender.py",
            "--input-file",
            "actions.txt",
            "--host",
            SCALAR_ROBOT_HOST,
            "--port",
            str(SCALAR_ROBOT_PORT),
            "--duration-scale",
            str(MOTION_DURATION_SCALE),
        ]
        sender_name = "llm_move_sender.py"
    elif MOTION_SENDER in {"standard", "standard_index"}:
        sender_cmd = [
            sys.executable,
            "standard_index_motion_sender.py",
            "--input-file",
            "actions.txt",
            "--host",
            STANDARD_ROBOT_HOST,
            "--port",
            str(STANDARD_ROBOT_PORT),
        ]
        sender_name = "standard_index_motion_sender.py"
    elif MOTION_SENDER == "direct":
        sender_cmd = [
            sys.executable,
            "direct_motion_sender.py",
            "--input-file",
            "actions.txt",
            "--host",
            DIRECT_ROBOT_HOST,
            "--port",
            str(DIRECT_ROBOT_PORT),
        ]
        sender_name = "direct_motion_sender.py"
    elif MOTION_SENDER in {"legacy", "smpl", "smplx"}:
        sender_cmd = [
            sys.executable,
            "llm_move_sender.py",
            "--input-file",
            "actions.txt",
            "--host",
            LEGACY_ROBOT_HOST,
            "--port",
            str(LEGACY_ROBOT_PORT),
            "--duration-scale",
            str(MOTION_DURATION_SCALE),
        ]
        sender_name = "llm_move_sender.py"
    else:
        print(
            f"Unknown SOPHIA_MOTION_SENDER={MOTION_SENDER!r}; "
            "expected scalar_index, standard_index, direct, or legacy.",
            flush=True,
        )
        return

    try:
        result = subprocess.run(
            sender_cmd,
            cwd=BASE_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        print(f"Failed to run {sender_name}: {exc}", flush=True)
        return

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        print(f"{sender_name} exited with code {result.returncode}", flush=True)


def handle_output(output_text: str, speech_duration_sec: float | None, no_send: bool = False) -> None:
    latency_mark("normalize_start")
    normalized = normalize_action_output(output_text, speech_duration_sec)
    latency_mark("normalize_done")
    if normalized != output_text.strip():
        print("Normalized model output to valid action pairs.", flush=True)
        print(normalized, flush=True)
    latency_mark("write_actions_start")
    write_actions_file(normalized)
    latency_mark("write_actions_done")
    latency_mark("motion_sender_start")
    run_move_sender(no_send=no_send)
    latency_mark("motion_sender_done")


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_text_from_content(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in (
            "text",
            "message",
            "content",
            "answer",
            "response",
            "utterance",
            "output_text",
            "final_answer",
            "value",
            "parts",
        ):
            text = _text_from_content(value.get(key))
            if text:
                return text
    return ""


def _role_from_item(item: dict[str, Any]) -> str:
    for key in ("role", "speaker", "sender", "author", "from", "type"):
        value = item.get(key)
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, dict):
            for nested_key in ("role", "name", "type"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested.strip().lower()
    return ""


def _walk_history_json(data: Any):
    if isinstance(data, list):
        for item in data:
            yield from _walk_history_json(item)
        return

    if not isinstance(data, dict):
        return

    if _role_from_item(data) and _text_from_content(data):
        yield data

    known_keys = (
        "messages",
        "history",
        "conversation",
        "conversations",
        "chat_history",
        "items",
        "data",
        "records",
        "turns",
    )
    walked_known = False
    for key in known_keys:
        if key in data:
            walked_known = True
            yield from _walk_history_json(data[key])

    if not walked_known:
        for value in data.values():
            if isinstance(value, (dict, list)):
                yield from _walk_history_json(value)


def _read_history_items(path: Path) -> list[dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        items: list[dict[str, Any]] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                walked = [nested for nested in _walk_history_json(item) if isinstance(nested, dict)]
                items.extend(walked or [item])
        return items

    return [item for item in _walk_history_json(parsed) if isinstance(item, dict)]


def extract_latest_ai_text_from_history(path: Path) -> tuple[str, tuple]:
    latest_item: dict[str, Any] | None = None
    latest_index = 0

    for index, item in enumerate(_read_history_items(path), start=1):
        role = _role_from_item(item)
        text = _text_from_content(item)
        if role in {"ai", "assistant", "robot"} and text:
            latest_item = item
            latest_index = index

    if not latest_item:
        return "", ("chat_history", 0)

    text = _text_from_content(latest_item)
    signature = (
        "chat_history",
        str(path),
        latest_item.get("time", ""),
        latest_item.get("sequence", ""),
        latest_index,
        text,
    )
    return text, signature


def write_extracted_input_text(text: str) -> None:
    try:
        old_text = INPUT_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        old_text = ""

    if old_text == text.strip():
        return
    INPUT_PATH.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Extracted latest AI text to {INPUT_PATH}", flush=True)


def read_motion_source() -> tuple[str, tuple]:
    for history_path in CHAT_HISTORY_PATHS:
        if not history_path.exists():
            continue
        stat = history_path.stat()
        content, signature = extract_latest_ai_text_from_history(history_path)
        if content:
            write_extracted_input_text(content)
            return content, signature
        return "", ("chat_history_no_robot_text", str(history_path), stat.st_mtime_ns, stat.st_size)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file does not exist yet: {INPUT_PATH}")

    stat = INPUT_PATH.stat()
    content = INPUT_PATH.read_text(encoding="utf-8").strip()
    return content, ("input", stat.st_mtime_ns, stat.st_size)


def run_motion_turn(
    raw_text: str,
    *,
    api_key: str,
    duration_hint: float | None = None,
    source_latency: dict[str, Any] | None = None,
    no_send: bool = False,
) -> None:
    global current_latency

    request = parse_motion_request(raw_text)
    if not request.spoken_text:
        print("Empty motion request; ignoring.", flush=True)
        return

    duration = request.speech_duration_sec or duration_hint
    request = MotionRequest(request.spoken_text, duration)
    duration = effective_speech_duration(request)

    current_latency = {
        "turn_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "spoken_text": request.spoken_text,
        "spoken_chars": len(request.spoken_text),
        "speech_duration_sec": duration,
    }
    if source_latency:
        current_latency.update(source_latency)
    latency_mark("turn_start")

    latency_mark("candidate_prompt_build_start")
    candidate_prompt = build_candidate_prompt(request.spoken_text, duration)
    current_latency["candidate_prompt_chars"] = len(candidate_prompt)
    latency_mark("candidate_prompt_built")

    print(f"[Agent:DeepSeek] planning for {duration:.2f}s spoken text", flush=True)
    latency_mark("candidate_request_start")
    candidate_output = deepseek_chat_completion(
        candidate_prompt,
        api_key=api_key,
        max_tokens=DEEPSEEK_CANDIDATE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    latency_mark("candidate_response_done")
    candidate_output = finalize_response_text(candidate_output)
    current_latency["candidate_output_chars"] = len(candidate_output)
    print("[Agent:DeepSeek] planner candidates:", flush=True)
    print(candidate_output, flush=True)

    latency_mark("judge_prompt_build_start")
    judge_prompt = build_judge_prompt(request.spoken_text, duration, candidate_output)
    current_latency["judge_prompt_chars"] = len(judge_prompt)
    latency_mark("judge_prompt_built")

    latency_mark("judge_request_start")
    judge_output = deepseek_chat_completion(
        judge_prompt,
        api_key=api_key,
        max_tokens=DEEPSEEK_JUDGE_MAX_TOKENS,
    )
    latency_mark("judge_response_done")
    judge_output = finalize_response_text(judge_output)
    current_latency["judge_output_chars"] = len(judge_output)
    print("[Agent:DeepSeek] selected motion:", flush=True)
    print(judge_output, flush=True)

    handle_output(judge_output, duration, no_send=no_send)
    latency_mark("turn_done")
    write_latency_record()


def read_source_with_latency() -> tuple[str, tuple, dict[str, Any]]:
    source_latency: dict[str, Any] = {"source_read_start": time.perf_counter()}
    content, signature = read_motion_source()
    source_latency["source_read_done"] = time.perf_counter()
    source_latency["motion_source"] = signature[0] if signature else "unknown"
    if len(signature) > 1 and isinstance(signature[1], str):
        source_latency["motion_source_path"] = signature[1]
    elif signature and signature[0] == "input":
        source_latency["motion_source_path"] = str(INPUT_PATH)
    return content, signature, source_latency


def dry_run_once() -> int:
    try:
        source_latency = {"source_read_start": time.perf_counter()}
        content, signature = read_motion_source()
        source_latency["source_read_done"] = time.perf_counter()
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return 1

    if not content:
        print("No latest ai/assistant/robot text was extracted yet.", flush=True)
        return 0

    request = parse_motion_request(content)
    duration = effective_speech_duration(request)
    prompt = build_candidate_prompt(request.spoken_text, duration)
    print("Dry run only. No DeepSeek request and no robot command were sent.", flush=True)
    print(f"source_signature = {signature}", flush=True)
    print(f"spoken_chars = {len(request.spoken_text)}", flush=True)
    print(f"speech_duration_sec = {duration:.2f}", flush=True)
    print(f"candidate_prompt_chars = {len(prompt)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Sophia nonverbal motions with DeepSeek Chat Completions."
    )
    parser.add_argument("--once", action="store_true", help="Process the current input once and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Read input and build the prompt without calling DeepSeek.")
    parser.add_argument("--no-send", action="store_true", help="Write actions.txt but do not send commands to the robot.")
    args = parser.parse_args()

    print("starting DeepSeek nonverbal motion agent...")
    print(f"BASE_URL = {DEEPSEEK_BASE_URL}")
    print(f"MODEL = {DEEPSEEK_MODEL}")
    print(f"THINKING = {DEEPSEEK_THINKING_TYPE}")
    print(f"PROMPT = {PROMPT_PATH}")
    print(f"ACTIONS = {ACTIONS_PATH}")
    print(f"INPUT = {INPUT_PATH}")
    print(f"CHAT_HISTORY = {', '.join(str(path) for path in CHAT_HISTORY_PATHS)}")
    print(f"DURATION = {DURATION_PATH}")
    print(f"MOTION_SENDER = {MOTION_SENDER}")
    print(f"MOTION_DURATION_SCALE = {MOTION_DURATION_SCALE}")
    print(f"LATENCY_LOG = {LATENCY_LOG_PATH}")
    print(f"LATENCY_REPORT = {LATENCY_REPORT_PATH}")
    if MOTION_SENDER in {"scalar", "scalar_index", "motor_index", "index"}:
        print(f"SCALAR_ROBOT = {SCALAR_ROBOT_HOST}:{SCALAR_ROBOT_PORT}")
    elif MOTION_SENDER in {"standard", "standard_index"}:
        print(f"STANDARD_ROBOT = {STANDARD_ROBOT_HOST}:{STANDARD_ROBOT_PORT}")
    elif MOTION_SENDER == "direct":
        print(f"DIRECT_ROBOT = {DIRECT_ROBOT_HOST}:{DIRECT_ROBOT_PORT}")
    else:
        print(f"LEGACY_ROBOT = {LEGACY_ROBOT_HOST}:{LEGACY_ROBOT_PORT}")

    if args.dry_run:
        return dry_run_once()

    api_key = load_deepseek_api_key()
    last_sent_signature: tuple | None = None
    last_empty_signature: tuple | None = None
    missing_reported = False

    while True:
        try:
            content, signature, source_latency = read_source_with_latency()
            missing_reported = False
        except FileNotFoundError as exc:
            if not missing_reported:
                print(str(exc), flush=True)
                missing_reported = True
            if args.once:
                return 1
            time.sleep(POLL_INTERVAL_SEC)
            continue
        except KeyboardInterrupt:
            print("\nmanual exit...")
            return 0
        except Exception as exc:
            print(f"Failed to read motion input source: {exc}", flush=True)
            if args.once:
                return 1
            time.sleep(POLL_INTERVAL_SEC)
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
            time.sleep(POLL_INTERVAL_SEC)
            continue

        if signature == last_sent_signature:
            if args.once:
                return 0
            time.sleep(POLL_INTERVAL_SEC)
            continue

        duration_hint = read_duration_hint()
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
            print(f"\nDeepSeek motion generation failed: {exc}\n", flush=True)
            if args.once:
                return 1

        if args.once:
            return 0
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

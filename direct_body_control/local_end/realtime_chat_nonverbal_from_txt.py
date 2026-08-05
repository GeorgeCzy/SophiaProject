import json
import os
import subprocess
import ssl
import sys
import threading
from datetime import datetime

import certifi
import websocket
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from nonverbal_motion_agent import (
    MotionRequest,
    build_candidate_prompt,
    build_judge_prompt,
    effective_speech_duration,
    normalize_action_output,
    parse_motion_request,
)


# ===== Settings ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
ACTIONS_PATH = os.path.join(BASE_DIR, "actions.txt")
SYNC_DIR = Path(os.getenv("SOPHIA_ROBOT_SYNC_DIR", "/tmp/robot_sync"))
DURATION_PATH = Path(os.getenv("SOPHIA_NONVERBAL_DURATION_FILE", str(SYNC_DIR / "audio_response.duration")))
SOPHIA_FACE_HCI_DIR = Path(BASE_DIR) / "Sophia_Face_HCI"
if str(SOPHIA_FACE_HCI_DIR) not in sys.path:
    sys.path.insert(0, str(SOPHIA_FACE_HCI_DIR))

# Optional local convenience config.
# By default this script prefers settings.py when it is available. Set
# USE_DIRECT_CONFIG = True only if you want to ignore settings.py and use the
# direct values below.
# Keep this file private if you paste a real API key here.
USE_DIRECT_CONFIG = False
DEFAULT_API_KEY = ""
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_URL = ""
DEFAULT_REALTIME_PROXY = "http://127.0.0.1:7897"


class _RealtimeConfig:
    def __init__(self, api_key: str, realtime_url: str, realtime_proxy: str | bool | None):
        self.api_key = api_key
        self.realtime_url = realtime_url
        self.realtime_proxy = realtime_proxy


def _load_api_key_from_dotenv() -> str:
    for dotenv_path in (Path(BASE_DIR) / ".env", SOPHIA_FACE_HCI_DIR / ".env"):
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"OPENAI_API_KEY", "OpenAI_API_KEY"}:
                return value.strip().strip("'\"")
    return ""


def _resolve_realtime_url() -> str:
    realtime_url = os.getenv("OPENAI_REALTIME_URL") or os.getenv("OpenAI_REALTIME_URL")
    if realtime_url and realtime_url.strip():
        return realtime_url.strip()
    if DEFAULT_REALTIME_URL.strip():
        return DEFAULT_REALTIME_URL.strip()

    realtime_model = (
        os.getenv("OPENAI_REALTIME_MODEL", "").strip()
        or os.getenv("OpenAI_REALTIME_MODEL", "").strip()
        or DEFAULT_REALTIME_MODEL
    )
    return f"wss://api.openai.com/v1/realtime?model={quote(realtime_model, safe='')}"


def _resolve_realtime_proxy() -> str | bool | None:
    proxy = os.getenv("OPENAI_WS_PROXY")
    if proxy is None:
        proxy = os.getenv("OpenAI_WS_PROXY")
    if proxy is None:
        proxy = DEFAULT_REALTIME_PROXY
    if proxy is None:
        return None
    proxy = proxy.strip()
    return proxy or None


def _load_local_config() -> _RealtimeConfig:
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("OpenAI_API_KEY", "").strip()
        or DEFAULT_API_KEY.strip()
        or _load_api_key_from_dotenv()
    )
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Export OPENAI_API_KEY or add it to "
            f"{Path(BASE_DIR) / '.env'}."
        )
    return _RealtimeConfig(
        api_key=api_key,
        realtime_url=_resolve_realtime_url(),
        realtime_proxy=_resolve_realtime_proxy(),
    )


if USE_DIRECT_CONFIG:
    CONFIG = _load_local_config()
    CONFIG_SOURCE = "direct config in realtime_chat_nonverbal_from_txt.py"
else:
    try:
        from settings import load_config as _load_config_from_settings
    except ModuleNotFoundError as exc:
        if exc.name != "settings":
            raise
        CONFIG = _load_local_config()
        CONFIG_SOURCE = "direct config fallback because settings.py was not found"
    else:
        CONFIG = _load_config_from_settings()
        CONFIG_SOURCE = "settings.py"

URL = CONFIG.realtime_url
LOG_ALL_EVENTS = os.getenv("REALTIME_LOG_ALL_EVENTS", "1").strip().lower() in {"1", "true", "yes"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


INPUT_PATH = Path(
    os.getenv(
        "SOPHIA_NONVERBAL_INPUT_FILE",
        str(Path(BASE_DIR) / "input.txt"),
    )
)
CHAT_HISTORY_ENV = os.getenv("SOPHIA_CHAT_HISTORY_FILE")
if CHAT_HISTORY_ENV:
    CHAT_HISTORY_PATHS = [Path(CHAT_HISTORY_ENV)]
else:
    CHAT_HISTORY_PATHS = [
        Path(BASE_DIR).parent / "chat_history.json",
        Path(BASE_DIR).parent / "chat_history.jsonl",
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
LATENCY_PROFILING = os.getenv("SOPHIA_NONVERBAL_LATENCY", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LATENCY_LOG_PATH = Path(
    os.getenv(
        "SOPHIA_NONVERBAL_LATENCY_LOG",
        str(Path(BASE_DIR) / "motion_latency_log.json"),
    )
)
LATENCY_REPORT_PATH = Path(
    os.getenv(
        "SOPHIA_NONVERBAL_LATENCY_REPORT",
        str(Path(BASE_DIR) / "motion_latency_report.txt"),
    )
)

HEADERS = [
    "Authorization: Bearer " + CONFIG.api_key,
]
# ======================================================================

stop_event = threading.Event()
response_done = threading.Event()
response_done.set()
response_chunks = []
response_active = False
agent_stage = "idle"
current_request: MotionRequest | None = None
candidate_output = ""
current_latency: dict = {}


def load_prompt(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYS_PROMPT = load_prompt(PROMPT_PATH)


def latency_mark(name: str) -> None:
    if LATENCY_PROFILING:
        current_latency[name] = time.perf_counter()


def latency_elapsed(start_name: str, end_name: str) -> float | None:
    start = current_latency.get(start_name)
    end = current_latency.get(end_name)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return max(0.0, end - start)
    return None


def print_latency(label: str, start_name: str, end_name: str) -> None:
    return


def print_latency_summary() -> None:
    return


def latency_metrics() -> dict[str, float]:
    pairs = [
        ("extract_latest_conversation", "source_read_start", "source_read_done"),
        ("source_to_candidate_prompt_sent", "source_read_done", "candidate_prompt_sent"),
        ("candidate_first_token", "candidate_prompt_sent", "candidate_first_delta"),
        ("candidate_complete", "candidate_prompt_sent", "candidate_response_done"),
        ("judge_first_token", "judge_prompt_sent", "judge_first_delta"),
        ("judge_complete", "judge_prompt_sent", "judge_response_done"),
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


def append_latency_json(record: dict) -> None:
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


def append_latency_report(record: dict) -> None:
    metrics = record.get("metrics_sec") or {}
    LATENCY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "",
        "=" * 72,
        f"Turn {record.get('turn_id')} | {record.get('time')}",
        "-" * 72,
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


def _model_name_from_url(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    values = query.get("model")
    return values[0] if values else ""


def _websocket_proxy_kwargs(proxy: str | bool | None) -> dict:
    if not proxy:
        return {}
    if proxy is True:
        proxy = "http://127.0.0.1:7897"
    parsed = urlparse(str(proxy))
    if not parsed.scheme or not parsed.hostname:
        raise RuntimeError(f"Invalid realtime proxy setting: {proxy!r}")
    result = {
        "http_proxy_host": parsed.hostname,
        "http_proxy_port": parsed.port or (443 if parsed.scheme == "https" else 80),
        "proxy_type": parsed.scheme,
    }
    if parsed.username:
        result["http_proxy_auth"] = (
            parsed.username,
            parsed.password or "",
        )
    return result


def send_text_message(ws, text: str):
    """Send one user message to Realtime and request a text response."""
    ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    )

    ws.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {"output_modalities": ["text"]},
            }
        )
    )


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


def start_motion_turn(
    ws,
    raw_text: str,
    duration_hint: float | None = None,
    source_latency: dict | None = None,
):
    global agent_stage, current_request, candidate_output, current_latency
    request = parse_motion_request(raw_text)
    if not request.spoken_text:
        print("Empty motion request; ignoring.", flush=True)
        response_done.set()
        return

    duration = request.speech_duration_sec or duration_hint
    request = MotionRequest(request.spoken_text, duration)
    duration = effective_speech_duration(request)
    current_request = MotionRequest(request.spoken_text, duration)
    candidate_output = ""
    current_latency = {
        "turn_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
        "spoken_text": request.spoken_text,
        "spoken_chars": len(request.spoken_text),
        "speech_duration_sec": duration,
    }
    if source_latency:
        current_latency.update(source_latency)
        print_latency("extract latest conversation", "source_read_start", "source_read_done")
    latency_mark("turn_start")
    agent_stage = "candidates"
    latency_mark("candidate_prompt_build_start")
    prompt = build_candidate_prompt(request.spoken_text, duration)
    current_latency["candidate_prompt_chars"] = len(prompt)
    latency_mark("candidate_prompt_built")
    print(f"[Agent] planning for {duration:.2f}s spoken text", flush=True)
    print_latency("candidate prompt build", "candidate_prompt_build_start", "candidate_prompt_built")
    latency_mark("candidate_prompt_send_start")
    send_text_message(ws, prompt)
    latency_mark("candidate_prompt_sent")
    print_latency("candidate prompt send", "candidate_prompt_send_start", "candidate_prompt_sent")


def finalize_response_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def extract_text_from_response_done(event: dict) -> str:
    response = event.get("response") or {}
    output = response.get("output") or []
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            for key in ("text", "transcript"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value)
    return "\n".join(chunks).strip()


def write_actions_file(output_text: str):
    print(f"Writing actions ({len(output_text)} chars) to {ACTIONS_PATH}", flush=True)
    with open(ACTIONS_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)
        if output_text and not output_text.endswith("\n"):
            f.write("\n")

    print(f"Saved actions to {ACTIONS_PATH}", flush=True)


def run_move_sender():

    if APPEND_RANDOM_GESTURE:
        try:
            subprocess.run(
                [sys.executable, "generating_random_gesture.py"],
                cwd=BASE_DIR,
                text=True,
                check=False,
            )
        except Exception as e:
            print(f"Failed to run generating_random_gesture.py: {e}", flush=True)

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
    except Exception as e:
        print(f"Failed to run {sender_name}: {e}", flush=True)
        return

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        print(f"{sender_name} exited with code {result.returncode}", flush=True)


def handle_output(output_text: str, speech_duration_sec: float | None):
    latency_mark("normalize_start")
    normalized = normalize_action_output(output_text, speech_duration_sec)
    latency_mark("normalize_done")
    print_latency("normalize output", "normalize_start", "normalize_done")
    if normalized != output_text.strip():
        print("Normalized model output to valid action pairs.", flush=True)
        print(normalized, flush=True)
    latency_mark("write_actions_start")
    write_actions_file(normalized)
    latency_mark("write_actions_done")
    print_latency("write actions file", "write_actions_start", "write_actions_done")
    latency_mark("motion_sender_start")
    run_move_sender()
    latency_mark("motion_sender_done")
    print_latency(
        "motion sender subprocess (includes action hold time)",
        "motion_sender_start",
        "motion_sender_done",
    )


def _text_from_content(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_text_from_content(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "message", "content", "answer", "response", "utterance"):
            text = _text_from_content(value.get(key))
            if text:
                return text
    return ""


def _role_from_item(item: dict) -> str:
    for key in ("role", "speaker", "sender", "author", "from"):
        value = item.get(key)
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, dict):
            for nested_key in ("role", "name", "type"):
                nested = value.get(nested_key)
                if isinstance(nested, str):
                    return nested.strip().lower()
    return ""


def _walk_history_json(data):
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


def _read_history_items(path: Path) -> list[dict]:
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        items: list[dict] = []
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
        return items

    return [item for item in _walk_history_json(parsed) if isinstance(item, dict)]


def extract_latest_ai_text_from_history(path: Path) -> tuple[str, tuple]:
    """Return the newest AI/robot utterance from a JSON or JSONL chat history."""
    latest_item: dict | None = None
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
    """Mirror extracted robot speech to input.txt for visibility and debugging."""
    try:
        old_text = INPUT_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        old_text = ""

    if old_text == text.strip():
        return
    INPUT_PATH.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Extracted latest AI text to {INPUT_PATH}", flush=True)


def read_motion_source() -> tuple[str, tuple]:
    """Read either a chat-history JSON/JSONL file or plain input.txt."""
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
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        content = file.read().strip()
    return content, ("input", stat.st_mtime_ns, stat.st_size)


def file_input_loop(ws):
    """Poll Sophia's latest spoken answer and plan motion when it updates."""
    print(
        "Watching chat-history files when present: "
        + ", ".join(str(path) for path in CHAT_HISTORY_PATHS)
    )
    print(f"Watching/extracting spoken-answer file: {INPUT_PATH}")
    print(f"Watching duration hint: {DURATION_PATH}")

    last_sent_signature: tuple | None = None
    last_empty_signature: tuple | None = None
    missing_reported = False

    while not stop_event.is_set():
        response_done.wait()

        try:
            source_latency = {"source_read_start": time.perf_counter()}
            content, signature = read_motion_source()
            source_latency["source_read_done"] = time.perf_counter()
            source_latency["motion_source"] = signature[0] if signature else "unknown"
            if len(signature) > 1 and isinstance(signature[1], str):
                source_latency["motion_source_path"] = signature[1]
            elif signature and signature[0] == "input":
                source_latency["motion_source_path"] = str(INPUT_PATH)
            missing_reported = False
        except FileNotFoundError as e:
            if not missing_reported:
                print(str(e), flush=True)
                missing_reported = True
            stop_event.wait(0.5)
            continue
        except Exception as e:
            print(f"Failed to read motion input source: {e}", flush=True)
            stop_event.wait(0.5)
            continue

        # ✅ 空文件 → 不发送
        if not content:
            if signature != last_empty_signature:
                print(
                    "Chat history is present, but no latest ai/assistant/robot text "
                    "was extracted yet; waiting for robot reply.",
                    flush=True,
                )
                last_empty_signature = signature
            stop_event.wait(0.5)
            continue

        if signature == last_sent_signature:
            stop_event.wait(0.5)
            continue

        duration_hint = read_duration_hint()
        if duration_hint:
            print(f"\n[Spoken Answer] duration={duration_hint:.2f}s\n{content}\n", flush=True)
        else:
            print(f"\n[Spoken Answer] duration=estimated\n{content}\n", flush=True)

        response_done.clear()
        response_chunks.clear()

        try:
            start_motion_turn(
                ws,
                content,
                duration_hint=duration_hint,
                source_latency=source_latency,
            )
            last_sent_signature = signature
            print("Sent spoken answer to realtime motion agent.", flush=True)
        except Exception as e:
            print(f"\nSend failed: {e}\n", flush=True)
            response_done.set()

        # 控制轮询频率（避免CPU占用）
        stop_event.wait(0.5)


def on_open(ws):
    print("WS connected. Entering text mode...")
    print(f"[{datetime.now().isoformat(timespec='seconds')}] sending session.update", flush=True)
    try:
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": SYS_PROMPT,
                        "output_modalities": ["text"],
                    },
                }
            )
        )
    except Exception as e:
        print(f"Failed to send session.update: {e}", flush=True)
        stop_event.set()
        response_done.set()
        ws.close()
        return

    threading.Thread(target=file_input_loop, args=(ws,), daemon=True).start()


def on_message(ws, message):
    global response_active, agent_stage, current_request, candidate_output
    data = json.loads(message)
    t = data.get("type")
    if LOG_ALL_EVENTS:
        print(f"[event] {t}", flush=True)

    if t in {"response.output_text.delta", "response.text.delta"}:
        response_active = True
        delta = data.get("delta", "")
        response_chunks.append(delta)
        if agent_stage == "candidates" and "candidate_first_delta" not in current_latency:
            latency_mark("candidate_first_delta")
            print_latency("candidate first token", "candidate_prompt_sent", "candidate_first_delta")
        elif agent_stage == "judge" and "judge_first_delta" not in current_latency:
            latency_mark("judge_first_delta")
            print_latency("judge first token", "judge_prompt_sent", "judge_first_delta")

    elif t == "response.done":
        response_active = False
        full_text = "".join(response_chunks).strip()
        if not full_text:
            full_text = extract_text_from_response_done(data)
        output_text = finalize_response_text(full_text)
        response_chunks.clear()

        if agent_stage == "candidates" and current_request is not None:
            latency_mark("candidate_response_done")
            current_latency["candidate_output_chars"] = len(output_text)
            print_latency("candidate complete", "candidate_prompt_sent", "candidate_response_done")
            candidate_output = output_text
            print("[Agent] planner candidates:", flush=True)
            print(candidate_output, flush=True)
            agent_stage = "judge"
            latency_mark("judge_prompt_build_start")
            judge_prompt = build_judge_prompt(
                current_request.spoken_text,
                effective_speech_duration(current_request),
                candidate_output,
            )
            current_latency["judge_prompt_chars"] = len(judge_prompt)
            latency_mark("judge_prompt_built")
            print_latency("judge prompt build", "judge_prompt_build_start", "judge_prompt_built")
            try:
                latency_mark("judge_prompt_send_start")
                send_text_message(ws, judge_prompt)
                latency_mark("judge_prompt_sent")
                print_latency("judge prompt send", "judge_prompt_send_start", "judge_prompt_sent")
            except Exception as e:
                print(f"\nJudge send failed: {e}\n", flush=True)
                agent_stage = "idle"
                response_done.set()
            return

        if agent_stage == "judge":
            latency_mark("judge_response_done")
            current_latency["judge_output_chars"] = len(output_text)
            print_latency("judge complete", "judge_prompt_sent", "judge_response_done")

        print("[Agent] selected motion:", flush=True)
        print(output_text, flush=True)
        if not output_text:
            print("Realtime returned an empty text response; actions.txt will use fallback motion.", flush=True)
        speech_duration = current_request.speech_duration_sec if current_request else None
        handle_output(output_text, speech_duration)
        latency_mark("turn_done")
        print_latency_summary()
        write_latency_record()
        agent_stage = "idle"
        current_request = None
        candidate_output = ""
        response_done.set()

    elif t == "error":
        print("\nserver error:")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        response_active = False
        agent_stage = "idle"
        response_done.set()
    elif t and "error" in t:
        print("\nserver reported nonstandard error event:")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        if data.get("code") == "insufficient_permissions":
            print(
                "\nHint: your API key cannot access this model. "
                "Try another realtime model with OPENAI_REALTIME_MODEL, "
                "or enable this model in your provider console.",
                flush=True,
            )
            print(
                "Example: OPENAI_REALTIME_MODEL=<allowed_realtime_model> "
                "OPENAI_WS_PROXY=http://127.0.0.1:7897 "
                "python realtime_chat_nonverbal_from_txt.py",
                flush=True,
            )
        response_active = False
        agent_stage = "idle"
        response_done.set()
    elif LOG_ALL_EVENTS:
        # Keep unknown event payload visible for protocol mismatch debugging.
        # print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        print("ok")


def on_error(ws, err):
    global agent_stage
    print(f"Error: {type(err).__name__}: {err}", flush=True)
    agent_stage = "idle"
    response_done.set()


def on_close(ws, code, reason):
    global agent_stage
    print(f"WS closed ({code}/{reason})", flush=True)
    print(
        f"Close diagnostics: response_active={response_active}, "
        f"thread={threading.current_thread().name}",
        flush=True,
    )
    if code is None and reason is None:
        print(
            "No close frame received (abnormal close). "
            "Most common causes: local proxy reset, gateway/protocol mismatch, or TLS/network interruption.",
            flush=True,
        )
    stop_event.set()
    agent_stage = "idle"
    response_done.set()


if __name__ == "__main__":
    websocket.enableTrace(False)

    proxy_kwargs = _websocket_proxy_kwargs(CONFIG.realtime_proxy)

    print("starting websocket...")
    print(f"CONFIG = {CONFIG_SOURCE}")
    print(f"URL    = {URL}")
    print(f"MODEL  = {_model_name_from_url(URL)}")
    print(f"PROMPT = {PROMPT_PATH}")
    print(f"ACTIONS = {ACTIONS_PATH}")
    print(f"INPUT  = {INPUT_PATH}")
    print(f"CHAT_HISTORY = {', '.join(str(path) for path in CHAT_HISTORY_PATHS)}")
    print(f"DURATION = {DURATION_PATH}")
    print(f"APPEND_RANDOM_GESTURE = {APPEND_RANDOM_GESTURE}")
    print(f"MOTION_SENDER = {MOTION_SENDER}")
    print(f"MOTION_DURATION_SCALE = {MOTION_DURATION_SCALE}")
    if MOTION_SENDER in {"scalar", "scalar_index", "motor_index", "index"}:
        print(f"SCALAR_ROBOT = {SCALAR_ROBOT_HOST}:{SCALAR_ROBOT_PORT}")
    elif MOTION_SENDER in {"standard", "standard_index"}:
        print(f"STANDARD_ROBOT = {STANDARD_ROBOT_HOST}:{STANDARD_ROBOT_PORT}")
    elif MOTION_SENDER == "direct":
        print(f"DIRECT_ROBOT = {DIRECT_ROBOT_HOST}:{DIRECT_ROBOT_PORT}")
    else:
        print(f"LEGACY_ROBOT = {LEGACY_ROBOT_HOST}:{LEGACY_ROBOT_PORT}")
    if CONFIG.realtime_proxy:
        print(f"PROXY = {CONFIG.realtime_proxy}")
    else:
        print("PROXY = disabled (direct connection)")

    ws = websocket.WebSocketApp(
        URL,
        header=HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    try:
        run_kwargs = {
            "sslopt": {
                "cert_reqs": ssl.CERT_REQUIRED,
                "ca_certs": certifi.where(),
            },
            "ping_interval": 20,
            "ping_timeout": 10,
        }
        run_kwargs.update(proxy_kwargs)
        ws.run_forever(**run_kwargs)
    except KeyboardInterrupt:
        print("\nmanual exit...")
        stop_event.set()
        ws.close()

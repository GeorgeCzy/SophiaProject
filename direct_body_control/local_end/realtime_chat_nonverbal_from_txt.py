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
from urllib.parse import parse_qs, urlparse

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

from settings import load_config

CONFIG = load_config()
URL = CONFIG.realtime_url
LOG_ALL_EVENTS = os.getenv("REALTIME_LOG_ALL_EVENTS", "1").strip().lower() in {"1", "true", "yes"}
INPUT_PATH = Path(
    os.getenv(
        "SOPHIA_NONVERBAL_INPUT_FILE",
        str(Path(BASE_DIR) / "input.txt"),
    )
)
CHAT_HISTORY_PATH = Path(
    os.getenv(
        "SOPHIA_CHAT_HISTORY_FILE",
        str(Path(BASE_DIR) / "chat_history.jsonl"),
    )
)
APPEND_RANDOM_GESTURE = os.getenv("SOPHIA_NONVERBAL_APPEND_RANDOM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MOTION_SENDER = os.getenv("SOPHIA_MOTION_SENDER", "scalar_index").strip().lower()
SCALAR_ROBOT_HOST = os.getenv("SOPHIA_SCALAR_ROBOT_HOST", "10.0.0.10")
SCALAR_ROBOT_PORT = int(os.getenv("SOPHIA_SCALAR_ROBOT_PORT", "5005"))
STANDARD_ROBOT_HOST = os.getenv("SOPHIA_STANDARD_ROBOT_HOST", "10.0.0.10")
STANDARD_ROBOT_PORT = int(os.getenv("SOPHIA_STANDARD_ROBOT_PORT", "5005"))
DIRECT_ROBOT_HOST = os.getenv("SOPHIA_DIRECT_ROBOT_HOST", "10.0.0.10")
DIRECT_ROBOT_PORT = int(os.getenv("SOPHIA_DIRECT_ROBOT_PORT", "5006"))
LEGACY_ROBOT_HOST = os.getenv("SOPHIA_LEGACY_ROBOT_HOST", "10.0.0.10")
LEGACY_ROBOT_PORT = int(os.getenv("SOPHIA_LEGACY_ROBOT_PORT", "5005"))

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


def load_prompt(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYS_PROMPT = load_prompt(PROMPT_PATH)


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


def start_motion_turn(ws, raw_text: str, duration_hint: float | None = None):
    global agent_stage, current_request, candidate_output
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
    agent_stage = "candidates"
    prompt = build_candidate_prompt(request.spoken_text, duration)
    print(f"[Agent] planning for {duration:.2f}s spoken text", flush=True)
    send_text_message(ws, prompt)


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

    wait_start = time.time()
    wait_reported = False
    while not (SYNC_DIR / "audio_response.ready").exists():
        if not wait_reported and time.time() - wait_start > 2.0:
            print(
                f"Waiting for {SYNC_DIR / 'audio_response.ready'} before sending motions...",
                flush=True,
            )
            wait_reported = True
        if time.time() - wait_start > 30.0:
            print(
                "Timed out waiting for audio_response.ready; actions.txt was written, "
                "but the motion sender was not run.",
                flush=True,
            )
            return
        time.sleep(0.1)
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
    normalized = normalize_action_output(output_text, speech_duration_sec)
    if normalized != output_text.strip():
        print("Normalized model output to valid action pairs.", flush=True)
        print(normalized, flush=True)
    write_actions_file(normalized)
    run_move_sender()


def extract_latest_ai_text_from_history(path: Path) -> tuple[str, tuple]:
    """Return the newest AI/robot utterance from a JSONL chat history."""
    latest_item: dict | None = None
    latest_line_no = 0

    with path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = str(item.get("role", "")).strip().lower()
            text = str(item.get("text", "")).strip()
            if role in {"ai", "assistant", "robot"} and text:
                latest_item = item
                latest_line_no = line_no

    if not latest_item:
        return "", ("chat_history", 0)

    text = str(latest_item.get("text", "")).strip()
    signature = (
        "chat_history",
        latest_item.get("time", ""),
        latest_item.get("sequence", ""),
        latest_line_no,
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
    """Read either chat_history.jsonl or plain input.txt."""
    if CHAT_HISTORY_PATH.exists():
        content, signature = extract_latest_ai_text_from_history(CHAT_HISTORY_PATH)
        if content:
            write_extracted_input_text(content)
        return content, signature

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file does not exist yet: {INPUT_PATH}")

    stat = INPUT_PATH.stat()
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        content = file.read().strip()
    return content, ("input", stat.st_mtime_ns, stat.st_size)


def file_input_loop(ws):
    """Poll Sophia's latest spoken answer and plan motion when it updates."""
    print(f"Watching chat-history file when present: {CHAT_HISTORY_PATH}")
    print(f"Watching/extracting spoken-answer file: {INPUT_PATH}")
    print(f"Watching duration hint: {DURATION_PATH}")

    last_sent_signature: tuple | None = None
    missing_reported = False

    while not stop_event.is_set():
        response_done.wait()

        try:
            content, signature = read_motion_source()
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
            start_motion_turn(ws, content, duration_hint=duration_hint)
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

    elif t == "response.done":
        response_active = False
        full_text = "".join(response_chunks).strip()
        if not full_text:
            full_text = extract_text_from_response_done(data)
        output_text = finalize_response_text(full_text)
        response_chunks.clear()

        if agent_stage == "candidates" and current_request is not None:
            candidate_output = output_text
            print("[Agent] planner candidates:", flush=True)
            print(candidate_output, flush=True)
            agent_stage = "judge"
            judge_prompt = build_judge_prompt(
                current_request.spoken_text,
                effective_speech_duration(current_request),
                candidate_output,
            )
            try:
                send_text_message(ws, judge_prompt)
            except Exception as e:
                print(f"\nJudge send failed: {e}\n", flush=True)
                agent_stage = "idle"
                response_done.set()
            return

        print("[Agent] selected motion:", flush=True)
        print(output_text, flush=True)
        if not output_text:
            print("Realtime returned an empty text response; actions.txt will use fallback motion.", flush=True)
        speech_duration = current_request.speech_duration_sec if current_request else None
        handle_output(output_text, speech_duration)
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
    print(f"URL    = {URL}")
    print(f"MODEL  = {_model_name_from_url(URL)}")
    print(f"PROMPT = {PROMPT_PATH}")
    print(f"ACTIONS = {ACTIONS_PATH}")
    print(f"INPUT  = {INPUT_PATH}")
    print(f"CHAT_HISTORY = {CHAT_HISTORY_PATH}")
    print(f"DURATION = {DURATION_PATH}")
    print(f"APPEND_RANDOM_GESTURE = {APPEND_RANDOM_GESTURE}")
    print(f"MOTION_SENDER = {MOTION_SENDER}")
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

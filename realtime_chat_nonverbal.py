import json
import os
import subprocess
import ssl
import threading

import certifi
import websocket


# ===== Settings ==========================================================
API_KEY = os.getenv("OMG_API_KEY")
if not API_KEY:
    raise RuntimeError("invalid API key")

MODEL = "gpt-realtime-mini"
URL = f"wss://api.ohmygpt.com/v1/realtime?model={MODEL}"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
ACTIONS_PATH = os.path.join(BASE_DIR, "actions.txt")

HEADERS = [
    "Authorization: Bearer " + API_KEY,
    "OpenAI-Beta: realtime=v1",
]
# ======================================================================

stop_event = threading.Event()
response_done = threading.Event()
response_done.set()
response_chunks = []


def load_prompt(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYS_PROMPT = load_prompt(PROMPT_PATH)


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
                "response": {"modalities": ["text"]},
            }
        )
    )


def finalize_response_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped

    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def write_actions_file(output_text: str):
    with open(ACTIONS_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)
        if output_text and not output_text.endswith("\n"):
            f.write("\n")

    print(f"Saved actions to {ACTIONS_PATH}", flush=True)


def run_move_sender():
    try:
        result = subprocess.run(
            ["python", "llm_move_sender.py", "--input-file", "actions.txt"],
            cwd=BASE_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as e:
        print(f"Failed to run llm_move_sender.py: {e}", flush=True)
        return

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        print(f"llm_move_sender exited with code {result.returncode}", flush=True)


def handle_output(output_text: str):
    write_actions_file(output_text)
    run_move_sender()


def keyboard_input_loop(ws):
    """Keyboard input loop."""
    print("connected. Type input now.")
    print("Type scene description and press Enter; type exit/quit to stop.\n")

    while not stop_event.is_set():
        response_done.wait()
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "quit"

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Exiting...")
            stop_event.set()
            ws.close()
            break

        response_done.clear()
        response_chunks.clear()
        try:
            send_text_message(ws, user_text)
        except Exception as e:
            print(f"\nSend failed: {e}\n", flush=True)
            response_done.set()


def on_open(ws):
    print("WS connected. Entering text mode...")

    ws.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "instructions": SYS_PROMPT,
                },
            }
        )
    )

    threading.Thread(target=keyboard_input_loop, args=(ws,), daemon=True).start()


def on_message(ws, message):
    data = json.loads(message)
    t = data.get("type")

    if t == "response.text.delta":
        delta = data.get("delta", "")
        response_chunks.append(delta)

    elif t == "response.done":
        full_text = "".join(response_chunks).strip()
        output_text = finalize_response_text(full_text)
        print("LLM:", flush=True)
        print(output_text, flush=True)
        handle_output(output_text)
        response_done.set()

    elif t == "error":
        print("\nserver error:")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        response_done.set()


def on_error(ws, err):
    print(f"Error: {err}")
    response_done.set()


def on_close(ws, code, reason):
    print(f"WS closed ({code}/{reason})")
    stop_event.set()
    response_done.set()


if __name__ == "__main__":
    websocket.enableTrace(False)

    print("starting websocket...")
    print(f"URL    = {URL}")
    print(f"MODEL  = {MODEL}")
    print(f"PROMPT = {PROMPT_PATH}")
    print(f"ACTIONS = {ACTIONS_PATH}")

    ws = websocket.WebSocketApp(
        URL,
        header=HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    try:
        ws.run_forever(
            sslopt={
                "cert_reqs": ssl.CERT_REQUIRED,
                "ca_certs": certifi.where(),
            },
            http_proxy_host="127.0.0.1",
            http_proxy_port=7897,
            proxy_type="socks5h",
            ping_interval=20,
            ping_timeout=10,
        )
    except KeyboardInterrupt:
        print("\nmanual exit...")
        stop_event.set()
        ws.close()

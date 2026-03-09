import argparse
import json
import os
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

HEADERS = [
    "Authorization: Bearer " + API_KEY,
    "OpenAI-Beta: realtime=v1",
]
# ======================================================================

stop_event = threading.Event()
response_done = threading.Event()
response_done.set()
response_chunks = []
single_input_text = None


def load_prompt(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYS_PROMPT = load_prompt(PROMPT_PATH)


def send_text_message(ws, text: str):
    """把一条用户文本消息发送给 Realtime 会话，并请求模型生成回复。"""
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


def keyboard_input_loop(ws):
    """键盘输入线程。"""
    print("✅ 已连接。现在可以键盘输入。")
    print("输入场景描述后按回车发送；输入 exit 或 quit 退出。\n")

    while not stop_event.is_set():
        response_done.wait()
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_text = "quit"

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("正在退出...")
            stop_event.set()
            ws.close()
            break

        response_done.clear()
        response_chunks.clear()
        try:
            send_text_message(ws, user_text)
        except Exception as e:
            print(f"\n发送失败: {e}\n", flush=True)
            response_done.set()


def on_open(ws):
    print("✅ WS connected. 进入 text 模态 ...")

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

    if single_input_text is not None:
        response_done.clear()
        response_chunks.clear()
        print(f"单次测试输入: {single_input_text}")
        send_text_message(ws, single_input_text)
        return

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
        response_done.set()
        if single_input_text is not None:
            stop_event.set()
            ws.close()

    elif t == "error":
        print("\n❌ 服务端错误：")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
        response_done.set()
        if single_input_text is not None:
            stop_event.set()
            ws.close()


def on_error(ws, err):
    print(f"❌ {err}")
    response_done.set()


def on_close(ws, code, reason):
    print(f"WS closed ({code}/{reason})")
    stop_event.set()
    response_done.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        help="Send one scene description and exit after the response.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    single_input_text = args.once

    websocket.enableTrace(False)

    print("starting websocket...")
    print(f"URL   = {URL}")
    print(f"MODEL = {MODEL}")
    print(f"PROMPT = {PROMPT_PATH}")

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
        print("\n手动退出 ...")
        stop_event.set()
        ws.close()
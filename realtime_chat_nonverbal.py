import os
import ssl
import json
import threading
import websocket
import certifi
import sys

# ===== Settings ==========================================================
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("invalid OPENAI_API_KEY")

MODEL = "gpt-realtime-mini"
URL = f"wss://api.ohmygpt.com/v1/realtime?model={MODEL}"

HEADERS = [
    "Authorization: Bearer " + API_KEY,
    "OpenAI-Beta: realtime=v1"
]
# ======================================================================

stop_event = threading.Event()
response_lock = threading.Lock()


def send_text_message(ws, text: str):
    """
    把一条用户文本消息发送给 Realtime 会话，并请求模型生成回复。
    """
    # 1) 把用户消息加入会话
    ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text
                }
            ]
        }
    }))

    # 2) 请求模型生成回复
    ws.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["text"]
        }
    }))


def keyboard_input_loop(ws):
    """
    键盘输入线程：
    - 输入普通文本：发送给 LLM
    - 输入 exit / quit：退出程序
    """
    print("✅ 已连接。现在可以键盘输入。")
    print("输入内容后按回车发送；输入 exit 或 quit 退出。\n")

    while not stop_event.is_set():
        try:
            user_text = input("你: ").strip()
        except EOFError:
            user_text = "quit"
        except KeyboardInterrupt:
            user_text = "quit"

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("正在退出...")
            stop_event.set()
            ws.close()
            break

        with response_lock:
            print("LLM: ", end="", flush=True)
            try:
                send_text_message(ws, user_text)
            except Exception as e:
                print(f"\n发送失败: {e}\n", flush=True)


def on_open(ws):
    print("✅ WS connected. 进入 text 模态 …")

    # 更新 session，声明使用 text
    ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "modalities": ["text"]
        }
    }))

    threading.Thread(target=keyboard_input_loop, args=(ws,), daemon=True).start()


def on_message(ws, message):
    data = json.loads(message)
    t = data.get("type")

    if t == "response.text.delta":
        sys.stdout.write(data.get("delta", ""))
        sys.stdout.flush()

    elif t == "response.done":
        print("\n回复结束。\n", flush=True)

    elif t == "error":
        print("\n❌ 服务端错误：")
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)

    else:
        # 调试时可以打开
        # print("DEBUG EVENT:", json.dumps(data, ensure_ascii=False))
        pass


def on_error(ws, err):
    print(f"❌ {err}")


def on_close(ws, code, reason):
    print(f"WS closed ({code}/{reason})")
    stop_event.set()


if __name__ == "__main__":
    websocket.enableTrace(False)

    print("starting websocket...")
    print(f"URL   = {URL}")
    print(f"MODEL = {MODEL}")

    ws = websocket.WebSocketApp(
        URL,
        header=HEADERS,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    try:
        ws.run_forever(
            sslopt={
                "cert_reqs": ssl.CERT_REQUIRED,
                "ca_certs": certifi.where()
            },
            http_proxy_host="127.0.0.1",
            http_proxy_port=7897,
            proxy_type="socks5h",
            ping_interval=20,
            ping_timeout=10
        )
    except KeyboardInterrupt:
        print("\n手动退出 …")
        stop_event.set()
        ws.close()
import os, ssl, json, time, base64, queue, threading
import numpy as np
import sounddevice as sd
import websocket, certifi, sys
import wave, datetime


# ===== Settings ==========================================================
API_KEY = os.getenv("OPENAI_API_KEY", API_KEY)
# MODEL   = "gpt-4o-realtime-preview-2024-12-17"
MODEL = "gpt-realtime-mini"
# URL  = f"wss://cfwus02.opapi.win/v1/realtime?model={MODEL}"
URL = f"wss://api.ohmygpt.com/v1/realtime?model={MODEL}"

HEADERS      = [
    "Authorization: Bearer " + API_KEY,
    "OpenAI-Beta: realtime=v1"
]

SAMPLE_RATE  = 24_000      
CHUNK_MS     = 500          
BYTES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)  
MIN_SAMPLES = int(SAMPLE_RATE * 0.1)  
# ======================================================================

captions = []
audio_chunks: list[bytes] = []           
reply_index        = 0                   


play_stream = sd.RawOutputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='int16',
    blocksize=0 
)
play_stream.start()

def float32_to_b64(float32_array: np.ndarray) -> str:
    int16 = np.clip(float32_array, -1.0, 1.0) * 32767
    pcm   = int16.astype("<i2").tobytes()        
    return base64.b64encode(pcm).decode("ascii")


mic_queue: queue.Queue[np.ndarray] = queue.Queue()

def record_audio():
    def callback(indata, frames, t, status):
        if status: 
            print("⚠️", status, file=sys.stderr)
        mic_queue.put(indata.copy())             

    with sd.InputStream(channels=1,
                        samplerate=SAMPLE_RATE,
                        dtype='float32',
                        blocksize=BYTES_PER_CHUNK,
                        callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)  

def save_wav():
    global audio_chunks, reply_index
    if not audio_chunks:          
        return

    fname = f"assistant_reply_{reply_index:03d}.wav"
    pcm_data = b"".join(audio_chunks)
    with wave.open(fname, "wb") as wf:
        wf.setnchannels(1)        # 单声道
        wf.setsampwidth(2)        # 16-bit -> 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    print(f"\n 已保存到 {fname}")
    reply_index  += 1
    audio_chunks.clear()

            
def on_open(ws):
    print("✅ WS connected. 进入 audio+text 模态 …")

    ws.send(json.dumps({
        "type": "session.update",
        "session": { "modalities": ['text','audio'] }
    }))

    threading.Thread(target=record_audio, daemon=True).start()

    def _uploader():
        THRESH      = 0.008      
        HANG_MS     = 300              
        HANG_SAMPLES = int(SAMPLE_RATE * HANG_MS / 1000)

        buf_samples = 0                
        last_voice  = time.time()      

        while not stop_event.is_set():
            try:
                chunk = mic_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = np.sqrt(np.mean(chunk**2))
            voiced = rms > THRESH

            if voiced:
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": float32_to_b64(chunk.flatten())
                }))
                buf_samples += len(chunk)
                last_voice = time.time()

            else:
                
                print("no voice detected")

                silence_len = time.time() - last_voice
                if buf_samples >= MIN_SAMPLES and silence_len * SAMPLE_RATE >= HANG_SAMPLES:
                    ws.send(json.dumps({ "type": "input_audio_buffer.commit" }))
                    buf_samples = 0    


    threading.Thread(target=_uploader, daemon=True).start()

def on_message(ws, message):
    data = json.loads(message)
    t    = data.get("type")

    if t == "response.audio.delta":
        pcm = base64.b64decode(data["delta"])
        play_stream.write(pcm)
        audio_chunks.append(pcm)  

    elif t == "response.text.delta":
        sys.stdout.write(data["delta"])
        sys.stdout.flush()
        
    elif t == "response.audio_transcript.delta":
        word = data["delta"]
        captions.append(word)
        line = "".join(captions)
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
    
    elif t == "response.audio_transcript.done":
        captions.clear()

    elif t == "response.done":
        save_wav()
        print("\n 回复结束。\n")
        
    elif t == "input_audio_buffer.speech_started":
        print("\n 语音开始 …", end="", flush=True)
        
    elif t == "input_audio_buffer.speech_ended":
        print("\n 语音结束。")






def on_error(ws, err):
    print("❌", err)

def on_close(ws, code, reason):
    print(f" WS closed ({code}/{reason})")
    stop_event.set()
    play_stream.stop(); play_stream.close()

if __name__ == "__main__":
    stop_event = threading.Event()
    websocket.enableTrace(False)

    print("starting websocket...")
    print(f"URL = {URL}")
    print(f"MODEL = {MODEL}")

    ws = websocket.WebSocketApp(
        URL,
        header=HEADERS,
        on_open   = on_open,
        on_message= on_message,
        on_error  = on_error,
        on_close  = on_close
    )

    try:
        ws.run_forever(
            sslopt={
                "cert_reqs": ssl.CERT_REQUIRED,
                "ca_certs": certifi.where()
            },


            http_proxy_host = "127.0.0.1",
            http_proxy_port = "7897",
            proxy_type = "socks5h",
            ping_interval=20,
            ping_timeout=10
        )
        
    except KeyboardInterrupt:
        print("\n 手动退出 …")
        stop_event.set()
        ws.close()

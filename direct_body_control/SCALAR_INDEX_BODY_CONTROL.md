# Simple Scalar Index Body Control

This is the simplified replacement for:

```text
bodycontrol_tcp_standard.py + realtime_chat_nonverbal_from_txt.py
```

The local TCP client stays unchanged: `Sophia_control.py`.

## Protocol

`Sophia_control.py` already sends:

```json
{"index":0,"value":-1.2217}
```

The new robot-end bridge interprets it literally:

```text
one index -> one motor -> one scalar radian value
```

No `[x, y, z]`, no axis-angle, no SMPL-X mapping.

## Robot End

Copy only this file to the robot:

```text
bodycontrol_tcp_scalar_index.py
```

Run it on the same port used by `Sophia_control.py`:

```bash
python3 bodycontrol_tcp_scalar_index.py --port 5005
```

If Sophia's real motor IDs are different, edit `MOTOR_INDEX_TO_ACTUATOR` near
the top of `bodycontrol_tcp_scalar_index.py`. The same table must match
`llm_move_sender.py` on the local end.

If the chat history file is stored on the robot, run this robot-end helper in a
second terminal so the local motion agent can read the newest conversation:

```bash
python3 sync_chat_history_to_local.py \
  --source ../chat_history.json \
  --dest ywguo@10.0.0.111:/home/ywguo/Documents/Sophia_VLA/chat_history.json
```

Edit the `--source` path if the robot writes `chat_history.json` somewhere
else. Edit the `--dest` host/path to match the local machine running
`realtime_chat_nonverbal_from_txt.py`.

For quick testing without SSH keys, the helper can use `sshpass` with a local
password file. On the first run, if `sync_password.txt` is missing, the helper
asks for the SSH password once and saves it locally:

```bash
python3 sync_chat_history_to_local.py
```

After that, run sync normally and it will autofill from `sync_password.txt`:

```bash
python3 sync_chat_history_to_local.py \
  --source ../chat_history.json \
  --dest ywguo@10.0.0.111:/home/ywguo/Documents/Sophia_VLA/chat_history.json
```

## Local End

Copy or keep these files together locally, in the same folder:

```text
realtime_chat_nonverbal_from_txt.py
nonverbal_motion_agent.py
system_prompt.txt
motion_repo.py
llm_move_sender.py
Sophia_control.py
extract_input_from_chat_history.py
input.txt
actions.txt
```

Run:

```powershell
$env:SOPHIA_MOTION_SENDER="scalar_index"
$env:SOPHIA_SCALAR_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_SCALAR_ROBOT_PORT="5005"
python realtime_chat_nonverbal_from_txt.py
```

By default, `realtime_chat_nonverbal_from_txt.py` watches `input.txt` in the
same folder. Edit or overwrite that file to trigger the planner and judge
agents:

```bash
echo "Hello, nice to meet you. Let me explain how this works." > input.txt
```

If `../chat_history.json` exists relative to `local_end/`, the realtime script
watches that file instead, extracts the newest `role:"ai"` message into
`input.txt`, and sends only that latest robot utterance to the motion agents.
For older setups, it also falls back to `../chat_history.jsonl`.

Typical layout:

```text
direct_body_control/chat_history.json
direct_body_control/local_end/realtime_chat_nonverbal_from_txt.py
```

Manual extraction:

```bash
python extract_input_from_chat_history.py
```

Continuous extraction:

```bash
python extract_input_from_chat_history.py --watch
```

Then run face/speech as before:

```powershell
cd Sophia_Face_HCI
python main.py
```

Dry-run without robot:

```powershell
python llm_move_sender.py --input-file actions.txt --dry-run
```

## Important Note About Simultaneous Motion

Keeping `Sophia_control.py` unchanged means it sends one motor per TCP request.
That is simpler and matches the old local interface, but true same-packet batch
movement would require changing `Sophia_control.py`.

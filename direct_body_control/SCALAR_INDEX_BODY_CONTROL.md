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

If `../chat_history.jsonl` exists relative to `local_end/`, the realtime script
watches that file instead, extracts the newest `role:"ai"` message into
`input.txt`, and sends only that latest robot utterance to the motion agents.

Typical layout:

```text
direct_body_control/chat_history.jsonl
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

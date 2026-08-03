# Sophia Body Control

This folder contains the simplified scalar-index body-control path for Sophia's
LLM nonverbal motion pipeline.

Old path:

```text
motion_repo keyframe -> SMPL/index mapping -> axis-angle [x,y,z] -> bodycontrol_tcp_standard.py
```

Recommended path:

```text
motion_repo keyframe -> scalar motor index + radian value -> batch TCP -> bodycontrol_tcp_scalar_index.py
```

The old single-motor `Sophia_control.py` request still works for manual tests.
The motion sender now uses a batch request by default so one keyframe reaches
the robot as one TCP message.

## Robot End

Copy only this file to the robot:

```text
robot_end/bodycontrol_tcp_scalar_index.py
```

Run it on the same port that `Sophia_control.py` already uses:

```bash
python3 bodycontrol_tcp_scalar_index.py --port 5005
```

The robot bridge accepts:

```json
{"index":0,"value":-1.2217}
```

and sends that scalar radian value to one actuator. It also accepts batch
commands:

```json
{"unit":"rad","commands":[{"index":0,"value":-1.2217},{"index":3,"value":1.8849}]}
```

Batch commands are published in one `/hr/actuators/pose` message, so left/right
compound actions can start together.

If Sophia's official motor IDs are different, edit `MOTOR_INDEX_TO_ACTUATOR` in
both `robot_end/bodycontrol_tcp_scalar_index.py` and `local_end/llm_move_sender.py`.

If the chat history file is produced on the robot, also copy this helper to the
robot:

```text
robot_end/sync_chat_history_to_local.py
```

Run it in another robot terminal to keep the local motion computer updated:

```bash
python3 sync_chat_history_to_local.py \
  --source ../chat_history.json \
  --dest ywguo@10.0.0.111:/home/ywguo/Documents/Sophia_VLA/chat_history.json
```

The helper watches the file and uses `scp` only when it changes. For this to run
without repeated password prompts, set up SSH login from the robot to the local
machine first.

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

`sync_password.txt` is ignored by Git. If `sshpass` is missing on the robot,
install it or use SSH keys instead.

## Local End

Use these files locally, in the same folder:

```text
local_end/realtime_chat_nonverbal_from_txt.py
local_end/nonverbal_motion_agent.py
local_end/system_prompt.txt
local_end/motion_repo.py
local_end/llm_move_sender.py
local_end/Sophia_control.py
local_end/extract_input_from_chat_history.py
local_end/input.txt
local_end/actions.txt
```

For the full speech + motion loop, place/copy the local-end files into the
same folder that contains `Sophia_Face_HCI/`, then run:

```powershell
$env:SOPHIA_MOTION_SENDER="scalar_index"
$env:SOPHIA_SCALAR_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_SCALAR_ROBOT_PORT="5005"
python realtime_chat_nonverbal_from_txt.py
```

Low-latency defaults are already enabled:

```text
SOPHIA_NONVERBAL_AGENT_MODE=fast
SOPHIA_SCALAR_RESET_FIRST=0
SOPHIA_SCALAR_BATCH_COMMANDS=1
REALTIME_LOG_ALL_EVENTS=0
```

The fast mode uses one LLM response. To compare with the older planner plus
judge flow:

```bash
SOPHIA_NONVERBAL_AGENT_MODE=two_stage python realtime_chat_nonverbal_from_txt.py
```

To debug against an older robot bridge without batch support:

```bash
SOPHIA_SCALAR_BATCH_COMMANDS=0 python realtime_chat_nonverbal_from_txt.py
```

By default, `realtime_chat_nonverbal_from_txt.py` watches `input.txt` in the
same folder. Edit or overwrite that file to trigger the planner and judge
agents:

```powershell
"Hello, nice to meet you. Let me explain how this works." | Set-Content input.txt
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

```powershell
python extract_input_from_chat_history.py
```

Continuous extraction if another program is writing the chat-history file:

```powershell
python extract_input_from_chat_history.py --watch
```

In another terminal, run the Sophia Face HCI realtime speech program:

```powershell
cd Sophia_Face_HCI
python main.py
```

Dry run:

```powershell
python llm_move_sender.py --input-file actions.txt --dry-run --no-reset-first
```

Quick robot test:

```powershell
"rightHandRaise 0.5`nstandby 0.5" | python llm_move_sender.py --host 10.0.0.10 --port 5005
```

## Simultaneous Motion Note

`+` compound actions still merge keyframes locally:

```text
leftHandReachOut+rightHandReachOut 0.8
standby 0.6
```

`llm_move_sender.py` sends each compound keyframe as one batch TCP request. If
the robot bridge is old and rejects batch format, it automatically falls back
to the older one-command-per-socket behavior.

## Comparison Paths

`standard_index_motion_sender.py` and the direct actuator-name files are still
kept only for comparison. The recommended path above avoids both SMPL-X and the
old `[x, y, z]` vector packet.

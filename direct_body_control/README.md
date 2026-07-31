# Sophia Body Control

This folder contains the simplified scalar-index body-control path for Sophia's
LLM nonverbal motion pipeline.

Old path:

```text
motion_repo keyframe -> SMPL/index mapping -> axis-angle [x,y,z] -> bodycontrol_tcp_standard.py
```

Recommended path:

```text
motion_repo keyframe -> scalar motor index + radian value -> Sophia_control.py -> bodycontrol_tcp_scalar_index.py
```

The local TCP client remains `Sophia_control.py`.

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

and sends that scalar radian value to one actuator.

If Sophia's official motor IDs are different, edit `MOTOR_INDEX_TO_ACTUATOR` in
both `robot_end/bodycontrol_tcp_scalar_index.py` and `local_end/llm_move_sender.py`.

## Local End

Use these files locally, in the same folder:

```text
local_end/realtime_chat_nonverbal_from_txt.py
local_end/nonverbal_motion_agent.py
local_end/system_prompt.txt
local_end/motion_repo.py
local_end/llm_move_sender.py
local_end/Sophia_control.py
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

In another terminal, run the Sophia Face HCI realtime speech program:

```powershell
cd Sophia_Face_HCI
python main.py
```

Dry run:

```powershell
python llm_move_sender.py --input-file actions.example.txt --dry-run
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

Because `Sophia_control.py` is unchanged, `llm_move_sender.py` sends those
motors one by one very quickly. True same-packet batch movement would require
changing `Sophia_control.py`.

## Comparison Paths

`standard_index_motion_sender.py` and the direct actuator-name files are still
kept only for comparison. The recommended path above avoids both SMPL-X and the
old `[x, y, z]` vector packet.

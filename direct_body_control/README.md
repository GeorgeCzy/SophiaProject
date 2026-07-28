# Sophia Body Control

This folder contains simplified body-control options for Sophia's LLM nonverbal
motion pipeline.

The old path was:

```text
LLM action -> motion_repo keyframe -> SMPL/index mapping -> axis-angle value -> robot bridge -> actuator
```

The recommended path now is:

```text
LLM action -> motion_repo keyframe -> standard index packet with direct radians -> bodycontrol_tcp_standard.py
```

This keeps the original `bodycontrol_tcp_standard.py` robot-end protocol, but
removes the local SMPL/axis-angle conversion.

## Recommended Robot End

Run your existing standard bridge on the robot:

```bash
python3 bodycontrol_tcp_standard.py
```

Default port: `5005`.

## Recommended Local End

Use:

```text
local_end/realtime_chat_nonverbal_from_txt.py
local_end/nonverbal_motion_agent.py
local_end/system_prompt.txt
local_end/motion_repo.py
local_end/standard_index_motion_sender.py
local_end/direct_robot_protocol.py
```

For the full speech + motion loop, place/copy the local-end files into the
same folder that contains `Sophia_Face_HCI/`, then run:

```powershell
$env:SOPHIA_MOTION_SENDER="standard_index"
$env:SOPHIA_STANDARD_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_STANDARD_ROBOT_PORT="5005"
python realtime_chat_nonverbal_from_txt.py
```

In another terminal, run the Sophia Face HCI realtime speech program:

```powershell
cd Sophia_Face_HCI
python main.py
```

Quick dry run:

```powershell
python standard_index_motion_sender.py --input-file actions.example.txt --dry-run
```

Quick robot test:

```powershell
"rightHandRaise 0.5`nstandby 0.5" | python standard_index_motion_sender.py --host 10.0.0.10 --port 5005
```

## What The Standard-Index Sender Does

It sends the same packet shape expected by `bodycontrol_tcp_standard.py`:

```json
{"index":17,"value":[-1.2217,0.0,0.0]}
```

But it fills the vector directly from `motion_repo.py` angles:

```text
RightShoulderPitch -70 degrees -> index 17 slot 0 -> -1.2217 radians
```

There is no SMPL-X interpretation and no fake axis-angle vector.

## Optional Direct Actuator-Name Path

The earlier direct actuator-name bridge is still included for comparison:

Robot end:

```text
robot_end/bodycontrol_tcp_direct.py
robot_end/direct_robot_protocol.py
```

Local end:

```text
local_end/direct_motion_sender.py
local_end/direct_robot_protocol.py
```

Run it only if you intentionally want the actuator-name protocol:

```powershell
$env:SOPHIA_MOTION_SENDER="direct"
$env:SOPHIA_DIRECT_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_DIRECT_ROBOT_PORT="5006"
python realtime_chat_nonverbal_from_txt.py
```

## Simultaneous Motions

Use `+` to merge keyframes into one command:

```text
leftHandReachOut+rightHandReachOut 0.8
standby 0.6
```

For the standard-index path, this creates one merged pose and sends the needed
standard index packets without axis-angle conversion. If two keyframes in the
same compound action control the same actuator, the later keyframe wins and the
sender prints a warning.

## Legacy Comparison

Use `SOPHIA_MOTION_SENDER=legacy` only when comparing against the old
`bodycontrol_tcp_standard.py` + `llm_move_sender.py` SMPL/index path.

# Sophia Direct Body Control

This folder contains the simplified direct-control version for Sophia's LLM
nonverbal motion pipeline.

The old path was:

```text
LLM action -> motion_repo keyframe -> SMPL/index mapping -> axis-angle value -> robot bridge -> actuator
```

The new path is:

```text
LLM action -> motion_repo keyframe -> actuator names + degree values -> robot bridge -> actuator
```

## Robot End

Copy these files to the same folder on the robot:

```text
robot_end/bodycontrol_tcp_direct.py
robot_end/direct_robot_protocol.py
```

Run the direct bridge after ROS and the HR actuator services are available:

```bash
cd <robot-folder>
python3 bodycontrol_tcp_direct.py
```

The bridge listens on TCP port `5006` and accepts payloads like:

```json
{"actuators":{"RightShoulderPitch":-70,"RightElbowPitch":108},"unit":"deg"}
```

## Local End

Use these files on the local computer:

```text
local_end/realtime_chat_nonverbal_from_txt.py
local_end/nonverbal_motion_agent.py
local_end/system_prompt.txt
local_end/motion_repo.py
local_end/direct_motion_sender.py
local_end/direct_robot_protocol.py
```

For the full speech + motion loop, place/copy the local-end files into the
same folder that contains `Sophia_Face_HCI/`, then run:

```powershell
$env:SOPHIA_MOTION_SENDER="direct"
$env:SOPHIA_DIRECT_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_DIRECT_ROBOT_PORT="5006"
python realtime_chat_nonverbal_from_txt.py
```

In another terminal, run the Sophia Face HCI realtime speech program:

```powershell
cd Sophia_Face_HCI
python main.py
```

`main.py` writes Sophia's spoken answer to `Sophia_Face_HCI/answers.txt` and
audio duration to `/tmp/robot_sync/audio_response.duration`. The nonverbal
planner watches those files, plans actions, and sends them to the direct robot
bridge.

## Quick Motion Test

From the local computer:

```powershell
"leftHandReachOut+rightHandReachOut 0.8`nstandby 0.6" | python direct_motion_sender.py --host 10.0.0.10 --port 5006
```

Use dry run before connecting to the robot:

```powershell
python direct_motion_sender.py --input-file actions.example.txt --dry-run
```

## Simultaneous Motions

Use `+` to merge keyframes into one command:

```text
leftHandReachOut+rightHandReachOut 0.8
standby 0.6
```

That sends both arms in one TCP request. If two keyframes in the same compound
action control the same actuator, the later keyframe wins and the sender prints
a warning.

## Legacy Comparison

Use `SOPHIA_MOTION_SENDER=legacy` only when comparing against the old
`bodycontrol_tcp_standard.py` + `llm_move_sender.py` SMPL/index path.

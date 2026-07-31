# Sophia Body Control

This folder contains simplified body-control options for Sophia's LLM nonverbal
motion pipeline.

The old path was:

```text
LLM action -> motion_repo keyframe -> SMPL/index mapping -> axis-angle value -> robot bridge -> actuator
```

The recommended path now is:

```text
LLM action -> motion_repo keyframe -> scalar motor-index packet -> bodycontrol_tcp_scalar_index.py
```

This uses one integer motor index and one scalar radian value per motor. It
does not send `[x, y, z]` vectors.

## Recommended Robot End

Copy these files to the robot:

```text
robot_end/bodycontrol_tcp_scalar_index.py
robot_end/scalar_index_protocol.py
robot_end/direct_robot_protocol.py
```

Run the scalar-index bridge:

```bash
python3 bodycontrol_tcp_scalar_index.py --port 5007
```

Default port: `5007`.

## Recommended Local End

Use:

```text
local_end/realtime_chat_nonverbal_from_txt.py
local_end/nonverbal_motion_agent.py
local_end/system_prompt.txt
local_end/motion_repo.py
local_end/scalar_index_motion_sender.py
local_end/scalar_index_protocol.py
local_end/direct_robot_protocol.py
```

For the full speech + motion loop, place/copy the local-end files into the
same folder that contains `Sophia_Face_HCI/`, then run:

```powershell
$env:SOPHIA_MOTION_SENDER="scalar_index"
$env:SOPHIA_SCALAR_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_SCALAR_ROBOT_PORT="5007"
python realtime_chat_nonverbal_from_txt.py
```

In another terminal, run the Sophia Face HCI realtime speech program:

```powershell
cd Sophia_Face_HCI
python main.py
```

Quick dry run:

```powershell
python scalar_index_motion_sender.py --input-file actions.example.txt --dry-run
```

Quick robot test:

```powershell
"rightHandRaise 0.5`nstandby 0.5" | python scalar_index_motion_sender.py --host 10.0.0.10 --port 5007
```

## Motor Index Map

The default map is in `scalar_index_protocol.py` and assigns `0..N` to the
actuator names used by `motion_repo.py`.

If the robot has an official motor-index table, put it in a JSON file and pass
the same file to both ends:

```json
{"0":"RightShoulderPitch","1":"RightShoulderRoll"}
```

Robot:

```bash
python3 bodycontrol_tcp_scalar_index.py --port 5007 --motor-map motor_index_map.json
```

Local:

```powershell
$env:SOPHIA_SCALAR_MOTOR_MAP="motor_index_map.json"
python realtime_chat_nonverbal_from_txt.py
```

## What The Scalar-Index Sender Does

It sends a batch of scalar motor commands:

```json
{"unit":"rad","commands":[{"index":0,"value":-1.2217},{"index":3,"value":1.8850}]}
```

Each value comes directly from `motion_repo.py` after degrees-to-radians
conversion and actuator-limit clamping.

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

For the scalar-index path, this creates one merged pose and sends one batch TCP
request. The robot bridge publishes one `TargetPosture` message for all motors.
If two keyframes in the same compound action control the same actuator, the
later keyframe wins and the sender prints a warning.

## Standard-Index Comparison

`standard_index_motion_sender.py` is still included for comparison with
`bodycontrol_tcp_standard.py`, but that path still uses the old vector-shaped
packet:

```json
{"index":17,"value":[-1.2217,0.0,0.0]}
```

Use it only when intentionally comparing against the existing standard bridge.

## Legacy Comparison

Use `SOPHIA_MOTION_SENDER=legacy` only when comparing against the old
`bodycontrol_tcp_standard.py` + `llm_move_sender.py` SMPL/index path.

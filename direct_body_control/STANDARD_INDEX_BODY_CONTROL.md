# Standard Index Body Control

Use this only for comparison with the existing robot-end
`bodycontrol_tcp_standard.py`.

For one-index/one-radian motor control, use `SCALAR_INDEX_BODY_CONTROL.md`.

It keeps the original robot-end packet format:

```json
{"index": 17, "value": [-1.2217, 0.0, 0.0]}
```

but avoids the old local SMPL/axis-angle conversion. The local sender writes
the motion keyframe angles directly into the vector slots that
`bodycontrol_tcp_standard.py` extracts.

That means this path removes `to_axisangle()`, but it does not remove the old
3D vector slots from the robot-end bridge.

## Robot End

Copy/run the existing standard bridge on the robot:

```bash
python3 bodycontrol_tcp_standard.py
```

Default port: `5005`.

## Local End

Use:

```text
standard_index_motion_sender.py
direct_robot_protocol.py
motion_repo.py
actions.txt
```

Quick dry run:

```powershell
python standard_index_motion_sender.py --input-file actions.txt --dry-run
```

Quick robot test:

```powershell
"rightHandRaise 0.5`nstandby 0.5" | python standard_index_motion_sender.py --host 10.0.0.10 --port 5005
```

For the realtime nonverbal pipeline:

```powershell
$env:SOPHIA_MOTION_SENDER="standard_index"
$env:SOPHIA_STANDARD_ROBOT_HOST="10.0.0.10"
$env:SOPHIA_STANDARD_ROBOT_PORT="5005"
python realtime_chat_nonverbal_from_txt.py
```

## What Changed From `llm_move_sender.py`

Old local path:

```text
motion_repo degrees -> radians -> fake axis-angle vector -> bodycontrol extracts components
```

New local path:

```text
motion_repo degrees -> radians in bodycontrol_tcp_standard.py's exact vector slots
```

Example:

```text
RightShoulderPitch = -70 deg
```

is sent as:

```json
{"index": 17, "value": [-1.2217, 0.0, 0.0]}
```

because `bodycontrol_tcp_standard.py` maps index `17` slot `0` to
`RightShoulderPitch`.

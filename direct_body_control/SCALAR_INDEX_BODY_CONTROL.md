# Scalar Index Body Control Path

This path removes the SMPL-X vector shape completely.

## Protocol

Each TCP command means:

```text
one motor index -> one actuator -> one scalar radian value
```

Single motor:

```json
{"index":0,"value":-1.2217,"unit":"rad"}
```

Multiple motors in one command:

```json
{"commands":[{"index":0,"value":-1.2217},{"index":13,"value":1.2217}],"unit":"rad"}
```

Reset all mapped motors to zero:

```json
{"command":"reset"}
```

No request in this path uses `[x, y, z]`.

## Motor Index Map

The shared map is in `scalar_index_protocol.py`.

By default it assigns indexes `0..N` to the actuator names already used by
`motion_repo.py`. If the robot has an official motor-index table, create a JSON
file and use it on both ends:

```json
{
  "0": "RightShoulderPitch",
  "1": "RightShoulderRoll",
  "2": "RightShoulderYaw"
}
```

The JSON values must be actuator names from `motion_repo.py`.

## Robot End

Copy these files to the same folder on the robot:

```text
bodycontrol_tcp_scalar_index.py
scalar_index_protocol.py
direct_robot_protocol.py
```

Run:

```bash
python3 bodycontrol_tcp_scalar_index.py --port 5007
```

If using an official motor map:

```bash
python3 bodycontrol_tcp_scalar_index.py --port 5007 --motor-map motor_index_map.json
```

## Local End

Keep these files together locally:

```text
realtime_chat_nonverbal_from_txt.py
nonverbal_motion_agent.py
system_prompt.txt
motion_repo.py
scalar_index_motion_sender.py
scalar_index_protocol.py
direct_robot_protocol.py
actions.txt
Sophia_Face_HCI/main.py
```

Dry-run an action file:

```bash
python scalar_index_motion_sender.py --input-file actions.txt --dry-run
```

Send to robot:

```bash
python scalar_index_motion_sender.py --input-file actions.txt --host 10.0.0.10 --port 5007
```

With an official motor map:

```bash
python scalar_index_motion_sender.py --input-file actions.txt --host 10.0.0.10 --port 5007 --motor-map motor_index_map.json
```

The realtime scripts now default to this path:

```bash
export SOPHIA_MOTION_SENDER=scalar_index
export SOPHIA_SCALAR_ROBOT_HOST=10.0.0.10
export SOPHIA_SCALAR_ROBOT_PORT=5007
python realtime_chat_nonverbal_from_txt.py
```

## Simultaneous Movement

Use `+` to merge keyframes:

```text
leftHandReachOut+rightHandReachOut 0.8
standby 0.6
```

The local sender turns that merged keyframe into one batch TCP request, and the
robot bridge publishes one `TargetPosture` message containing all involved
motors.

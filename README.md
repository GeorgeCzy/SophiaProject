# SophiaProject

## Direct Body Control

The simplified Sophia robot body-control pipeline is in
[`direct_body_control/`](direct_body_control/).

It bypasses the old SMPL/index mapping path and sends LLM-selected keyframes
directly as actuator-name commands.

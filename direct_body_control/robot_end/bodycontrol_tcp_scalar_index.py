#!/usr/bin/env python3
"""
Simple scalar-index TCP bridge for Sophia body control.

This is the robot-end replacement for bodycontrol_tcp_standard.py when the
local side uses the existing Sophia_control.py client unchanged.

Sophia_control.py sends:

    {"index": <motor_index>, "value": <target_radians>}

This bridge interprets that literally:

    one index -> one actuator -> one scalar radian target

No SMPL-X mapping, no axis-angle vector, no [x, y, z] value.
"""

import argparse
import json
import math
import socket
import threading
from typing import Any, Dict, List, Tuple

import rospy
from hr_msgs.msg import TargetPosture
from hr_msgs.srv import SetActuatorsControl, SetActuatorsControlRequest


def deg(value):
    return float(value) * math.pi / 180.0


# Keep this table identical to llm_move_sender.py.
# If the robot has official motor IDs, edit only these integer keys.
MOTOR_INDEX_TO_ACTUATOR = {
    0: "RightShoulderPitch",
    1: "RightShoulderRoll",
    2: "RightShoulderYaw",
    3: "RightElbowPitch",
    4: "RightElbowYaw",
    5: "RightWristPitch",
    6: "RightWristRoll",
    7: "RightThumbRoll",
    8: "RightThumbFinger",
    9: "RightIndexFinger",
    10: "RightMiddleFinger",
    11: "RightRingFinger",
    12: "RightPinkyFinger",
    13: "LeftShoulderPitch",
    14: "LeftShoulderRoll",
    15: "LeftShoulderYaw",
    16: "LeftElbowPitch",
    17: "LeftElbowYaw",
    18: "LeftWristPitch",
    19: "LeftWristRoll",
    20: "LeftThumbRoll",
    21: "LeftThumbFinger",
    22: "LeftIndexFinger",
    23: "LeftMiddleFinger",
    24: "LeftRingFinger",
    25: "LeftPinkyFinger",
}


ACTUATOR_LIMITS = {
    "RightShoulderPitch": (deg(-145), deg(35)),
    "RightShoulderRoll": (deg(-4), deg(101)),
    "RightShoulderYaw": (deg(-66), deg(83)),
    "RightElbowPitch": (deg(-127), deg(119)),
    "RightElbowYaw": (deg(-123), deg(123)),
    "RightWristPitch": (deg(-35), deg(35)),
    "RightWristRoll": (deg(-35), deg(35)),
    "RightThumbRoll": (deg(-31), deg(22)),
    "RightThumbFinger": (deg(-75), deg(44)),
    "RightIndexFinger": (deg(-123), deg(18)),
    "RightMiddleFinger": (deg(-18), deg(132)),
    "RightRingFinger": (deg(-18), deg(136)),
    "RightPinkyFinger": (deg(-75), deg(4)),
    "LeftShoulderPitch": (deg(-35), deg(145)),
    "LeftShoulderRoll": (deg(-101), deg(4)),
    "LeftShoulderYaw": (deg(-83), deg(66)),
    "LeftElbowPitch": (deg(-119), deg(127)),
    "LeftElbowYaw": (deg(-123), deg(123)),
    "LeftWristPitch": (deg(-35), deg(35)),
    "LeftWristRoll": (deg(-35), deg(35)),
    "LeftThumbRoll": (deg(-31), deg(22)),
    "LeftThumbFinger": (deg(-44), deg(75)),
    "LeftIndexFinger": (deg(-18), deg(123)),
    "LeftMiddleFinger": (deg(-18), deg(132)),
    "LeftRingFinger": (deg(-18), deg(136)),
    "LeftPinkyFinger": (deg(-4), deg(75)),
}


def clamp(actuator, value):
    lo, hi = ACTUATOR_LIMITS[actuator]
    return max(lo, min(hi, float(value)))


def parse_scalar_command(request):
    return parse_scalar_commands(request)[0]


def parse_scalar_commands(request):
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    if request.get("command") == "reset":
        return [
            (index, MOTOR_INDEX_TO_ACTUATOR[index], 0.0)
            for index in sorted(MOTOR_INDEX_TO_ACTUATOR)
        ]

    commands = []
    if isinstance(request.get("commands"), list):
        for item in request["commands"]:
            commands.append(parse_scalar_command_item(item))
    elif isinstance(request.get("indices"), list) and isinstance(request.get("values"), list):
        indices = request["indices"]
        values = request["values"]
        if len(indices) != len(values):
            raise ValueError("indices and values must have the same length")
        for index, value in zip(indices, values):
            commands.append(parse_scalar_index_value(index, value))
    elif "index" in request and "value" in request:
        commands.append(parse_scalar_index_value(request["index"], request["value"]))
    else:
        raise ValueError(
            "request must include index/value, commands, indices/values, or command='reset'"
        )

    if not commands:
        raise ValueError("no scalar motor commands in request")

    merged = {}
    for index, actuator, value in commands:
        merged[index] = (actuator, value)
    return [
        (index, actuator, value)
        for index, (actuator, value) in sorted(merged.items())
    ]


def parse_scalar_command_item(item: Any):
    if isinstance(item, dict):
        if "index" not in item or "value" not in item:
            raise ValueError("each command object must include index and value")
        return parse_scalar_index_value(item["index"], item["value"])

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return parse_scalar_index_value(item[0], item[1])

    raise ValueError("each command must be an object or [index, value] pair")


def parse_scalar_index_value(raw_index, raw_value):
    index = int(raw_index)
    if index not in MOTOR_INDEX_TO_ACTUATOR:
        raise ValueError("unknown motor index: %s" % index)

    if isinstance(raw_value, (dict, list, tuple)):
        raise ValueError("value must be one scalar radian number, not a vector/list/object")

    actuator = MOTOR_INDEX_TO_ACTUATOR[index]
    value = clamp(actuator, float(raw_value))
    return index, actuator, value


def reset_pose():
    names = []
    values = []
    for index in sorted(MOTOR_INDEX_TO_ACTUATOR):
        actuator = MOTOR_INDEX_TO_ACTUATOR[index]
        names.append(actuator)
        values.append(0.0)
    return names, values


class ScalarIndexBodyBridgeServer:
    def __init__(self, host="0.0.0.0", port=5005):
        rospy.loginfo("[ScalarIndexBodyBridge] waiting for /hr/actuators/set_control ...")
        rospy.wait_for_service("/hr/actuators/set_control")

        self.pose_pub = rospy.Publisher("/hr/actuators/pose", TargetPosture, queue_size=1)
        self.set_control = rospy.ServiceProxy("/hr/actuators/set_control", SetActuatorsControl)

        self._set_manual_for_actuators()
        self._publish_startup_reset()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        self.sock.settimeout(0.5)
        rospy.loginfo("[ScalarIndexBodyBridge] listening on %s:%s", host, port)

    def _set_manual_for_actuators(self):
        req = SetActuatorsControlRequest()
        req.control = SetActuatorsControlRequest.CONTROL_MANUAL
        req.actuators = sorted(MOTOR_INDEX_TO_ACTUATOR.values())
        self.set_control(req)
        rospy.loginfo(
            "[ScalarIndexBodyBridge] set MANUAL control for %s actuators",
            len(req.actuators),
        )

    def serve_forever(self):
        try:
            while not rospy.is_shutdown():
                try:
                    conn, addr = self.sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if rospy.is_shutdown():
                        break
                    raise
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            rospy.loginfo("[ScalarIndexBodyBridge] Ctrl+C received, shutting down")
            rospy.signal_shutdown("Ctrl+C")
        finally:
            self.shutdown()

    def shutdown(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def _handle(self, conn, addr):
        try:
            raw = conn.recv(65536)
            if not raw:
                return

            request = json.loads(raw.decode("utf-8"))
            commands = parse_scalar_commands(request)
            names = [actuator for _, actuator, _ in commands]
            values = [value for _, _, value in commands]
            indices = [index for index, _, _ in commands]
            self._publish(names, values)
            self._send(
                conn,
                code=0,
                result={
                    "count": len(commands),
                    "indices": indices,
                    "sent": dict(zip(names, values)),
                },
            )
        except Exception as exc:
            self._send(conn, code=99, error=str(exc))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _publish(self, names, values):
        msg = TargetPosture()
        msg.names = list(names)
        msg.values = list(values)
        self.pose_pub.publish(msg)

    def _publish_startup_reset(self, repeats=10, dt=0.1):
        names, values = reset_pose()
        for _ in range(repeats):
            self._publish(names, values)
            rospy.sleep(dt)
        rospy.loginfo(
            "[ScalarIndexBodyBridge] sent startup zero pose (%s repeats)",
            repeats,
        )

    def _send(self, conn, code, result=None, error=""):
        response = {"code": code}
        if code == 0:
            response["result"] = result
        else:
            response["error"] = error
        conn.sendall(json.dumps(response).encode("utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple scalar-index TCP bridge for Sophia body actuators."
    )
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host.")
    parser.add_argument("--port", type=int, default=5005, help="TCP bind port.")
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("sophia_body_bridge_scalar_index", anonymous=True)
    server = ScalarIndexBodyBridgeServer(host=args.host, port=args.port)
    rospy.on_shutdown(server.shutdown)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

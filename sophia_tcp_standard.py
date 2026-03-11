#!/usr/bin/env python3
"""
Standard TCP JSON -> HR ROS body actuators bridge.

Client sends: {"index": <int>, "value": [x, y, z]}
Server replies: {"code": 0, "result": {...}} or {"code": nonzero, "error": "..."}

This version keeps the same index-to-actuator mapping as the current bridge,
but removes custom direction/sign tweaks and gain scaling. The command sent to
each actuator is simply:

    clamp(OFFSET[actuator] + extracted_component)

This makes it a simpler baseline bridge for sending rotations to the robot.
"""

import json
import math
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import rospy
from hr_msgs.msg import TargetPosture
from hr_msgs.srv import SetActuatorsControl, SetActuatorsControlRequest


def deg(x: float) -> float:
    return x * math.pi / 180.0


# Conservative actuator limits in radians.
DEFAULT_LIMITS: Dict[str, Tuple[float, float]] = {
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

LIMITS = DEFAULT_LIMITS

# A-pose startup bias. Remove/comment entries if you want motor=0 pose instead.
OFFSET: Dict[str, float] = {
    "RightShoulderRoll": deg(45.0),
    "RightElbowPitch": deg(-127.0),
    "LeftShoulderRoll": deg(-45.0),
    "LeftElbowPitch": deg(127.0),
}


def clamp(actuator: str, value: float) -> float:
    lo, hi = LIMITS[actuator]
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def send_a_pose(
    pose_pub,
    limits: Dict[str, Tuple[float, float]],
    repeats: int = 10,
    dt: float = 0.1,
):
    """Publish the startup pose using only OFFSET values."""
    names: List[str] = []
    values: List[float] = []
    for name, (lo, hi) in limits.items():
        value = OFFSET.get(name, 0.0)
        if value < lo:
            value = lo
        elif value > hi:
            value = hi
        names.append(name)
        values.append(value)

    msg = TargetPosture()
    msg.names = names
    msg.values = values
    for _ in range(repeats):
        pose_pub.publish(msg)
        rospy.sleep(dt)


@dataclass(frozen=True)
class ActuatorCmd:
    actuator: str
    extractor: Callable[[List[float]], float]


def _need_vec3(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("value must be a list/tuple of 3 floats: [x, y, z]")
    return [float(value[0]), float(value[1]), float(value[2])]


INDEX_MAP: Dict[int, List[ActuatorCmd]] = {
    16: [
        ActuatorCmd("LeftShoulderPitch", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("LeftShoulderRoll", lambda v: _need_vec3(v)[2]),
    ],
    17: [
        ActuatorCmd("RightShoulderPitch", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("RightShoulderRoll", lambda v: _need_vec3(v)[2]),
    ],
    18: [
        ActuatorCmd("LeftShoulderYaw", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("LeftElbowPitch", lambda v: _need_vec3(v)[1]),
    ],
    19: [
        ActuatorCmd("RightShoulderYaw", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("RightElbowPitch", lambda v: _need_vec3(v)[1]),
    ],
    20: [
        ActuatorCmd("LeftElbowYaw", lambda v: _need_vec3(v)[0]),
    ],
    21: [
        ActuatorCmd("RightElbowYaw", lambda v: _need_vec3(v)[0]),
    ],
    25: [ActuatorCmd("LeftIndexFinger", lambda v: _need_vec3(v)[2])],
    28: [ActuatorCmd("LeftMiddleFinger", lambda v: _need_vec3(v)[2])],
    31: [ActuatorCmd("LeftPinkyFinger", lambda v: _need_vec3(v)[2])],
    34: [ActuatorCmd("LeftRingFinger", lambda v: _need_vec3(v)[2])],
    37: [
        ActuatorCmd("LeftThumbRoll", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("LeftThumbFinger", lambda v: _need_vec3(v)[2]),
    ],
    40: [ActuatorCmd("RightIndexFinger", lambda v: _need_vec3(v)[2])],
    43: [ActuatorCmd("RightMiddleFinger", lambda v: _need_vec3(v)[2])],
    46: [ActuatorCmd("RightPinkyFinger", lambda v: _need_vec3(v)[2])],
    49: [ActuatorCmd("RightRingFinger", lambda v: _need_vec3(v)[2])],
    52: [
        ActuatorCmd("RightThumbRoll", lambda v: _need_vec3(v)[0]),
        ActuatorCmd("RightThumbFinger", lambda v: _need_vec3(v)[2]),
    ],
}


class BodyBridgeServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5005):
        rospy.loginfo("[BodyBridgeStandard] waiting for /hr/actuators/set_control ...")
        rospy.wait_for_service("/hr/actuators/set_control")

        self.pose_pub = rospy.Publisher("/hr/actuators/pose", TargetPosture, queue_size=1)
        self.set_control = rospy.ServiceProxy("/hr/actuators/set_control", SetActuatorsControl)

        self._set_manual_for_whitelist()
        send_a_pose(self.pose_pub, LIMITS)
        rospy.loginfo("[BodyBridgeStandard] sent startup A-pose")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        rospy.loginfo(f"[BodyBridgeStandard] listening on {host}:{port}")

    def _set_manual_for_whitelist(self):
        whitelist = sorted(LIMITS.keys())
        req = SetActuatorsControlRequest()
        req.control = SetActuatorsControlRequest.CONTROL_MANUAL
        req.actuators = whitelist
        self.set_control(req)
        rospy.loginfo(f"[BodyBridgeStandard] set MANUAL control for {len(whitelist)} actuators")

    def serve_forever(self):
        try:
            while not rospy.is_shutdown():
                conn, addr = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            rospy.loginfo("Ctrl+C received, shutting down")
        finally:
            try:
                self.sock.close()
            except Exception:
                pass

    def _handle(self, conn: socket.socket, addr):
        try:
            raw = conn.recv(4096)
            if not raw:
                return

            req = json.loads(raw.decode("utf-8"))
            if not isinstance(req, dict) or "index" not in req or "value" not in req:
                self._send(conn, code=1, error="request must be dict with keys: index, value")
                return

            idx = int(req["index"])
            value = req["value"]

            if idx not in INDEX_MAP:
                self._send(conn, code=2, error=f"index {idx} not allowed (no mapping)")
                return

            names: List[str] = []
            values: List[float] = []
            for cmd in INDEX_MAP[idx]:
                if cmd.actuator not in LIMITS:
                    self._send(conn, code=3, error=f"no limits for actuator {cmd.actuator}")
                    return

                delta = float(cmd.extractor(value))
                target = clamp(cmd.actuator, OFFSET.get(cmd.actuator, 0.0) + delta)
                names.append(cmd.actuator)
                values.append(target)

            msg = TargetPosture()
            msg.names = names
            msg.values = values
            self.pose_pub.publish(msg)

            self._send(conn, code=0, result={"index": idx, "sent": dict(zip(names, values))})

        except Exception as e:
            self._send(conn, code=99, error=str(e))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _send(self, conn: socket.socket, code: int, result=None, error: str = ""):
        resp = {"code": code}
        if code == 0:
            resp["result"] = result
        else:
            resp["error"] = error
        conn.sendall(json.dumps(resp).encode("utf-8"))


if __name__ == "__main__":
    rospy.init_node("sophia_body_bridge_standard", anonymous=True)
    server = BodyBridgeServer(host="0.0.0.0", port=5005)
    server.serve_forever()

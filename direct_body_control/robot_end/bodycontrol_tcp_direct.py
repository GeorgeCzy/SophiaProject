#!/usr/bin/env python3
"""
Direct TCP JSON -> HR ROS body actuators bridge.

This is the simplified robot-end replacement for the SMPL/index bridge.
It does not know about SMPL-X, axis-angle, or index mappings.

Client request examples:

    {"actuators": {"RightShoulderPitch": -70, "RightElbowPitch": 108}, "unit": "deg"}
    {"names": ["RightShoulderPitch"], "values": [-1.22], "unit": "rad"}
    {"command": "reset"}

Server replies:

    {"code": 0, "result": {"sent": {"RightShoulderPitch": -1.2217}}}
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import rospy
from hr_msgs.msg import TargetPosture
from hr_msgs.srv import SetActuatorsControl, SetActuatorsControlRequest

from direct_robot_protocol import (
    DIRECT_ACTUATORS,
    clamp_rad,
    deg_to_rad,
    filter_supported_angles_deg,
    standby_angles_deg,
)


class DirectBodyBridgeServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5006):
        rospy.loginfo("[DirectBodyBridge] waiting for /hr/actuators/set_control ...")
        rospy.wait_for_service("/hr/actuators/set_control")

        self.pose_pub = rospy.Publisher("/hr/actuators/pose", TargetPosture, queue_size=1)
        self.set_control = rospy.ServiceProxy("/hr/actuators/set_control", SetActuatorsControl)

        self._set_manual_for_direct_actuators()
        self._publish_degrees(standby_angles_deg())
        rospy.loginfo("[DirectBodyBridge] sent startup standby pose")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        rospy.loginfo("[DirectBodyBridge] listening on %s:%s", host, port)

    def _set_manual_for_direct_actuators(self) -> None:
        req = SetActuatorsControlRequest()
        req.control = SetActuatorsControlRequest.CONTROL_MANUAL
        req.actuators = list(DIRECT_ACTUATORS)
        self.set_control(req)
        rospy.loginfo("[DirectBodyBridge] set MANUAL control for %s actuators", len(DIRECT_ACTUATORS))

    def serve_forever(self) -> None:
        try:
            while not rospy.is_shutdown():
                conn, addr = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            rospy.loginfo("[DirectBodyBridge] Ctrl+C received, shutting down")
        finally:
            try:
                self.sock.close()
            except Exception:
                pass

    def _handle(self, conn: socket.socket, addr) -> None:
        try:
            raw = conn.recv(65536)
            if not raw:
                return
            req = json.loads(raw.decode("utf-8"))
            names, values = self._parse_request(req)
            self._publish_radians(names, values)
            self._send(conn, code=0, result={"sent": dict(zip(names, values))})
        except Exception as exc:
            self._send(conn, code=99, error=str(exc))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _parse_request(self, req: Any) -> tuple[list[str], list[float]]:
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")

        if req.get("command") == "reset":
            return self._degrees_to_radians(standby_angles_deg())

        unit = str(req.get("unit", "deg")).strip().lower()
        if unit not in {"deg", "degree", "degrees", "rad", "radian", "radians"}:
            raise ValueError("unit must be 'deg' or 'rad'")

        if isinstance(req.get("actuators"), dict):
            actuator_values = req["actuators"]
            names = [str(name) for name in actuator_values]
            values = [float(actuator_values[name]) for name in actuator_values]
        elif isinstance(req.get("names"), list) and isinstance(req.get("values"), list):
            names = [str(name) for name in req["names"]]
            values = [float(value) for value in req["values"]]
        else:
            raise ValueError("request must include actuators dict, names/values lists, or command='reset'")

        if len(names) != len(values):
            raise ValueError("names and values must have the same length")
        if not names:
            raise ValueError("no actuators requested")

        if unit.startswith("deg"):
            return self._degrees_to_radians(dict(zip(names, values)))
        return self._radians_to_radians(dict(zip(names, values)))

    def _degrees_to_radians(self, angles_deg: dict[str, float]) -> tuple[list[str], list[float]]:
        supported, ignored = filter_supported_angles_deg(angles_deg)
        if ignored:
            rospy.logwarn("[DirectBodyBridge] ignored unsupported actuators: %s", sorted(ignored))
        if not supported:
            raise ValueError("no supported actuators in request")
        names = list(supported)
        values = [deg_to_rad(supported[name]) for name in names]
        return names, values

    def _radians_to_radians(self, angles_rad: dict[str, float]) -> tuple[list[str], list[float]]:
        names: list[str] = []
        values: list[float] = []
        ignored: list[str] = []
        for name, value in angles_rad.items():
            if name not in DIRECT_ACTUATORS:
                ignored.append(name)
                continue
            names.append(name)
            values.append(clamp_rad(name, float(value)))
        if ignored:
            rospy.logwarn("[DirectBodyBridge] ignored unsupported actuators: %s", sorted(ignored))
        if not names:
            raise ValueError("no supported actuators in request")
        return names, values

    def _publish_degrees(self, angles_deg: dict[str, float]) -> None:
        names, values = self._degrees_to_radians(angles_deg)
        self._publish_radians(names, values)

    def _publish_radians(self, names: list[str], values: list[float]) -> None:
        msg = TargetPosture()
        msg.names = names
        msg.values = values
        self.pose_pub.publish(msg)

    def _send(self, conn: socket.socket, code: int, result=None, error: str = "") -> None:
        resp = {"code": code}
        if code == 0:
            resp["result"] = result
        else:
            resp["error"] = error
        conn.sendall(json.dumps(resp).encode("utf-8"))


if __name__ == "__main__":
    rospy.init_node("sophia_body_bridge_direct", anonymous=True)
    server = DirectBodyBridgeServer(host="0.0.0.0", port=5006)
    server.serve_forever()

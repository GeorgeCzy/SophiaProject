#!/usr/bin/env python3
"""
Scalar-index TCP JSON -> HR ROS body actuators bridge.

This is the minimal index-based robot-end path:

    {"index": 0, "value": -1.22, "unit": "rad"}
    {"commands": [{"index": 0, "value": -1.22}, {"index": 13, "value": 1.22}], "unit": "rad"}
    {"command": "reset"}

Each index maps to exactly one actuator, and each value is one scalar target
angle. No SMPL-X joint mapping and no axis-angle vector is used.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading

import rospy
from hr_msgs.msg import TargetPosture
from hr_msgs.srv import SetActuatorsControl, SetActuatorsControlRequest

from scalar_index_protocol import (
    load_motor_index_map,
    request_to_scalar_commands,
    scalar_commands_to_names_values,
    standby_scalar_commands,
)


class ScalarIndexBodyBridgeServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5007,
        motor_map_path: str | None = None,
    ):
        self.motor_map = load_motor_index_map(motor_map_path)

        rospy.loginfo("[ScalarIndexBodyBridge] waiting for /hr/actuators/set_control ...")
        rospy.wait_for_service("/hr/actuators/set_control")

        self.pose_pub = rospy.Publisher("/hr/actuators/pose", TargetPosture, queue_size=1)
        self.set_control = rospy.ServiceProxy("/hr/actuators/set_control", SetActuatorsControl)

        self._set_manual_for_scalar_actuators()
        self._publish_scalar_commands(standby_scalar_commands(self.motor_map))
        rospy.loginfo("[ScalarIndexBodyBridge] sent startup scalar reset")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(16)
        rospy.loginfo("[ScalarIndexBodyBridge] listening on %s:%s", host, port)

    def _set_manual_for_scalar_actuators(self) -> None:
        req = SetActuatorsControlRequest()
        req.control = SetActuatorsControlRequest.CONTROL_MANUAL
        req.actuators = sorted(self.motor_map.values())
        self.set_control(req)
        rospy.loginfo(
            "[ScalarIndexBodyBridge] set MANUAL control for %s actuators",
            len(req.actuators),
        )

    def serve_forever(self) -> None:
        try:
            while not rospy.is_shutdown():
                conn, addr = self.sock.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            rospy.loginfo("[ScalarIndexBodyBridge] Ctrl+C received, shutting down")
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

            request = json.loads(raw.decode("utf-8"))
            commands = request_to_scalar_commands(request, self.motor_map)
            sent = self._publish_scalar_commands(commands)
            self._send(conn, code=0, result={"sent": sent})
        except Exception as exc:
            self._send(conn, code=99, error=str(exc))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _publish_scalar_commands(self, commands: list[tuple[int, float]]) -> dict[str, float]:
        names, values = scalar_commands_to_names_values(commands, self.motor_map)
        if not names:
            raise ValueError("no mapped actuator commands to publish")

        msg = TargetPosture()
        msg.names = names
        msg.values = values
        self.pose_pub.publish(msg)
        return {
            f"{index}:{self.motor_map[index]}": value
            for index, value in commands
            if index in self.motor_map
        }

    def _send(self, conn: socket.socket, code: int, result=None, error: str = "") -> None:
        response = {"code": code}
        if code == 0:
            response["result"] = result
        else:
            response["error"] = error
        conn.sendall(json.dumps(response).encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scalar-index TCP bridge for Sophia body actuators."
    )
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host.")
    parser.add_argument("--port", type=int, default=5007, help="TCP bind port.")
    parser.add_argument(
        "--motor-map",
        default="",
        help="Optional JSON motor index map shared with the local sender.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rospy.init_node("sophia_body_bridge_scalar_index", anonymous=True)
    server = ScalarIndexBodyBridgeServer(
        host=args.host,
        port=args.port,
        motor_map_path=args.motor_map or None,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

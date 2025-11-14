#!/usr/bin/env python3
"""Minimal pure pursuit follower for the simplified path."""

from __future__ import annotations

import json
import math
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
from unitree_api.msg import Request


class SimplePathFollower(Node):
    """Follow the latest local path with a proportional controller."""

    def __init__(self) -> None:
        super().__init__("path_follower_simple")

        self.declare_parameter("path_frame", "sensor")
        self.declare_parameter("cmd_frame", "sensor")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("max_speed", 0.6)
        self.declare_parameter("speed_gain", 0.55)
        self.declare_parameter("lookahead_distance", 0.4)
        self.declare_parameter("goal_tolerance", 0.2)
        self.declare_parameter("angular_gain", 1.5)
        self.declare_parameter("max_yaw_rate", 1.0)  # rad/s
        self.declare_parameter("is_real_robot", False)

        self.path_frame = self.get_parameter("path_frame").get_parameter_value().string_value
        self.cmd_frame = self.get_parameter("cmd_frame").get_parameter_value().string_value
        self.max_speed = self.get_parameter("max_speed").get_parameter_value().double_value
        self.speed_gain = self.get_parameter("speed_gain").get_parameter_value().double_value
        self.lookahead = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.angular_gain = self.get_parameter("angular_gain").get_parameter_value().double_value
        self.max_yaw_rate = self.get_parameter("max_yaw_rate").get_parameter_value().double_value
        self.is_real_robot = self.get_parameter("is_real_robot").get_parameter_value().bool_value

        publish_rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value

        self.current_path: Optional[Path] = None
        self.request_seq = 0

        self.create_subscription(Path, "/path", self._path_callback, 5)
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.request_pub = self.create_publisher(Request, "/api/sport/request", 10)

        self.timer = self.create_timer(1.0 / max(publish_rate, 1e-3), self._on_timer)
        self.get_logger().info("Simple path follower ready (pure pursuit, no joystick).")

    def _path_callback(self, path: Path) -> None:
        if path.header.frame_id and path.header.frame_id != self.path_frame:
            self.get_logger().warn_once(
                f"Received path in '{path.header.frame_id}' but configured for '{self.path_frame}'. "
                "Continuing, but behavior may be undefined."
            )
        self.current_path = path

    def _on_timer(self) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.cmd_frame

        if not self.current_path or len(self.current_path.poses) == 0:
            self.cmd_pub.publish(cmd)
            if self.is_real_robot:
                self._publish_sport_request(cmd)
            return

        last_pose = self.current_path.poses[-1].pose.position
        goal_distance = math.hypot(last_pose.x, last_pose.y)
        if goal_distance < self.goal_tolerance:
            self.cmd_pub.publish(cmd)
            if self.is_real_robot:
                self._publish_sport_request(cmd)
            return

        target = self._select_target_point(self.current_path)
        heading = math.atan2(target[1], target[0])

        speed_cmd = min(self.max_speed, self.speed_gain * goal_distance)
        cmd.twist.linear.x = speed_cmd * math.cos(heading)
        cmd.twist.linear.y = speed_cmd * math.sin(heading)

        yaw_rate = self.angular_gain * heading
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))
        cmd.twist.angular.z = yaw_rate

        self.cmd_pub.publish(cmd)
        if self.is_real_robot:
            self._publish_sport_request(cmd)

    def _select_target_point(self, path: Path) -> tuple[float, float]:
        for pose in path.poses:
            point = pose.pose.position
            distance = math.hypot(point.x, point.y)
            if distance >= self.lookahead:
                return (point.x, point.y)
        last = path.poses[-1].pose.position
        return (last.x, last.y)

    def _publish_sport_request(self, cmd: TwistStamped) -> None:
        req = Request()
        req.header.identity.id = self.request_seq
        self.request_seq += 1

        lin_zero = abs(cmd.twist.linear.x) < 1e-4 and abs(cmd.twist.linear.y) < 1e-4
        ang_zero = abs(cmd.twist.angular.z) < 1e-4

        if lin_zero and ang_zero:
            req.header.identity.api_id = 1003  # ROBOT_SPORT_API_ID_STOPMOVE
            req.parameter = ""
        else:
            req.header.identity.api_id = 1008  # ROBOT_SPORT_API_ID_MOVE
            req.parameter = json.dumps(
                {"x": cmd.twist.linear.x, "y": cmd.twist.linear.y, "z": cmd.twist.angular.z}
            )
        self.request_pub.publish(req)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = SimplePathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

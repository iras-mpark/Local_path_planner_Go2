#!/usr/bin/env python3
"""Build a short-horizon LiDAR map anchored to a stable frame (e.g., odom)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import quaternion_matrix


@dataclass
class SlidingScan:
    stamp: float
    points: List[Tuple[float, float, float]]


class LocalSlidingMap(Node):
    """Aggregate the latest LiDAR scans into a rolling map."""

    def __init__(self) -> None:
        super().__init__("local_sliding_map")

        self.declare_parameter("input_topic", "/points_raw")
        self.declare_parameter("output_topic", "/local_sliding_map")
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("fallback_source_frame", "")
        self.declare_parameter("publish_rate_hz", 8.0)
        self.declare_parameter("window_seconds", 3.0)
        self.declare_parameter("max_scans", 15)
        self.declare_parameter("max_points", 40000)
        self.declare_parameter("min_points_to_publish", 200)
        self.declare_parameter("tf_timeout", 0.05)
        self.declare_parameter("max_range", 8.0)
        self.declare_parameter("min_range", 0.15)
        self.declare_parameter("z_min", -1.0)
        self.declare_parameter("z_max", 1.5)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.fallback_source_frame = (
            self.get_parameter("fallback_source_frame").get_parameter_value().string_value
        )
        self.publish_rate_hz = max(1e-3, self.get_parameter("publish_rate_hz").get_parameter_value().double_value)
        self.window_seconds = max(0.0, self.get_parameter("window_seconds").get_parameter_value().double_value)
        self.max_scans = max(1, self.get_parameter("max_scans").get_parameter_value().integer_value)
        self.max_points = max(100, self.get_parameter("max_points").get_parameter_value().integer_value)
        self.min_points_to_publish = max(
            0, self.get_parameter("min_points_to_publish").get_parameter_value().integer_value
        )
        self.tf_timeout = max(0.0, self.get_parameter("tf_timeout").get_parameter_value().double_value)
        self.min_range = max(0.0, self.get_parameter("min_range").get_parameter_value().double_value)
        self.max_range = max(self.min_range, self.get_parameter("max_range").get_parameter_value().double_value)
        self.z_min = self.get_parameter("z_min").get_parameter_value().double_value
        self.z_max = self.get_parameter("z_max").get_parameter_value().double_value

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self, qos=5)
        self.tf_timeout_duration = Duration(seconds=self.tf_timeout)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.scans: Deque[SlidingScan] = deque()
        self.total_points = 0
        self._last_warn_time = 0.0

        self.subscription = self.create_subscription(PointCloud2, self.input_topic, self._scan_callback, sensor_qos)
        self.publisher = self.create_publisher(PointCloud2, self.output_topic, map_qos)

        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._on_timer)
        self.get_logger().info(
            "Local sliding map ready "
            f"(window={self.window_seconds:.1f}s, target_frame={self.target_frame}, "
            f"input={self.input_topic}, output={self.output_topic})."
        )

    # ------------------------------------------------------------------ Callbacks
    def _scan_callback(self, cloud: PointCloud2) -> None:
        source_frame = cloud.header.frame_id or self.fallback_source_frame
        if not source_frame:
            self.get_logger().warn_once("Incoming cloud has no frame_id and no fallback was provided; ignoring.")
            return

        stamp = cloud.header.stamp
        lookup_time = Time(seconds=stamp.sec, nanoseconds=stamp.nanosec)
        if stamp.sec == 0 and stamp.nanosec == 0:
            lookup_time = Time()

        transform = self._lookup_transform(source_frame, lookup_time)
        if transform is None:
            return

        rotation = quaternion_matrix(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        )
        rot = (
            (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])),
            (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])),
            (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])),
        )
        translation = (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        )

        filtered_points: List[Tuple[float, float, float]] = []

        for x, y, z in point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            if not (self.z_min <= z <= self.z_max):
                continue
            tx, ty, tz = self._transform_point(x, y, z, rot, translation)
            distance = math.sqrt(tx * tx + ty * ty)
            if distance < self.min_range or distance > self.max_range:
                continue
            filtered_points.append((tx, ty, tz))

        if not filtered_points:
            return

        stamp_sec = float(cloud.header.stamp.sec) + float(cloud.header.stamp.nanosec) * 1e-9
        self.scans.append(SlidingScan(stamp_sec, filtered_points))
        self.total_points += len(filtered_points)
        self._prune_scans(stamp_sec)

    def _on_timer(self) -> None:
        if not self.scans or self.total_points < self.min_points_to_publish:
            return

        all_points: List[Tuple[float, float, float]] = []
        for scan in self.scans:
            all_points.extend(scan.points)

        header = Header()
        header.frame_id = self.target_frame
        header.stamp = self.get_clock().now().to_msg()

        if all_points:
            cloud = point_cloud2.create_cloud_xyz32(header, all_points)
        else:
            cloud = PointCloud2()
            cloud.header = header

        self.publisher.publish(cloud)

    # ------------------------------------------------------------------ Helpers
    def _prune_scans(self, current_stamp: float) -> None:
        cutoff = current_stamp - self.window_seconds
        while self.scans and (self.scans[0].stamp < cutoff):
            removed = self.scans.popleft()
            self.total_points -= len(removed.points)

        while len(self.scans) > self.max_scans:
            removed = self.scans.popleft()
            self.total_points -= len(removed.points)

        while self.total_points > self.max_points and self.scans:
            removed = self.scans.popleft()
            self.total_points -= len(removed.points)

        self.total_points = max(0, self.total_points)

    def _lookup_transform(self, source_frame: str, lookup_time: Time) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                lookup_time,
                timeout=self.tf_timeout_duration,
            )
        except TransformException as exc:
            try:
                return self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
            except TransformException:
                now_sec = self.get_clock().now().nanoseconds / 1e9
                if now_sec - self._last_warn_time > 1.0:
                    self.get_logger().warn(
                        f"Sliding map failed to get transform {self.target_frame} <- {source_frame}: {exc}"
                    )
                    self._last_warn_time = now_sec
                return None

    @staticmethod
    def _transform_point(
        x: float,
        y: float,
        z: float,
        rotation: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]],
        translation: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        tx = rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z + translation[0]
        ty = rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z + translation[1]
        tz = rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z + translation[2]
        return tx, ty, tz


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = LocalSlidingMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

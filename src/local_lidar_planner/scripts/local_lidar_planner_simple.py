#!/usr/bin/env python3
"""Lightweight LiDAR-only local planner.

Each scan is voxelized into a local occupancy grid, obstacles are inflated
by the configured safety radius, and an A* search produces a short collision-
free path toward the goal expressed in `path_frame`. The resulting plan plus
visual aids (`/goal_preview`, `/goal_raw`, `/local_obstacles`) make it easy to
debug in RViz while remaining standalone.
"""

from __future__ import annotations

import math
import heapq
from typing import Dict, List, Optional, Sequence, Tuple, Set

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import quaternion_matrix

class SimpleLocalPlanner(Node):
    """Generate a short collision-free path directly toward the waypoint."""

    def __init__(self) -> None:
        super().__init__("local_lidar_planner_simple")

        # Path construction parameters
        self.declare_parameter("path_frame", "sensor")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("path_resolution", 0.25)
        self.declare_parameter("max_path_length", 3.0)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("goal_offset", 1.0)

        # Obstacle handling parameters
        self.declare_parameter("safety_radius", 0.45)
        self.declare_parameter("max_considered_range", 4.0)
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("obstacle_z_min", -1.0)
        self.declare_parameter("obstacle_z_max", 1.0)
        self.declare_parameter("scan_topic", "/terrain_map_ext")
        self.declare_parameter("scan_frame", "map")
        self.declare_parameter("scan_tf_timeout", 0.05)
        self.declare_parameter("goal_tf_frame", "")
        self.declare_parameter("goal_tf_timeout", 0.5)
        self.declare_parameter("obstacle_topic", "/local_obstacles")
        self.declare_parameter("publish_debug_topics", False)

        self.path_frame: str = self.get_parameter("path_frame").get_parameter_value().string_value
        publish_rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.path_resolution = self.get_parameter("path_resolution").get_parameter_value().double_value
        self.max_path_length = self.get_parameter("max_path_length").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.goal_offset = self.get_parameter("goal_offset").get_parameter_value().double_value
        self.safety_radius = self.get_parameter("safety_radius").get_parameter_value().double_value
        self.max_considered_range = self.get_parameter("max_considered_range").get_parameter_value().double_value
        self.grid_resolution = max(
            0.05, self.get_parameter("grid_resolution").get_parameter_value().double_value
        )
        self.grid_radius_cells = max(1, int(math.ceil(self.max_considered_range / self.grid_resolution)))
        self.inflation_cells = max(0, int(math.ceil(self.safety_radius / self.grid_resolution)))
        self.obstacle_z_min = self.get_parameter("obstacle_z_min").get_parameter_value().double_value
        self.obstacle_z_max = self.get_parameter("obstacle_z_max").get_parameter_value().double_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.scan_frame = self.get_parameter("scan_frame").get_parameter_value().string_value
        self.scan_tf_timeout = max(0.0, self.get_parameter("scan_tf_timeout").get_parameter_value().double_value)
        self._scan_tf_timeout_duration = Duration(seconds=self.scan_tf_timeout)
        self.goal_tf_frame = self.get_parameter("goal_tf_frame").get_parameter_value().string_value
        if not self.goal_tf_frame:
            raise ValueError("goal_tf_frame parameter must be set (e.g., 'suitcase_frame').")
        self.goal_tf_timeout = self.get_parameter("goal_tf_timeout").get_parameter_value().double_value
        self.publish_debug_topics = self.get_parameter("publish_debug_topics").get_parameter_value().bool_value

        self.latest_obstacles: List[Tuple[float, float]] = []

        # TF tracking
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self, qos=5)
        self._last_tf_warn_time = 0.0
        self._last_goal_tf_stamp: Optional[Tuple[int, int]] = None
        self._last_goal_stale_warn_time = 0.0
        self._last_cloud_tf_warn_time = 0.0

        # Interfaces
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        debug_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(PointCloud2, self.scan_topic, self._scan_callback, sensor_qos)
        self.path_pub = self.create_publisher(Path, "/path", 10)
        self.goal_pub = self.create_publisher(PointStamped, "/goal_preview", 5)
        self.goal_raw_pub = self.create_publisher(PointStamped, "/goal_raw", 5)
        obstacle_topic = self.get_parameter("obstacle_topic").get_parameter_value().string_value
        if self.publish_debug_topics:
            self.obstacle_pub = self.create_publisher(PointCloud2, obstacle_topic, debug_qos)
            self.grid_pub = self.create_publisher(OccupancyGrid, "/local_obstacles_grid", debug_qos)
            self.potential_pub = self.create_publisher(OccupancyGrid, "/local_potential_grid", debug_qos)
        else:
            self.obstacle_pub = None
            self.grid_pub = None
            self.potential_pub = None

        self.timer = self.create_timer(1.0 / max(publish_rate_hz, 1e-3), self._on_timer)
        self.get_logger().info("Simple local planner ready (LiDAR-only, joystick-free).")

    # ------------------------------------------------------------------ Callbacks
    def _scan_callback(self, cloud: PointCloud2) -> None:
        """Convert the incoming point cloud into a filtered obstacle list."""
        obstacle_points: List[Tuple[float, float, float]] = []
        xy_points: List[Tuple[float, float]] = []

        source_frame = cloud.header.frame_id or self.scan_frame
        if not source_frame:
            self.get_logger().warn_once("Incoming cloud missing frame_id and no scan_frame fallback set.")
            return

        stamp_msg = cloud.header.stamp
        lookup_time = Time(seconds=stamp_msg.sec, nanoseconds=stamp_msg.nanosec)
        if stamp_msg.sec == 0 and stamp_msg.nanosec == 0:
            lookup_time = Time()

        transform = self._lookup_cloud_transform(source_frame, lookup_time)
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

        for x, y, z in point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            if not (self.obstacle_z_min <= z <= self.obstacle_z_max):
                continue

            tx, ty, tz = self._transform_point(x, y, z, rot, translation)
            distance = math.hypot(tx, ty)
            if distance < 1e-3 or distance > self.max_considered_range:
                continue

            obstacle_points.append((tx, ty, tz))
            xy_points.append((tx, ty))

        now_msg = self.get_clock().now().to_msg()
        if self.obstacle_pub is not None:
            obstacle_header = cloud.header
            obstacle_header.frame_id = self.path_frame
            if obstacle_points:
                obstacle_msg = point_cloud2.create_cloud_xyz32(obstacle_header, obstacle_points)
            else:
                obstacle_msg = PointCloud2()
                obstacle_msg.header = obstacle_header
            obstacle_msg.header.stamp = now_msg
            self.obstacle_pub.publish(obstacle_msg)
            self.grid_pub.publish(self._build_grid_map(xy_points, inflated=False, stamp=now_msg))
            self.potential_pub.publish(self._build_grid_map(xy_points, inflated=True, stamp=now_msg))
        self.latest_obstacles = xy_points

    # ------------------------------------------------------------------ Timer
    def _on_timer(self) -> None:
        path_msg = self._build_path()
        self.path_pub.publish(path_msg)

    # ------------------------------------------------------------------ Helpers
    def _build_path(self) -> Path:
        """Return a short path that respects the latest goal and LiDAR data."""
        now = self.get_clock().now()
        path = Path()
        path.header.stamp = now.to_msg()
        path.header.frame_id = self.path_frame

        rel_goal = self._lookup_goal_in_vehicle(now)
        if rel_goal is None:
            self._publish_goal_marker(0.0, 0.0, now)
            path.poses.append(self._pose_at(0.0, 0.0, 0.0, now))
            return path

        raw_distance = math.hypot(rel_goal[0], rel_goal[1])
        desired_heading = math.atan2(rel_goal[1], rel_goal[0])

        if raw_distance <= self.goal_offset:
            path.poses.append(self._pose_at(0.0, 0.0, desired_heading, now))
            self._publish_goal_marker(0.0, 0.0, now)
            return path

        goal_distance = max(0.0, raw_distance - self.goal_offset)

        if goal_distance < self.goal_tolerance:
            path.poses.append(self._pose_at(0.0, 0.0, desired_heading, now))
            self._publish_goal_marker(0.0, 0.0, now)
            return path

        max_travel = self.max_path_length
        if self.max_considered_range > 0:
            max_travel = min(self.max_path_length, self.max_considered_range)
        if max_travel <= 0:
            max_travel = goal_distance
        if goal_distance > max_travel and goal_distance > 1e-6:
            scale = max_travel / goal_distance
            goal_distance = max_travel
            rel_goal = (rel_goal[0] * scale, rel_goal[1] * scale)

        path_points = self._plan_path_astar(rel_goal[0], rel_goal[1])
        if not path_points:
            path.poses.append(self._pose_at(0.0, 0.0, desired_heading, now))
            self._publish_goal_marker(0.0, 0.0, now)
            return path

        prev = path_points[0]
        for current in path_points[1:]:
            heading = math.atan2(current[1] - prev[1], current[0] - prev[0])
            path.poses.append(self._pose_at(current[0], current[1], heading, now))
            prev = current

        if path.poses:
            last_pose = path.poses[-1].pose.position
            self._publish_goal_marker(last_pose.x, last_pose.y, now)
        else:
            self._publish_goal_marker(0.0, 0.0, now)
        return path

    def _lookup_goal_in_vehicle(self, stamp: Time) -> Optional[Tuple[float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.path_frame,
                self.goal_tf_frame,
                Time(),
                timeout=Duration(seconds=self.goal_tf_timeout),
            )
            rel_x = transform.transform.translation.x
            rel_y = transform.transform.translation.y
            raw_msg = PointStamped()
            raw_msg.header.stamp = stamp.to_msg()
            raw_msg.header.frame_id = self.path_frame
            raw_msg.point.x = rel_x
            raw_msg.point.y = rel_y
            self.goal_raw_pub.publish(raw_msg)
            tf_stamp = (transform.header.stamp.sec, transform.header.stamp.nanosec)
            if (
                self._last_goal_tf_stamp is not None
                and tf_stamp == self._last_goal_tf_stamp
                and tf_stamp != (0, 0)
            ):
                now_sec = self.get_clock().now().nanoseconds / 1e9
                if now_sec - self._last_goal_stale_warn_time > 2.0:
                    self.get_logger().warn("Goal TF timestamp is stale. Holding position.")
                    self._last_goal_stale_warn_time = now_sec
                return None
            self._last_goal_tf_stamp = tf_stamp
            return rel_x, rel_y
        except TransformException as exc:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - self._last_tf_warn_time > 2.0:
                self.get_logger().warn(f"TF lookup failed ({self.goal_tf_frame}): {exc}")
                self._last_tf_warn_time = now_sec
            return None

    def _lookup_cloud_transform(self, source_frame: str, lookup_time: Time) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                self.path_frame,
                source_frame,
                lookup_time,
                timeout=self._scan_tf_timeout_duration,
            )
        except TransformException as exc:
            # If the transform is only slightly out of date/future, fall back to the latest available transform.
            exc_text = str(exc).lower()
            if "extrapolation" in exc_text or "would require" in exc_text:
                try:
                    return self.tf_buffer.lookup_transform(
                        self.path_frame,
                        source_frame,
                        Time(),
                        timeout=Duration(seconds=0.0),
                    )
                except TransformException:
                    pass

            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - self._last_cloud_tf_warn_time > 1.5:
                self.get_logger().warn(
                    f"Failed to transform cloud from {source_frame} to {self.path_frame}: {exc}"
                )
                self._last_cloud_tf_warn_time = now_sec
            return None

    def _build_occupancy(self) -> Set[Tuple[int, int]]:
        occupancy: Set[Tuple[int, int]] = set()
        if not self.latest_obstacles:
            return occupancy

        max_cells = self.grid_radius_cells
        for cell in self._cells_from_points(self.latest_obstacles):
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                    if dx * dx + dy * dy > self.inflation_cells * self.inflation_cells:
                        continue
                    occ = (cell[0] + dx, cell[1] + dy)
                    if abs(occ[0]) <= max_cells and abs(occ[1]) <= max_cells:
                        occupancy.add(occ)
        return occupancy

    def _plan_path_astar(self, goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        occupancy = self._build_occupancy()
        grid_limit = self.grid_radius_cells * self.grid_resolution
        goal_distance = math.hypot(goal_x, goal_y)
        if goal_distance > grid_limit and goal_distance > 1e-6:
            scale = grid_limit / goal_distance
            goal_x *= scale
            goal_y *= scale
        start_cell = (0, 0)
        goal_cell = self._world_to_cell(goal_x, goal_y)

        if goal_cell == start_cell:
            return [(0.0, 0.0)]

        if goal_cell in occupancy:
            goal_cell = self._find_nearest_free(goal_cell, occupancy)
            if goal_cell is None:
                return None

        def heuristic(cell: Tuple[int, int]) -> float:
            return math.hypot(goal_cell[0] - cell[0], goal_cell[1] - cell[1])

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start_cell), 0.0, start_cell))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        max_cells = self.grid_radius_cells
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current == goal_cell:
                return self._reconstruct_path(came_from, current)

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if abs(neighbor[0]) > max_cells or abs(neighbor[1]) > max_cells:
                    continue
                if neighbor != start_cell and neighbor in occupancy:
                    continue
                step = math.hypot(dx, dy)
                tentative_cost = current_cost + step
                if tentative_cost < g_cost.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_cost
                    priority = tentative_cost + heuristic(neighbor)
                    heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

        return None

    def _reconstruct_path(
        self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        points = [self._cell_to_world(cell) for cell in cells]
        # drop the duplicated start point to avoid zero-length segment
        if len(points) > 1 and math.hypot(points[1][0], points[1][1]) < 1e-6:
            points = points[1:]
        return points

    def _find_nearest_free(
        self, start_cell: Tuple[int, int], occupancy: Set[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if start_cell not in occupancy:
            return start_cell
        max_cells = self.grid_radius_cells
        max_radius = max_cells
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    candidate = (start_cell[0] + dx, start_cell[1] + dy)
                    if abs(candidate[0]) > max_cells or abs(candidate[1]) > max_cells:
                        continue
                    if candidate not in occupancy:
                        return candidate
        return None

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(round(x / self.grid_resolution)),
            int(round(y / self.grid_resolution)),
        )

    def _cell_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        return (cell[0] * self.grid_resolution, cell[1] * self.grid_resolution)

    def _cells_from_points(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        cells: List[Tuple[int, int]] = []
        max_range = self.max_considered_range
        for x, y in points:
            if math.hypot(x, y) > max_range:
                continue
            cells.append(self._world_to_cell(x, y))
        return cells

    def _build_grid_map(self, points: List[Tuple[float, float]], inflated: bool, stamp) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.path_frame
        grid.info.resolution = self.grid_resolution
        width = height = self.grid_radius_cells * 2 + 1
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = -self.grid_radius_cells * self.grid_resolution
        grid.info.origin.position.y = -self.grid_radius_cells * self.grid_resolution
        grid.info.origin.orientation.w = 1.0

        total_cells = width * height
        data = [-1] * total_cells

        cells = set(self._cells_from_points(points))
        if inflated and self.inflation_cells > 0:
            inflated_cells: Set[Tuple[int, int]] = set()
            max_cells = self.grid_radius_cells
            for cell in cells:
                for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                    for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                        if dx * dx + dy * dy > self.inflation_cells * self.inflation_cells:
                            continue
                        occ = (cell[0] + dx, cell[1] + dy)
                        if abs(occ[0]) <= max_cells and abs(occ[1]) <= max_cells:
                            inflated_cells.add(occ)
            cells = inflated_cells

        for cell in cells:
            idx = self._cell_to_grid_index(cell, width)
            if 0 <= idx < total_cells:
                data[idx] = 100 if inflated else 75

        grid.data = data
        return grid

    def _cell_to_grid_index(self, cell: Tuple[int, int], width: int) -> int:
        x_idx = cell[0] + self.grid_radius_cells
        y_idx = cell[1] + self.grid_radius_cells
        return y_idx * width + x_idx

    def _pose_at(self, x: float, y: float, heading: float, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp.to_msg()
        pose.header.frame_id = self.path_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(heading / 2.0)
        pose.pose.orientation.w = math.cos(heading / 2.0)
        return pose

    def _publish_goal_marker(self, x: float, y: float, stamp) -> None:
        goal = PointStamped()
        goal.header.stamp = stamp.to_msg()
        goal.header.frame_id = self.path_frame
        goal.point.x = x
        goal.point.y = y
        self.goal_pub.publish(goal)

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
    node = SimpleLocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

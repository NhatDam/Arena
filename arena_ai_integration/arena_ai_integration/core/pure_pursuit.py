"""Pure Pursuit path tracking controller."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class PurePursuitController:
    """Track a local-frame waypoint sequence with standard Pure Pursuit."""

    def __init__(
        self,
        look_ahead_dist: float = 0.5,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 1.5,
        min_look_ahead_dist: float = 0.1,
    ):
        self.L_d = look_ahead_dist
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.min_L_d = max(min_look_ahead_dist, 1e-3)

    def compute_control(
        self,
        current_pos: np.ndarray,
        current_yaw: float,
        waypoints: np.ndarray,
        dt: float = 0.1,
    ) -> Tuple[float, float]:
        if len(waypoints) == 0:
            return 0.0, 0.0

        target = self._find_lookahead_point(current_pos, waypoints)

        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]

        cos_yaw = math.cos(current_yaw)
        sin_yaw = math.sin(current_yaw)

        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw

        alpha = math.atan2(local_y, local_x)
        lookahead = max(math.hypot(local_x, local_y), self.min_L_d)
        curvature = 2.0 * math.sin(alpha) / lookahead

        heading_factor = math.cos(alpha) ** 2
        linear_vel = self.max_linear_vel * heading_factor
        angular_vel = linear_vel * curvature
        angular_vel = float(np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel))
        linear_vel = float(np.clip(linear_vel, 0.0, self.max_linear_vel))

        return linear_vel, angular_vel

    def compute_twist_from_local_waypoints(self, waypoints: np.ndarray):
        """Return a geometry_msgs/Twist for robot-at-origin local waypoints."""
        from geometry_msgs.msg import Twist

        cmd = Twist()
        if waypoints is None or len(waypoints) == 0:
            return cmd

        target = self._simple_lookahead(waypoints)
        local_x = float(target[0])
        local_y = float(target[1])
        lookahead = max(math.hypot(local_x, local_y), self.min_L_d)
        alpha = math.atan2(local_y, local_x)
        curvature = 2.0 * math.sin(alpha) / lookahead

        forward_gain = max(0.0, math.cos(alpha))
        linear = self.max_linear_vel * forward_gain * forward_gain
        if 0.0 < linear < 0.05 and abs(alpha) < 1.2:
            linear = 0.05

        angular = linear * curvature
        if linear == 0.0 and abs(alpha) > 0.05:
            angular = math.copysign(
                min(self.max_angular_vel, max(0.2, abs(alpha))),
                alpha,
            )

        cmd.linear.x = max(-self.max_linear_vel, min(self.max_linear_vel, linear))
        cmd.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, angular))
        return cmd

    def _simple_lookahead(self, waypoints: np.ndarray) -> np.ndarray:
        for wp in waypoints:
            if math.hypot(float(wp[0]), float(wp[1])) >= self.L_d:
                return wp
        return waypoints[-1]

    def _find_lookahead_point(
        self,
        current_pos: np.ndarray,
        waypoints: np.ndarray,
    ) -> np.ndarray:
        closest_idx = int(np.argmin(np.linalg.norm(waypoints - current_pos, axis=1)))

        for i in range(closest_idx, len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            intersect = self._circle_segment_intersect(current_pos, self.L_d, p1, p2)
            if intersect is not None:
                return intersect

        return waypoints[-1]

    @staticmethod
    def _circle_segment_intersect(
        center: np.ndarray,
        radius: float,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> Optional[np.ndarray]:
        d = p2 - p1
        f = p1 - center

        a = float(np.dot(d, d))
        b = float(2.0 * np.dot(f, d))
        c = float(np.dot(f, f) - radius ** 2)

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        for t in (t2, t1):
            if 0.0 <= t <= 1.0:
                return p1 + t * d

        return None

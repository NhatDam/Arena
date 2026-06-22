"""DWB trajectory hard-gate selection and waypoint regeneration."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
from dwb_msgs.msg import LocalPlanEvaluation
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DWBHardGateAdapter:
    """Select DWB candidates that pass through a gated AI waypoint."""

    def __init__(
        self,
        gate_waypoint_index: int = 3,
        waypoint_gate_radius: float = 0.25,
        regeneration_max_attempts: int = 3,
        regeneration_min_wp4_delta: float = 0.20,
        rejected_waypoint_memory: int = 12,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 1.5,
    ):
        self.gate_waypoint_index = gate_waypoint_index
        self.waypoint_gate_radius = waypoint_gate_radius
        self.regeneration_max_attempts = max(0, regeneration_max_attempts)
        self.regeneration_min_wp4_delta = regeneration_min_wp4_delta
        self.rejected_gate_waypoints: Deque[np.ndarray] = deque(
            maxlen=max(1, rejected_waypoint_memory)
        )
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

    def clear(self) -> None:
        self.rejected_gate_waypoints.clear()

    def select_best_candidate(
        self,
        waypoints: np.ndarray,
        eval_msg: LocalPlanEvaluation,
        current_odom: Odometry,
        logger=None,
    ) -> Tuple[Twist, np.ndarray]:
        context = self._dwb_eval_context(eval_msg, current_odom)
        selected_waypoints = np.array(waypoints, copy=True)
        local_rejections = []

        for attempt in range(self.regeneration_max_attempts + 1):
            selected = self._select_gated_dwb_candidate(selected_waypoints, eval_msg, context)
            if selected is not None:
                cmd, _best_cand = selected
                if logger is not None:
                    logger.info(
                        f"DWB hard gate accepted candidate after regen_attempt={attempt}",
                        throttle_duration_sec=1.0,
                    )
                self.rejected_gate_waypoints.clear()
                return cmd, selected_waypoints

            gate_wp = self._gate_waypoint(selected_waypoints)
            if gate_wp is not None:
                rejected = np.array(gate_wp, dtype=np.float32)
                local_rejections.append(rejected)
                self.rejected_gate_waypoints.append(rejected)

            if attempt >= self.regeneration_max_attempts:
                break

            regenerated = self._regenerate_waypoints_from_dwb(
                waypoints,
                eval_msg,
                context,
                local_rejections,
            )
            if regenerated is None:
                break
            selected_waypoints = regenerated

        if logger is not None:
            logger.warn(
                "No DWB trajectory passed through the AI waypoint gate; holding position.",
                throttle_duration_sec=2.0,
            )
        return Twist(), selected_waypoints

    def _dwb_eval_context(self, eval_msg: LocalPlanEvaluation, odom: Odometry):
        rot = odom.pose.pose.orientation
        yaw = math.atan2(
            2 * (rot.w * rot.z + rot.x * rot.y),
            1 - 2 * (rot.y ** 2 + rot.z ** 2),
        )
        frame_id = (eval_msg.header.frame_id or '').lower()
        return {
            'cx': float(odom.pose.pose.position.x),
            'cy': float(odom.pose.pose.position.y),
            'yaw': yaw,
            'is_global': 'odom' in frame_id or 'map' in frame_id,
        }

    def _trajectory_to_local_array(self, poses, context) -> np.ndarray:
        if not poses:
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.zeros((len(poses), 2), dtype=np.float32)
        yaw = context['yaw']
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cx = context['cx']
        cy = context['cy']
        for idx, pose in enumerate(poses):
            px = float(pose.x)
            py = float(pose.y)
            if context['is_global']:
                dx = px - cx
                dy = py - cy
                arr[idx] = [
                    dx * cos_yaw + dy * sin_yaw,
                    -dx * sin_yaw + dy * cos_yaw,
                ]
            else:
                arr[idx] = [px, py]
        return arr

    def _gate_waypoint_index(self, waypoints: np.ndarray) -> Optional[int]:
        if waypoints is None or len(waypoints) == 0:
            return None
        return min(max(self.gate_waypoint_index, 0), len(waypoints) - 1)

    def _gate_waypoint(self, waypoints: np.ndarray) -> Optional[np.ndarray]:
        gate_idx = self._gate_waypoint_index(waypoints)
        if gate_idx is None:
            return None
        return np.asarray(waypoints[gate_idx], dtype=np.float32)

    def _select_gated_dwb_candidate(self, waypoints, eval_msg, context):
        gate_wp = self._gate_waypoint(waypoints)
        if gate_wp is None:
            return None

        best_cmd = None
        best_score = float('inf')
        best_cand = None

        for twist in eval_msg.twists:
            if twist.total < 0.0 or not twist.traj.poses:
                continue

            traj_arr = self._trajectory_to_local_array(twist.traj.poses, context)
            if len(traj_arr) == 0:
                continue

            gate_dist = float(np.min(np.linalg.norm(traj_arr - gate_wp, axis=1)))
            if gate_dist > self.waypoint_gate_radius:
                continue

            combined = gate_dist + 0.1 * float(twist.total)
            if combined < best_score:
                best_score = combined
                best_cand = {'gate_dist': gate_dist, 'cost': float(twist.total)}
                best_cmd = Twist()
                best_cmd.linear.x = float(twist.traj.velocity.x)
                best_cmd.angular.z = float(twist.traj.velocity.theta)

        if best_cmd is None:
            return None

        best_cmd.linear.x = max(-self.max_linear_vel, min(self.max_linear_vel, best_cmd.linear.x))
        best_cmd.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, best_cmd.angular.z))
        return best_cmd, best_cand

    def _regenerate_waypoints_from_dwb(
        self,
        base_waypoints: np.ndarray,
        eval_msg: LocalPlanEvaluation,
        context,
        local_rejections,
    ) -> Optional[np.ndarray]:
        gate_idx = self._gate_waypoint_index(base_waypoints)
        if gate_idx is None:
            return None

        old_gate = np.asarray(base_waypoints[gate_idx], dtype=np.float32)
        rejected = list(self.rejected_gate_waypoints) + list(local_rejections)
        best_point = None
        best_score = float('inf')

        for twist in eval_msg.twists:
            if twist.total < 0.0 or not twist.traj.poses:
                continue
            traj_arr = self._trajectory_to_local_array(twist.traj.poses, context)
            for point in traj_arr:
                if float(np.linalg.norm(point)) < 0.05:
                    continue
                if any(
                    float(np.linalg.norm(point - prev)) < self.regeneration_min_wp4_delta
                    for prev in rejected
                ):
                    continue
                score = float(np.linalg.norm(point - old_gate)) + 0.1 * float(twist.total)
                if score < best_score:
                    best_score = score
                    best_point = np.array(point, dtype=np.float32)

        if best_point is None:
            return None

        return self._repair_waypoints_to_gate(base_waypoints, gate_idx, best_point)

    @staticmethod
    def _repair_waypoints_to_gate(
        waypoints: np.ndarray,
        gate_idx: int,
        new_gate: np.ndarray,
    ) -> np.ndarray:
        repaired = np.array(waypoints, copy=True)
        old_gate = np.asarray(repaired[gate_idx], dtype=np.float32)
        delta = new_gate - old_gate
        count = len(repaired)
        for idx in range(count):
            if idx <= gate_idx:
                factor = float(idx + 1) / float(gate_idx + 1)
            else:
                denom = max(1, count - gate_idx - 1)
                factor = max(0.0, 1.0 - float(idx - gate_idx) / float(denom))
            repaired[idx] = repaired[idx] + factor * delta
        repaired[gate_idx] = new_gate
        return repaired

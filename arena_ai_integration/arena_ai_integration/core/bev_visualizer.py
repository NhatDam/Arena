"""Bird's-eye-view matplotlib visualization for AI + DWB debug."""

from __future__ import annotations

import math
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cv_bridge import CvBridge
from dwb_msgs.msg import LocalPlanEvaluation
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.time import Time
from sensor_msgs.msg import Image


class BEVVisualizer:
    """Render a local-frame BEV debug image and publish as sensor_msgs/Image."""

    def __init__(self, cv_bridge: Optional[CvBridge] = None):
        self.cv_bridge = cv_bridge or CvBridge()

    def render(
        self,
        *,
        agent_label: str,
        ai_waypoints: np.ndarray,
        current_odom: Odometry,
        odom_history,
        latest_eval: Optional[LocalPlanEvaluation],
        last_eval_time: Optional[Time],
        max_eval_staleness_sec: float,
        latest_global_path: Optional[NavPath],
        latest_ai_path: Optional[NavPath],
        phase_local_goal_odom,
        path_phase: str,
        local_goal_relock_mode: str,
        phase_local_goal_lock_count: int,
        phase_local_goal_reached_radius: float,
        path_waypoint_index_fn: Callable[[np.ndarray], Optional[int]],
        path_to_local_array_fn: Callable,
        dwb_traj_to_local_array_fn: Callable,
        best_dwb_eval_trajectory_fn: Callable,
        point_in_odom_to_local_fn: Callable,
        distance_to_phase_local_goal_fn: Callable,
        goal_position_in_odom_frame_fn: Callable,
        enable_human_tracking: bool,
        human_tracker,
        robot_frame: str,
        clock_now_fn: Callable,
        dwb_integration_mode: str = 'path_adapter',
        shaped_path_num_waypoints: int = 4,
    ) -> Image:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=80)

        cx = current_odom.pose.pose.position.x
        cy = current_odom.pose.pose.position.y
        rot = current_odom.pose.pose.orientation
        yaw = math.atan2(
            2 * (rot.w * rot.z + rot.x * rot.y),
            1 - 2 * (rot.y ** 2 + rot.z ** 2),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        plot_bounds = [np.asarray([[0.0, 0.0]], dtype=np.float32)]

        ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=2.5, label="Robot", zorder=5)
        ax.annotate(
            "",
            xy=(0.5, 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
            zorder=5,
        )

        if len(odom_history) > 1:
            trail_points = []
            for px, py, _ in odom_history:
                dx = px - cx
                dy = py - cy
                trail_points.append((
                    dx * cos_yaw + dy * sin_yaw,
                    -dx * sin_yaw + dy * cos_yaw,
                ))
            trail_local = np.asarray(trail_points, dtype=np.float32)
            plot_bounds.append(trail_local)
            ax.plot(
                trail_local[:, 0],
                trail_local[:, 1],
                color='green',
                linewidth=2.2,
                marker='.',
                markersize=4,
                label="Actual robot path (AI+DWB)",
                zorder=3,
            )

        ai_segment = None
        wp_idx = None
        inserted_wp = None
        if ai_waypoints is not None and len(ai_waypoints) > 0:
            wp_idx = path_waypoint_index_fn(ai_waypoints)
            if wp_idx is not None:
                ai_segment = np.asarray(ai_waypoints[wp_idx:], dtype=np.float32)
                inserted_wp = np.asarray(ai_waypoints[wp_idx], dtype=np.float32)

        eval_msg = None
        if latest_eval is not None and last_eval_time is not None:
            eval_age = (clock_now_fn() - last_eval_time).nanoseconds / 1e9
            if eval_age <= max_eval_staleness_sec:
                eval_msg = latest_eval

        if eval_msg is not None and eval_msg.twists:
            target_draw_count = 24
            step = max(1, len(eval_msg.twists) // target_draw_count)
            for idx, twist in enumerate(eval_msg.twists):
                if twist.total < 0.0 or not twist.traj.poses or idx % step != 0:
                    continue
                traj_local = dwb_traj_to_local_array_fn(
                    twist.traj.poses,
                    eval_msg.header.frame_id,
                )
                if traj_local is not None and len(traj_local) > 1:
                    plot_bounds.append(traj_local)
                    ax.plot(
                        traj_local[:, 0],
                        traj_local[:, 1],
                        color='gray',
                        alpha=0.22,
                        linewidth=1.0,
                        zorder=1,
                    )

        # selected_traj = best_dwb_eval_trajectory_fn(eval_msg)
        # if selected_traj is not None and len(selected_traj) > 1:
        #     plot_bounds.append(selected_traj)
        #     ax.plot(
        #         selected_traj[:, 0],
        #         selected_traj[:, 1],
        #         color='#0066FF',
        #         linewidth=3.2,
        #         label="DWB baseline path (no AI)",
        #         zorder=4,
        #     )

        if ai_segment is not None and inserted_wp is not None and wp_idx is not None:
            plot_bounds.append(ai_segment)
            if len(ai_segment) > 1:
                ax.plot(
                    ai_segment[:, 0],
                    ai_segment[:, 1],
                    'r--',
                    linewidth=3.0,
                    label=f"AI waypoints WP{wp_idx + 1}-WP{len(ai_waypoints)}",
                    zorder=6,
                )
            ax.scatter(ai_segment[:, 0], ai_segment[:, 1], c='red', s=40, zorder=7)
            ax.scatter(
                [inserted_wp[0]],
                [inserted_wp[1]],
                c='orange',
                s=140,
                marker='D',
                edgecolors='black',
                label=f"AI WP{wp_idx + 1}",
                zorder=8,
            )

        if enable_human_tracking and human_tracker.is_ready():
            h_pos_hist = human_tracker.get_human_positions()
            if h_pos_hist is not None and len(h_pos_hist) > 0:
                current_humans = h_pos_hist[-1]
                hx_list, hy_list = [], []
                for h in current_humans:
                    if abs(h[0]) > 0.01 or abs(h[1]) > 0.01:
                        dxh = h[0] - cx
                        dyh = h[1] - cy
                        hx_list.append(dxh * cos_yaw + dyh * sin_yaw)
                        hy_list.append(-dxh * sin_yaw + dyh * cos_yaw)
                if hx_list:
                    ax.scatter(
                        hx_list,
                        hy_list,
                        c='blue',
                        marker='o',
                        s=80,
                        edgecolors='black',
                        label="Humans",
                        zorder=7,
                    )

        ax.set_title(
            f"{agent_label} AI-DWB BEV",
            fontsize=12,
            fontweight='bold',
        )
        ax.set_xlabel("Local X (m) - Forward", fontsize=11)
        ax.set_ylabel("Local Y (m) - Left", fontsize=11)
        ax.axis("equal")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

        all_points = np.vstack([arr for arr in plot_bounds if arr is not None and len(arr) > 0])
        x_min = min(-0.5, float(np.min(all_points[:, 0])) - 0.6)
        x_max = max(5.5, float(np.max(all_points[:, 0])) + 0.6)
        y_min = min(-3.5, float(np.min(all_points[:, 1])) - 0.6)
        y_max = max(3.5, float(np.max(all_points[:, 1])) + 0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        fig.canvas.draw()
        try:
            img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            img_np = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        img_msg = self.cv_bridge.cv2_to_imgmsg(img_np, encoding="rgb8")
        img_msg.header.stamp = clock_now_fn().to_msg()
        img_msg.header.frame_id = robot_frame
        return img_msg

#!/usr/bin/env python3
"""Unified ROS 2 AI controller node with DWB path adaptation."""

from __future__ import annotations

import copy
import json
import math
import sys
import threading
import traceback
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from dwb_msgs.msg import LocalPlanEvaluation
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Odometry, Path as NavPath
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from arena_ai_integration.agents.base_agent import BaseAgent, PredictionContext
from arena_ai_integration.core.dwb_adapter import DWBHardGateAdapter
from arena_ai_integration.core.human_tracker import HumanPositionTracker

try:
    import torch
    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on deployment env
    torch = None
    TORCH_IMPORT_ERROR = exc

class BaseAINode(Node):
    def __init__(self, agent: BaseAgent):
        super().__init__('ai_controller')
        self.agent = agent
        self.topic_prefix = agent.topic_prefix

        self.declare_parameters(
            namespace='',
            parameters=[
                ('agent_name',           '', ParameterDescriptor()),
                ('model_config_path',     str(agent.default_config_path()), ParameterDescriptor()),
                ('model_checkpoint_path', '', ParameterDescriptor()),
                ('history_length',        8,  ParameterDescriptor()),
                ('control_frequency',     agent.config.control_frequency, ParameterDescriptor()),
                ('arrival_threshold',     agent.config.arrival_threshold, ParameterDescriptor()),
                ('use_arrival_completion', False, ParameterDescriptor()),
                ('controller_reset_enabled', False, ParameterDescriptor(
                    description='Allow this AI controller to call reset_task; keep false for benchmark runs')),
                ('goal_completion_radius', 0.25, ParameterDescriptor()),
                ('reset_on_eval_dropout', False, ParameterDescriptor()),
                ('max_eval_staleness_sec', 2.0, ParameterDescriptor()),
                ('initial_eval_wait_sec', 2.0, ParameterDescriptor()),
                ('startup_data_timeout_sec', 30.0, ParameterDescriptor()),
                ('fail_on_startup_timeout', False, ParameterDescriptor()),
                ('odom_staleness_sec', 2.0, ParameterDescriptor()),
                ('image_staleness_sec', 2.0, ParameterDescriptor()),
                # fallback_to_dwb vẫn giữ để dùng khi model AI lỗi liên tiếp
                ('fallback_to_dwb', False, ParameterDescriptor()),
                ('dwb_cmd_staleness_sec', 1.0, ParameterDescriptor()),
                ('enable_bev_visualization', False, ParameterDescriptor()),
                ('bev_visualization_period_sec', 1.0, ParameterDescriptor()),
                ('look_ahead_distance',   agent.config.look_ahead_distance, ParameterDescriptor()),
                ('max_linear_velocity',   1.0, ParameterDescriptor()),
                ('max_angular_velocity',  1.5, ParameterDescriptor()),
                ('enable_human_tracking', True, ParameterDescriptor()),
                ('max_humans',            10,  ParameterDescriptor()),
                ('human_context_radius',  10.0, ParameterDescriptor(
                    description='Only pass humans within this robot-local radius to SocialNav; <= 0 disables filtering')),
                ('path_waypoint_index',    agent.config.path_waypoint_index, ParameterDescriptor(
                    description='Zero-based AI waypoint index inserted into the DWB FollowPath path')),
                ('ai_rejoin_skip_distance', agent.config.rejoin_skip_distance, ParameterDescriptor(
                    description='Meters to advance along the global path after the nearest AI waypoint rejoin point')),
                ('phase_local_goal_enabled', True, ParameterDescriptor(
                    description='Use AI local subgoals before returning to the benchmark/global goal')),
                ('phase_local_goal_reached_radius', 0.45, ParameterDescriptor(
                    description='Distance in meters for considering the AI local waypoint reached')),
                ('phase_local_goal_timeout_sec', 8.0, ParameterDescriptor(
                    description='Maximum seconds to chase the AI local waypoint before returning to the benchmark goal; <= 0 disables timeout')),
                ('local_goal_relock_mode', 'periodic', ParameterDescriptor(
                    description='AI local goal mode: once, reached, or periodic')),
                ('rolling_local_goal_final_radius', 0.8, ParameterDescriptor(
                    description='Switch from rolling AI local goals to the benchmark/global goal inside this radius; <= 0 disables')),
                ('rolling_relock_period_sec', 1.5, ParameterDescriptor(
                    description='Period for refreshing the AI local subgoal in periodic mode')),
                ('rolling_min_relock_shift', 0.25, ParameterDescriptor(
                    description='Minimum odom-frame shift required before replacing the current local subgoal')),
                ('rolling_min_local_goal_distance', 0.25, ParameterDescriptor(
                    description='Reject AI local subgoals closer than this distance from the robot; <= 0 disables')),
                ('rolling_max_local_goal_distance', 4.0, ParameterDescriptor(
                    description='Reject AI local subgoals farther than this distance from the robot; <= 0 disables')),
                ('path_update_period_sec', 1.5, ParameterDescriptor(
                    description='Minimum period between SocialNav path requests sent to Nav2')),
                ('compute_path_timeout_sec', 2.5, ParameterDescriptor(
                    description='Maximum seconds to wait for ComputePathToPose before sending a minimal AI FollowPath path; <= 0 disables timeout')),
                ('planner_action_name', '', ParameterDescriptor(
                    description='Nav2 ComputePathToPose action name; defaults to <robot_namespace>/compute_path_to_pose')),
                ('follow_path_action_name', '', ParameterDescriptor(
                    description='Nav2 FollowPath action name; defaults to <robot_namespace>/follow_path')),
                ('planner_id', 'GridBased', ParameterDescriptor()),
                ('follow_path_controller_id', 'FollowPath', ParameterDescriptor()),
                ('follow_path_goal_checker_id', '', ParameterDescriptor()),
                ('robot_namespace', '/task_generator_node/turtlebot', ParameterDescriptor()),
                ('instruction_topic', '/nav_instruction', ParameterDescriptor()),
                ('human_detections_topic', '/detections/humans', ParameterDescriptor()),
                ('dwb_cmd_topic', '', ParameterDescriptor()),
                ('reset_service_name', '', ParameterDescriptor()),
                ('robot_frame', '', ParameterDescriptor()),
                ('image_topic', '', ParameterDescriptor()),
                ('publish_odom_tf', False, ParameterDescriptor(
                    description='Publish odom->base TF from odometry only if no other TF source exists')),
                # ai_inference_fail_threshold: số tick lỗi liên tiếp trước khi chuyển DWB fallback
                ('ai_inference_fail_threshold', 3, ParameterDescriptor(
                    description='Consecutive AI inference failures before falling back to DWB')),
                ('goal_frame', 'odom', ParameterDescriptor(
                    description='Deprecated; AI waypoints are inserted into FollowPath paths')),
                ('flip_y_axis', agent.config.flip_y_axis, ParameterDescriptor(
                    description='Flip sign of model output Y to match ROS convention (y positive = left)')),
                ('coordinate_mode', str(agent.config.extra_params.get('coordinate_mode', '')), ParameterDescriptor(
                    description='Agent-specific waypoint coordinate conversion mode')),
                ('waypoint_scale', float(agent.config.extra_params.get('waypoint_scale', 1.0)), ParameterDescriptor(
                    description='Optional agent-specific waypoint output scale')),
                ('dwb_integration_mode', 'path_adapter', ParameterDescriptor(
                    description=(
                        'AI-DWB integration: none, path_adapter, shaped_path, '
                        'one_waypoint_replace, or hard_gate'
                    ))),
                ('shaped_path_num_waypoints', 4, ParameterDescriptor(
                    description='Number of leading AI waypoints inserted into the shaped FollowPath path')),
                ('social_cost_hard_radius', 0.35, ParameterDescriptor()),
                ('social_cost_personal_radius', 1.0, ParameterDescriptor()),
                ('social_cost_social_radius', 1.5, ParameterDescriptor()),
                ('social_cost_w_hard', 100.0, ParameterDescriptor()),
                ('social_cost_w_personal', 5.0, ParameterDescriptor()),
                ('social_cost_w_social', 1.0, ParameterDescriptor()),
                ('social_cost_w_global', 0.25, ParameterDescriptor()),
                ('social_cost_w_progress', 0.5, ParameterDescriptor()),
                ('social_cost_w_smooth', 0.1, ParameterDescriptor()),
                ('use_dwb_hard_gate', False, ParameterDescriptor(
                    description='Use DWBHardGateAdapter to select a DWB candidate through the AI waypoint')),
                ('dwb_hard_gate_radius', 0.25, ParameterDescriptor(
                    description='Maximum distance in meters between a DWB candidate and the gated AI waypoint')),
                ('dwb_hard_gate_regeneration_max_attempts', 3, ParameterDescriptor(
                    description='Waypoint regeneration attempts before holding position in hard-gate mode')),
            ]
        )

        self.history_length        = self.get_parameter('history_length').value
        self.agent_name           = self.get_parameter('agent_name').value
        self.control_freq          = self.get_parameter('control_frequency').value
        self.arrival_threshold     = self.get_parameter('arrival_threshold').value
        self.use_arrival_completion = self.get_parameter('use_arrival_completion').value
        self.controller_reset_enabled = self.get_parameter('controller_reset_enabled').value
        self.goal_completion_radius = self.get_parameter('goal_completion_radius').value
        self.reset_on_eval_dropout = self.get_parameter('reset_on_eval_dropout').value
        self.max_eval_staleness_sec = self.get_parameter('max_eval_staleness_sec').value
        self.initial_eval_wait_sec = self.get_parameter('initial_eval_wait_sec').value
        self.startup_data_timeout_sec = self.get_parameter('startup_data_timeout_sec').value
        self.fail_on_startup_timeout = self.get_parameter('fail_on_startup_timeout').value
        self.odom_staleness_sec = self.get_parameter('odom_staleness_sec').value
        self.image_staleness_sec = self.get_parameter('image_staleness_sec').value
        self.fallback_to_dwb = self.get_parameter('fallback_to_dwb').value
        self.dwb_cmd_staleness_sec = self.get_parameter('dwb_cmd_staleness_sec').value
        self.enable_bev_visualization = self.get_parameter('enable_bev_visualization').value
        self.bev_visualization_period_sec = self.get_parameter('bev_visualization_period_sec').value
        self.look_ahead_dist       = self.get_parameter('look_ahead_distance').value
        self.max_linear_vel        = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel       = self.get_parameter('max_angular_velocity').value
        self.enable_human_tracking = self.get_parameter('enable_human_tracking').value
        self.max_humans            = self.get_parameter('max_humans').value
        self.human_context_radius  = float(self.get_parameter('human_context_radius').value)
        self.path_waypoint_index   = int(self.get_parameter('path_waypoint_index').value)
        self.ai_rejoin_skip_distance = max(
            0.0,
            float(self.get_parameter('ai_rejoin_skip_distance').value),
        )
        self.phase_local_goal_enabled = bool(self.get_parameter('phase_local_goal_enabled').value)
        self.phase_local_goal_reached_radius = max(
            0.05,
            float(self.get_parameter('phase_local_goal_reached_radius').value),
        )
        self.phase_local_goal_timeout_sec = max(
            0.0,
            float(self.get_parameter('phase_local_goal_timeout_sec').value),
        )
        self.local_goal_relock_mode = str(
            self.get_parameter('local_goal_relock_mode').value
        ).strip().lower()
        valid_relock_modes = {'once', 'reached', 'periodic'}
        if self.local_goal_relock_mode not in valid_relock_modes:
            self.get_logger().warn(
                f"Invalid local_goal_relock_mode='{self.local_goal_relock_mode}', "
                "falling back to 'periodic'."
            )
            self.local_goal_relock_mode = 'periodic'
        self.rolling_local_goal_final_radius = max(
            0.0,
            float(self.get_parameter('rolling_local_goal_final_radius').value),
        )
        self.rolling_relock_period_sec = max(
            0.05,
            float(self.get_parameter('rolling_relock_period_sec').value),
        )
        self.rolling_min_relock_shift = max(
            0.0,
            float(self.get_parameter('rolling_min_relock_shift').value),
        )
        self.rolling_min_local_goal_distance = max(
            0.0,
            float(self.get_parameter('rolling_min_local_goal_distance').value),
        )
        self.rolling_max_local_goal_distance = max(
            0.0,
            float(self.get_parameter('rolling_max_local_goal_distance').value),
        )
        if (
            self.rolling_max_local_goal_distance > 0.0
            and self.rolling_max_local_goal_distance < self.rolling_min_local_goal_distance
        ):
            self.get_logger().warn(
                "rolling_max_local_goal_distance is smaller than "
                "rolling_min_local_goal_distance; disabling max-distance filtering."
            )
            self.rolling_max_local_goal_distance = 0.0
        self.path_update_period_sec = max(0.05, float(self.get_parameter('path_update_period_sec').value))
        self.compute_path_timeout_sec = max(
            0.0,
            float(self.get_parameter('compute_path_timeout_sec').value),
        )
        self.robot_namespace       = self.get_parameter('robot_namespace').value.rstrip('/')
        self.instruction_topic     = self.get_parameter('instruction_topic').value
        self.human_detections_topic = self.get_parameter('human_detections_topic').value
        self.dwb_cmd_topic         = self._resolve_dwb_cmd_topic(
            self.get_parameter('dwb_cmd_topic').value
        )
        self.planner_action_name = self._resolve_nav2_action_name(
            self.get_parameter('planner_action_name').value,
            'compute_path_to_pose',
        )
        self.follow_path_action_name = self._resolve_nav2_action_name(
            self.get_parameter('follow_path_action_name').value,
            'follow_path',
        )
        self.planner_id = str(self.get_parameter('planner_id').value).strip()
        self.follow_path_controller_id = str(
            self.get_parameter('follow_path_controller_id').value
        ).strip()
        self.follow_path_goal_checker_id = str(
            self.get_parameter('follow_path_goal_checker_id').value
        ).strip()
        self.reset_service_name    = self._resolve_reset_service_name(
            self.get_parameter('reset_service_name').value
        )
        self.robot_frame_param     = str(self.get_parameter('robot_frame').value).strip()
        self.robot_frame           = self.robot_frame_param or self._default_robot_frame()
        configured_image_topic     = str(self.get_parameter('image_topic').value).strip()
        self.publish_odom_tf       = bool(self.get_parameter('publish_odom_tf').value)
        self.ai_inference_fail_threshold = max(1, int(
            self.get_parameter('ai_inference_fail_threshold').value
        ))
        self.goal_frame            = str(self.get_parameter('goal_frame').value).strip() or 'map'
        self.flip_y_axis = self.get_parameter('flip_y_axis').value
        self.agent.config.flip_y_axis = self.flip_y_axis
        coordinate_mode = str(self.get_parameter('coordinate_mode').value).strip()
        if coordinate_mode:
            self.agent.config.extra_params['coordinate_mode'] = coordinate_mode
        self.agent.config.extra_params['waypoint_scale'] = float(
            self.get_parameter('waypoint_scale').value
        )
        self.dwb_integration_mode = str(
            self.get_parameter('dwb_integration_mode').value
        ).strip().lower().replace('-', '_')
        self.use_dwb_hard_gate = bool(self.get_parameter('use_dwb_hard_gate').value)
        if self.use_dwb_hard_gate:
            self.dwb_integration_mode = 'hard_gate'
        valid_integration_modes = {
            'none',
            'path_adapter',
            'shaped_path',
            'one_waypoint_replace',
            'hard_gate',
        }
        if self.dwb_integration_mode not in valid_integration_modes:
            self.get_logger().warn(
                f"Invalid dwb_integration_mode='{self.dwb_integration_mode}', "
                "falling back to 'path_adapter'."
            )
            self.dwb_integration_mode = 'path_adapter'
        self.use_dwb_hard_gate = self.dwb_integration_mode == 'hard_gate'
        self.shaped_path_num_waypoints = max(
            1,
            int(self.get_parameter('shaped_path_num_waypoints').value),
        )
        self.social_cost_hard_radius = max(0.01, float(self.get_parameter('social_cost_hard_radius').value))
        self.social_cost_personal_radius = max(0.01, float(self.get_parameter('social_cost_personal_radius').value))
        self.social_cost_social_radius = max(0.01, float(self.get_parameter('social_cost_social_radius').value))
        self.social_cost_w_hard = float(self.get_parameter('social_cost_w_hard').value)
        self.social_cost_w_personal = float(self.get_parameter('social_cost_w_personal').value)
        self.social_cost_w_social = float(self.get_parameter('social_cost_w_social').value)
        self.social_cost_w_global = float(self.get_parameter('social_cost_w_global').value)
        self.social_cost_w_progress = float(self.get_parameter('social_cost_w_progress').value)
        self.social_cost_w_smooth = float(self.get_parameter('social_cost_w_smooth').value)
        self.dwb_hard_gate_radius = max(
            0.0,
            float(self.get_parameter('dwb_hard_gate_radius').value),
        )
        self.dwb_hard_gate_regeneration_max_attempts = max(
            0,
            int(self.get_parameter('dwb_hard_gate_regeneration_max_attempts').value),
        )
        self.dwb_hard_gate_adapter = None
        if self.use_dwb_hard_gate:
            self.dwb_hard_gate_adapter = DWBHardGateAdapter(
                gate_waypoint_index=self.path_waypoint_index,
                waypoint_gate_radius=self.dwb_hard_gate_radius,
                regeneration_max_attempts=self.dwb_hard_gate_regeneration_max_attempts,
                max_linear_vel=self.max_linear_vel,
                max_angular_vel=self.max_angular_vel,
            )

        self.image_topic = configured_image_topic or f'{self.robot_namespace}/rgbd_camera/image'
        self.odom_topic = f'{self.robot_namespace}/odom'
        self.evaluation_topic = f'{self.robot_namespace}/evaluation'
        self.cmd_vel_topic = f'{self.robot_namespace}/cmd_vel'
        self.current_goal_topic = f'{self.robot_namespace}/goal_pose'

        model_config_path = Path(self.get_parameter('model_config_path').value)
        model_checkpoint_value = str(self.get_parameter('model_checkpoint_path').value).strip()
        if model_checkpoint_value:
            model_checkpoint_path = Path(model_checkpoint_value)
        else:
            model_checkpoint_path = self.agent.default_checkpoint_path(self.agent_name)

        self.device = 'cpu'
        self.model = None
        if TORCH_IMPORT_ERROR is not None:
            self.get_logger().error(
                f"Torch import failed; running AI controller in DWB fallback mode: {TORCH_IMPORT_ERROR}"
            )
        elif self.agent.config.allow_model_soft_fail:
            model_paths_valid = model_config_path.exists() and model_checkpoint_path.exists()
            if not model_paths_valid:
                self.get_logger().error(
                    f"AI model paths missing (config={model_config_path}, ckpt={model_checkpoint_path}); "
                    "running in DWB fallback mode."
                )
            elif self.agent.load(model_config_path, model_checkpoint_path, self.get_logger()):
                self.model = self.agent
                self.device = self.agent.device
        elif self.agent.load(model_config_path, model_checkpoint_path, self.get_logger()):
            self.model = self.agent
            self.device = self.agent.device

        if self.agent.name == 'lelan' and hasattr(self.agent, 'context_size'):
            self.history_length = int(getattr(self.agent, 'context_size'))

        self._cuda_stream = None
        if torch is not None and self.device == 'cuda' and torch.cuda.is_available():
            try:
                self._cuda_stream = torch.cuda.Stream()
            except Exception as exc:
                self.get_logger().warn(
                    f"Failed to create AI CUDA stream; using default stream: {exc}",
                    throttle_duration_sec=2.0,
                )

        self.human_tracker = HumanPositionTracker(
            history_length=self.history_length,
            max_humans=self.max_humans
        )
        # State
        self.image_history       = deque(maxlen=self.history_length)
        self._image_lock         = threading.Lock()
        self.odom_history        = deque(maxlen=self.history_length)

        self.current_instruction = "Navigate safely to the goal and avoid pedestrians"
        self.current_odom        = None
        self.latest_eval         = None
        self.latest_dwb_cmd      = None
        self.episode_active      = False
        self.task_complete       = True
        self.last_eval_time      = self.get_clock().now()
        self.last_dwb_cmd_time   = self.get_clock().now()
        self.dwb_cmd_count       = 0
        self.last_dwb_cmd_log_time = None
        self.last_episode_time   = self.get_clock().now()
        self.last_odom_time      = None
        self.last_image_time     = None
        self.last_bev_time       = self.get_clock().now()
        self.last_path_request_time = None
        self.path_request_in_progress = False
        self.latest_ai_waypoints = None
        self.latest_global_path = None
        self.latest_benchmark_global_path = None
        self.latest_no_ai_dwb_trajectory = None
        self.latest_ai_path = None
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None
        self.last_compute_path_goal_handle = None
        self._pending_compute_path_goal_kind = None
        self.last_follow_path_goal_handle = None
        self.last_follow_path_send_time = None
        self.follow_path_send_count = 0
        self.active_follow_path_send_count = 0
        self._follow_path_owner: str = 'none'
        self._force_infer: bool = False
        self._pending_bt_cancel_future = None
        self._subgoal_phase: str = 'idle'
        self.reset_in_progress   = False
        self.PHASE_AI_LOCAL_GOAL = 'ai_local_goal'
        self.PHASE_GLOBAL_GOAL = 'global_goal'
        self.path_phase = (
            self.PHASE_AI_LOCAL_GOAL
            if self.phase_local_goal_enabled
            else self.PHASE_GLOBAL_GOAL
        )
        self.phase_local_goal_odom = None
        self.phase_local_goal_start_time = None
        self.phase_local_goal_wp_idx = None
        self.phase_local_goal_last_distance = None
        self.phase_local_goal_lock_count = 0

        # Biến đếm lỗi AI liên tiếp để quyết định fallback
        self._ai_consecutive_failures = 0

        instruction_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        goal_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # Subscribers
        self.create_subscription(
            String,
            self.instruction_topic,
            self.instruction_callback,
            instruction_qos,
        )
        self.create_subscription(PoseStamped,
            self.current_goal_topic,
            self.goal_callback, goal_qos)
        self.create_subscription(Image,
            self.image_topic,
            self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry,
            self.odom_topic,
            self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(LocalPlanEvaluation,
            self.evaluation_topic,
            self.eval_callback, qos_profile_sensor_data)
        self.create_subscription(Twist,
            self.dwb_cmd_topic,
            self.dwb_cmd_callback, qos_profile_sensor_data)
        self.create_subscription(String,
            self.human_detections_topic,
            self.human_detection_callback, 10)

        # Publishers
        # Nav2 publishes DWB raw commands to cmd_vel_nav_raw in SocialNav mode;
        # this node relays those commands to the robot cmd_vel after sending
        # a AI-adapted path to FollowPath.
        self.cmd_pub     = self.create_publisher(Twist,       self.cmd_vel_topic, 10)
        self.path_pub    = self.create_publisher(NavPath,     f'/{self.topic_prefix}/path', 10)
        self.arrival_pub = self.create_publisher(Float32,     f'/{self.topic_prefix}/arrival_score', 10)
        self.viz_pub     = self.create_publisher(Image,       f'/{self.topic_prefix}/bev_viz', 10)
        self.reset_client = self.create_client(Empty, self.reset_service_name)
        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            self.planner_action_name,
        )
        self.follow_path_client = ActionClient(
            self,
            FollowPath,
            self.follow_path_action_name,
        )

        self.control_timer  = self.create_timer(1.0 / self.control_freq, self.control_loop_callback)
        self.watchdog_timer = self.create_timer(0.2, self.watchdog_callback)

        self.get_logger().info(
            "AI DWB Path Adapter initialized "
            f"(robot_ns={self.robot_namespace}, robot_frame={self.robot_frame}, "
            f"image_topic={self.image_topic}, "
            f"goal_topic={self.current_goal_topic}, "
            f"dwb_cmd_topic={self.dwb_cmd_topic}, "
            f"cmd_vel_topic={self.cmd_vel_topic}, "
            f"planner_action={self.planner_action_name}, "
            f"follow_path_action={self.follow_path_action_name}, "
            f"reset_service={self.reset_service_name}, "
            f"path_waypoint_index={self.path_waypoint_index}, "
            f"ai_rejoin_skip_distance={self.ai_rejoin_skip_distance:.2f}m, "
            f"phase_local_goal_enabled={self.phase_local_goal_enabled}, "
            f"phase_local_goal_reached_radius={self.phase_local_goal_reached_radius:.2f}m, "
            f"phase_local_goal_timeout_sec={self.phase_local_goal_timeout_sec:.1f}s, "
            f"local_goal_relock_mode={self.local_goal_relock_mode}, "
            f"rolling_final_radius={self.rolling_local_goal_final_radius:.2f}m, "
            f"rolling_relock_period={self.rolling_relock_period_sec:.2f}s, "
            f"dwb_integration_mode={self.dwb_integration_mode}, "
            f"use_dwb_hard_gate={self.use_dwb_hard_gate}, "
            f"model_ready={self.model is not None})"
        )

        self.current_goal = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_odom_tf else None

    # ──────────────────────────── Callbacks ────────────────────────────

    def _image_msg_to_rgb_array(self, msg: Image) -> np.ndarray:
        """Convert a ROS Image to contiguous RGB uint8 without cv_bridge."""
        encoding = (msg.encoding or "").lower()
        channel_counts = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "8uc1": 1,
            "8uc3": 3,
            "8uc4": 4,
        }
        channels = channel_counts.get(encoding)
        if channels is None:
            raise ValueError(f"Unsupported image encoding '{msg.encoding}'")

        row_width = int(msg.width) * channels
        step = int(msg.step) if msg.step else row_width
        if step < row_width:
            raise ValueError(
                f"Invalid image step={step} for width={msg.width}, channels={channels}"
            )

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        needed = int(msg.height) * step
        if raw.size < needed:
            raise ValueError(
                f"Image data too short: got {raw.size} bytes, expected at least {needed}"
            )

        rows = raw[:needed].reshape((int(msg.height), step))
        arr = rows[:, :row_width].reshape((int(msg.height), int(msg.width), channels))

        if encoding in ("mono8", "8uc1"):
            arr = np.repeat(arr, 3, axis=2)
        elif encoding in ("bgr8",):
            arr = arr[:, :, ::-1]
        elif encoding in ("rgba8", "8uc4"):
            arr = arr[:, :, :3]
        elif encoding == "bgra8":
            arr = arr[:, :, :3][:, :, ::-1]

        return np.ascontiguousarray(arr, dtype=np.uint8)

    def _rgb_array_to_image_msg(self, image: np.ndarray) -> Image:
        """Convert contiguous RGB uint8 image to sensor_msgs/Image without cv_bridge."""
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape={arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        arr = np.ascontiguousarray(arr)

        msg = Image()
        msg.height = int(arr.shape[0])
        msg.width = int(arr.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(arr.shape[1] * 3)
        msg.data = arr.tobytes()
        return msg

    def image_callback(self, msg):
        try:
            img = self._image_msg_to_rgb_array(msg)
            with self._image_lock:
                self.image_history.append(img.copy())
            self.last_image_time = self.get_clock().now()
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to convert RGB image; holding until images recover: {exc}",
                throttle_duration_sec=2.0,
            )

    def instruction_callback(self, msg):
        self.current_instruction = msg.data
        self.current_goal = None

        self.episode_active = False
        self.task_complete = True
        self.latest_eval = None
        self.latest_dwb_cmd = None
        self.dwb_cmd_count = 0
        self.last_dwb_cmd_log_time = None
        self.latest_ai_waypoints = None
        self.latest_global_path = None
        self.latest_benchmark_global_path = None
        self.latest_no_ai_dwb_trajectory = None
        self.latest_ai_path = None
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None
        self.path_request_in_progress = False
        self.last_compute_path_goal_handle = None
        self._pending_compute_path_goal_kind = None
        self.last_follow_path_goal_handle = None
        self.last_follow_path_send_time = None
        self.follow_path_send_count = 0
        self.active_follow_path_send_count = 0
        self._follow_path_owner = 'none'
        self._force_infer = False
        self._pending_bt_cancel_future = None
        self._subgoal_phase = 'idle'
        with self._image_lock:
            self.image_history.clear()
        self.odom_history.clear()
        self.reset_in_progress = False
        self._ai_consecutive_failures = 0
        self._reset_phase_state()
        if self.dwb_hard_gate_adapter is not None:
            self.dwb_hard_gate_adapter.clear()

        self.get_logger().info(f"[INFO] New instruction received: {msg.data}. Waiting for goal_pose.")

    def odom_callback(self, msg):
        self.current_odom = msg
        self.last_odom_time = self.get_clock().now()
        if not self.robot_frame_param and msg.child_frame_id:
            self.robot_frame = msg.child_frame_id

        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        rot = msg.pose.pose.orientation
        siny = 2 * (rot.w * rot.z + rot.x * rot.y)
        cosy = 1 - 2 * (rot.y**2 + rot.z**2)
        pyaw = math.atan2(siny, cosy)

        self.odom_history.append((px, py, pyaw))

        if self.tf_broadcaster is not None and msg.child_frame_id:
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = msg.header.frame_id or 'odom'
            t.child_frame_id = msg.child_frame_id
            t.transform.translation.x = msg.pose.pose.position.x
            t.transform.translation.y = msg.pose.pose.position.y
            t.transform.translation.z = msg.pose.pose.position.z
            t.transform.rotation = msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)

    def eval_callback(self, msg: LocalPlanEvaluation):
        # Visualization only. Control remains delegated to DWB FollowPath.
        self.latest_eval = msg
        self.last_eval_time = self.get_clock().now()
        if self._follow_path_owner != 'ai':
            self.latest_no_ai_dwb_trajectory = self._best_dwb_eval_trajectory_points(msg)

    def dwb_cmd_callback(self, msg: Twist):
        now = self.get_clock().now()
        previous_time = self.last_dwb_cmd_time
        self.latest_dwb_cmd = msg
        self.last_dwb_cmd_time = now
        self.dwb_cmd_count += 1

        should_log = self.dwb_cmd_count <= 3
        if self.last_dwb_cmd_log_time is None:
            should_log = True
        else:
            log_age = (now - self.last_dwb_cmd_log_time).nanoseconds / 1e9
            should_log = should_log or log_age >= 2.0

        if should_log:
            dt = (now - previous_time).nanoseconds / 1e9 if previous_time is not None else 0.0
            self.last_dwb_cmd_log_time = now
            self.get_logger().info(
                "[DWB_DEBUG] raw_cmd_received "
                f"topic={self.dwb_cmd_topic} count={self.dwb_cmd_count} "
                f"dt={dt:.3f}s v={msg.linear.x:.3f} w={msg.angular.z:.3f} "
                f"episode_active={self.episode_active} task_complete={self.task_complete}",
                throttle_duration_sec=0.5,
            )

    def human_detection_callback(self, msg: String):
        try:
            human_positions = json.loads(msg.data)
            detections = [(float(x), float(y)) for x, y in human_positions]
            self.human_tracker.update(detections)
        except Exception as e:
            self.get_logger().warn(f"Failed to parse human detections: {e}",
                                   throttle_duration_sec=2.0)

    def _prepare_human_context_for_model(self, human_positions, human_mask):
        """Convert tracked map-frame humans to robot-local coordinates for social_film."""
        if human_positions is None:
            return None, None, 0

        positions = np.asarray(human_positions, dtype=np.float32)
        if positions.ndim != 3 or positions.shape[-1] != 2:
            self.get_logger().warn(
                f"Invalid human_positions shape for SocialNav: {positions.shape}",
                throttle_duration_sec=2.0,
            )
            return None, None, 0

        t_count, p_count, _ = positions.shape
        if human_mask is not None:
            mask = np.asarray(human_mask, dtype=bool)
            if mask.shape != (t_count, p_count):
                self.get_logger().warn(
                    f"Ignoring invalid human_mask shape {mask.shape}; expected {(t_count, p_count)}",
                    throttle_duration_sec=2.0,
                )
                mask = np.zeros((t_count, p_count), dtype=bool)
            else:
                mask = mask.copy()
        else:
            mask = np.zeros((t_count, p_count), dtype=bool)

        pose_history = list(self.odom_history)
        if len(pose_history) >= t_count:
            pose_history = pose_history[-t_count:]
        else:
            if self.current_odom is not None:
                pos = self.current_odom.pose.pose.position
                rot = self.current_odom.pose.pose.orientation
                yaw = math.atan2(
                    2 * (rot.w * rot.z + rot.x * rot.y),
                    1 - 2 * (rot.y**2 + rot.z**2),
                )
                fallback_pose = (pos.x, pos.y, yaw)
            elif pose_history:
                fallback_pose = pose_history[0]
            else:
                return None, None, 0
            pose_history = [fallback_pose] * (t_count - len(pose_history)) + pose_history

        local_positions = np.zeros_like(positions, dtype=np.float32)
        for t_idx, (cx, cy, cyaw) in enumerate(pose_history):
            dx = positions[t_idx, :, 0] - cx
            dy = positions[t_idx, :, 1] - cy
            cos_yaw = math.cos(cyaw)
            sin_yaw = math.sin(cyaw)

            local_positions[t_idx, :, 0] = dx * cos_yaw + dy * sin_yaw
            local_positions[t_idx, :, 1] = -dx * sin_yaw + dy * cos_yaw

            finite = np.isfinite(local_positions[t_idx]).all(axis=1)
            invalid = mask[t_idx] | ~finite
            if self.human_context_radius > 0.0:
                distances = np.linalg.norm(local_positions[t_idx], axis=1)
                invalid |= distances > self.human_context_radius

            mask[t_idx] = invalid
            local_positions[t_idx, invalid, :] = 0.0

        valid_count = int(np.count_nonzero(~mask[-1])) if t_count > 0 else 0
        return local_positions, mask, valid_count

    def watchdog_callback(self):
        """Dừng robot nếu không nhận được command DWB thô quá lâu."""
        if not self.episode_active or self.task_complete or self.reset_in_progress:
            return

        if self._subgoal_phase == 'chasing_waypoint' and self._follow_path_owner == 'ai':
            return

        elapsed = (self.get_clock().now() - self.last_dwb_cmd_time).nanoseconds / 1e9
        if elapsed > self.dwb_cmd_staleness_sec:
            self.cmd_pub.publish(Twist())

    def goal_callback(self, msg: PoseStamped):
        """Nhận và lưu global goal thật của benchmark/Nav2."""
        self.current_goal = msg
        now = self.get_clock().now()

        pending_cancel = None
        if self.last_follow_path_goal_handle is not None:
            try:
                pending_cancel = self.last_follow_path_goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warn(f"goal_callback: cancel BT handle failed: {exc}")
            self.last_follow_path_goal_handle = None

        self.episode_active = True
        self.task_complete = False
        self.reset_in_progress = False
        self.last_episode_time = now
        self.last_eval_time = now
        self.last_dwb_cmd_time = now
        self.dwb_cmd_count = 0
        self.last_dwb_cmd_log_time = None
        self.latest_eval = None
        self.latest_dwb_cmd = None
        self.latest_ai_waypoints = None
        self.latest_global_path = None
        self.latest_benchmark_global_path = None
        self.latest_no_ai_dwb_trajectory = None
        self.latest_ai_path = None
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None
        self.path_request_in_progress = False
        self.last_compute_path_goal_handle = None
        self._pending_compute_path_goal_kind = None
        self.last_follow_path_goal_handle = None
        self.last_follow_path_send_time = None
        self.follow_path_send_count = 0
        self.active_follow_path_send_count = 0
        self.last_path_request_time = None
        self._pending_bt_cancel_future = pending_cancel
        self._subgoal_phase = (
            'cancelling_bt'
            if self._pending_bt_cancel_future is not None
            else 'chasing_waypoint'
        )
        self._follow_path_owner = 'none'
        self._force_infer = False
        with self._image_lock:
            self.image_history.clear()
        self.odom_history.clear()
        self._ai_consecutive_failures = 0
        self._reset_phase_state()
        if self.dwb_hard_gate_adapter is not None:
            self.dwb_hard_gate_adapter.clear()
        self.get_logger().info(
            "[INFO] Benchmark goal received; starting SocialNav DWB path-adapter episode."
        )

    # ──────────────────────────── Helper Utilities ────────────────────────────

    def _resolve_reset_service_name(self, configured_name: str) -> str:
        configured_name = str(configured_name).strip()
        if configured_name:
            return configured_name

        namespace_parts = [part for part in self.robot_namespace.split('/') if part]
        if namespace_parts:
            namespace_parts = namespace_parts[:-1]
        task_namespace = '/' + '/'.join(namespace_parts) if namespace_parts else ''
        return f"{task_namespace}/reset_task" if task_namespace else "/reset_task"

    def _resolve_dwb_cmd_topic(self, configured_name: str) -> str:
        configured_name = str(configured_name).strip()
        if configured_name:
            return configured_name
        return f"{self.robot_namespace}/cmd_vel_nav_raw"

    def _resolve_nav2_action_name(self, configured_name: str, basename: str) -> str:
        configured_name = str(configured_name).strip()
        if configured_name:
            return configured_name
        return f"{self.robot_namespace}/{basename}"

    def _default_robot_frame(self) -> str:
        robot_name = self.robot_namespace.strip('/').split('/')[-1]
        return f"{robot_name}/base_link" if robot_name else "base_link"

    @staticmethod
    def _frame_name(frame_id: str | None) -> str:
        return str(frame_id or '').strip().lstrip('/')

    @staticmethod
    def _transform_point_2d(x: float, y: float, transform) -> tuple[float, float]:
        rot = transform.transform.rotation
        yaw = math.atan2(
            2 * (rot.w * rot.z + rot.x * rot.y),
            1 - 2 * (rot.y ** 2 + rot.z ** 2),
        )
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        return (
            tx + x * math.cos(yaw) - y * math.sin(yaw),
            ty + x * math.sin(yaw) + y * math.cos(yaw),
        )

    def _goal_position_in_odom_frame(self) -> tuple[float, float] | None:
        if self.current_goal is None or self.current_odom is None:
            return None

        odom_frame = self._frame_name(self.current_odom.header.frame_id) or 'odom'
        goal_frame = self._frame_name(self.current_goal.header.frame_id) or odom_frame
        gx = float(self.current_goal.pose.position.x)
        gy = float(self.current_goal.pose.position.y)

        if goal_frame == odom_frame:
            return gx, gy

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_frame,
                goal_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            return self._transform_point_2d(gx, gy, transform)
        except Exception as exc:
            self.get_logger().warn(
                f"Unable to transform benchmark goal from {goal_frame} to {odom_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _distance_to_goal(self) -> float | None:
        if self.current_goal is None or self.current_odom is None:
            return None

        cx = self.current_odom.pose.pose.position.x
        cy = self.current_odom.pose.pose.position.y
        goal_odom = self._goal_position_in_odom_frame()
        if goal_odom is None:
            return None
        gx, gy = goal_odom
        return math.hypot(gx - cx, gy - cy)

    def _seconds_since(self, stamp) -> float | None:
        if stamp is None:
            return None
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _episode_age(self) -> float:
        return (self.get_clock().now() - self.last_episode_time).nanoseconds / 1e9

    def _using_sim_time(self) -> bool:
        try:
            return bool(self.get_parameter('use_sim_time').value)
        except Exception:
            return False

    def _clock_ready_for_nav2(self) -> bool:
        if not self._using_sim_time():
            return True
        return self.get_clock().now().nanoseconds > 0

    def _maybe_fail_startup_wait(self, reason: str) -> bool:
        if not self.episode_active:
            return False
        if not self.fail_on_startup_timeout or self.startup_data_timeout_sec <= 0.0:
            return False
        episode_age = self._episode_age()
        if episode_age < self.startup_data_timeout_sec:
            return False

        self._request_episode_reset(
            success=False,
            reason=f"{reason} after {episode_age:.1f}s startup wait",
            distance_to_goal=self._distance_to_goal(),
        )
        return True

    def _request_episode_reset(self, *, success: bool, reason: str, distance_to_goal: float | None = None) -> None:
        status = "SUCCESS" if success else "FAILURE"
        distance_log = (
            f", dist_to_goal={distance_to_goal:.2f}m"
            if distance_to_goal is not None
            else ""
        )
        if not self.controller_reset_enabled:
            self.get_logger().warn(
                f"AI controller self-termination suppressed [{status}] due to {reason}{distance_log}; "
                "benchmark/Nav2 owns episode completion.",
                throttle_duration_sec=2.0,
            )
            self.cmd_pub.publish(Twist())
            return

        if self.reset_in_progress:
            return

        self.get_logger().warn(
            f"Episode termination requested [{status}] due to {reason}{distance_log}",
            throttle_duration_sec=1.0,
        )

        self.task_complete = True
        self.episode_active = False
        self.reset_in_progress = True
        self._stop()

        if not self.reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().error(
                f"Reset service {self.reset_service_name} is unavailable; keeping robot stopped."
            )
            return

        future = self.reset_client.call_async(Empty.Request())
        future.add_done_callback(self._on_reset_response)

    def _on_reset_response(self, future) -> None:
        try:
            future.result()
            self.get_logger().info("Task reset request accepted by task_generator.")
        except Exception as exc:
            self.get_logger().error(f"Task reset request failed: {exc}")

    # ──────────────────────────── Core: DWB Path Adapter ───────────────────────

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        return math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2),
        )

    @staticmethod
    def _set_yaw(pose_msg, yaw: float) -> None:
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = math.sin(yaw * 0.5)
        pose_msg.orientation.w = math.cos(yaw * 0.5)

    def _path_waypoint_index(self, waypoints: np.ndarray) -> int | None:
        if waypoints is None or len(waypoints) == 0:
            return None
        return min(max(self.path_waypoint_index, 0), len(waypoints) - 1)

    def _path_waypoint(self, waypoints: np.ndarray) -> np.ndarray | None:
        wp_idx = self._path_waypoint_index(waypoints)
        if wp_idx is None:
            return None
        return np.asarray(waypoints[wp_idx], dtype=np.float32)

    def _reset_phase_state(self) -> None:
        self.path_phase = (
            self.PHASE_AI_LOCAL_GOAL
            if self.phase_local_goal_enabled
            else self.PHASE_GLOBAL_GOAL
        )
        self.phase_local_goal_odom = None
        self.phase_local_goal_start_time = None
        self.phase_local_goal_wp_idx = None
        self.phase_local_goal_last_distance = None
        self.phase_local_goal_lock_count = 0

    def _phase_local_goal_active(self) -> bool:
        return (
            self.phase_local_goal_enabled
            and self.path_phase == self.PHASE_AI_LOCAL_GOAL
        )

    def _distance_to_phase_local_goal(self) -> float | None:
        if self.phase_local_goal_odom is None:
            return None
        robot_pose = self._robot_pose_in_odom()
        if robot_pose is None:
            return None
        cx, cy, _ = robot_pose
        gx, gy = self.phase_local_goal_odom
        return math.hypot(gx - cx, gy - cy)

    def _switch_to_global_phase(self, reason: str) -> None:
        if self.path_phase == self.PHASE_GLOBAL_GOAL:
            return

        dist = self._distance_to_phase_local_goal()
        dist_str = f"{dist:.2f}m" if dist is not None else "n/a"
        self.get_logger().info(
            f"SocialNav phase transition: ai_local_goal -> global_goal "
            f"reason={reason} local_goal_dist={dist_str}"
        )
        self.path_phase = self.PHASE_GLOBAL_GOAL
        self.phase_local_goal_odom = None
        self.phase_local_goal_start_time = None
        self.phase_local_goal_wp_idx = None
        self.phase_local_goal_last_distance = None
        self.phase_local_goal_lock_count = 0
        self.last_path_request_time = None
        self.path_request_in_progress = False
        self.last_compute_path_goal_handle = None

    def _phase_local_goal_candidate(
        self,
        waypoints: np.ndarray,
    ) -> tuple[int, np.ndarray, tuple[float, float], float] | None:
        waypoint = self._path_waypoint(waypoints)
        wp_idx = self._path_waypoint_index(waypoints)
        if waypoint is None:
            return None

        waypoint_odom = self._local_waypoint_to_odom(waypoint)
        if waypoint_odom is None:
            self.get_logger().warn(
                "Unable to compute AI local goal because odom is unavailable.",
                throttle_duration_sec=2.0,
            )
            return None

        robot_pose = self._robot_pose_in_odom()
        if robot_pose is None:
            return None

        distance = math.hypot(
            waypoint_odom[0] - robot_pose[0],
            waypoint_odom[1] - robot_pose[1],
        )
        if (
            self.rolling_min_local_goal_distance > 0.0
            and distance < self.rolling_min_local_goal_distance
        ):
            self.get_logger().warn(
                "Rejecting AI local goal candidate because it is too close: "
                f"dist={distance:.2f}m min={self.rolling_min_local_goal_distance:.2f}m",
                throttle_duration_sec=2.0,
            )
            return None
        if (
            self.rolling_max_local_goal_distance > 0.0
            and distance > self.rolling_max_local_goal_distance
        ):
            self.get_logger().warn(
                "Rejecting AI local goal candidate because it is too far: "
                f"dist={distance:.2f}m max={self.rolling_max_local_goal_distance:.2f}m",
                throttle_duration_sec=2.0,
            )
            return None

        return wp_idx, waypoint, waypoint_odom, distance

    def _lock_phase_local_goal(self, waypoints: np.ndarray, reason: str) -> bool:
        candidate = self._phase_local_goal_candidate(waypoints)
        if candidate is None:
            return False

        wp_idx, waypoint, waypoint_odom, distance = candidate
        previous_goal = self.phase_local_goal_odom
        shift = None
        if previous_goal is not None:
            shift = math.hypot(
                waypoint_odom[0] - previous_goal[0],
                waypoint_odom[1] - previous_goal[1],
            )
        self.phase_local_goal_odom = waypoint_odom
        self.phase_local_goal_start_time = self.get_clock().now()
        self.phase_local_goal_wp_idx = wp_idx
        self.phase_local_goal_last_distance = distance
        self.phase_local_goal_lock_count += 1
        self.last_path_request_time = None
        self.path_request_in_progress = False
        self.last_compute_path_goal_handle = None
        shift_str = f"{shift:.2f}m" if shift is not None else "n/a"
        self.get_logger().info(
            "SocialNav rolling local goal locked: "
            f"mode={self.local_goal_relock_mode} reason={reason} "
            f"subgoal_count={self.phase_local_goal_lock_count} "
            f"wp_idx={wp_idx} wp_local=({waypoint[0]:.2f},{waypoint[1]:.2f}) "
            f"goal_odom=({waypoint_odom[0]:.2f},{waypoint_odom[1]:.2f}) "
            f"dist={distance:.2f}m shift={shift_str}"
        )
        return True

    def _ensure_phase_local_goal(self, waypoints: np.ndarray) -> bool:
        if not self._phase_local_goal_active():
            return False
        if self.phase_local_goal_odom is not None:
            return True
        if not self._lock_phase_local_goal(waypoints, "initial"):
            self._switch_to_global_phase("no valid AI local goal available")
            return False
        return True

    def _periodic_relock_due(self) -> bool:
        if self.local_goal_relock_mode != 'periodic':
            return False
        if self.phase_local_goal_start_time is None:
            return True
        elapsed = (
            self.get_clock().now() - self.phase_local_goal_start_time
        ).nanoseconds / 1e9
        return elapsed >= self.rolling_relock_period_sec

    def _try_periodic_relock(self, waypoints: np.ndarray) -> bool:
        if not self._periodic_relock_due():
            return False

        candidate = self._phase_local_goal_candidate(waypoints)
        if candidate is None:
            return False

        _, _, waypoint_odom, _ = candidate
        if self.phase_local_goal_odom is not None:
            shift = math.hypot(
                waypoint_odom[0] - self.phase_local_goal_odom[0],
                waypoint_odom[1] - self.phase_local_goal_odom[1],
            )
            if shift < self.rolling_min_relock_shift:
                self.get_logger().info(
                    "SocialNav periodic relock skipped: "
                    f"candidate shift={shift:.2f}m "
                    f"min={self.rolling_min_relock_shift:.2f}m",
                    throttle_duration_sec=1.0,
                )
                return False

        return self._lock_phase_local_goal(
            waypoints,
            f"periodic relock every {self.rolling_relock_period_sec:.2f}s",
        )

    def _maybe_update_phase_state(self, waypoints: np.ndarray) -> None:
        if not self.phase_local_goal_enabled:
            self.path_phase = self.PHASE_GLOBAL_GOAL
            return
        if self.path_phase == self.PHASE_GLOBAL_GOAL:
            return

        dist_to_goal = self._distance_to_goal()
        if (
            self.rolling_local_goal_final_radius > 0.0
            and dist_to_goal is not None
            and dist_to_goal <= self.rolling_local_goal_final_radius
        ):
            self._switch_to_global_phase(
                f"inside rolling final radius {self.rolling_local_goal_final_radius:.2f}m"
            )
            return

        if not self._ensure_phase_local_goal(waypoints):
            return

        dist = self._distance_to_phase_local_goal()
        if dist is None:
            return
        self.phase_local_goal_last_distance = dist

        if dist <= self.phase_local_goal_reached_radius:
            if self.local_goal_relock_mode == 'once':
                self._switch_to_global_phase(
                    f"reached local waypoint radius {self.phase_local_goal_reached_radius:.2f}m"
                )
                return
            if not self._lock_phase_local_goal(
                waypoints,
                f"reached previous local waypoint radius {self.phase_local_goal_reached_radius:.2f}m",
            ):
                self._switch_to_global_phase(
                    "reached local waypoint but no valid next AI local goal is available"
                )
            return

        if (
            self.phase_local_goal_timeout_sec > 0.0
            and self.phase_local_goal_start_time is not None
        ):
            elapsed = (
                self.get_clock().now() - self.phase_local_goal_start_time
            ).nanoseconds / 1e9
            if elapsed >= self.phase_local_goal_timeout_sec:
                if self.local_goal_relock_mode in ('once', 'reached'):
                    self._switch_to_global_phase(
                        f"local waypoint timeout {self.phase_local_goal_timeout_sec:.1f}s"
                    )
                    return
                if not self._lock_phase_local_goal(
                    waypoints,
                    f"local waypoint timeout {self.phase_local_goal_timeout_sec:.1f}s",
                ):
                    self._switch_to_global_phase(
                        f"local waypoint timeout {self.phase_local_goal_timeout_sec:.1f}s "
                        "and no valid replacement is available"
                    )
                return

        if self._try_periodic_relock(waypoints):
            dist = self._distance_to_phase_local_goal()
            if dist is not None:
                self.phase_local_goal_last_distance = dist

        self.get_logger().info(
            f"SocialNav phase=ai_local_goal wp_idx={self.phase_local_goal_wp_idx} "
            f"local_goal_dist={dist:.2f}m mode={self.local_goal_relock_mode}",
            throttle_duration_sec=1.0,
        )

    def _odom_frame(self) -> str:
        if self.current_odom is None:
            return 'odom'
        return self._frame_name(self.current_odom.header.frame_id) or 'odom'

    def _robot_pose_in_odom(self) -> tuple[float, float, float] | None:
        if self.current_odom is None:
            return None
        pose = self.current_odom.pose.pose
        return (
            float(pose.position.x),
            float(pose.position.y),
            self._yaw_from_quaternion(pose.orientation),
        )

    def _local_waypoint_to_odom(self, waypoint: np.ndarray) -> tuple[float, float] | None:
        robot_pose = self._robot_pose_in_odom()
        if robot_pose is None:
            return None
        cx, cy, yaw = robot_pose
        lx = float(waypoint[0])
        ly = float(waypoint[1])
        return (
            cx + lx * math.cos(yaw) - ly * math.sin(yaw),
            cy + lx * math.sin(yaw) + ly * math.cos(yaw),
        )

    def _point_in_odom_to_local(self, x: float, y: float) -> tuple[float, float] | None:
        robot_pose = self._robot_pose_in_odom()
        if robot_pose is None:
            return None
        cx, cy, yaw = robot_pose
        dx = float(x) - cx
        dy = float(y) - cy
        return (
            dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw),
        )

    def _points_in_frame_to_local_array(
        self,
        points,
        source_frame: str,
    ) -> np.ndarray | None:
        robot_pose = self._robot_pose_in_odom()
        if robot_pose is None:
            return None

        cx, cy, yaw = robot_pose
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        source = self._frame_name(source_frame)
        odom_frame = self._odom_frame()
        robot_frame = self._frame_name(self.robot_frame)
        treat_as_local = not source or source == robot_frame
        transform = None

        if not treat_as_local and source != odom_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    odom_frame,
                    source,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"Unable to transform visualization points from {source} to {odom_frame}: {exc}",
                    throttle_duration_sec=2.0,
                )
                return None

        local_points = []
        for point in points:
            px = float(point[0])
            py = float(point[1])
            if treat_as_local:
                local_points.append((px, py))
                continue

            if transform is not None:
                px, py = self._transform_point_2d(px, py, transform)

            dx = px - cx
            dy = py - cy
            local_points.append((
                dx * cos_yaw + dy * sin_yaw,
                -dx * sin_yaw + dy * cos_yaw,
            ))

        if not local_points:
            return None
        return np.asarray(local_points, dtype=np.float32)

    def _point_between_frames(
        self,
        x: float,
        y: float,
        source_frame: str,
        target_frame: str,
    ) -> tuple[float, float] | None:
        source = self._frame_name(source_frame)
        target = self._frame_name(target_frame)
        if source == target:
            return float(x), float(y)

        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            return self._transform_point_2d(float(x), float(y), transform)
        except Exception as exc:
            self.get_logger().warn(
                f"Unable to transform point from {source} to {target}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _point_in_frame_to_local(
        self,
        x: float,
        y: float,
        source_frame: str,
    ) -> tuple[float, float] | None:
        odom_point = self._point_between_frames(
            float(x),
            float(y),
            source_frame,
            self._odom_frame(),
        )
        if odom_point is None:
            return None
        return self._point_in_odom_to_local(odom_point[0], odom_point[1])

    def _path_to_local_array(self, path: NavPath | None, max_points: int | None = None) -> np.ndarray | None:
        if path is None or not path.poses:
            return None
        frame_id = self._frame_name(path.header.frame_id) or self._odom_frame()
        poses = path.poses[:max_points] if max_points is not None else path.poses
        points = [(pose.pose.position.x, pose.pose.position.y) for pose in poses]
        return self._points_in_frame_to_local_array(points, frame_id)

    def _dwb_traj_to_local_array(self, poses, frame_id: str) -> np.ndarray | None:
        if not poses:
            return None
        points = [(pose.x, pose.y) for pose in poses]
        return self._points_in_frame_to_local_array(points, frame_id)

    def _best_dwb_eval_trajectory_points(
        self,
        eval_msg: LocalPlanEvaluation | None,
    ) -> tuple[list[tuple[float, float]], str] | None:
        if eval_msg is None:
            return None
        best_twist = None
        best_cost = float('inf')
        for twist in eval_msg.twists:
            if twist.total < 0.0 or not twist.traj.poses:
                continue
            if twist.total < best_cost:
                best_cost = float(twist.total)
                best_twist = twist
        if best_twist is None:
            return None
        points = [(pose.x, pose.y) for pose in best_twist.traj.poses]
        return points, eval_msg.header.frame_id

    def _best_dwb_eval_trajectory(self, eval_msg: LocalPlanEvaluation | None) -> np.ndarray | None:
        best = self._best_dwb_eval_trajectory_points(eval_msg)
        if best is None:
            return None
        points, frame_id = best
        return self._points_in_frame_to_local_array(points, frame_id)

    def _stored_dwb_trajectory_to_local(
        self,
        stored_trajectory: tuple[list[tuple[float, float]], str] | None,
    ) -> np.ndarray | None:
        if stored_trajectory is None:
            return None
        points, frame_id = stored_trajectory
        return self._points_in_frame_to_local_array(points, frame_id)

    def _robot_pose_in_frame(self, target_frame: str) -> tuple[float, float, float] | None:
        robot_pose = self._robot_pose_in_odom()
        if robot_pose is None:
            return None

        odom_frame = self._odom_frame()
        target = self._frame_name(target_frame) or odom_frame
        x, y, yaw = robot_pose
        transformed = self._point_between_frames(x, y, odom_frame, target)
        if transformed is None:
            return None

        if self._frame_name(odom_frame) == self._frame_name(target):
            return transformed[0], transformed[1], yaw

        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                odom_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            tyaw = self._yaw_from_quaternion(transform.transform.rotation)
            return transformed[0], transformed[1], yaw + tyaw
        except Exception:
            return transformed[0], transformed[1], yaw

    def _local_waypoint_to_frame(
        self,
        waypoint: np.ndarray,
        target_frame: str,
    ) -> tuple[float, float] | None:
        odom_point = self._local_waypoint_to_odom(waypoint)
        if odom_point is None:
            return None
        return self._point_between_frames(
            odom_point[0],
            odom_point[1],
            self._odom_frame(),
            target_frame,
        )

    def _goal_pose_in_frame(self, target_frame: str, stamp=None) -> PoseStamped | None:
        if self.current_goal is None:
            return None

        target = self._frame_name(target_frame)
        goal_frame = self._frame_name(self.current_goal.header.frame_id) or target
        goal = copy.deepcopy(self.current_goal)
        goal.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()

        if goal_frame == target:
            goal.header.frame_id = target
            return goal

        transformed = self._point_between_frames(
            goal.pose.position.x,
            goal.pose.position.y,
            goal_frame,
            target,
        )
        if transformed is None:
            return None

        goal.header.frame_id = target
        goal.pose.position.x = transformed[0]
        goal.pose.position.y = transformed[1]
        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                goal_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            goal_yaw = self._yaw_from_quaternion(self.current_goal.pose.orientation)
            frame_yaw = self._yaw_from_quaternion(transform.transform.rotation)
            self._set_yaw(goal.pose, goal_yaw + frame_yaw)
        except Exception:
            pass
        return goal

    def _phase_local_goal_pose_for_planner(self, stamp=None) -> PoseStamped | None:
        """Return the locked AI waypoint as a temporary Nav2 planner goal."""
        if self.phase_local_goal_odom is None:
            return None

        odom_frame = self._odom_frame()
        target_candidates = []
        if self.current_goal is not None:
            target_candidates.append(self._frame_name(self.current_goal.header.frame_id))
        target_candidates.extend(['map', odom_frame])

        seen = set()
        for target_frame in target_candidates:
            target = self._frame_name(target_frame)
            if not target or target in seen:
                continue
            seen.add(target)

            target_xy = self._point_between_frames(
                self.phase_local_goal_odom[0],
                self.phase_local_goal_odom[1],
                odom_frame,
                target,
            )
            if target_xy is None:
                continue

            yaw = 0.0
            robot_pose = self._robot_pose_in_frame(target)
            if robot_pose is not None:
                rx, ry, ryaw = robot_pose
                dx = target_xy[0] - rx
                dy = target_xy[1] - ry
                yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-3 else ryaw

            return self._pose_stamped(
                target,
                target_xy[0],
                target_xy[1],
                yaw,
                stamp,
            )

        self.get_logger().warn(
            "Unable to convert AI local subgoal into a planner goal pose.",
            throttle_duration_sec=2.0,
        )
        return None

    def _pose_stamped(self, frame_id: str, x: float, y: float, yaw: float, stamp=None) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        self._set_yaw(pose.pose, yaw)
        return pose

    @staticmethod
    def _pose_xy(pose_stamped: PoseStamped) -> tuple[float, float]:
        return (
            float(pose_stamped.pose.position.x),
            float(pose_stamped.pose.position.y),
        )

    def _nearest_path_index(self, path: NavPath, x: float, y: float) -> int:
        if not path.poses:
            return 0
        best_idx = 0
        best_dist = float('inf')
        for idx, pose in enumerate(path.poses):
            px, py = self._pose_xy(pose)
            dist = math.hypot(px - x, py - y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _advance_path_index_by_distance(
        self,
        path: NavPath,
        start_idx: int,
        distance_m: float,
    ) -> tuple[int, float]:
        if not path.poses:
            return 0, 0.0

        start_idx = min(max(int(start_idx), 0), len(path.poses) - 1)
        if distance_m <= 0.0 or start_idx >= len(path.poses) - 1:
            return start_idx, 0.0

        traveled = 0.0
        prev_x, prev_y = self._pose_xy(path.poses[start_idx])
        for idx in range(start_idx + 1, len(path.poses)):
            px, py = self._pose_xy(path.poses[idx])
            traveled += math.hypot(px - prev_x, py - prev_y)
            if traveled >= distance_m:
                return idx, traveled
            prev_x, prev_y = px, py

        return len(path.poses) - 1, traveled

    def _set_intermediate_orientations(self, path: NavPath) -> None:
        for idx in range(len(path.poses) - 1):
            x0, y0 = self._pose_xy(path.poses[idx])
            x1, y1 = self._pose_xy(path.poses[idx + 1])
            yaw = math.atan2(y1 - y0, x1 - x0)
            self._set_yaw(path.poses[idx].pose, yaw)

    def _goal_to_local(self) -> tuple[float, float] | None:
        if self.current_goal is None:
            return None
        return self._point_in_frame_to_local(
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
            self.current_goal.header.frame_id,
        )

    def _append_pose_if_distinct(
        self,
        path: NavPath,
        x: float,
        y: float,
        yaw: float,
        stamp,
        min_distance: float = 0.05,
    ) -> bool:
        if path.poses:
            px, py = self._pose_xy(path.poses[-1])
            if math.hypot(float(x) - px, float(y) - py) < min_distance:
                return False
        path.poses.append(self._pose_stamped(path.header.frame_id, x, y, yaw, stamp))
        return True

    def _select_ai_candidate(
        self,
        candidates: np.ndarray,
        arrival_scores: np.ndarray | None,
        human_positions: np.ndarray | None,
        human_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, int, float, float]:
        arr = np.asarray(candidates, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[-1] < 2 or arr.shape[0] == 0:
            raise ValueError(f"Expected AI candidates shape [K,T,2], got {arr.shape}")
        arr = arr[:, :, :2]

        if self.dwb_integration_mode not in (
            'shaped_path',
            'one_waypoint_replace',
            'hard_gate',
        ) or arr.shape[0] == 1:
            return arr[0], 0, 0.0, float('nan')

        humans = None
        if human_positions is not None:
            humans = np.asarray(human_positions, dtype=np.float32)
            if human_mask is not None and humans.ndim == 3:
                mask = np.asarray(human_mask, dtype=bool)
                if mask.shape == humans.shape[:2]:
                    valid_points = []
                    for t_idx in range(humans.shape[0]):
                        valid = humans[t_idx, ~mask[t_idx], :2]
                        if len(valid) > 0:
                            valid_points.append(valid)
                    humans = np.concatenate(valid_points, axis=0) if valid_points else None
            elif humans.ndim == 3:
                humans = humans.reshape(-1, humans.shape[-1])[:, :2]
            if humans is not None and len(humans) == 0:
                humans = None

        global_path_local = self._path_to_local_array(self.latest_global_path, max_points=200)
        goal_local = self._goal_to_local()
        scores = []
        min_human_dists = []
        for cand in arr:
            cost = 0.0
            min_human_dist = float('inf')
            if humans is not None and len(humans) > 0:
                dists = np.linalg.norm(cand[:, None, :] - humans[None, :, :], axis=2)
                min_per_wp = np.min(dists, axis=1)
                min_human_dist = float(np.min(min_per_wp))
                cost += self.social_cost_w_hard * float(np.count_nonzero(min_per_wp < self.social_cost_hard_radius))
                personal_penalty = np.maximum(0.0, self.social_cost_personal_radius - min_per_wp) ** 2
                cost += self.social_cost_w_personal * float(np.sum(personal_penalty))
                cost += self.social_cost_w_social * float(
                    np.sum(np.exp(-(min_per_wp ** 2) / (self.social_cost_social_radius ** 2)))
                )

            if global_path_local is not None and len(global_path_local) > 0:
                d_global = np.linalg.norm(cand[:, None, :] - global_path_local[None, :, :], axis=2)
                cost += self.social_cost_w_global * float(np.mean(np.min(d_global, axis=1)))

            if len(cand) > 0:
                progress_idx = min(self.shaped_path_num_waypoints - 1, len(cand) - 1)
                if goal_local is not None:
                    start_goal_dist = math.hypot(goal_local[0], goal_local[1])
                    end_goal_dist = math.hypot(goal_local[0] - cand[progress_idx, 0], goal_local[1] - cand[progress_idx, 1])
                    progress = start_goal_dist - end_goal_dist
                else:
                    progress = float(np.linalg.norm(cand[progress_idx]))
                cost -= self.social_cost_w_progress * progress

            if len(cand) >= 3:
                second_diff = cand[2:] - 2.0 * cand[1:-1] + cand[:-2]
                cost += self.social_cost_w_smooth * float(np.sum(np.linalg.norm(second_diff, axis=1)))

            scores.append(cost)
            min_human_dists.append(min_human_dist)

        best_idx = int(np.argmin(np.asarray(scores, dtype=np.float32)))
        best_score = float(scores[best_idx])
        best_min_human_dist = float(min_human_dists[best_idx])
        self.get_logger().info(
            "AI candidate selection: "
            f"mode={self.dwb_integration_mode} best_k={best_idx} "
            f"score={best_score:.3f} min_human_dist={best_min_human_dist:.3f}",
            throttle_duration_sec=1.0,
        )
        return arr[best_idx], best_idx, best_score, best_min_human_dist

    def _build_ai_shaped_path(self, global_path: NavPath | None, waypoints: np.ndarray) -> NavPath | None:
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None
        if self.current_goal is None:
            return None

        source_global_path = global_path
        if (
            source_global_path is None
            and self.latest_benchmark_global_path is not None
            and self.latest_benchmark_global_path.poses
        ):
            source_global_path = self.latest_benchmark_global_path
            self.get_logger().info(
                "AI shaped path using cached benchmark global path.",
                throttle_duration_sec=1.0,
            )
        if source_global_path is None or not source_global_path.poses:
            self.get_logger().warn(
                "AI shaped path requires a Nav2 global path; skipping shaped FollowPath update.",
                throttle_duration_sec=2.0,
            )
            return None

        path_frame = ''
        if source_global_path is not None:
            path_frame = self._frame_name(source_global_path.header.frame_id)
        if not path_frame:
            path_frame = self._frame_name(self.current_goal.header.frame_id)
        if not path_frame:
            path_frame = self._odom_frame()

        robot_pose = self._robot_pose_in_frame(path_frame)
        if robot_pose is None:
            return None

        stamp = self.get_clock().now().to_msg()
        path = NavPath()
        path.header.stamp = stamp
        path.header.frame_id = path_frame
        shaped_waypoint_path = NavPath()
        shaped_waypoint_path.header.stamp = stamp
        shaped_waypoint_path.header.frame_id = path_frame

        rx, ry, ryaw = robot_pose
        path.poses.append(self._pose_stamped(path_frame, rx, ry, ryaw, stamp))

        arr = np.asarray(waypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            return None

        last_ai_xy = None
        inserted_waypoints_local = []
        for wp_i, waypoint in enumerate(arr[:self.shaped_path_num_waypoints]):
            xy = self._local_waypoint_to_frame(waypoint[:2], path_frame)
            if xy is None:
                continue
            if wp_i == 3:
                roundtrip_local = self._point_in_frame_to_local(xy[0], xy[1], path_frame)
                roundtrip_str = (
                    f"({float(roundtrip_local[0]):.3f},{float(roundtrip_local[1]):.3f})"
                    if roundtrip_local is not None
                    else "(n/a,n/a)"
                )
                coord_mode = str(self.agent.config.extra_params.get('coordinate_mode', '')).strip() or 'default'
                self.get_logger().info(
                    "[WP_DEBUG] stage=shaped_local_to_frame wp_idx=3 "
                    f"raw=(n/a,n/a) "
                    f"ros=({float(waypoint[0]):.3f},{float(waypoint[1]):.3f}) "
                    f"selected=({float(waypoint[0]):.3f},{float(waypoint[1]):.3f}) "
                    f"frame_xy=({float(xy[0]):.3f},{float(xy[1]):.3f}) "
                    f"roundtrip_local={roundtrip_str} "
                    f"robot_yaw={float(ryaw):.3f} "
                    f"mode={self.dwb_integration_mode} coord_mode={coord_mode}",
                    throttle_duration_sec=0.5,
                )
            last_ai_xy = xy
            if self._append_pose_if_distinct(path, xy[0], xy[1], ryaw, stamp):
                inserted_waypoints_local.append(np.asarray(waypoint[:2], dtype=np.float32))
                shaped_waypoint_path.poses.append(copy.deepcopy(path.poses[-1]))

        goal_pose = self._goal_pose_in_frame(path_frame, stamp)
        if goal_pose is None:
            return None

        robot_global_idx = self._nearest_path_index(source_global_path, rx, ry)
        rejoin_seed_idx = robot_global_idx
        nearest_ai_idx = None
        if last_ai_xy is not None:
            nearest_ai_idx = self._nearest_path_index(source_global_path, last_ai_xy[0], last_ai_xy[1])
            rejoin_seed_idx = max(robot_global_idx, nearest_ai_idx)

        tail_seed = max(1, rejoin_seed_idx) if len(source_global_path.poses) > 1 else rejoin_seed_idx
        tail_start, actual_skip = self._advance_path_index_by_distance(
            source_global_path,
            tail_seed,
            self.ai_rejoin_skip_distance if last_ai_xy is not None else 0.0,
        )
        tail_start = max(tail_start, max(1, robot_global_idx) if len(source_global_path.poses) > 1 else robot_global_idx)
        self.get_logger().info(
            "AI shaped path rejoin: "
            f"robot_idx={robot_global_idx} "
            f"nearest_ai_idx={nearest_ai_idx if nearest_ai_idx is not None else 'none'} "
            f"rejoin_idx={tail_start} "
            f"skip={actual_skip:.2f}m target={self.ai_rejoin_skip_distance:.2f}m "
            f"ai_wps={len(inserted_waypoints_local)}/{min(self.shaped_path_num_waypoints, len(arr))} "
            f"global_poses={len(source_global_path.poses)}",
            throttle_duration_sec=1.0,
        )
        for pose in source_global_path.poses[tail_start:]:
            tail_pose = copy.deepcopy(pose)
            tail_pose.header.stamp = path.header.stamp
            tail_pose.header.frame_id = path_frame
            if path.poses:
                last_x, last_y = self._pose_xy(path.poses[-1])
                pose_x, pose_y = self._pose_xy(tail_pose)
                if math.hypot(pose_x - last_x, pose_y - last_y) < 0.05:
                    continue
            path.poses.append(tail_pose)

        if len(path.poses) < 2:
            self.get_logger().warn(
                "AI shaped path had no usable AI/global tail poses.",
                throttle_duration_sec=2.0,
            )
            return None

        if math.hypot(
            path.poses[-1].pose.position.x - goal_pose.pose.position.x,
            path.poses[-1].pose.position.y - goal_pose.pose.position.y,
        ) > 0.05:
            path.poses.append(goal_pose)
        else:
            path.poses[-1].pose.orientation = goal_pose.pose.orientation

        self._set_intermediate_orientations(path)
        if inserted_waypoints_local:
            self.latest_shaped_ai_waypoints = np.asarray(inserted_waypoints_local, dtype=np.float32)
            self.latest_shaped_ai_waypoint_path = copy.deepcopy(shaped_waypoint_path)
        self.get_logger().info(
            "AI shaped FollowPath: "
            f"poses={len(path.poses)} ai_wps={len(inserted_waypoints_local)} "
            f"frame={path_frame}",
            throttle_duration_sec=1.0,
        )
        return path

    def _build_one_waypoint_replace_path(
        self,
        global_path: NavPath | None,
        waypoints: np.ndarray,
    ) -> NavPath | None:
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None
        if self.current_goal is None:
            return None

        source_global_path = global_path
        if (
            source_global_path is None
            and self.latest_benchmark_global_path is not None
            and self.latest_benchmark_global_path.poses
        ):
            source_global_path = self.latest_benchmark_global_path
            self.get_logger().info(
                "AI one-waypoint replace using cached benchmark global path.",
                throttle_duration_sec=1.0,
            )
        if source_global_path is None or not source_global_path.poses:
            self.get_logger().warn(
                "AI one-waypoint replace requires a Nav2 global path; skipping FollowPath update.",
                throttle_duration_sec=2.0,
            )
            return None

        path_frame = self._frame_name(source_global_path.header.frame_id)
        if not path_frame:
            path_frame = self._frame_name(self.current_goal.header.frame_id)
        if not path_frame:
            path_frame = self._odom_frame()

        robot_pose = self._robot_pose_in_frame(path_frame)
        if robot_pose is None:
            return None

        stamp = self.get_clock().now().to_msg()
        rx, ry, ryaw = robot_pose
        robot_global_idx = self._nearest_path_index(source_global_path, rx, ry)
        tail_start = robot_global_idx + 1
        if len(source_global_path.poses) > 1:
            tail_start = max(1, tail_start)
        tail_start = min(tail_start, len(source_global_path.poses))

        path = NavPath()
        path.header.stamp = stamp
        path.header.frame_id = path_frame
        path.poses.append(self._pose_stamped(path_frame, rx, ry, ryaw, stamp))

        replacement_idx = None
        replacement_xy = None
        replacement_wp_idx = self._path_waypoint_index(waypoints)
        replacement_reason = "ok"
        replacement_deviation = None

        waypoint = self._path_waypoint(waypoints)
        if waypoint is None or replacement_wp_idx is None:
            replacement_reason = "no_ai_waypoint"
        else:
            candidate_idx = robot_global_idx + replacement_wp_idx + 1
            if candidate_idx >= len(source_global_path.poses) - 1:
                replacement_reason = (
                    f"target_idx_out_of_range target={candidate_idx} "
                    f"global_poses={len(source_global_path.poses)}"
                )
            else:
                ai_xy = self._local_waypoint_to_frame(waypoint[:2], path_frame)
                ai_local = None
                if ai_xy is not None:
                    ai_local = self._point_in_frame_to_local(ai_xy[0], ai_xy[1], path_frame)
                if ai_xy is None or ai_local is None:
                    replacement_reason = "ai_transform_unavailable"
                elif ai_local[0] <= 0.05:
                    replacement_reason = f"ai_not_ahead local_x={ai_local[0]:.2f}"
                else:
                    ai_distance = math.hypot(ai_xy[0] - rx, ai_xy[1] - ry)
                    if (
                        self.rolling_min_local_goal_distance > 0.0
                        and ai_distance < self.rolling_min_local_goal_distance
                    ):
                        replacement_reason = (
                            f"ai_too_close dist={ai_distance:.2f} "
                            f"min={self.rolling_min_local_goal_distance:.2f}"
                        )
                    elif (
                        self.rolling_max_local_goal_distance > 0.0
                        and ai_distance > self.rolling_max_local_goal_distance
                    ):
                        replacement_reason = (
                            f"ai_too_far dist={ai_distance:.2f} "
                            f"max={self.rolling_max_local_goal_distance:.2f}"
                        )
                    else:
                        original_x, original_y = self._pose_xy(source_global_path.poses[candidate_idx])
                        replacement_deviation = math.hypot(ai_xy[0] - original_x, ai_xy[1] - original_y)
                        replacement_idx = candidate_idx
                        replacement_xy = ai_xy

        if replacement_idx is None:
            self.get_logger().warn(
                "AI one-waypoint replace falling back to original global path: "
                f"reason={replacement_reason}",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().info(
                "AI one-waypoint replace: "
                f"robot_idx={robot_global_idx} target_idx={replacement_idx} "
                f"wp_idx={replacement_wp_idx} "
                f"ai_local=({float(waypoint[0]):.2f},{float(waypoint[1]):.2f}) "
                f"frame_xy=({replacement_xy[0]:.2f},{replacement_xy[1]:.2f}) "
                f"deviation={replacement_deviation:.2f}m "
                f"global_poses={len(source_global_path.poses)}",
                throttle_duration_sec=1.0,
            )

        for idx in range(tail_start, len(source_global_path.poses)):
            if idx == replacement_idx and replacement_xy is not None:
                tail_pose = self._pose_stamped(
                    path_frame,
                    replacement_xy[0],
                    replacement_xy[1],
                    ryaw,
                    stamp,
                )
            else:
                tail_pose = copy.deepcopy(source_global_path.poses[idx])
                tail_pose.header.stamp = path.header.stamp
                tail_pose.header.frame_id = path_frame

            if path.poses:
                last_x, last_y = self._pose_xy(path.poses[-1])
                pose_x, pose_y = self._pose_xy(tail_pose)
                if math.hypot(pose_x - last_x, pose_y - last_y) < 0.05:
                    continue
            path.poses.append(tail_pose)

        goal_pose = self._goal_pose_in_frame(path_frame, stamp)
        if goal_pose is None:
            return None
        if not path.poses or math.hypot(
            path.poses[-1].pose.position.x - goal_pose.pose.position.x,
            path.poses[-1].pose.position.y - goal_pose.pose.position.y,
        ) > 0.05:
            path.poses.append(goal_pose)
        else:
            path.poses[-1].pose.orientation = goal_pose.pose.orientation

        if len(path.poses) < 2:
            self.get_logger().warn(
                "AI one-waypoint replace built fewer than two poses.",
                throttle_duration_sec=2.0,
            )
            return None

        self._set_intermediate_orientations(path)
        return path

    def _build_ai_adapted_path(self, global_path: NavPath | None, waypoints: np.ndarray) -> NavPath | None:
        if self.dwb_integration_mode == 'shaped_path':
            return self._build_ai_shaped_path(global_path, waypoints)
        if self.dwb_integration_mode == 'one_waypoint_replace':
            return self._build_one_waypoint_replace_path(global_path, waypoints)
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None

        if self.current_goal is None:
            return None

        path_frame = ''
        if global_path is not None:
            path_frame = self._frame_name(global_path.header.frame_id)
        if not path_frame:
            path_frame = self._frame_name(self.current_goal.header.frame_id)
        if not path_frame:
            path_frame = self._odom_frame()

        robot_pose = self._robot_pose_in_frame(path_frame)
        if robot_pose is None:
            return None

        stamp = self.get_clock().now().to_msg()
        path = NavPath()
        path.header.stamp = stamp
        path.header.frame_id = path_frame

        rx, ry, ryaw = robot_pose
        path.poses.append(self._pose_stamped(path_frame, rx, ry, ryaw, stamp))

        if self._phase_local_goal_active():
            target_xy = None
            if self.phase_local_goal_odom is not None:
                target_xy = self._point_between_frames(
                    self.phase_local_goal_odom[0],
                    self.phase_local_goal_odom[1],
                    self._odom_frame(),
                    path_frame,
                )
            else:
                waypoint = self._path_waypoint(waypoints)
                if waypoint is not None:
                    target_xy = self._local_waypoint_to_frame(waypoint, path_frame)
            if target_xy is None:
                return None

            wx, wy = target_xy
            path.poses.clear()
            path.poses.append(self._pose_stamped(path_frame, rx, ry, ryaw, stamp))

            if global_path is not None and global_path.poses:
                nearest_idx = self._nearest_path_index(global_path, wx, wy)
                for pose in global_path.poses[1:nearest_idx]:
                    tail_pose = copy.deepcopy(pose)
                    tail_pose.header.stamp = stamp
                    tail_pose.header.frame_id = path_frame
                    path.poses.append(tail_pose)

            path.poses.append(self._pose_stamped(path_frame, wx, wy, ryaw, stamp))
            self._set_intermediate_orientations(path)
            dist = self._distance_to_phase_local_goal()
            dist_str = f"{dist:.2f}m" if dist is not None else "n/a"
            self.get_logger().info(
                "SocialNav FollowPath phase=ai_local_goal: "
                f"mode={self.local_goal_relock_mode} "
                f"subgoal_count={self.phase_local_goal_lock_count} "
                f"wp_idx={self.phase_local_goal_wp_idx} poses={len(path.poses)} "
                f"local_goal_dist={dist_str} frame={path_frame} "
                f"last_xy=({path.poses[-1].pose.position.x:.2f},{path.poses[-1].pose.position.y:.2f})",
                throttle_duration_sec=1.0,
            )
            return path

        goal_pose = self._goal_pose_in_frame(path_frame, stamp)
        if goal_pose is None:
            return None

        if self.phase_local_goal_enabled and self.path_phase == self.PHASE_GLOBAL_GOAL:
            if global_path is not None and global_path.poses:
                nearest_idx = self._nearest_path_index(global_path, rx, ry)
                tail_start = nearest_idx
                if len(global_path.poses) > 1:
                    tail_start = max(1, tail_start)
                for pose in global_path.poses[tail_start:]:
                    tail_pose = copy.deepcopy(pose)
                    tail_pose.header.stamp = path.header.stamp
                    tail_pose.header.frame_id = path_frame
                    path.poses.append(tail_pose)

            if not path.poses or math.hypot(
                path.poses[-1].pose.position.x - goal_pose.pose.position.x,
                path.poses[-1].pose.position.y - goal_pose.pose.position.y,
            ) > 0.05:
                path.poses.append(goal_pose)
            else:
                path.poses[-1].pose.orientation = goal_pose.pose.orientation

            self._set_intermediate_orientations(path)
            self.get_logger().info(
                "SocialNav FollowPath phase=global_goal: "
                f"poses={len(path.poses)} frame={path_frame} "
                f"global_poses={len(global_path.poses) if global_path is not None else 0}",
                throttle_duration_sec=1.0,
            )
            return path

        waypoint = self._path_waypoint(waypoints)
        if waypoint is None:
            return None

        waypoint_xy = self._local_waypoint_to_frame(waypoint, path_frame)
        if waypoint_xy is None:
            return None

        wx, wy = waypoint_xy
        path.poses.append(self._pose_stamped(path_frame, wx, wy, ryaw, stamp))

        if global_path is not None and global_path.poses:
            nearest_idx = self._nearest_path_index(global_path, wx, wy)
            tail_seed = nearest_idx
            if len(global_path.poses) > 1:
                tail_seed = max(1, tail_seed)
            tail_start, actual_skip = self._advance_path_index_by_distance(
                global_path,
                tail_seed,
                self.ai_rejoin_skip_distance,
            )
            self.get_logger().info(
                "SocialNav rejoin: "
                f"nearest_idx={nearest_idx} rejoin_idx={tail_start} "
                f"skip={actual_skip:.2f}m target={self.ai_rejoin_skip_distance:.2f}m "
                f"global_poses={len(global_path.poses)}",
                throttle_duration_sec=1.0,
            )
            for pose in global_path.poses[tail_start:]:
                tail_pose = copy.deepcopy(pose)
                tail_pose.header.stamp = path.header.stamp
                tail_pose.header.frame_id = path_frame
                path.poses.append(tail_pose)

        if not path.poses or math.hypot(
            path.poses[-1].pose.position.x - goal_pose.pose.position.x,
            path.poses[-1].pose.position.y - goal_pose.pose.position.y,
        ) > 0.05:
            path.poses.append(goal_pose)
        else:
            path.poses[-1].pose.orientation = goal_pose.pose.orientation

        self._set_intermediate_orientations(path)
        return path

    def _path_update_due(self) -> bool:
        if self.last_path_request_time is None:
            return not self.path_request_in_progress
        elapsed = (self.get_clock().now() - self.last_path_request_time).nanoseconds / 1e9
        return elapsed >= self.path_update_period_sec and not self.path_request_in_progress

    def _request_ai_path_update(self, waypoints: np.ndarray) -> None:
        if self.current_goal is None:
            self.get_logger().info(
                "[PATH_DEBUG] request_skip reason=no_current_goal",
                throttle_duration_sec=1.0,
            )
            return
        if not self._clock_ready_for_nav2():
            self.get_logger().warn(
                "Waiting for non-zero simulation clock before sending SocialNav path to Nav2.",
                throttle_duration_sec=2.0,
            )
            return
        now = self.get_clock().now()
        if self.last_path_request_time is None:
            if self.path_request_in_progress:
                self.get_logger().info(
                    "[PATH_DEBUG] request_skip reason=in_progress_without_timestamp",
                    throttle_duration_sec=0.5,
                )
                return
        else:
            elapsed = (now - self.last_path_request_time).nanoseconds / 1e9
            if self.path_request_in_progress:
                if self.compute_path_timeout_sec > 0.0 and elapsed >= self.compute_path_timeout_sec:
                    self.get_logger().warn(
                        "[PATH_DEBUG] compute_path_timeout "
                        f"elapsed={elapsed:.3f}s timeout={self.compute_path_timeout_sec:.3f}s "
                        "fallback=minimal_follow_path",
                        throttle_duration_sec=0.5,
                    )
                    if self.last_compute_path_goal_handle is not None:
                        try:
                            self.last_compute_path_goal_handle.cancel_goal_async()
                        except Exception as exc:
                            self.get_logger().warn(
                                f"ComputePathToPose cancel after timeout failed: {exc}",
                                throttle_duration_sec=2.0,
                            )
                    self.last_compute_path_goal_handle = None
                    self._pending_compute_path_goal_kind = None
                    self.path_request_in_progress = False
                    self.last_path_request_time = now
                    self.latest_ai_waypoints = np.array(waypoints, copy=True)
                    self._send_ai_follow_path(None, self.latest_ai_waypoints)
                    return
                self.get_logger().info(
                    "[PATH_DEBUG] request_skip "
                    f"reason=compute_path_in_progress elapsed={elapsed:.3f}s",
                    throttle_duration_sec=0.5,
                )
                return
            if elapsed < self.path_update_period_sec:
                self.get_logger().info(
                    "[PATH_DEBUG] request_skip "
                    f"reason=not_due elapsed={elapsed:.3f}s "
                    f"period={self.path_update_period_sec:.3f}s",
                    throttle_duration_sec=0.5,
                )
                return

        wp_idx = self._path_waypoint_index(waypoints)
        wp = waypoints[wp_idx] if wp_idx is not None else [0.0, 0.0]
        self.get_logger().info(
            "[PATH_DEBUG] request_start "
            f"mode={self.dwb_integration_mode} phase={self.path_phase} "
            f"in_progress={self.path_request_in_progress} wp_idx={wp_idx} "
            f"wp_local=({float(wp[0]):.3f},{float(wp[1]):.3f})",
            throttle_duration_sec=0.5,
        )
        if not self._path_update_due():
            self.get_logger().info(
                "[PATH_DEBUG] request_skip reason=path_update_due_false_after_checks",
                throttle_duration_sec=0.5,
            )
            return

        self.latest_ai_waypoints = np.array(waypoints, copy=True)
        self.last_path_request_time = now

        self.path_request_in_progress = True

        if not self.compute_path_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().info(
                "[PATH_DEBUG] compute_path_unavailable fallback=minimal_follow_path",
                throttle_duration_sec=0.5,
            )
            self.get_logger().warn(
                f"ComputePathToPose action {self.planner_action_name} unavailable; "
                "sending a minimal SocialNav path directly to FollowPath.",
                throttle_duration_sec=2.0,
            )
            self.path_request_in_progress = False
            self._pending_compute_path_goal_kind = None
            self._send_ai_follow_path(None, self.latest_ai_waypoints)
            return

        goal_msg = ComputePathToPose.Goal()
        planner_goal = None
        planner_goal_kind = 'benchmark_goal'
        if (
            self._phase_local_goal_active()
            and self.dwb_integration_mode not in ('shaped_path', 'one_waypoint_replace')
        ):
            planner_goal = self._phase_local_goal_pose_for_planner(now.to_msg())
            planner_goal_kind = 'ai_local_subgoal'
            if planner_goal is None:
                self.get_logger().warn(
                    "[PATH_DEBUG] local_subgoal_pose_unavailable fallback=minimal_follow_path",
                    throttle_duration_sec=2.0,
                )
                self.path_request_in_progress = False
                self._pending_compute_path_goal_kind = None
                self._send_ai_follow_path(None, self.latest_ai_waypoints)
                return
        else:
            planner_goal = copy.deepcopy(self.current_goal)
            planner_goal.header.stamp = now.to_msg()

        goal_msg.goal = planner_goal
        if hasattr(goal_msg, 'planner_id'):
            goal_msg.planner_id = self.planner_id
        if hasattr(goal_msg, 'use_start'):
            goal_msg.use_start = False

        self.get_logger().info(
            "[PATH_DEBUG] compute_path_goal_send "
            f"planner_action={self.planner_action_name} "
            f"target={planner_goal_kind} "
            f"frame={planner_goal.header.frame_id} "
            f"xy=({planner_goal.pose.position.x:.2f},{planner_goal.pose.position.y:.2f})",
            throttle_duration_sec=0.5,
        )
        self._pending_compute_path_goal_kind = planner_goal_kind
        future = self.compute_path_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_compute_path_goal_response)

    def _on_compute_path_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.path_request_in_progress = False
            self.last_compute_path_goal_handle = None
            self._pending_compute_path_goal_kind = None
            self.get_logger().warn(f"ComputePathToPose request failed: {exc}", throttle_duration_sec=2.0)
            return

        if not goal_handle.accepted:
            self.path_request_in_progress = False
            self.last_compute_path_goal_handle = None
            self._pending_compute_path_goal_kind = None
            self.get_logger().warn("ComputePathToPose goal rejected.", throttle_duration_sec=2.0)
            return

        self.last_compute_path_goal_handle = goal_handle
        self.get_logger().info(
            "[PATH_DEBUG] compute_path_goal_accepted",
            throttle_duration_sec=0.5,
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_compute_path_result)

    def _on_compute_path_result(self, future) -> None:
        self.path_request_in_progress = False
        self.last_compute_path_goal_handle = None
        planner_goal_kind = self._pending_compute_path_goal_kind
        self._pending_compute_path_goal_kind = None
        try:
            result_wrapper = future.result()
            global_path = result_wrapper.result.path
        except Exception as exc:
            self.get_logger().warn(
                f"ComputePathToPose result failed; using minimal SocialNav path: {exc}",
                throttle_duration_sec=2.0,
            )
            global_path = None

        self.latest_global_path = copy.deepcopy(global_path) if global_path is not None else None
        if planner_goal_kind == 'benchmark_goal' and global_path is not None:
            self.latest_benchmark_global_path = copy.deepcopy(global_path)
        self.get_logger().info(
            "[PATH_DEBUG] compute_path_result "
            f"target={planner_goal_kind or 'unknown'} "
            f"global_poses={len(global_path.poses) if global_path is not None else 0} "
            f"has_latest_ai_waypoints={self.latest_ai_waypoints is not None}",
            throttle_duration_sec=0.5,
        )
        if self.latest_ai_waypoints is None:
            return
        self._send_ai_follow_path(global_path, self.latest_ai_waypoints)

    def _send_ai_follow_path(
        self,
        global_path: NavPath | None,
        waypoints: np.ndarray,
    ) -> bool:
        self.get_logger().info(
            "[PATH_DEBUG] follow_path_build_start "
            f"mode={self.dwb_integration_mode} "
            f"global_poses={len(global_path.poses) if global_path is not None else 0} "
            f"waypoints={len(waypoints) if waypoints is not None else 0}",
            throttle_duration_sec=0.5,
        )
        path = self._build_ai_adapted_path(global_path, waypoints)
        if path is None or len(path.poses) < 2:
            self.get_logger().warn(
                "Unable to build SocialNav FollowPath path; holding current DWB command.",
                throttle_duration_sec=2.0,
            )
            return False

        self.latest_ai_path = copy.deepcopy(path)
        self.path_pub.publish(path)
        self.get_logger().info(
            "[PATH_DEBUG] follow_path_built "
            f"poses={len(path.poses)} frame={path.header.frame_id}",
            throttle_duration_sec=0.5,
        )

        if not self.follow_path_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn(
                f"FollowPath action {self.follow_path_action_name} unavailable.",
                throttle_duration_sec=2.0,
            )
            return False

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        if hasattr(goal_msg, 'controller_id'):
            goal_msg.controller_id = self.follow_path_controller_id
        if hasattr(goal_msg, 'goal_checker_id'):
            goal_msg.goal_checker_id = self.follow_path_goal_checker_id

        self.last_follow_path_send_time = self.get_clock().now()
        self.follow_path_send_count += 1
        send_count = self.follow_path_send_count
        future = self.follow_path_client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda done_future, seq=send_count: self._on_follow_path_goal_response(done_future, seq)
        )

        wp_idx = self._path_waypoint_index(waypoints)
        wp = waypoints[wp_idx] if wp_idx is not None else [0.0, 0.0]
        self.get_logger().info(
            f"Sent SocialNav path to DWB FollowPath: poses={len(path.poses)} "
            f"phase={self.path_phase} mode={self.local_goal_relock_mode} "
            f"subgoal_count={self.phase_local_goal_lock_count} "
            f"wp_idx={wp_idx} wp_local=({wp[0]:.2f},{wp[1]:.2f}) "
            f"final_frame={path.header.frame_id}",
            throttle_duration_sec=1.0,
        )
        return True

    def _on_follow_path_goal_response(self, future, send_count: int) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f"FollowPath request failed: {exc}", throttle_duration_sec=2.0)
            return

        if not goal_handle.accepted:
            self.get_logger().warn(
                "[PATH_DEBUG] follow_path_goal_rejected "
                f"action={self.follow_path_action_name} "
                f"send_count={send_count} "
                f"raw_cmd_status=({self._dwb_raw_cmd_status()})",
                throttle_duration_sec=2.0,
            )
            return

        self.last_follow_path_goal_handle = goal_handle
        self.active_follow_path_send_count = send_count
        self._follow_path_owner = 'ai'
        self.get_logger().info(
            "[PATH_DEBUG] follow_path_goal_accepted "
            f"action={self.follow_path_action_name} "
            f"send_count={send_count} owner={self._follow_path_owner}",
            throttle_duration_sec=0.5,
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda done_future, seq=send_count: self._on_follow_path_result(done_future, seq)
        )

    def _on_follow_path_result(self, future, send_count: int) -> None:
        try:
            result_wrapper = future.result()
        except Exception as exc:
            self.get_logger().warn(
                f"[PATH_DEBUG] follow_path_result_error send_count={send_count} error={exc}",
                throttle_duration_sec=2.0,
            )
            return

        status = getattr(result_wrapper, 'status', 'unknown')
        result_msg = getattr(result_wrapper, 'result', None)
        error_code = getattr(result_msg, 'error_code', 'n/a')
        error_msg = getattr(result_msg, 'error_msg', '')
        is_current = send_count == self.active_follow_path_send_count
        state = "current" if is_current else "stale"
        self.get_logger().info(
            "[PATH_DEBUG] follow_path_result "
            f"status={status} action={self.follow_path_action_name} "
            f"send_count={send_count} active_send_count={self.active_follow_path_send_count} "
            f"state={state} phase={self._subgoal_phase} owner={self._follow_path_owner} "
            f"error_code={error_code} error_msg={error_msg!r} "
            f"raw_cmd_count={self.dwb_cmd_count}",
            throttle_duration_sec=0.5,
        )
        if not is_current:
            return

        self.last_follow_path_goal_handle = None
        self.active_follow_path_send_count = 0
        self._follow_path_owner = 'none'

        if status == 4 and self._subgoal_phase == 'chasing_waypoint':
            self.get_logger().info(
                "[SUBGOAL] FollowPath SUCCEEDED at waypoint; triggering immediate re-inference"
            )
            self._subgoal_phase = 'waypoint_reached'
            self._force_infer = True
            self.last_path_request_time = None
            self.path_request_in_progress = False
            self.phase_local_goal_odom = None
            self.control_loop_callback()
            return

        if status == 5 and self._subgoal_phase == 'chasing_waypoint':
            self.get_logger().warn(
                "[SUBGOAL] FollowPath CANCELED externally; re-asserting AI ownership"
            )
            self._subgoal_phase = 'cancelling_bt'
            self._pending_bt_cancel_future = None
            self.last_path_request_time = None
            self.path_request_in_progress = False
            return

        if status == 6:
            self.last_path_request_time = None

    # ──────────────────────────── Startup / Fallback Helpers ────────────────────

    def _publish_dwb_fallback(self, reason: str) -> bool:
        """Fallback an toàn: dừng robot hoặc relay DWB cmd thô khi AI không sẵn sàng."""
        dist = self._distance_to_goal()
        dist_str = f"{dist:.3f}m" if dist is not None else "n/a"
        self.get_logger().warn(
            f"\033[91m[FALLBACK] dist_to_goal={dist_str} reason={reason}\033[0m",
            throttle_duration_sec=0.5,
        )
        if (
            not self.episode_active
            or self.task_complete
            or self.reset_in_progress
        ):
            self.cmd_pub.publish(Twist())
            return False

        if not self.fallback_to_dwb:
            self.get_logger().warn(
                f"{reason}; DWB fallback disabled, holding position.",
                throttle_duration_sec=2.0,
            )
            self.cmd_pub.publish(Twist())
            return False

        self._publish_status_visualization(reason)

        cmd = self._fresh_dwb_raw_cmd()
        if cmd is None:
            self.get_logger().warn(
                f"{reason}; no fresh DWB fallback command available, holding position.",
                throttle_duration_sec=2.0,
            )
            self.cmd_pub.publish(Twist())
            return False

        self.cmd_pub.publish(cmd)
        self.get_logger().warn(
            f"{reason}; publishing raw DWB fallback: "
            f"v={cmd.linear.x:.3f} w={cmd.angular.z:.3f}",
            throttle_duration_sec=2.0,
        )
        return True

    def _dwb_raw_cmd_status(self) -> str:
        now = self.get_clock().now()
        cmd_age = None
        if self.latest_dwb_cmd is not None and self.last_dwb_cmd_time is not None:
            cmd_age = (now - self.last_dwb_cmd_time).nanoseconds / 1e9

        follow_age = None
        if self.last_follow_path_send_time is not None:
            follow_age = (now - self.last_follow_path_send_time).nanoseconds / 1e9

        try:
            raw_publishers = self.count_publishers(self.dwb_cmd_topic)
        except Exception:
            raw_publishers = -1

        try:
            cmd_subscribers = self.count_subscribers(self.cmd_vel_topic)
        except Exception:
            cmd_subscribers = -1

        cmd_state = "none"
        if self.latest_dwb_cmd is not None:
            stale = cmd_age is not None and cmd_age > self.dwb_cmd_staleness_sec
            cmd_state = (
                f"age={cmd_age:.3f}s stale={stale} "
                f"v={self.latest_dwb_cmd.linear.x:.3f} "
                f"w={self.latest_dwb_cmd.angular.z:.3f}"
            )

        follow_state = "none"
        if follow_age is not None:
            follow_state = f"age={follow_age:.3f}s count={self.follow_path_send_count}"

        return (
            f"topic={self.dwb_cmd_topic} raw_publishers={raw_publishers} "
            f"cmd_subscribers={cmd_subscribers} raw_count={self.dwb_cmd_count} "
            f"cmd={cmd_state} follow_path={follow_state} "
            f"has_follow_goal={self.last_follow_path_goal_handle is not None} "
            f"episode_active={self.episode_active} task_complete={self.task_complete} "
            f"reset_in_progress={self.reset_in_progress}"
        )

    def _fresh_dwb_raw_cmd(self) -> Twist | None:
        if self.latest_dwb_cmd is None:
            return None
        cmd_age = (self.get_clock().now() - self.last_dwb_cmd_time).nanoseconds / 1e9
        if cmd_age > self.dwb_cmd_staleness_sec:
            return None
        return self._copy_twist(self.latest_dwb_cmd)

    def _fresh_dwb_eval(self) -> LocalPlanEvaluation | None:
        if self.latest_eval is None:
            return None
        eval_age = (self.get_clock().now() - self.last_eval_time).nanoseconds / 1e9
        if eval_age > self.max_eval_staleness_sec:
            return None
        return self.latest_eval

    def _relay_dwb_raw_cmd(self, reason: str) -> bool:
        cmd = self._fresh_dwb_raw_cmd()
        if cmd is None:
            self.get_logger().warn(
                f"{reason}; waiting for fresh DWB cmd_vel_nav_raw. "
                f"[DWB_DEBUG] {self._dwb_raw_cmd_status()}",
                throttle_duration_sec=2.0,
            )
            self.cmd_pub.publish(Twist())
            return False

        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            "[DWB_DEBUG] raw_cmd_relayed "
            f"reason={reason} v={cmd.linear.x:.3f} w={cmd.angular.z:.3f} "
            f"{self._dwb_raw_cmd_status()}",
            throttle_duration_sec=2.0,
        )
        return True

    def _publish_dwb_hard_gate_cmd(self, waypoints: np.ndarray) -> bool:
        if self.dwb_hard_gate_adapter is None:
            return self._relay_dwb_raw_cmd("DWB hard gate disabled")

        eval_msg = self._fresh_dwb_eval()
        if eval_msg is None:
            return self._relay_dwb_raw_cmd("DWB hard gate waiting for fresh LocalPlanEvaluation")

        if self.current_odom is None:
            return self._relay_dwb_raw_cmd("DWB hard gate waiting for odom")

        cmd, selected_waypoints = self.dwb_hard_gate_adapter.select_best_candidate(
            waypoints,
            eval_msg,
            self.current_odom,
            logger=self.get_logger(),
        )
        self.latest_ai_waypoints = np.array(selected_waypoints, copy=True)
        self._publish_path(selected_waypoints)
        self.cmd_pub.publish(cmd)

        wp_idx = self._path_waypoint_index(selected_waypoints)
        wp = selected_waypoints[wp_idx] if wp_idx is not None else [0.0, 0.0]
        self.get_logger().info(
            "DWB hard gate command published: "
            f"wp_idx={wp_idx} wp_local=({wp[0]:.2f},{wp[1]:.2f}) "
            f"v={cmd.linear.x:.3f} w={cmd.angular.z:.3f}",
            throttle_duration_sec=1.0,
        )
        return True

    # ──────────────────────────── Main Loop ────────────────────────────

    def control_loop_callback(self):
        if not self.episode_active:
            self.cmd_pub.publish(Twist())
            return

        if self._subgoal_phase == 'cancelling_bt':
            if self._pending_bt_cancel_future is not None:
                if not self._pending_bt_cancel_future.done():
                    self.cmd_pub.publish(Twist())
                    return
                try:
                    self._pending_bt_cancel_future.result()
                except Exception as exc:
                    self.get_logger().warn(f"BT cancel result error (non-fatal): {exc}")
                self._pending_bt_cancel_future = None

            self._subgoal_phase = 'chasing_waypoint'
            self._follow_path_owner = 'none'
            self.last_path_request_time = None

        if self.task_complete:
            self._stop()
            return

        # --- Kiểm tra odom ---
        odom_age = self._seconds_since(self.last_odom_time)
        if self.current_odom is None:
            if self._maybe_fail_startup_wait("missing odom"):
                return
            self._publish_dwb_fallback("waiting for odom")
            return

        if odom_age is not None and odom_age > self.odom_staleness_sec:
            if self._maybe_fail_startup_wait(f"stale odom (age={odom_age:.2f}s)"):
                return
            self._publish_dwb_fallback(f"waiting for fresh odom (age={odom_age:.2f}s)")
            return

        if self.task_complete:
            self._stop()
            return

        if self.model is None:
            self._publish_dwb_fallback("AI model unavailable")
            return

        # --- Kiểm tra RGB history ---
        with self._image_lock:
            images_snapshot = list(self.image_history)

        if len(images_snapshot) < self.history_length:
            if self._maybe_fail_startup_wait(
                f"missing RGB history ({len(images_snapshot)}/{self.history_length})"
            ):
                return
            self._publish_dwb_fallback(
                f"waiting for RGB history ({len(images_snapshot)}/{self.history_length})"
            )
            return

        image_age = self._seconds_since(self.last_image_time)
        if image_age is not None and image_age > self.image_staleness_sec:
            if self._maybe_fail_startup_wait(f"stale RGB image (age={image_age:.2f}s)"):
                return
            self._publish_dwb_fallback(f"waiting for fresh RGB image (age={image_age:.2f}s)")
            return

        # --- AI Inference ---
        try:
            # 1. Lấy human positions nếu tracking được bật
            human_positions = None
            human_mask = None
            human_valid_count = 0
            if self.enable_human_tracking and self.human_tracker.is_ready():
                human_positions_global = self.human_tracker.get_human_positions()
                human_mask_global = self.human_tracker.get_human_mask()
                human_positions, human_mask, human_valid_count = self._prepare_human_context_for_model(
                    human_positions_global,
                    human_mask_global,
                )

            # 2. Tính ego_hist_xy trong robot local frame
            ego_hist_xy = []
            cx = self.current_odom.pose.pose.position.x
            cy = self.current_odom.pose.pose.position.y
            rot = self.current_odom.pose.pose.orientation
            cyaw = math.atan2(2*(rot.w*rot.z + rot.x*rot.y), 1 - 2*(rot.y**2 + rot.z**2))

            for (px, py, _) in self.odom_history:
                dx = px - cx
                dy = py - cy
                lx =  dx * math.cos(cyaw) + dy * math.sin(cyaw)
                ly = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
                ego_hist_xy.append([lx, ly])

            while len(ego_hist_xy) < self.history_length:
                ego_hist_xy.insert(0, [0.0, 0.0])

            ego_hist_np = np.array(ego_hist_xy, dtype=np.float32)

            # 3. Agent predict waypoints
            pred_context = PredictionContext(
                human_positions=human_positions,
                human_mask=human_mask,
                ego_hist_xy=ego_hist_np,
                cuda_stream=self._cuda_stream,
            )
            candidates, arrival_scores = self.model.predict_candidates(
                images_snapshot,
                self.current_instruction,
                pred_context,
            )
            if human_positions is not None and human_mask is not None:
                self.get_logger().info(
                    "\033[96m[AI] human_context=robot_local "
                    f"valid_now={human_valid_count}/{human_mask.shape[1]} "
                    f"radius={self.human_context_radius:.1f}m\033[0m",
                    throttle_duration_sec=1.0,
                )
            ros_candidates = self.agent.to_ros_candidates(candidates)
            ros_waypoints, best_k, best_cost, best_min_human_dist = self._select_ai_candidate(
                ros_candidates,
                arrival_scores,
                human_positions,
                human_mask,
            )
            debug_wp_idx = 3
            raw_arr = np.asarray(candidates, dtype=np.float32)
            ros_arr = np.asarray(ros_candidates, dtype=np.float32)
            raw_wp = None
            ros_wp = None
            if raw_arr.ndim == 3 and raw_arr.shape[1] > debug_wp_idx:
                raw_wp = raw_arr[min(best_k, raw_arr.shape[0] - 1), debug_wp_idx, :2]
            elif raw_arr.ndim == 2 and raw_arr.shape[0] > debug_wp_idx:
                raw_wp = raw_arr[debug_wp_idx, :2]
            if ros_arr.ndim == 3 and ros_arr.shape[1] > debug_wp_idx:
                ros_wp = ros_arr[min(best_k, ros_arr.shape[0] - 1), debug_wp_idx, :2]
            elif ros_arr.ndim == 2 and ros_arr.shape[0] > debug_wp_idx:
                ros_wp = ros_arr[debug_wp_idx, :2]
            if ros_waypoints is not None and len(ros_waypoints) > debug_wp_idx:
                selected_wp = ros_waypoints[debug_wp_idx, :2]
            else:
                selected_wp = None
            raw_str = (
                f"({float(raw_wp[0]):.3f},{float(raw_wp[1]):.3f})"
                if raw_wp is not None
                else "(n/a,n/a)"
            )
            ros_str = (
                f"({float(ros_wp[0]):.3f},{float(ros_wp[1]):.3f})"
                if ros_wp is not None
                else "(n/a,n/a)"
            )
            selected_str = (
                f"({float(selected_wp[0]):.3f},{float(selected_wp[1]):.3f})"
                if selected_wp is not None
                else "(n/a,n/a)"
            )
            coord_mode = str(self.agent.config.extra_params.get('coordinate_mode', '')).strip() or 'default'
            self.get_logger().info(
                "[WP_DEBUG] stage=after_to_ros wp_idx=3 "
                f"raw={raw_str} ros={ros_str} selected={selected_str} "
                f"frame_xy=(n/a,n/a) robot_yaw={float(cyaw):.3f} "
                f"mode={self.dwb_integration_mode} coord_mode={coord_mode}",
                throttle_duration_sec=0.5,
            )
            arrival_arr = np.asarray(arrival_scores, dtype=np.float32).reshape(-1)
            arrival_score = float(arrival_arr[min(best_k, len(arrival_arr) - 1)]) if len(arrival_arr) else 0.0

            wp_idx = self._path_waypoint_index(ros_waypoints)
            if wp_idx is not None:
                self.get_logger().info(
                    f"Selected AI waypoint[{wp_idx}] Y: {ros_waypoints[wp_idx, 1]:.3f}",
                    throttle_duration_sec=1.0,
                )

            # Inference thành công → reset bộ đếm lỗi
            self._ai_consecutive_failures = 0

            # 5. Publish arrival score và AI path (cho visualization)
            self.arrival_pub.publish(Float32(data=float(arrival_score)))
            self._publish_path(ros_waypoints)

            # 6. Kiểm tra hoàn thành dựa trên goal benchmark thực sự
            dist_to_goal = self._distance_to_goal()
            if dist_to_goal is not None:
                self.get_logger().info(
                    f"\033[92m[AI] dist_to_goal={dist_to_goal:.3f}m ({self.agent.name} inference active)\033[0m",
                    throttle_duration_sec=0.5,
                )
            if (
                self.use_arrival_completion
                and self.controller_reset_enabled
                and dist_to_goal is not None
                and dist_to_goal <= self.goal_completion_radius
            ):
                self._request_episode_reset(
                    success=True,
                    reason=f"goal distance <= goal_completion_radius ({self.goal_completion_radius:.2f}m)",
                    distance_to_goal=dist_to_goal,
                )
                return

            if self._subgoal_phase == 'waypoint_reached':
                if (
                    self.rolling_local_goal_final_radius > 0.0
                    and dist_to_goal is not None
                    and dist_to_goal <= self.rolling_local_goal_final_radius
                ):
                    self._subgoal_phase = 'returning_to_goal'
                    self._switch_to_global_phase("waypoint reached, inside final radius")
                    self._force_infer = False
                    self._relay_dwb_raw_cmd("returning to benchmark goal")
                    return

                if self._lock_phase_local_goal(ros_waypoints, "chaining after SUCCEEDED"):
                    self._subgoal_phase = 'chasing_waypoint'
                    self._force_infer = False
                    self._request_ai_path_update(ros_waypoints)
                else:
                    self._subgoal_phase = 'returning_to_goal'
                    self._switch_to_global_phase("no valid next waypoint after SUCCEEDED")
                    self._force_infer = False
                return

            self._force_infer = False

            # 7. State machine: phase 1 chases the selected AI waypoint,
            #    phase 2 returns to the benchmark/global goal.
            if self.dwb_integration_mode == 'none':
                self._relay_dwb_raw_cmd("AI integration mode is none")
            elif self.dwb_integration_mode in ('shaped_path', 'one_waypoint_replace'):
                self._request_ai_path_update(ros_waypoints)
                self._relay_dwb_raw_cmd(f"following AI {self.dwb_integration_mode} DWB path")
            else:
                self._maybe_update_phase_state(ros_waypoints)
                self._request_ai_path_update(ros_waypoints)
                if self.use_dwb_hard_gate:
                    self._publish_dwb_hard_gate_cmd(ros_waypoints)
                else:
                    self._relay_dwb_raw_cmd("following AI-adapted DWB path")

            self.get_logger().info(
                f"[INFO] AI path adapter: mode={self.dwb_integration_mode} phase={self.path_phase} "
                f"best_k={best_k} wp_idx={wp_idx} arrival={arrival_score:.3f} cost={best_cost:.3f} "
                f"dist_to_benchmark={dist_to_goal:.2f}m" if dist_to_goal is not None else
                f"[INFO] AI path adapter: mode={self.dwb_integration_mode} phase={self.path_phase} "
                f"best_k={best_k} wp_idx={wp_idx} arrival={arrival_score:.3f} cost={best_cost:.3f}",
                throttle_duration_sec=1.0,
            )

            # 8. BEV Visualization
            bev_age = (self.get_clock().now() - self.last_bev_time).nanoseconds / 1e9
            if self.enable_bev_visualization and bev_age >= self.bev_visualization_period_sec:
                self._publish_bev_visualization(ros_waypoints)
                self.last_bev_time = self.get_clock().now()

        except Exception as e:
            self._ai_consecutive_failures += 1
            self.get_logger().error(
                f"AI inference error (consecutive={self._ai_consecutive_failures}): {e}\n"
                f"{traceback.format_exc()}"
            )

            if self._ai_consecutive_failures < self.ai_inference_fail_threshold:
                self.cmd_pub.publish(Twist())
            else:
                # Quá nhiều lỗi liên tiếp → chuyển sang DWB thuần hoặc dừng
                self._publish_dwb_fallback(
                    f"AI inference failed {self._ai_consecutive_failures} consecutive times"
                )

        finally:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ──────────────────────────── DWB Command Utilities ─────────────────────────

    def _copy_twist(self, msg: Twist) -> Twist:
        cmd = Twist()
        cmd.linear.x = float(msg.linear.x)
        cmd.linear.y = float(msg.linear.y)
        cmd.linear.z = float(msg.linear.z)
        cmd.angular.x = float(msg.angular.x)
        cmd.angular.y = float(msg.angular.y)
        cmd.angular.z = float(msg.angular.z)
        cmd.linear.x = max(-self.max_linear_vel, min(self.max_linear_vel, cmd.linear.x))
        cmd.angular.z = max(-self.max_angular_vel, min(self.max_angular_vel, cmd.angular.z))
        return cmd

    # ──────────────────────────── Visualization ────────────────────────────

    def _publish_status_visualization(self, reason: str):
        if not self.enable_bev_visualization:
            return

        bev_age = (self.get_clock().now() - self.last_bev_time).nanoseconds / 1e9
        if bev_age < self.bev_visualization_period_sec:
            return

        try:
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=80)
            ax.axis("off")
            odom_age = self._seconds_since(self.last_odom_time)
            image_age = self._seconds_since(self.last_image_time)
            cmd_age = self._seconds_since(self.last_dwb_cmd_time)
            with self._image_lock:
                image_count = len(self.image_history)
            status = (
                f"odom_age={odom_age:.1f}s" if odom_age is not None else "odom=none"
            )
            status += " | "
            status += (
                f"rgb={image_count}/{self.history_length}, age={image_age:.1f}s"
                if image_age is not None
                else f"rgb={image_count}/{self.history_length}"
            )
            status += " | "
            status += (
                f"dwb_cmd_age={cmd_age:.1f}s"
                if self.latest_dwb_cmd is not None and cmd_age is not None
                else "dwb_cmd=none"
            )

            ax.text(0.5, 0.68, "SocialNav DWB path adapter waiting", ha="center", va="center",
                    fontsize=18, fontweight="bold")
            ax.text(0.5, 0.48, reason, ha="center", va="center", fontsize=12, wrap=True)
            ax.text(0.5, 0.30, status, ha="center", va="center", fontsize=10)

            fig.canvas.draw()
            try:
                img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                img_np = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)

            img_msg = self._rgb_array_to_image_msg(img_np)
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.robot_frame
            self.viz_pub.publish(img_msg)
            self.last_bev_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().warn(f"Status visualization error: {e}", throttle_duration_sec=2.0)

    def _publish_bev_visualization(self, socialnav_waypoints: np.ndarray):
        """Vẽ BEV tối giản: robot, humans, DWB candidates, baseline, and AI waypoints."""
        try:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=80)

            cx = self.current_odom.pose.pose.position.x
            cy = self.current_odom.pose.pose.position.y
            rot = self.current_odom.pose.pose.orientation
            yaw = math.atan2(2 * (rot.w * rot.z + rot.x * rot.y), 1 - 2 * (rot.y**2 + rot.z**2))
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            plot_bounds = [np.asarray([[0.0, 0.0]], dtype=np.float32)]
            inserted_wp = None

            # Vẽ Robot
            ax.plot(0, 0, 'kx', markersize=15, markeredgewidth=2.5, label="Robot", zorder=5)

            # Vẽ hướng robot (mũi tên ngắn)
            ax.annotate("", xy=(0.5, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2.0), zorder=5)

            baseline_local = self._path_to_local_array(
                self.latest_benchmark_global_path,
                max_points=300,
            )
            if baseline_local is None:
                goal_local = self._goal_to_local()
                if goal_local is not None:
                    baseline_local = np.asarray(
                        [[0.0, 0.0], [goal_local[0], goal_local[1]]],
                        dtype=np.float32,
                    )
            if baseline_local is not None and len(baseline_local) > 1:
                plot_bounds.append(baseline_local)
                ax.plot(
                    baseline_local[:, 0],
                    baseline_local[:, 1],
                    color='green',
                    linewidth=2.2,
                    label="Global planner",
                    zorder=3,
                )

            ai_segment = None
            anchored_ai_segment = None
            wp_idx = None
            ai_label = "AI shaped waypoints"
            if self.dwb_integration_mode == 'shaped_path' and self.latest_shaped_ai_waypoints is not None:
                anchored_ai_segment = self._path_to_local_array(self.latest_shaped_ai_waypoint_path)
                if socialnav_waypoints is not None and len(socialnav_waypoints) > 0:
                    wp_idx = self._path_waypoint_index(socialnav_waypoints)
                    ai_segment = np.asarray(socialnav_waypoints, dtype=np.float32)
                    if wp_idx is not None:
                        inserted_wp = np.asarray(socialnav_waypoints[wp_idx], dtype=np.float32)
                if ai_segment is None and anchored_ai_segment is None:
                    ai_segment = np.asarray(self.latest_shaped_ai_waypoints, dtype=np.float32)
                if ai_segment is not None and len(ai_segment) > 0:
                    if inserted_wp is None:
                        inserted_wp = np.asarray(ai_segment[0], dtype=np.float32)
                    if wp_idx is not None:
                        ai_label = f"AI waypoints WP1-WP{len(socialnav_waypoints)}"
                    else:
                        ai_label = f"Current AI shaped proposal x{len(ai_segment)}"
            elif socialnav_waypoints is not None and len(socialnav_waypoints) > 0:
                wp_idx = self._path_waypoint_index(socialnav_waypoints)
                ai_segment = np.asarray(socialnav_waypoints, dtype=np.float32)
                if wp_idx is not None:
                    inserted_wp = np.asarray(socialnav_waypoints[wp_idx], dtype=np.float32)
                    ai_label = f"AI waypoints WP1-WP{len(socialnav_waypoints)}"
                else:
                    inserted_wp = np.asarray(socialnav_waypoints[0], dtype=np.float32)
                    ai_label = f"AI waypoints WP1-WP{len(socialnav_waypoints)}"

            # Vẽ các trajectory ứng viên DWB từ LocalPlanEvaluation (chỉ để quan sát).
            eval_age = self._seconds_since(self.last_eval_time)
            eval_msg = (
                self.latest_eval
                if self.latest_eval is not None
                and eval_age is not None
                and eval_age <= self.max_eval_staleness_sec
                else None
            )
            if eval_msg is not None and eval_msg.twists:
                target_draw_count = 24
                step = max(1, len(eval_msg.twists) // target_draw_count)
                for idx, twist in enumerate(eval_msg.twists):
                    if twist.total < 0.0 or not twist.traj.poses or idx % step != 0:
                        continue
                    traj_local = self._dwb_traj_to_local_array(
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

            # Vẽ trajectory tốt nhất hiện tại của DWB theo FollowPath đang active.
            selected_traj = self._best_dwb_eval_trajectory(eval_msg)
            if selected_traj is not None and len(selected_traj) > 1:
                plot_bounds.append(selected_traj)
                ax.plot(
                    selected_traj[:, 0],
                    selected_traj[:, 1],
                    color='#0066FF',
                    linewidth=3.2,
                    label="DWB best trajectory",
                    zorder=4,
                )

            # no_ai_selected_traj = self._stored_dwb_trajectory_to_local(
            #     self.latest_no_ai_dwb_trajectory
            # )
            # if no_ai_selected_traj is not None and len(no_ai_selected_traj) > 1:
            #     plot_bounds.append(no_ai_selected_traj)
            #     ax.plot(
            #         no_ai_selected_traj[:, 0],
            #         no_ai_selected_traj[:, 1],
            #         color='gold',
            #         linewidth=2.8,
            #         label="DWB selected no-AI",
            #         zorder=4.5,
            #     )

            # Vẽ waypoint đã insert vào FollowPath trước đó và proposal mới nhất.
            # if anchored_ai_segment is not None and len(anchored_ai_segment) > 0:
            #     plot_bounds.append(anchored_ai_segment)
            #     if len(anchored_ai_segment) > 1:
            #         ax.plot(
            #             anchored_ai_segment[:, 0],
            #             anchored_ai_segment[:, 1],
            #             color='#8B0000',
            #             linestyle=':',
            #             linewidth=2.2,
            #             alpha=0.75,
            #             label=f"Inserted shaped WPs x{len(anchored_ai_segment)}",
            #             zorder=5.5,
            #         )
            #     ax.scatter(
            #         anchored_ai_segment[:, 0],
            #         anchored_ai_segment[:, 1],
            #         c='#8B0000',
            #         s=24,
            #         alpha=0.75,
            #         zorder=6,
            #     )

            if ai_segment is not None and inserted_wp is not None and len(ai_segment) > 0:
                plot_bounds.append(ai_segment)
                if len(ai_segment) > 1:
                    ax.plot(
                        ai_segment[:, 0],
                        ai_segment[:, 1],
                        'r--',
                        linewidth=3.0,
                        label=ai_label,
                        zorder=6,
                    )
                ax.scatter(ai_segment[:, 0], ai_segment[:, 1],
                           c='red', s=40, zorder=7)
                ax.scatter([inserted_wp[0]], [inserted_wp[1]], c='orange', s=140, marker='D',
                           edgecolors='black', label="Selected AI WP", zorder=8)

            # Vẽ con người
            if self.enable_human_tracking and self.human_tracker.is_ready():
                h_pos_hist = self.human_tracker.get_human_positions()
                if h_pos_hist is not None and len(h_pos_hist) > 0:
                    current_humans = h_pos_hist[-1]
                    hx_list, hy_list = [], []
                    for h in current_humans:
                        if abs(h[0]) > 0.01 or abs(h[1]) > 0.01:
                            dxh = h[0] - cx
                            dyh = h[1] - cy
                            lx = dxh * cos_yaw + dyh * sin_yaw
                            ly = -dxh * sin_yaw + dyh * cos_yaw
                            hx_list.append(lx)
                            hy_list.append(ly)
                    if hx_list:
                        ax.scatter(hx_list, hy_list, c='blue', marker='o', s=80,
                                   edgecolors='black', label="Humans", zorder=7)

            ax.set_title(
                "AI-DWB BEV",
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

            img_msg = self._rgb_array_to_image_msg(img_np)
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = self.robot_frame
            self.viz_pub.publish(img_msg)

        except Exception as e:
            self.get_logger().error(f"BEV visualization error: {e}", throttle_duration_sec=2.0)
            self.get_logger().error(traceback.format_exc(), throttle_duration_sec=2.0)

    def _publish_path(self, waypoints: np.ndarray):
        path_msg = NavPath()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.robot_frame
        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _stop(self):
        """Dừng hoàn toàn robot và xóa lịch sử."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        with self._image_lock:
            self.image_history.clear()
        self.odom_history.clear()
        self.latest_eval = None
        self.latest_ai_waypoints = None
        self.latest_global_path = None
        self.latest_benchmark_global_path = None
        self.latest_no_ai_dwb_trajectory = None
        self.latest_ai_path = None
        self.latest_shaped_ai_waypoints = None
        self.latest_shaped_ai_waypoint_path = None
        self.path_request_in_progress = False
        self.last_compute_path_goal_handle = None
        self._pending_compute_path_goal_kind = None
        self._ai_consecutive_failures = 0
        self._reset_phase_state()

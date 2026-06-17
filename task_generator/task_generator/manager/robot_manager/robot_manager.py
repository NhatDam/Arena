import asyncio
import os
import subprocess
import time
import typing

import action_msgs.msg
import ament_index_python
import arena_evaluation_msgs.srv as arena_evaluation_srvs
import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import attrs
import geometry_msgs.msg
import launch.launch_description_sources
import launch_ros
import lifecycle_msgs.msg
import nav_msgs.msg as nav_msgs
import rclpy
import rclpy.client
import rclpy.logging
import rclpy.publisher
import rclpy.timer
import sensor_msgs.msg
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from nav2_msgs.srv import ClearCostmapAroundRobot, ClearEntireCostmap

import launch
import task_generator.utils.arena as Utils
from task_generator import NodeInterface
from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.shared import Orientation, Pose, Position, Robot

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

import rclpy.node


class RobotManager(NodeInterface):
    """
    The robot manager manages the goal and start
    position of a robot for all task modes.

    Args:
        namespace (Namespace): The namespace for the robot.
        environment_manager (EnvironmentManager): The environment manager.
        robot (Robot): The robot instance.
    """

    _namespace: Namespace
    _environment_manager: EnvironmentManager
    _start_pos: Pose
    _goal_pos: Pose
    _pose: Pose
    _robot_radius: float
    _goal_tolerance_distance: float
    _goal_tolerance_angle: float
    _robot: Robot
    _move_base_pub: rclpy.publisher.Publisher
    _goal_pub: rclpy.publisher.Publisher
    _pub_goal_timer: rclpy.timer.Timer
    _clear_costmap_around_robot_srv: rclpy.client.Client
    _is_goal_reached: bool
    _rate_setup: rclpy.timer.Rate
    _config: RobotView

    @property
    def robot(self) -> Robot:
        """Get the robot instance.

        Returns:
            Robot: The robot instance.
        """
        return self._robot

    @property
    def start_pos(self) -> Pose:
        """Get the start position.

        Returns:
            Pose: The start position.
        """
        return self._start_pos

    @property
    def goal_pos(self) -> Pose:
        """Get the goal position.

        Returns:
            Pose: The goal position.
        """
        return self._goal_pos

    def __init__(
        self,
        *args,
        namespace: Namespace,
        environment_manager: EnvironmentManager,
        robot: Robot,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rate_setup = self.node.create_rate(.1)

        self._config = robot.model.resolve_sync()

        self._namespace = namespace
        self._environment_manager = environment_manager

        self._start_pos = Pose()
        self._goal_pos = Pose()
        self._is_goal_reached = False
        self._goal_active = False
        self._robot_radius = 0.25

        self._goal_tolerance_distance = self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value
        self._goal_tolerance_angle = self.node.conf.Robot.GOAL_TOLERANCE_ANGLE.value
        self._safety_distance = self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value

        self._robot = self.node._environment_manager.realize(robot)
        self._robot.extra.setdefault('namespace', self.namespace)
        self._pose = self._start_pos
        self._goal_timer = None

        self._publish_goal_task: typing.Optional[asyncio.Task] = None
        self._goal_publishing_suspended = False
        self._launch_tasks: list[asyncio.Task] = []
        self._nav_stack_ready: bool = False
        self._nav_action_client = None
        self._nav_goal_handle = None
        self._nav_goal_result_task: typing.Optional[asyncio.Task] = None
        self._nav_goal_sequence = 0
        self._last_dwb_cmd_log_time = 0.0
        self._last_dwb_odom_log_time = 0.0
        self._last_dwb_joint_log_time = 0.0

    async def _do_launch(self, launch_description: launch.LaunchDescription):
        """Launch and retain task handles so robot sidecars can be cancelled on destroy."""
        task = await self.node._launch_manager.launch_description(launch_description)
        self._launch_tasks.append(task)
        return task

    async def _shutdown_launch_tasks(self):
        """Shutdown launch services owned by this robot manager."""
        if not self._launch_tasks:
            return

        shutdown_task = getattr(self.node._launch_manager, "shutdown_task", None)
        if callable(shutdown_task):
            await asyncio.gather(
                *(shutdown_task(task) for task in self._launch_tasks),
                return_exceptions=True,
            )
        else:
            for task in self._launch_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._launch_tasks, return_exceptions=True)

        self._launch_tasks.clear()

    def _namespace_node_paths(self, node_paths: set[str]) -> list[str]:
        namespace_prefix = str(self.namespace)
        return sorted(
            path for path in node_paths
            if path == namespace_prefix or path.startswith(f"{namespace_prefix}/")
        )

    async def _wait_for_namespace_nodes_gone(
        self,
        node_paths: set[str],
        *,
        timeout: float = 15.0,
    ) -> bool:
        """Wait for previously launched robot ROS nodes to disappear."""
        deadline = asyncio.get_running_loop().time() + timeout
        last_live: list[str] = []

        while True:
            live = self._namespace_node_paths(node_paths)
            if not live:
                return True

            last_live = live
            if asyncio.get_running_loop().time() >= deadline:
                self._logger.warn(
                    "Timed out waiting for old navigation nodes to stop: "
                    f"{last_live[:8]}"
                )
                await self._actively_kill_nav_nodes()
                cleanup_deadline = asyncio.get_running_loop().time() + 10.0
                while asyncio.get_running_loop().time() < cleanup_deadline:
                    live = self._namespace_node_paths(node_paths)
                    if not live:
                        return True
                    last_live = live
                    await asyncio.sleep(0.5)

                self._logger.warn(
                    "Old navigation nodes still visible after cleanup; "
                    f"continuing after targeted process termination: {last_live[:8]}"
                )
                for path in last_live:
                    node_paths.discard(path)
                return True

            await asyncio.sleep(0.25)

    async def _actively_kill_nav_nodes(self):
        """Actively terminate Nav2 nodes in the namespace to force cleanup."""
        namespace = str(self.namespace)
        namespace_token = f"__ns:={namespace}"
        compact_namespace = namespace.strip("/").replace("/", "_")
        # patterns = [
        #     namespace_token,
        #     rf"(controller_server|planner_server|bt_navigator|behavior_server|waypoint_follower|velocity_smoother|collision_monitor|smoother_server).*{namespace}",
        #     rf"(socialnav|urbannav|citywalker|human_states_bridge).*{compact_namespace}",
        # ]
        patterns = [
            namespace_token,
            rf"(controller_server|planner_server|bt_navigator|behavior_server|waypoint_follower|velocity_smoother|collision_monitor|smoother_server|map_server|lifecycle_manager).*{namespace}",
            rf"(socialnav|urbannav|citywalker|human_states_bridge|map_server).*{compact_namespace}",
        ]

        self._logger.warn(f"[Nav Stack] Terminating stale processes for {namespace}")
        for signal_name in ("-TERM", "-KILL"):
            for pattern in patterns:
                try:
                    subprocess.run(
                        ["pkill", signal_name, "-f", pattern],
                        timeout=2,
                        capture_output=True,
                        check=False,
                    )
                except Exception as exc:
                    self._logger.debug(
                        f"[Nav Stack] Could not run pkill {signal_name} for {pattern}: {exc}"
                    )
            await asyncio.sleep(1.0)

    async def _wait_for_navigation_stack_active(
        self,
        *,
        timeout: float = 30.0,
    ) -> bool:
        """Wait until the relaunched Nav2 stack is usable."""
        lifecycle_nodes = [
            self.namespace("controller_server"),
            self.namespace("planner_server"),
            self.namespace("bt_navigator"),
            self.namespace("local_costmap/local_costmap"),
            self.namespace("global_costmap/global_costmap"),
        ]
        pending = {str(node) for node in lifecycle_nodes}
        last_state: dict[str, str] = {}
        deadline = asyncio.get_running_loop().time() + timeout

        while pending:
            for node_name in list(pending):
                try:
                    state = await self.node.get_lifecycle_state_async(node_name, timeout=1.0)
                    last_state[node_name] = f"{state.label or '<unknown>'}({state.id})"
                    if state.id == lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
                        pending.remove(node_name)
                except Exception as exc:
                    last_state[node_name] = type(exc).__name__

            if not pending:
                self._logger.info("Navigation stack is ACTIVE")
                return True

            if asyncio.get_running_loop().time() >= deadline:
                self._logger.warn(
                    "Navigation stack did not become ACTIVE before timeout: "
                    + ", ".join(
                        f"{name}={last_state.get(name, 'unknown')}"
                        for name in sorted(pending)
                    )
                )
                return False

            await asyncio.sleep(0.5)

    async def _odom_base_transform(self):
        """Launch a static transform publisher for odometry to base frame.
        """
        await self._do_launch(
            launch.LaunchDescription([

                # launch_ros.actions.Node(
                #     package="tf2_ros",
                #     executable="static_transform_publisher",
                #     name="odom_to_baseframe_publisher",
                #     arguments=["0", "0", "0", "0", "0", "0", "1", "odom", "base_link"],
                #     parameters=[{'use_sim_time': True}],
                # ),

                launch_ros.actions.Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="map_to_odom_publisher",
                    arguments=["0", "0", "0", "0", "0", "0", "1", "map", "odom"],
                    parameters=[{'use_sim_time': True}],
                )
            ])
        )

    async def set_up_robot(self, node_names: set[str]):
        """Set up the robot by configuring its model and spawning it in the environment.
        """

        self._robot.pose.position.z += self._config.model_params.z_offset
        self._robot = (await self._environment_manager.spawn_robot((self._robot,)))[0]

        _gen_goal_topic = self.namespace("goal_pose")
        goal_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self._goal_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            goal_qos,
        )

        action_topic = f"/{str(self.namespace).strip('/')}/navigate_to_pose"
        self._nav_action_client = ActionClient(
            self.node,
            NavigateToPose,
            action_topic,
        )
        self.node.get_logger().info(
            f"[RobotManager] Nav2 action client ready at: {action_topic}"
        )

        self.node.create_subscription(
            nav_msgs.Odometry,
            self.namespace("odom"),
            self._robot_pos_callback,
            10
        )

        self.node.create_subscription(
            geometry_msgs.msg.Twist,
            self.namespace("cmd_vel"),
            self._cmd_vel_callback,
            10
        )

        self.node.create_subscription(
            sensor_msgs.msg.JointState,
            self.namespace("joint_states"),
            self._joint_states_callback,
            10
        )

        self.node.create_subscription(
            action_msgs.msg.GoalStatusArray,
            self.namespace('navigate_to_pose', '_action', 'status'),
            self._goal_status_callback,
            1
        )

        await self._launch_robot(node_names)
        # Skipped: Isaac Sim's odom OmniGraph already publishes odom → base_link
        # dynamically from the prim world pose.  The static identity transform
        # conflicts with it (TF2 static buffer wins) and breaks the costmap
        # after teleport because it pins base_link at (0,0,0) in odom frame.
        await self._odom_base_transform()

        self._robot_radius = self.node.rosparam[float].get(
            'robot_radius',
            self._robot_radius,
        )

    @property
    def safe_distance(self) -> float:
        """Get the safe distance for the robot.

        Returns:
            float: The safe distance for the robot.
        """
        return self._robot_radius + self._safety_distance

    @property
    def model_name(self) -> str:
        """Get the model name of the robot.

        Returns:
            str: The model name of the robot.
        """
        return self._robot.model.name

    @property
    def name(self) -> str:
        """Get the name of the robot.

        Returns:
            str: The name of the robot.
        """
        return self._robot.name

    @property
    def frame(self) -> Namespace:
        """Get the tf2 frame of the robot.

        Returns:
            Namespace: The tf2 frame of the robot.
        """
        return self._robot.frame

    @property
    def namespace(self) -> Namespace:
        """Get the ROS2 namespace of the robot.

        Returns:
            Namespace: The ROS2 namespace of the robot.
        """
        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return Namespace(
                f"{self._namespace}{self._namespace}_{self.model_name}"
            )

        return self._namespace(self.name)

    @property
    async def is_done(self) -> bool:
        """Check if the robot has reached its goal.

        Returns:
            bool: True if the goal is reached, False otherwise.
        """
        return self._is_goal_reached

    async def move_robot_to_pos(self, pose: Pose):
        """Move the robot to the specified pose.

        Args:
            pose(Pose): The target pose for the robot.
        """
        pose.position.z += self._config.model_params.z_offset
        self.robot.pose = pose
        await self._environment_manager.move_robot((self.robot,))
        await asyncio.sleep(0.5)     
        # await self._clear_local_costmap(-1)
        # await asyncio.sleep(0.2)    

    async def _clear_local_costmap(self, reset_distance: float = -1) -> bool:
        """Clear the local costmap around the robot.

        Args:
            reset_distance(float, optional): The distance to reset the costmap. Defaults to - 1. If reset_distance is -1, the entire costmap will be cleared. If reset_distance is >= 0, only the costmap around the robot will be cleared.

        Returns:
            bool: True if the costmap was cleared successfully, False otherwise.
        """
        node_name = self.node.service_namespace(self.name, 'local_costmap/local_costmap')

        if reset_distance < 0:
            srv_name = os.path.abspath(node_name('../clear_entirely_local_costmap'))
            srv_type = ClearEntireCostmap
            req = ClearEntireCostmap.Request()
        else:
            srv_name = os.path.abspath(node_name('../clear_around_local_costmap'))
            srv_type = ClearCostmapAroundRobot
            req = ClearCostmapAroundRobot.Request()
            req.reset_distance = reset_distance

        state = await self.node.get_lifecycle_state_async(node_name)
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            return False

        self._logger.info(f"Service name: {srv_name}")
        cli = self.node.create_client_wrapper(
            srv_type,
            srv_name,
        )
        await cli.ensure()

        result = await cli.call_timeout(req)
        if result is None:
            self._logger.error(
                f"service call failed for {srv_name}")
            return False
        self._logger.info(
            f"successfull service call for {srv_name}"
        )
        return True

    async def _clear_global_costmap(self) -> bool:
        """Clear the entire global costmap (obstacle layer data)."""
        node_name = self.node.service_namespace(self.name, 'global_costmap/global_costmap')
        srv_name = os.path.abspath(node_name('../clear_entirely_global_costmap'))

        state = await self.node.get_lifecycle_state_async(node_name)
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            return False

        self._logger.info(f"Clearing global costmap: {srv_name}")
        cli = self.node.create_client_wrapper(
            ClearEntireCostmap,
            srv_name,
        )
        await cli.ensure()

        result = await cli.call_timeout(ClearEntireCostmap.Request())
        if result is None:
            self._logger.error(f"service call failed for {srv_name}")
            return False
        self._logger.info(f"successfull service call for {srv_name}")
        return True

    async def clear_costmaps(self) -> None:
        """Clear both local and global costmaps entirely.

        Should be called after the simulator has unpaused so that TF is
        up-to-date and the costmap rolling window re-centres correctly.
        """
        await asyncio.gather(
            self._clear_local_costmap(-1),
            self._clear_global_costmap(),
        )

    def suspend_goal_publishing(self):
        """Cancel the goal publishing task to prevent premature navigation.

        Call this immediately after _task.reset() so that the BT navigator
        does not pick up a goal while the local costmap still contains
        stale observations from before the teleport.
        """
        if self._publish_goal_task is not None:
            self._publish_goal_task.cancel()
            self._publish_goal_task = None
        self._cancel_nav_goal_result_task()
        self._goal_publishing_suspended = True

    def resume_goal_publishing(self):
        """Restart the goal publishing loop.

        Call this after the costmap lifecycle cycle is complete so Nav2
        starts with a clean observation buffer.
        """
        self._goal_publishing_suspended = False
        if self._goal_pos is not None and not self._is_goal_reached:
            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()
            self._publish_goal_task = asyncio.create_task(
                self._publish_goal_loop()
            )

    async def cycle_local_costmap(self) -> bool:
        """Deactivate and reactivate the local costmap lifecycle node.

        This fully resets the observation buffer, purging any stale
        observations from a previous robot position.  Call this after the
        robot has been teleported and TF has settled at the new pose.

        Returns:
            bool: True if the cycle completed successfully.
        """
        node_name = self.node.service_namespace(
            self.name, 'local_costmap/local_costmap'
        )

        try:
            state = await self.node.get_lifecycle_state_async(node_name)
            if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
                self._logger.warn(
                    f"local costmap not active (state={state.id}), skipping cycle"
                )
                return False

            self._logger.warn(f"Deactivating local costmap: {node_name}")
            ok = await self.node.change_lifecycle_state_async(
                node_name,
                lifecycle_msgs.msg.Transition.TRANSITION_DEACTIVATE,
            )
            if not ok:
                self._logger.error("Failed to deactivate local costmap")
                return False

            # Brief pause to let deactivation complete
            await asyncio.sleep(0.3)

            self._logger.warn(f"Reactivating local costmap: {node_name}")
            ok = await self.node.change_lifecycle_state_async(
                node_name,
                lifecycle_msgs.msg.Transition.TRANSITION_ACTIVATE,
            )
            if not ok:
                self._logger.error("Failed to reactivate local costmap")
                return False

            self._logger.warn("Local costmap lifecycle cycle complete")
            return True
        except Exception as e:
            self._logger.error(
                f"Local costmap lifecycle cycle failed: {type(e).__name__}: {e}"
            )
            return False

    async def reset(
        self,
        start_pos: typing.Optional[Pose],
        goal_pos: typing.Optional[Pose],
    ) -> tuple[Pose, Pose]:
        """Reset the robot's position and / or goal.

        Args:
            start_pos(typing.Optional[Pose]): The new starting position of the robot.
            goal_pos(typing.Optional[Pose]): The new goal position of the robot.

        Returns:
            tuple[Pose, Pose]: The new starting and goal positions of the robot.
        """
        if start_pos is not None:
            self._start_pos = self._environment_manager.realize(start_pos)
            await self.move_robot_to_pos(start_pos)

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("start"),
                    [self.start_pos.position.x, self.start_pos.position.y, self.start_pos.orientation.to_yaw()]
                )
        if goal_pos is not None:
            self._goal_pos = self._environment_manager.realize(goal_pos)
            self._is_goal_reached = False  # reset so the new goal is actually navigated
            self._goal_active = False
            self._nav_goal_handle = None
            self._cancel_nav_goal_result_task()
            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()
            self._publish_goal_task = None
            if not self._goal_publishing_suspended:
                self._publish_goal_task = asyncio.create_task(self._publish_goal_loop())

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("goal"),
                    [self.goal_pos.position.x, self.goal_pos.position.y, self.goal_pos.orientation.to_yaw()]
                )
        return self._pose, self._goal_pos

    def _cancel_nav_goal_result_task(self):
        if self._nav_goal_result_task is not None:
            self._nav_goal_result_task.cancel()
            self._nav_goal_result_task = None

    @staticmethod
    def _goal_status_name(status: int) -> str:
        status_names = {
            action_msgs.msg.GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            action_msgs.msg.GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            action_msgs.msg.GoalStatus.STATUS_EXECUTING: "EXECUTING",
            action_msgs.msg.GoalStatus.STATUS_CANCELING: "CANCELING",
            action_msgs.msg.GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            action_msgs.msg.GoalStatus.STATUS_CANCELED: "CANCELED",
            action_msgs.msg.GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return status_names.get(status, f"STATUS_{status}")

    async def _wait_for_nav_action_server(
        self,
        action_topic: str,
        *,
        timeout: float,
    ) -> bool:
        if self._nav_action_client is None:
            self._logger.error(
                f"[Goal Publishing] Nav2 action client is not initialized for {action_topic}"
            )
            return False

        deadline = asyncio.get_running_loop().time() + timeout
        while not self._nav_action_client.server_is_ready():
            if asyncio.get_running_loop().time() >= deadline:
                self._logger.error(
                    f"[Goal Publishing] Nav2 action server {action_topic} "
                    f"not ready after {timeout:.1f}s"
                )
                return False
            await asyncio.sleep(0.25)

        return True

    async def _monitor_nav_goal_result(
        self,
        goal_handle,
        goal_sequence: int,
        action_topic: str,
    ):
        try:
            result_response = await self.node.await_ros(goal_handle.get_result_async())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if goal_sequence == self._nav_goal_sequence:
                self._logger.error(
                    f"[Goal Publishing] Nav2 goal result failed on {action_topic}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._goal_active = False
            return

        if goal_sequence != self._nav_goal_sequence:
            self._logger.debug(
                f"[Goal Publishing] Ignoring stale Nav2 result for {action_topic}"
            )
            return

        status = result_response.status
        status_name = self._goal_status_name(status)
        self._goal_active = False
        self._nav_goal_result_task = None

        if status == action_msgs.msg.GoalStatus.STATUS_SUCCEEDED:
            self._is_goal_reached = True
            self._logger.info(
                f"[Goal Publishing] Nav2 goal succeeded on {action_topic}"
            )
            return

        self._is_goal_reached = False
        self._logger.error(
            f"[Goal Publishing] Nav2 goal finished without success on "
            f"{action_topic}: {status_name}({status})"
        )

    async def _send_goal_to_nav2_action(
        self,
        target_pose: geometry_msgs.msg.PoseStamped,
    ) -> bool:
        action_topic = f"/{str(self.namespace).strip('/')}/navigate_to_pose"
        server_timeout = float(
            os.environ.get("ARENA_NAV2_ACTION_SERVER_TIMEOUT_SEC", "30")
        )
        if not await self._wait_for_nav_action_server(
            action_topic,
            timeout=server_timeout,
        ):
            return False

        action_client = self._nav_action_client
        if action_client is None:
            self._logger.error(
                f"[Goal Publishing] Nav2 action client disappeared for {action_topic}"
            )
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        goal_msg.behavior_tree = ""

        self._nav_goal_sequence += 1
        goal_sequence = self._nav_goal_sequence
        self._cancel_nav_goal_result_task()

        accept_timeout = float(
            os.environ.get("ARENA_NAV2_GOAL_ACCEPT_TIMEOUT_SEC", "10")
        )
        goal_future = None

        try:
            goal_future = action_client.send_goal_async(goal_msg)
            goal_handle = await asyncio.wait_for(
                self.node.await_ros(goal_future),
                timeout=accept_timeout,
            )
        except asyncio.TimeoutError:
            if goal_future is not None:
                goal_future.cancel()
            self._logger.error(
                f"[Goal Publishing] Nav2 did not acknowledge goal on "
                f"{action_topic} after {accept_timeout:.1f}s"
            )
            return False
        except Exception as exc:
            self._logger.error(
                f"[Goal Publishing] Failed to send Nav2 goal on {action_topic}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

        if goal_handle is None:
            self._logger.error(
                f"[Goal Publishing] Nav2 goal send returned no handle on {action_topic}"
            )
            return False

        if not goal_handle.accepted:
            self._logger.error(
                f"[Goal Publishing] Nav2 rejected goal on {action_topic}"
            )
            return False

        self._nav_goal_handle = goal_handle
        self._goal_active = True
        self._goal_start_time = self.node.get_clock().now()
        self._nav_goal_result_task = asyncio.create_task(
            self._monitor_nav_goal_result(
                goal_handle,
                goal_sequence,
                action_topic,
            )
        )
        self._logger.warn(
            f"[Goal Publishing] Nav2 goal accepted on {action_topic}"
        )
        return True

    async def _publish_goal_loop(self):
        """Publish goal via topic and send the same goal through Nav2 ActionClient."""
        
        self.node.get_logger().warn("🟢 [DEBUG] Waiting 1.5s for Isaac Sim TF & Costmap...")
        await asyncio.sleep(1.5) 
        
        if self._goal_publishing_suspended:
            self._logger.debug("[Goal Publishing] Goal publishing suspended")
            return
        
        if self._is_goal_reached:
            self._logger.debug("[Goal Publishing] Goal already reached")
            return

        if not self._nav_stack_ready:
            timeout = float(os.environ.get("ARENA_NAV2_GOAL_READY_TIMEOUT_SEC", "90"))
            self._logger.warn(
                f"[Goal Publishing] Waiting up to {timeout:.0f}s for Nav2 stack "
                "after Isaac TF/clock warmup..."
            )
            self._nav_stack_ready = await self._wait_for_navigation_stack_active(
                timeout=timeout,
            )
            if not self._nav_stack_ready:
                self._logger.error(
                    "[Goal Publishing] Navigation stack NOT ready after deferred wait. "
                    "Skipping this goal send so benchmark can reincarnate cleanly."
                )
                return

        goal = self._goal_pos
        self.node.get_logger().warn(f"🚀 [ROBOT MANAGER] Target Goal: x={goal.position.x}, y={goal.position.y}")

        if self._goal_timer is not None:
            self._goal_timer.cancel()
            self._goal_timer.destroy()

        # Publish the benchmark goal in the global map frame so Nav2, metrics,
        # and downstream benchmark tooling agree on the same target.
        target_pose = geometry_msgs.msg.PoseStamped()
        target_pose.header.frame_id = "map"
        target_pose.header.stamp.sec = 0
        target_pose.header.stamp.nanosec = 0
        target_pose.pose = goal.to_msg()
        self._goal_pub.publish(target_pose)

        goal_sent = await self._send_goal_to_nav2_action(target_pose)
        if not goal_sent:
            self._goal_active = False
            self._logger.error(
                "[Goal Publishing] Nav2 goal was not sent/accepted; "
                "robot will not receive DWB commands for this goal."
            )
        

    async def _launch_robot(self, node_paths: set[str]):
        """Launch the robot external nodes.
        """
        self._logger.info(f"LAUNCH ROBOT {self.name}")

        if Utils.get_arena_type() != Constants.ArenaType.TRAINING:
            launch_description = launch.LaunchDescription()
            current_log_level = rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()
            launch_description.add_action(NodeLogLevelExtension.SetGlobalLogLevelAction(current_log_level))  # type: ignore

            launch_arguments = {
                'robot': self.model_name,
                # 'simulator': self.node.conf.Arena.SIM.value.value,
                # 'name': self.name,
                'task_generator_node': os.path.join(self.node.get_namespace(), self.node.get_name()),
                'namespace': self.namespace,
                # 'use_namespace': 'True',
                'frame': self._robot.frame(''),  # trailing slash
                'inter_planner': self._robot.inter_planner,
                'global_planner': self._robot.global_planner,
                'local_planner': self._robot.local_planner,
                # 'complexity': self.node.declare_parameter('complexity', 1).value,
                'train_mode': str(self.node._train_mode).lower(),
                'agent_name': self._robot.agent,
                'map_file': self.node.conf.Arena.WORLD.value,
                'scenario_file': self.node.get_parameter("task.scenario.file").value if self.node.has_parameter("task.scenario.file") else '',
                'use_sim_time': 'True',
                'amcl': 'true' if self.node.conf.Arena.SIM.value in (Constants.SimSimulator.GAZEBO,) else 'false',
                'goal_tolerance_radius': str(self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value),
            }

            if self._robot.record_data_dir:
                launch_arguments.update({
                    'record_data_dir': self._robot.record_data_dir,
                })

            workspace_dir = os.environ.get("WORKSPACE_DIR", os.path.expanduser("~/arena5_ws"))
            source_robot_launch = os.path.join(
                workspace_dir,
                "src",
                "Arena",
                "arena_simulation_setup",
                "launch",
                "robot.launch.py",
            )
            robot_launch_path = (
                source_robot_launch
                if os.path.exists(source_robot_launch)
                else os.path.join(
                    ament_index_python.packages.get_package_share_directory('arena_simulation_setup'),
                    'launch/robot.launch.py'
                )
            )

            launch_description.add_action(
                launch.actions.IncludeLaunchDescription(
                    launch.launch_description_sources.PythonLaunchDescriptionSource(
                        robot_launch_path
                    ),
                    launch_arguments=launch_arguments.items(),
                )
            )
            await self._do_launch(launch_description)

            bt_node_path = str(self.namespace('bt_navigator'))
            self._logger.info(f'waiting for {bt_node_path}')
            while bt_node_path not in node_paths:
                await asyncio.sleep(0.01)

            self._nav_stack_ready = await self._wait_for_navigation_stack_active(
                timeout=float(os.environ.get("ARENA_NAV2_INITIAL_READY_TIMEOUT_SEC", "90")),
            )
            if self._nav_stack_ready:
                self._logger.info("[Nav Stack] Navigation stack is ready for goal publishing")
            else:
                self._logger.warn(
                    "[Nav Stack] Not ACTIVE during initial launch probe; "
                    "goal publishing will retry after Isaac TF/clock warmup."
                )

    def _robot_pos_callback(self, data: nav_msgs.Odometry):
        """Callback for robot position updates.

        Args:
            data(nav_msgs.Odometry): The odometry data containing the robot's position.
        """
        current_position = data.pose.pose
        quat = current_position.orientation

        self._pose = Pose(
            Position(
                current_position.position.x,
                current_position.position.y,
            ),
            Orientation.from_msg(quat)
        )
        self._log_dwb_velocity(
            "odom",
            data.twist.twist.linear.x,
            data.twist.twist.angular.z,
            "_last_dwb_odom_log_time",
        )

    def _should_log_dwb_baseline_velocity(self) -> bool:
        local_planner = str(getattr(self._robot, "local_planner", "") or "").lower()
        agent = str(getattr(self._robot, "agent", "") or "").strip()
        return local_planner == "dwb" and not agent and self._goal_active

    def _log_dwb_velocity(
        self,
        source: str,
        linear_velocity: float,
        angular_velocity: float,
        last_log_attr: str,
    ):
        if not self._should_log_dwb_baseline_velocity():
            return

        period = float(os.environ.get("ARENA_DWB_VELOCITY_LOG_PERIOD_SEC", "1.0"))
        now = time.monotonic()
        if now - getattr(self, last_log_attr) < period:
            return

        setattr(self, last_log_attr, now)
        self.node.get_logger().warn(
            f"[DWB VELOCITY] {self.name} {source}: "
            f"v={linear_velocity:.3f} m/s, w={angular_velocity:.3f} rad/s"
        )

    def _cmd_vel_callback(self, data: geometry_msgs.msg.Twist):
        self._log_dwb_velocity(
            "cmd_vel",
            data.linear.x,
            data.angular.z,
            "_last_dwb_cmd_log_time",
        )

    def _joint_states_callback(self, data: sensor_msgs.msg.JointState):
        if not self._should_log_dwb_baseline_velocity():
            return

        period = float(os.environ.get("ARENA_DWB_VELOCITY_LOG_PERIOD_SEC", "1.0"))
        now = time.monotonic()
        if now - self._last_dwb_joint_log_time < period:
            return

        self._last_dwb_joint_log_time = now
        velocities = dict(zip(data.name, data.velocity))
        wheel_names = ("left_wheel_joint", "right_wheel_joint")
        wheel_velocities = {name: velocities.get(name) for name in wheel_names}

        if all(value is not None for value in wheel_velocities.values()):
            self.node.get_logger().warn(
                f"[DWB VELOCITY] {self.name} joint_states: "
                f"left_wheel_joint={wheel_velocities['left_wheel_joint']:.3f} rad/s, "
                f"right_wheel_joint={wheel_velocities['right_wheel_joint']:.3f} rad/s"
            )
            return

        wheel_like_names = [name for name in data.name if "wheel" in name.lower()]
        self.node.get_logger().warn(
            f"[DWB VELOCITY] {self.name} joint_states: "
            f"missing target wheel joints {wheel_names}; "
            f"wheel-like joints={wheel_like_names}; total_joints={len(data.name)}"
        )

    def _goal_status_callback(self, data: action_msgs.msg.GoalStatusArray):
        """Callback for goal status updates.

        Args:
            data(action_msgs.msg.GoalStatusArray): The goal status data.
        """
        last_goal = next(reversed(list(data.status_list)), None)
        if last_goal is None:
            return
        if not self._goal_active:
            return
        self._is_goal_reached = \
            last_goal.status == action_msgs.msg.GoalStatus.STATUS_SUCCEEDED

    def _data_recorder_parameters(self) -> list[dict[str, typing.Any]]:
        scenario_file = ""
        if self.node.has_parameter("task.scenario.file"):
            scenario_file = self.node.get_parameter("task.scenario.file").value

        return [{
            "model": self.model_name,
            "local_planner": self._robot.local_planner,
            "inter_planner": self._robot.inter_planner,
            "agent_name": self._robot.agent,
            "map_file": self.node.conf.Arena.WORLD.value,
            "scenario_file": scenario_file,
        }]

    async def _launch_data_recorder(self, record_data_dir: str):
        self._logger.info(
            f"Launching data recorder for {self.name} in {record_data_dir}"
        )
        await self._do_launch(
            launch.LaunchDescription([
                launch_ros.actions.Node(
                    package='arena_evaluation',
                    executable='record',
                    namespace=str(self.namespace),
                    name='data_recorder',
                    arguments=['--dir', record_data_dir],
                    parameters=self._data_recorder_parameters(),
                )
            ])
        )

    async def _change_data_recorder_directory(self, record_data_dir: str):
        service_name = str(self.namespace("change_directory"))
        client = self.node.create_client(
            arena_evaluation_srvs.ChangeDirectory,
            service_name,
        )

        for _ in range(50):
            if client.service_is_ready():
                break
            await asyncio.sleep(0.1)

        if not client.service_is_ready():
            self._logger.warning(
                f"Data recorder service not available at {service_name}; directory update skipped"
            )
            return

        request = arena_evaluation_srvs.ChangeDirectory.Request()
        request.data = record_data_dir

        response = await self.node.await_ros(client.call_async(request))
        if response is None or not response.result:
            self._logger.warning(
                f"Data recorder at {service_name} rejected directory change to {record_data_dir}"
            )
            return

        self._logger.info(
            f"Data recorder for {self.name} switched to {record_data_dir}"
        )

    async def update(
        self,
        robot: typing.Optional[Robot] = None,
        *,
        node_paths: typing.Optional[set[str]] = None,
    ):
        """Live-update robot sidecars without respawning the Isaac robot."""
        if robot is None:
            return

        nav_config_changed = any(
            getattr(self._robot, attr) != getattr(robot, attr)
            for attr in (
                "inter_planner",
                "local_planner",
                "global_planner",
                "agent",
            )
        )
        old_record_data_dir = self._robot.record_data_dir
        new_record_data_dir = robot.record_data_dir

        self._robot = attrs.evolve(
            self._robot,
            inter_planner=robot.inter_planner,
            local_planner=robot.local_planner,
            global_planner=robot.global_planner,
            agent=robot.agent,
            record_data_dir=new_record_data_dir,
            extra=robot.extra,
        )
        self._robot.extra.setdefault('namespace', self.namespace)

        if nav_config_changed:
            self._logger.info(
                f"Relaunching navigation stack for {self.name} "
                f"(local_planner={self._robot.local_planner}, agent={self._robot.agent})"
            )
            self.suspend_goal_publishing()
            # Reset nav stack readiness flag during reincarnation
            self._nav_stack_ready = False

            await self._shutdown_launch_tasks()

            if node_paths is None:
                # Fallback for direct callers: launch without a live node-path
                # watcher instead of blocking forever. RobotsManagerROS passes
                # a watched set during benchmark reconfiguration.
                await asyncio.sleep(2.0)
                node_paths = {str(self.namespace('bt_navigator'))}
            else:
                await self._wait_for_namespace_nodes_gone(node_paths)
                for path in self._namespace_node_paths(node_paths):
                    node_paths.discard(path)

            await self._launch_robot(node_paths)
            return

        if old_record_data_dir == new_record_data_dir or not new_record_data_dir:
            return

        if not old_record_data_dir:
            await self._launch_data_recorder(new_record_data_dir)
            return

        await self._change_data_recorder_directory(new_record_data_dir)

    async def destroy(self):
        """Destroy robot and remove from simulation and navigation stack.
        """
        if self._goal_timer is not None:
            self._goal_timer.cancel()
            self._goal_timer.destroy()
        self._cancel_nav_goal_result_task()
        if self._nav_action_client is not None:
            self._nav_action_client.destroy()
            self._nav_action_client = None
        await self._shutdown_launch_tasks()
        await self._environment_manager.remove_robot((self.robot,))

import asyncio
import os
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
from arena_rclpy_mixins.shared import Namespace
from arena_robots.Robot import RobotView
from nav2_msgs.srv import ClearCostmapAroundRobot, ClearEntireCostmap

import launch
import task_generator.utils.arena as Utils
from task_generator import NodeInterface
from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.shared import Orientation, Pose, Position, Robot

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
        self._launch_tasks: list[asyncio.Task] = []

    async def _do_launch(self, launch_description: launch.LaunchDescription):
        """Launch and retain task handles so robot sidecars can be cancelled on destroy."""
        task = await self.node._launch_manager.launch_description(launch_description)
        self._launch_tasks.append(task)
        return task

    async def _odom_base_transform(self):
        """Launch a static transform publisher for odometry to base frame.
        """
        await self._do_launch(
            launch.LaunchDescription([
                launch_ros.actions.Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="odom_to_baseframe_publisher",
                    arguments=[
                        "0", "0", "0",
                        "0", "0", "0", "1",
                        self.frame(self._config.model_params.odom_frame),
                        self.frame(self._config.model_params.base_frame),
                    ],
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

        self._goal_pub = self.node.create_publisher(
            geometry_msgs.msg.PoseStamped,
            _gen_goal_topic,
            10,
        )

        self.node.create_subscription(
            nav_msgs.Odometry,
            self.namespace("odom"),
            self._robot_pos_callback,
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
        #await self._odom_base_transform()

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
        await self._clear_local_costmap(-1)
        await asyncio.sleep(0.2)    

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

    def resume_goal_publishing(self):
        """Restart the goal publishing loop.

        Call this after the costmap lifecycle cycle is complete so Nav2
        starts with a clean observation buffer.
        """
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
            if self._publish_goal_task is not None:
                self._publish_goal_task.cancel()
            self._publish_goal_task = asyncio.create_task(self._publish_goal_loop())

            if self._robot.record_data_dir:
                self.node.rosparam[list[float]].set(
                    self.namespace.robot_ns.ParamNamespace()("goal"),
                    [self.goal_pos.position.x, self.goal_pos.position.y, self.goal_pos.orientation.to_yaw()]
                )
        return self._pose, self._goal_pos

    async def _publish_goal_loop(self):
        """Publish the goal to the robot.
        """
        # only way to circumvent amcl absolutely trolling us is to create this loop

        with self.node.sim_time_rate(60.0, 60.0) as (done, rate):
            while not done.is_set():
                await rate.get()

                if self._is_goal_reached:
                    break

                goal = self._goal_pos
                self._logger.info(f"Publishing goal: x={goal.position.x}, y={goal.position.y}, orientation={goal.orientation.to_yaw()}")

                self._goal_pos = goal

                if self._goal_timer is not None:
                    self._goal_timer.cancel()
                    self._goal_timer.destroy()

                goal_msg = geometry_msgs.msg.PoseStamped()
                goal_msg.header.frame_id = "map"
                goal_msg.header.stamp = self.node.sim_time.to_msg()
                goal_msg.pose = goal.to_msg()
                self._goal_pub.publish(goal_msg)
                self._goal_active = True
                self._goal_start_time = self.node.sim_time

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
            }

            if self._robot.record_data_dir:
                launch_arguments.update({
                    'record_data_dir': self._robot.record_data_dir,
                })

            launch_description.add_action(
                launch.actions.IncludeLaunchDescription(
                    launch.launch_description_sources.PythonLaunchDescriptionSource(
                        os.path.join(
                            ament_index_python.packages.get_package_share_directory('arena_simulation_setup'),
                            'launch/robot.launch.py'
                        )
                    ),
                    launch_arguments=launch_arguments.items(),
                )
            )
            await self._do_launch(launch_description)

            bt_node_path = str(self.namespace('bt_navigator'))
            self._logger.info(f'waiting for {bt_node_path}')
            while bt_node_path not in node_paths:
                await asyncio.sleep(0.01)

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

        future = client.call_async(request)
        await future

        response = future.result()
        if response is None or not response.result:
            self._logger.warning(
                f"Data recorder at {service_name} rejected directory change to {record_data_dir}"
            )
            return

        self._logger.info(
            f"Data recorder for {self.name} switched to {record_data_dir}"
        )

    async def update(self, robot: typing.Optional[Robot] = None):
        """Live-update robot sidecars that can change without a full relaunch."""
        if robot is None:
            return

        old_record_data_dir = self._robot.record_data_dir
        new_record_data_dir = robot.record_data_dir

        self._robot = attrs.evolve(
            self._robot,
            record_data_dir=new_record_data_dir,
            extra=robot.extra,
        )

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
        for task in self._launch_tasks:
            if not task.done():
                task.cancel()
        if self._launch_tasks:
            await asyncio.gather(*self._launch_tasks, return_exceptions=True)
            self._launch_tasks.clear()
        await self._environment_manager.remove_robot((self.robot,))

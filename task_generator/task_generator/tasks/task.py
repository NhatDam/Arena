import asyncio
import typing
from collections.abc import Sequence

import rclpy
import rclpy.publisher
import rosgraph_msgs.msg as rosgraph_msgs
import std_msgs.msg as std_msgs
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import DefaultParameter, Namespace

from task_generator import NodeInterface
from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.robot_manager.robots_manager_ros import RobotsManager
from task_generator.manager.world_manager.world_manager_ros import WorldManager
from task_generator.shared import Pose, rosparam_set

from . import Namespaced, Props_
from .modules import TM_Module
from .obstacles import TM_Obstacles
from .robots import TM_Robots

# import training.srv as training_srvs


class _TaskRegistry(Namespaced):
    registry_obstacles: dict[Constants.TaskMode.TM_Obstacles, typing.Callable[[], type[TM_Obstacles]]] = {}
    registry_robots: dict[Constants.TaskMode.TM_Robots, typing.Callable[[], type[TM_Robots]]] = {}
    registry_module: dict[Constants.TaskMode.TM_Module, typing.Callable[[], type[TM_Module]]] = {}

    _namespace: typing.ClassVar[Namespace] = Namespaced.namespace('task')

    @classmethod
    def register_obstacles(cls, name: Constants.TaskMode.TM_Obstacles):
        def inner_wrapper(
                loader: typing.Callable[[], type[TM_Obstacles]]):
            assert (
                name not in cls.registry_obstacles
            ), f"TaskMode '{name}' for obstacles already exists!"

            def namespaced_loader():
                class Inner(loader()):
                    _namespace = cls._namespace(name.value)
                return Inner

            cls.registry_obstacles[name] = namespaced_loader
            return loader

        return inner_wrapper

    @classmethod
    def register_robots(cls, name: Constants.TaskMode.TM_Robots):
        def inner_wrapper(loader: typing.Callable[[], type[TM_Robots]]):
            assert (
                name not in cls.registry_obstacles
            ), f"TaskMode '{name}' for robots already exists!"

            def namespaced_loader():
                class Inner(loader()):
                    _namespace = cls._namespace(name.value)
                return Inner

            cls.registry_robots[name] = namespaced_loader
            return loader

        return inner_wrapper

    @classmethod
    def register_module(cls, name: Constants.TaskMode.TM_Module):
        def inner_wrapper(loader: typing.Callable[[], type[TM_Module]]):
            assert (
                name not in cls.registry_obstacles
            ), f"TaskMode '{name}' for module already exists!"

            def namespaced_loader():
                class Inner(loader()):
                    _namespace = cls._namespace(name.value)
                return Inner

            cls.registry_module[name] = namespaced_loader
            return loader

        return inner_wrapper


class Task(_TaskRegistry, NodeInterface, Props_):
    """Task class that comibnes task modes.
    """
    last_reset_time: int

    TOPIC_RESET_START = "reset_start"
    TOPIC_RESET_END = "reset_end"
    PARAM_RESETTING = "resetting"

    @classmethod
    def declare_parameters(cls, node: ROSParamServer):
        node.ROSParam[bool](cls.PARAM_RESETTING, True)

    __reset_start: rclpy.publisher.Publisher
    __reset_end: rclpy.publisher.Publisher
    __reset_mutex: bool

    PARAM_TM_ROBOTS = "tm_robots"
    PARAM_TM_OBSTACLES = "tm_obstacles"

    __param_tm_robots: Constants.TaskMode.TM_Robots
    __param_tm_obstacles: Constants.TaskMode.TM_Obstacles

    __tm_robots: TM_Robots
    __tm_obstacles: TM_Obstacles

    _force_reset: bool

    @classmethod
    async def create(
        cls,
        *,
        environment_manager: EnvironmentManager,
        robots_manager: RobotsManager,
        world_manager: WorldManager,
        modules: Sequence[Constants.TaskMode.TM_Module] = (),
        **kwargs,
    ):
        self = cls(
            environment_manager=environment_manager,
            robots_manager=robots_manager,
            world_manager=world_manager,
            modules=modules,
            **kwargs,
        )
        await self.robots_manager.set_up()
        return self

    def __init__(
        self,
        *args,
        environment_manager: EnvironmentManager,
        robots_manager: RobotsManager,
        world_manager: WorldManager,
        modules: Sequence[Constants.TaskMode.TM_Module] = (),
        **kwargs,
    ):
        """
        Initializes a CombinedTask object.

        Args:
            environment_manager (ObstacleManager): The obstacle manager for the task.
            robots_manager (RobotsManager): The dict of robot managers for the task.
            world_manager (WorldManager): The world manager for the task.
            namespace (str, optional): The namespace for the task. Defaults to "".
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self._force_reset = False
        self._is_resetting: bool = False

        self.environment_manager = environment_manager
        self.robots_manager = robots_manager
        self.world_manager = world_manager

        self._train_mode = self.node.get_parameter_or("/train_mode", DefaultParameter(False)).value

        self.__reset_start = self.node.create_publisher(std_msgs.Empty, 'reset_start', 1)
        self.__reset_end = self.node.create_publisher(std_msgs.Empty, 'reset_end', 1)
        self.__reset_mutex = False

        self.node.create_subscription(rosgraph_msgs.Clock, '/clock', self._clock_callback, 10)
        self.last_reset_time = 0
        self.clock = rosgraph_msgs.Clock()

        self.__param_tm_obstacles = None  # type: ignore
        self.__param_tm_robots = None  # type: ignore
        self._logger.info('initing modules')
        self.__modules = [
            self.registry_module[module]()(node=self.node, task=self) for module in modules
        ]

        if self._train_mode:
            self.set_tm_robots(Constants.TaskMode.TM_Robots(self.node.conf.TaskMode.TM_ROBOTS.value))
            self.set_tm_obstacles(Constants.TaskMode.TM_Obstacles(self.node.conf.TaskMode.TM_OBSTACLES.value))

    def set_tm_robots(self, tm_robots: Constants.TaskMode.TM_Robots):
        """
        Sets the task mode for robots.

        Args:
            tm_robots (Constants.TaskMode.TM_Robots): The task mode for robots.
        """
        assert tm_robots in self.registry_robots, f"TaskMode '{tm_robots}' for robots is not registered!"
        self.__tm_robots = self.registry_robots[tm_robots]()(node=self.node, props=self)
        self.__param_tm_robots = tm_robots

    def set_tm_obstacles(
            self, tm_obstacles: Constants.TaskMode.TM_Obstacles):
        """
        Sets the task mode for obstacles.

        Args:
            tm_obstacles (Constants.TaskMode.TM_Obstacles): The task mode for obstacles.
        """
        assert tm_obstacles in self.registry_obstacles, f"TaskMode '{tm_obstacles}' for obstacles is not registered!"
        self.__tm_obstacles = self.registry_obstacles[tm_obstacles]()(node=self.node, props=self)
        self.__param_tm_obstacles = tm_obstacles

    async def clear_all_costmaps(self):
        """Clear both local and global costmaps for every managed robot.

        Should be called after the simulator has unpaused so that TF/odom are
        current and the costmap rolling windows re-centre on the new position.
        """
        await asyncio.gather(*(
            rm.clear_costmaps()
            for rm in self.robots_manager.managers.values()
        ))

    def suspend_all_goal_publishing(self):
        """Cancel goal publishing for all robots.

        Call right after _task.reset() so that Nav2 does not start navigating
        before the costmap lifecycle cycle has purged stale observations.
        """
        for rm in self.robots_manager.managers.values():
            rm.suspend_goal_publishing()

    def resume_all_goal_publishing(self):
        """Restart goal publishing for all robots.

        Call after costmap lifecycle cycling is complete.
        """
        for rm in self.robots_manager.managers.values():
            rm.resume_goal_publishing()

    async def cycle_all_local_costmaps(self):
        """Deactivate and reactivate local costmaps for every managed robot.

        This fully resets observation buffers, purging stale observations
        from a previous robot position after a teleport.
        """
        await asyncio.gather(*(
            rm.cycle_local_costmap()
            for rm in self.robots_manager.managers.values()
        ))

    async def _reset_task(self, **kwargs):
        """
        Reset the task by updating task modes, resetting modules, and spawning obstacles.

        Args:
            **kwargs: Additional keyword arguments for resetting the task.

        Returns:
            None
        """
        try:
            self._is_resetting = True 
            self.__reset_start.publish(std_msgs.Empty())

            await self.robots_manager.set_up()

            if not self._train_mode:
                if (
                    new_tm_robots := self.node.conf.TaskMode.TM_ROBOTS.value
                ) != self.__param_tm_robots:
                    self.set_tm_robots(new_tm_robots)

                if (
                    new_tm_obstacles := self.node.conf.TaskMode.TM_OBSTACLES.value
                ) != self.__param_tm_obstacles:
                    self.set_tm_obstacles(new_tm_obstacles)

            for module in self.__modules:
                module.before_reset()

            # 1. Suspend nav goals so robot doesn't navigate with stale costmap
            self.suspend_all_goal_publishing()

            await self.__tm_robots.reset(**kwargs)
            obstacles, dynamic_obstacles = await self.__tm_obstacles.reset(**kwargs)

            async def respawn():
                await asyncio.gather(
                    self.environment_manager.spawn_dynamic_obstacles(dynamic_obstacles),
                    self.environment_manager.spawn_obstacles(obstacles),
                )

            await self.environment_manager.respawn(respawn)

            # 2. Wait briefly for Isaac Sim physics + TF to settle after teleport
            await asyncio.sleep(1.0)

            # 3. Clear any remaining stale costmap data
            await self.clear_all_costmaps()

            for module in self.__modules:
                module.after_reset()

            self.last_reset_time = self.clock.clock.sec
            # 4. Resume goal publishing now that costmaps are clean
            self.resume_all_goal_publishing()
        except Exception as e:
            self.node.get_logger().error(repr(e))
            raise

        finally:
            self._is_resetting = False            
            self.__reset_end.publish(std_msgs.Empty())

    def _mutex_reset_task(self, **kwargs):
        """
        Executes a reset task while ensuring mutual exclusion.

        This function acquires a mutex lock to ensure that only one reset task is executed at a time.
        It sets a parameter to indicate that the system is resetting, publishes a reset start message,
        performs the reset task, and then publishes a reset end message. If any exception occurs during
        the reset task, it logs the error, shuts down the ROS node, and raises an exception.

        Args:
            kwargs: Additional keyword arguments.

        Raises:
            Exception: If an error occurs during the reset task.

        """
        # TODO
        raise NotImplementedError("This method is deprecated. Use _reset_task instead.")
        while self.__reset_mutex:
            rclpy.sleep(0.001)
        self.__reset_mutex = True

        try:
            rosparam_set(self.PARAM_RESETTING, True)
            self._reset_task()

        except Exception as e:
            raise e

        finally:
            rosparam_set(self.PARAM_RESETTING, False)
            self.__reset_mutex = False

    async def reset(self, **kwargs):
        """
        Resets the task.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self._force_reset = False
        await self._reset_task(**kwargs)

    @property
    async def is_done(self) -> bool:
        """
        Checks if the task is done.

        Returns:
            bool: True if the task is done, False otherwise.
        """
        if self._is_resetting:
            return False
        return self._force_reset or await self.__tm_robots.done

    async def set_robot_position(self, pose: Pose):
        """
        Sets the position of the robot.

        Args:
            position (Pose): The position and orientation of the robot.
        """
        await self.__tm_robots.set_position(pose)

    async def set_robot_goal(self, pose: Pose):
        """
        Sets the goal position for the robot.

        Args:
            position (Pose): The goal position for the robot.
        """
        await self.__tm_robots.set_goal(pose)

    def force_reset(self):
        self._force_reset = True

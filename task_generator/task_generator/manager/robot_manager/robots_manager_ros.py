import abc
import asyncio
import contextlib
import os
import re
import typing

import arena_robots.SetupFile as robot_setup
import attrs
import rclpy
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_rclpy_mixins.shared import Namespace

from task_generator import NodeInterface
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.shared import Pose, Position, Robot

from .robot_manager import RobotManager


def _initialpose_generator(x: float, y: float, d: float):
    """Generate initial poses for the robot.

    Args:
        x (float): The initial x position.
        y (float): The initial y position.
        d (float): The distance between poses.

    Yields:
        Pose: The initial pose for the robot.
    """
    while True:
        yield Pose(Position(x=x, y=y))
        y += d


@attrs.frozen
class _RobotDiff:
    """Change to robot configurations to execute.
    """
    to_remove: list[str] = attrs.field(factory=list)
    to_add: dict[str, Robot] = attrs.field(factory=dict)
    to_update: dict[str, Robot] = attrs.field(factory=dict)


class RobotsManager(abc.ABC):
    """Abstract base class for managing multiple robots.
    """

    @property
    def managers(self) -> dict[str, RobotManager]:
        """Get the robot managers.

        Returns:
            dict[str, RobotManager]: The robot managers.
        """
        return self._managers

    @abc.abstractmethod
    async def set_up(self):
        """Set up the robot managers.
        """

    def __init__(self, *args, environment_manager: EnvironmentManager, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._environment_manager: EnvironmentManager = environment_manager
        self._managers: dict[str, RobotManager] = {}


class RobotsManagerROS(NodeInterface, RobotsManager):
    """ROS interface for dynamically loading multiple robots.

    Args:
        environment_manager (EnvironmentManager): The environment manager.
    """
    _initialpose: typing.Generator
    _robot_configurations: ROSParamT[_RobotDiff]
    _diff: _RobotDiff

    @contextlib.contextmanager
    def provide_node_paths(self, paths: set[str]):
        """Context manager to provide node paths.

        Args:
            paths (set[str]): The set to populate with node paths.

        Yields:
            asyncio.Task: The task that populates the node paths.
        """
        t: asyncio.Task | None = None
        try:
            async def task():
                while True:
                    latest = self.node.get_node_names_and_namespaces()
                    paths.update(os.path.join(ns, name) for name, ns in latest)
                    await asyncio.sleep(1.0)
            t = asyncio.create_task(task())
            yield t
        except asyncio.CancelledError:
            pass
        except Exception:
            self._logger.error('Error while providing node paths {e}\n{traceback.format_exc()}')
        finally:
            if t and not t.done():
                t.cancel()

    def _parse_robot_configurations(
        self,
        v: typing.Any
    ) -> _RobotDiff:
        """Parse robot configurations from the given value.

        Args:
            v (typing.Any): The value to parse.

        Raises:
            RuntimeError: If the parsing fails.

        Returns:
            _RobotDiff: The RobotDiff to execute.
        """

        robot_arg: list[str] = list(filter(len, str(v).split(',')))

        parsed_explicit: dict[str, Robot] = {}
        parsed_anonymous: dict[str, list[Robot]] = {}

        def add(base: robot_setup.Config):
            name = base.name
            config = Robot.from_setup(base)

            if name is None:  # anon
                parsed_anonymous.setdefault(
                    config.model.name, []
                ).append(config)

            else:  # explicit name
                if name in parsed_explicit:
                    raise RuntimeError(
                        f'naming conflict for robots with name {name}')

                parsed_explicit[name] = config

        for arg in robot_arg:
            if arg.endswith('.yaml'):
                for addition in robot_setup.RobotSetupIdentifier(arg).resolve_sync():
                    add(addition)
            elif (match := re.match(r'(.*)\[(\d+)\]', arg)):
                # multi-instantiations via model[count]
                for _ in range(int(match.group(2))):
                    add(robot_setup.Config(robot=match.group(1)))
            else:
                # just robot
                add(robot_setup.Config(robot=arg))

        existing = {
            k: v.robot
            for k, v
            in self.managers.items()
        }

        existing_keys = set(existing.keys())
        matchable_keys = set(existing_keys)

        to_add: dict[str, Robot] = {}
        to_update: dict[str, Robot] = {}

        # explicit naming first
        for prefix, config in parsed_explicit.items():
            match = next(
                (key for key in matchable_keys if existing[key] == config),
                None
            )

            if match is None:  # no matches
                to_add[prefix] = config
            else:  # exact match
                matchable_keys.remove(match)

        # anonymous naming
        for prefix, configs in parsed_anonymous.items():
            unassigned: list[Robot] = []

            for config in configs:
                match = next(
                    (
                        key
                        for key
                        in matchable_keys
                        if existing[key].compatible(config)
                    ),
                    None
                )

                if match is None:  # no similar robot found
                    unassigned.append(config)
                else:  # similar robot found, update
                    to_update[match] = config
                    matchable_keys.remove(match)

            i: int = 0
            for anon in unassigned:
                suffixed_key = prefix

                if len(configs) > 1:
                    suffixed_key = f'{prefix}_{i}'

                while suffixed_key in existing_keys:
                    i += 1
                    suffixed_key = f'{prefix}_{i}'

                to_add[suffixed_key] = anon
                existing_keys.add(suffixed_key)

        self._diff = _RobotDiff(
            to_remove=list(matchable_keys),
            to_add=to_add,
            to_update=to_update,
        )
        return self._diff

    async def set_up(self):
        futures: list[typing.Awaitable] = []
        for robot_name in self._diff.to_remove:
            futures.append(self.managers.pop(robot_name).destroy())
        self._diff.to_remove.clear()

        for robot_name, config in self._diff.to_update.items():
            futures.append(self.managers[robot_name].update(config))
            # TODO
        self._diff.to_update.clear()

        node_paths: set[str] = set()
        for robot_name, config in self._diff.to_add.items():
            config = attrs.evolve(config)
            config.name = robot_name
            config.pose = next(self._initialpose)
            manager = RobotManager(
                node=self.node,
                namespace=Namespace(self.node.get_namespace())(
                    self.node.get_name(),
                ),
                environment_manager=self._environment_manager,
                robot=config
            )
            futures.append(manager.set_up_robot(node_paths))
            self.managers[robot_name] = manager

        with self.provide_node_paths(node_paths) as fetch_task:
            await asyncio.wait(
                (
                    fetch_task,
                    asyncio.gather(*futures),
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
        self._diff.to_add.clear()

        self.node.rosparam[list[str]].set('robot_names', [robot.name for robot in self.managers.values()])

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initialpose = _initialpose_generator(-10, -10, -5)

        self._robot_configurations = self.node.ROSParam[_RobotDiff](
            'robot',
            type_=rclpy.Parameter.Type.STRING,
            parse=self._parse_robot_configurations,
        )

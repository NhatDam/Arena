from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import RobotGoal

from task_generator.shared import PositionRadius
from task_generator.tasks.robots import TM_Robots


class TM_Scenario(TM_Robots):
    """
    This class represents a scenario for robots in the task generator.
    It inherits from TM_Robots class and Node class.

    Attributes:
        _config (Config): The configuration object for the scenario.
    """

    _config: ROSParamT[list[RobotGoal]]

    def _parse_scenario(self, scenario: str) -> list[RobotGoal]:
        return WorldIdentifier(self.node._world_manager.world_name).resolve_sync().scenario(scenario).resolve_sync().load().robots

    async def reset(self, **kwargs):
        """
        Resets the scenario.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            None
        """

        await super().reset(**kwargs)

        SCENARIO_ROBOTS = self._config.value

        # --- ADVANCE-DEBUG: log what scenario is active ---
        _ros = self.node.get_logger()
        _param_val = self.node.get_parameter_or('task.scenario.file', None)
        _ros.warn(
            f"[ADVANCE-DEBUG] TM_Scenario.reset(): "
            f"param task.scenario.file={_param_val.value if _param_val else 'NOT_SET'}  "
            f"num_robots_in_scenario={len(SCENARIO_ROBOTS) if SCENARIO_ROBOTS else 0}")
        if SCENARIO_ROBOTS:
            for i, rc in enumerate(SCENARIO_ROBOTS):
                _ros.warn(
                    f"[ADVANCE-DEBUG]   robot[{i}] start=({rc.start.position.x:.2f}, {rc.start.position.y:.2f}) "
                    f"goal=({rc.goal.position.x:.2f}, {rc.goal.position.y:.2f})")

        # check robot manager length
        managed_robots = list(self._PROPS.robots.values())

        scenario_robots_length = len(SCENARIO_ROBOTS)
        setup_robot_length = len(managed_robots)

        if setup_robot_length > scenario_robots_length:
            managed_robots = managed_robots[:scenario_robots_length]
            self._logger.warn(
                "Robot setup contains more robots than the scenario file.", once=True)

        if scenario_robots_length > setup_robot_length:
            SCENARIO_ROBOTS = SCENARIO_ROBOTS[:setup_robot_length]
            self._logger.warn(
                "Scenario file contains more robots than setup.", once=True)

        for robot, config in zip(managed_robots, SCENARIO_ROBOTS):
            await robot.reset(start_pos=config.start, goal_pos=config.goal)
            self._PROPS.world_manager.forbid(
                [
                    PositionRadius(
                        x=config.start.position.x, y=config.start.position.y, radius=robot.safe_distance
                    ),
                    PositionRadius(
                        x=config.goal.position.x, y=config.goal.position.y, radius=robot.safe_distance
                    ),
                ]
            )

    def __init__(self, **kwargs):
        TM_Robots.__init__(self, **kwargs)

        self._config = self.node.ROSParam[list[RobotGoal]](
            self.namespace('file'),
            'default.json',
            parse=self._parse_scenario,
        )

from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario

from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


class TM_Scenario(TM_Obstacles):

    _config: ROSParamT[Scenario]

    def _parse_scenario(self, scenario: str) -> Scenario:
        return WorldIdentifier(self.node._world_manager.world_name).resolve_sync().scenario(scenario).resolve_sync().load()

    async def reset(self, **kwargs) -> Obstacles:
        return self._config.value.static, self._config.value.dynamic

    def __init__(self, **kwargs):
        TM_Obstacles.__init__(self, **kwargs)

        default_scenario: str | None = 'default'
        if default_scenario not in (scenarios := list(identifier_to_available(WorldIdentifier(self.node._world_manager.world_name).resolve_sync().scenario))):
            default_scenario = next(iter(scenarios), None)
        if default_scenario is None:
            raise ValueError(f"No scenarios found in world {self.node._world_manager.world_name}")

        self._config = self.node.ROSParam[Scenario](
            self.namespace('file'),
            default_scenario,
            parse=self._parse_scenario,
        )

"""Placeholder CityWalker agent (not yet integrated)."""

from __future__ import annotations

from pathlib import Path

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent


class CityWalkerAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='citywalker',
        topic_prefix='citywalker',
        default_config_filename='configs/citywalker.yaml',
        default_checkpoint_filename='ckpt/CityWalker.pth',
    )

    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        if logger is not None:
            logger.error("CityWalker agent is not yet integrated in arena_ai_integration")
        return False

    def predict(self, image_history, instruction, context=None):
        raise NotImplementedError("CityWalker agent is not yet integrated")

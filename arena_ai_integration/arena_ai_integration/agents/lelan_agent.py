"""Placeholder LeLan agent (not yet integrated)."""

from __future__ import annotations

from pathlib import Path

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent, PredictionContext


class LeLanAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='lelan',
        topic_prefix='lelan',
        default_config_filename='configs/lelan.yaml',
        default_checkpoint_filename='ckpt/LeLan.pth',
    )

    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        if logger is not None:
            logger.error("LeLan agent is not yet integrated in arena_ai_integration")
        return False

    def predict(self, image_history, instruction, context=None):
        raise NotImplementedError("LeLan agent is not yet integrated")

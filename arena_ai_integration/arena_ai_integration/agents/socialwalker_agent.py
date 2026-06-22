"""Placeholder SocialWalker agent (not yet integrated)."""

from __future__ import annotations

from pathlib import Path

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent


class SocialWalkerAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='socialwalker',
        topic_prefix='socialwalker',
        default_config_filename='configs/socialwalker.yaml',
        default_checkpoint_filename='ckpt/SocialWalker.pth',
    )

    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        if logger is not None:
            logger.error("SocialWalker agent is not yet integrated in arena_ai_integration")
        return False

    def predict(self, image_history, instruction, context=None):
        raise NotImplementedError("SocialWalker agent is not yet integrated")

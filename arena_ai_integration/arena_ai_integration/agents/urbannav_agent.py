"""UrbanNav agent wrapper."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent, PredictionContext

SOCIAL_NAV_ROOT = Path(__file__).resolve().parents[4] / 'social-nav'


def _ensure_social_nav_path(subdir: str) -> None:
    root = str(SOCIAL_NAV_ROOT / 'ros2_nodes' / subdir)
    if root not in sys.path:
        sys.path.insert(0, root)
    repo = str(SOCIAL_NAV_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class UrbanNavAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='urbannav',
        topic_prefix='urbannav',
        default_config_filename='configs/urbannav_film.yaml',
        default_checkpoint_filename='ckpt/UrbanNav_FiLM.pth',
        flip_y_axis=True,
        allow_model_soft_fail=False,
        rejoin_skip_distance=0.8,
        control_frequency=5.0,
        look_ahead_distance=0.5,
        arrival_threshold=0.7,
        path_waypoint_index=3,
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or self.DEFAULT_CONFIG)

    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        try:
            import torch
            _ensure_social_nav_path('urbannav')
            from urbannav_ros2_node import UrbanNavModel

            if not config_path.exists():
                raise FileNotFoundError(f"UrbanNav config not found: {config_path}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"UrbanNav checkpoint not found: {checkpoint_path}")

            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = UrbanNavModel(
                str(config_path),
                str(checkpoint_path),
                self._device,
            )
            if logger is not None:
                logger.info(f"UrbanNav model loaded from {checkpoint_path}")
            return True
        except Exception as exc:
            if logger is not None:
                logger.error(
                    "Failed to load UrbanNav model:\n"
                    f"{''.join(traceback.format_exception(exc))}"
                )
            self._model = None
            return False

    def predict(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        context: Optional[PredictionContext] = None,
    ) -> Tuple[np.ndarray, float]:
        if self._model is None:
            raise RuntimeError("UrbanNav model is not loaded")

        waypoints, arrival_score = self._model.predict(image_history, instruction)
        return waypoints, float(arrival_score)

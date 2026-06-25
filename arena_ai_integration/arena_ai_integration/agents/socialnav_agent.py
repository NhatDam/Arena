"""SocialNav agent wrapper."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent, PredictionContext


class SocialNavAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='socialnav',
        topic_prefix='socialnav',
        default_config_filename='config/models/socialnav_film.yaml',
        default_checkpoint_filename='checkpoints/SocialNav_1_path.pth',
        flip_y_axis=True,
        allow_model_soft_fail=True,
        rejoin_skip_distance=2.5,
        control_frequency=3.0,
        look_ahead_distance=1.25,
        arrival_threshold=2.5,
        path_waypoint_index=3,
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or self.DEFAULT_CONFIG)

    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        try:
            import torch
            from arena_ai_integration.models.socialnav.runtime import SocialNavModel

            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = SocialNavModel(
                str(config_path),
                str(checkpoint_path),
                self._device,
            )
            if logger is not None:
                logger.info(f"SocialNav model loaded from {checkpoint_path}")
            return True
        except Exception as exc:
            if logger is not None:
                logger.error(
                    "Failed to load SocialNav model; running in DWB fallback mode:\n"
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
            raise RuntimeError("SocialNav model is not loaded")

        context = context or PredictionContext()
        kwargs = {
            'human_positions': context.human_positions,
            'ego_hist_xy': context.ego_hist_xy,
            'human_mask': context.human_mask,
        }

        import torch

        if context.cuda_stream is not None:
            with torch.cuda.stream(context.cuda_stream):
                waypoints, arrival_score = self._model.predict(
                    image_history,
                    instruction,
                    **kwargs,
                )
            torch.cuda.current_stream().wait_stream(context.cuda_stream)
        else:
            waypoints, arrival_score = self._model.predict(
                image_history,
                instruction,
                **kwargs,
            )

        return waypoints, float(arrival_score)

    def predict_candidates(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        context: Optional[PredictionContext] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("SocialNav model is not loaded")

        context = context or PredictionContext()
        kwargs = {
            'human_positions': context.human_positions,
            'ego_hist_xy': context.ego_hist_xy,
            'human_mask': context.human_mask,
        }

        import torch

        if context.cuda_stream is not None:
            with torch.cuda.stream(context.cuda_stream):
                candidates, arrival_scores = self._model.predict_candidates(
                    image_history,
                    instruction,
                    **kwargs,
                )
            torch.cuda.current_stream().wait_stream(context.cuda_stream)
        else:
            candidates, arrival_scores = self._model.predict_candidates(
                image_history,
                instruction,
                **kwargs,
            )

        return candidates, arrival_scores

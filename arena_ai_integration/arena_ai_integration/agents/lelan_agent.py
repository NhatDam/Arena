"""LeLaN agent wrapper."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent, PredictionContext


class LeLanAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='lelan',
        topic_prefix='lelan',
        default_config_filename='config/models/lelan.yaml',
        default_checkpoint_filename='checkpoints/LeLan_latest.pth',
        flip_y_axis=False,
        allow_model_soft_fail=False,
        rejoin_skip_distance=0.8,
        control_frequency=5.0,
        look_ahead_distance=0.5,
        arrival_threshold=0.7,
        path_waypoint_index=3,
        extra_params={
            'coordinate_mode': 'xz_to_ros',
            'waypoint_scale': 1.0,
        },
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or self.DEFAULT_CONFIG)
        self.context_size = 8

    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        try:
            import torch
            from arena_ai_integration.models.lelan.runtime import LeLaNInferenceModel

            if not config_path.exists():
                raise FileNotFoundError(f"LeLaN config not found: {config_path}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"LeLaN checkpoint not found: {checkpoint_path}")

            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model = LeLaNInferenceModel(
                config_path=str(config_path),
                checkpoint_path=str(checkpoint_path),
                device=self._device,
            )
            self.context_size = int(getattr(self._model, 'context_size', self.context_size))
            if logger is not None:
                logger.info(f"LeLaN model loaded from {checkpoint_path}")
            return True
        except Exception as exc:
            if logger is not None:
                logger.error(
                    "Failed to load LeLaN model:\n"
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
            raise RuntimeError("LeLaN model is not loaded")

        waypoint_scale = float(self.config.extra_params.get('waypoint_scale', 1.0))
        waypoints, arrival_score = self._model.predict(
            image_history,
            instruction,
            waypoint_scale=waypoint_scale,
        )
        return waypoints, float(arrival_score)

    def to_ros_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        arr = np.asarray(waypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected LeLaN waypoints shape [T,2], got {arr.shape}")
        arr = arr[:, :2]

        coordinate_mode = str(self.config.extra_params.get('coordinate_mode', 'xz_to_ros'))
        if coordinate_mode == 'xz_to_ros':
            ros_waypoints = np.zeros_like(arr)
            ros_waypoints[:, 0] = arr[:, 1]
            ros_waypoints[:, 1] = arr[:, 0]
            return ros_waypoints
        if coordinate_mode == 'ros':
            return np.array(arr, copy=True)
        raise ValueError(
            f"Unsupported LeLaN coordinate_mode='{coordinate_mode}'. "
            "Use 'xz_to_ros' or 'ros'."
        )

"""UrbanNav agent wrapper."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from arena_ai_integration.agents.base_agent import AgentConfig, BaseAgent, PredictionContext


class UrbanNavAgent(BaseAgent):
    DEFAULT_CONFIG = AgentConfig(
        name='urbannav',
        topic_prefix='urbannav',
        default_config_filename='config/models/urbannav_film.yaml',
        default_checkpoint_filename='checkpoints/UrbanNav_FiLM.pth',
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
            from arena_ai_integration.models.socialnav.runtime import UrbanNavModel

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

        context = context or PredictionContext()
        human_positions = self._latest_valid_humans(context)

        import torch

        if context.cuda_stream is not None:
            with torch.cuda.stream(context.cuda_stream):
                waypoints, arrival_score = self._model.predict(
                    image_history,
                    instruction,
                    human_positions=human_positions,
                )
            torch.cuda.current_stream().wait_stream(context.cuda_stream)
        else:
            waypoints, arrival_score = self._model.predict(
                image_history,
                instruction,
                human_positions=human_positions,
            )

        waypoint_scale = float(self.config.extra_params.get('waypoint_scale', 1.0))
        if waypoint_scale != 1.0:
            waypoints = np.asarray(waypoints, dtype=np.float32) * waypoint_scale
        return waypoints, float(arrival_score)

    @staticmethod
    def _latest_valid_humans(context: PredictionContext) -> Optional[List[np.ndarray]]:
        if context.human_positions is None:
            return None

        humans = np.asarray(context.human_positions, dtype=np.float32)
        if humans.ndim == 3:
            latest = humans[-1]
            if context.human_mask is not None:
                mask = np.asarray(context.human_mask, dtype=bool)
                if mask.shape == humans.shape[:2]:
                    latest = latest[~mask[-1]]
        elif humans.ndim == 2:
            latest = humans
            if context.human_mask is not None:
                mask = np.asarray(context.human_mask, dtype=bool)
                if mask.shape == humans.shape[:1]:
                    latest = latest[~mask]
        else:
            return None

        if latest.size == 0:
            return None
        latest = latest[:, :2]
        finite = np.isfinite(latest).all(axis=1)
        latest = latest[finite]
        if len(latest) == 0:
            return None
        return [np.asarray(pos, dtype=np.float32) for pos in latest]

    def to_ros_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        arr = np.asarray(waypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected UrbanNav waypoints shape [T,2], got {arr.shape}")
        arr = arr[:, :2]

        coordinate_mode = str(self.config.extra_params.get('coordinate_mode', 'xz_to_ros'))
        if coordinate_mode in ('xz_to_ros', 'dataset_to_ros'):
            ros_waypoints = np.zeros_like(arr)
            ros_waypoints[:, 0] = arr[:, 1]
            ros_waypoints[:, 1] = arr[:, 0]
            return ros_waypoints
        if coordinate_mode == 'ros':
            return np.array(arr, copy=True)
        raise ValueError(
            f"Unsupported UrbanNav coordinate_mode='{coordinate_mode}'. "
            "Use 'xz_to_ros', 'dataset_to_ros', or 'ros'."
        )

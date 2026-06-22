"""Abstract interface for Arena AI navigation agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class PredictionContext:
    """Optional context passed to agents during inference."""

    human_positions: Optional[np.ndarray] = None
    human_mask: Optional[np.ndarray] = None
    ego_hist_xy: Optional[np.ndarray] = None
    cuda_stream: Any = None


@dataclass
class AgentConfig:
    name: str
    topic_prefix: str
    default_config_filename: str
    default_checkpoint_filename: str
    flip_y_axis: bool = True
    allow_model_soft_fail: bool = False
    rejoin_skip_distance: float = 2.5
    control_frequency: float = 3.0
    look_ahead_distance: float = 1.25
    arrival_threshold: float = 2.5
    path_waypoint_index: int = 3
    extra_params: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Strategy interface for end-to-end navigation models."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._model = None
        self._device = 'cpu'

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def topic_prefix(self) -> str:
        return self.config.topic_prefix

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def load(self, config_path: Path, checkpoint_path: Path, logger=None) -> bool:
        """Load model weights. Return True on success."""

    @abstractmethod
    def predict(
        self,
        image_history: List[np.ndarray],
        instruction: str,
        context: Optional[PredictionContext] = None,
    ) -> Tuple[np.ndarray, float]:
        """Return (waypoints, arrival_score) in model frame."""

    def to_ros_waypoints(self, waypoints: np.ndarray) -> np.ndarray:
        ros_waypoints = np.zeros_like(waypoints)
        ros_waypoints[:, 0] = waypoints[:, 0]
        if self.config.flip_y_axis:
            ros_waypoints[:, 1] = -waypoints[:, 1]
        else:
            ros_waypoints[:, 1] = waypoints[:, 1]
        return ros_waypoints

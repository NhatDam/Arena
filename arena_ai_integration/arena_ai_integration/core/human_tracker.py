"""Track human positions over time for social navigation context."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class HumanPositionTracker:
    """Rolling buffer of human (x, y) detections for model input."""

    def __init__(self, history_length: int = 8, max_humans: int = 10):
        self.history_length = history_length
        self.max_humans = max_humans
        self.human_positions_history = deque(maxlen=history_length)
        self.human_mask_history = deque(maxlen=history_length)

    def update(self, human_detections: List[Tuple[float, float]]) -> None:
        detections = list(human_detections[: self.max_humans])
        valid_count = len(detections)
        while len(detections) < self.max_humans:
            detections.append((0.0, 0.0))

        mask = np.ones(self.max_humans, dtype=bool)
        mask[:valid_count] = False

        self.human_positions_history.append(np.array(detections, dtype=np.float32))
        self.human_mask_history.append(mask)

    def get_human_positions(self) -> Optional[np.ndarray]:
        if len(self.human_positions_history) < self.history_length:
            return None
        return np.array(list(self.human_positions_history))

    def get_human_mask(self) -> Optional[np.ndarray]:
        if len(self.human_mask_history) < self.history_length:
            return None
        return np.array(list(self.human_mask_history), dtype=bool)

    def is_ready(self) -> bool:
        return len(self.human_positions_history) >= self.history_length

from typing import Optional
import numpy as np


class RankWeightedCentroid:
    """Per-game online estimator of the target word's position in embedding space.

    Weight of each observation = 1 / max(1, raw_distance). The answer word
    (distance=1) gets weight 1.0; a word at rank 5000 gets weight 0.0002.
    The normalised weighted average converges toward the true target embedding.
    """

    def __init__(self, emb_matrix: np.ndarray) -> None:
        self.emb = emb_matrix
        self._weighted_sum = np.zeros(emb_matrix.shape[1], dtype=np.float64)
        self._weight_total = 0.0
        self._centroid: Optional[np.ndarray] = None

    def update(self, idx: int, raw_distance: float) -> None:
        weight = 1.0 / max(1.0, float(raw_distance))
        self._weighted_sum += weight * self.emb[idx].astype(np.float64)
        self._weight_total += weight
        raw = self._weighted_sum / self._weight_total
        norm = np.linalg.norm(raw)
        self._centroid = (raw / norm).astype(np.float32) if norm > 1e-12 else None

    def get(self) -> Optional[np.ndarray]:
        return self._centroid

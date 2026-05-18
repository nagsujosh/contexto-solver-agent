import json
import os
from typing import Optional

import numpy as np


class CorrectionStore:
    """Persistent cross-game correction for the rank-weighted centroid.

    Learns the systematic offset between centroid estimates and true targets.
    Phase 1 (< min_games_for_matrix): exponential-moving-average bias vector.
    Phase 2 (>= min_games_for_matrix): full least-squares correction matrix.

    Storage: plain .npy arrays + JSON metadata.
    """

    def __init__(
        self,
        bias_path: str,
        matrix_path: str,
        meta_path: str,
        min_games_for_matrix: int = 20,
    ) -> None:
        self.bias_path = bias_path
        self.matrix_path = matrix_path
        self.meta_path = meta_path
        self.min_games_for_matrix = min_games_for_matrix

        self._bias: Optional[np.ndarray] = None
        self._matrix: Optional[np.ndarray] = None
        self.game_count: int = 0
        self._centroid_history: list = []
        self._target_history: list = []

        self._load()

    @property
    def has_matrix(self) -> bool:
        return self._matrix is not None

    def correct(self, centroid: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if centroid is None:
            return None
        v = centroid.astype(np.float64)
        if self._matrix is not None:
            corrected = self._matrix @ v
        elif self._bias is not None:
            alpha = min(0.4, self.game_count / 50.0)
            corrected = v + alpha * self._bias
        else:
            return centroid.astype(np.float32)
        norm = np.linalg.norm(corrected)
        return (corrected / norm).astype(np.float32) if norm > 1e-12 else centroid.astype(np.float32)

    def record_game(self, final_centroid: np.ndarray, target_emb: np.ndarray) -> None:
        c = final_centroid.astype(np.float64)
        t = target_emb.astype(np.float64)
        error = t - c
        self._bias = error if self._bias is None else 0.7 * self._bias + 0.3 * error
        self._centroid_history.append(c.astype(np.float32))
        self._target_history.append(t.astype(np.float32))
        self.game_count += 1
        if (
            self.game_count >= self.min_games_for_matrix
            and len(self._centroid_history) >= self.min_games_for_matrix
        ):
            self._fit_matrix()
        self._save()

    def _fit_matrix(self) -> None:
        X = np.vstack(self._centroid_history).astype(np.float64)
        Y = np.vstack(self._target_history).astype(np.float64)
        d = X.shape[1]
        # Ridge toward identity: min_M ||X M - Y||² + λ||M - I||²
        # Solution: M = (X^T X + λI)^{-1} (X^T Y + λI)
        # This falls back to M=I (no correction) in directions where X has no
        # data — avoiding the ~90° distortion of plain lstsq with n << d.
        lam = 1.0
        M = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y + lam * np.eye(d))
        self._matrix = M.T.astype(np.float32)

    def _save(self) -> None:
        if self._bias is not None:
            np.save(self.bias_path, self._bias.astype(np.float32))
        with open(self.meta_path, "w") as f:
            json.dump({"game_count": self.game_count}, f)
        if self._centroid_history:
            np.save(self._ch_path, np.vstack(self._centroid_history[-200:]).astype(np.float32))
            np.save(self._th_path, np.vstack(self._target_history[-200:]).astype(np.float32))
        if self._matrix is not None:
            np.save(self.matrix_path, self._matrix)

    def _load(self) -> None:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path) as f:
                    self.game_count = int(json.load(f).get("game_count", 0))
            except Exception:
                pass
        if os.path.exists(self.bias_path):
            try:
                self._bias = np.load(self.bias_path).astype(np.float64)
            except Exception:
                pass
        if os.path.exists(self._ch_path):
            try:
                self._centroid_history = list(np.load(self._ch_path))
            except Exception:
                pass
        if os.path.exists(self._th_path):
            try:
                self._target_history = list(np.load(self._th_path))
            except Exception:
                pass
        if os.path.exists(self.matrix_path):
            try:
                self._matrix = np.load(self.matrix_path).astype(np.float32)
            except Exception:
                pass

    @property
    def _ch_path(self) -> str:
        return self.bias_path.replace("bias", "ch")

    @property
    def _th_path(self) -> str:
        return self.bias_path.replace("bias", "th")

import random
from typing import List, Optional, Set, Tuple

import numpy as np

from .config import TOP_K_BEST


class HybridSolver:
    """Word selector using rank-weighted centroid + adaptive MMR scoring.

    lambda = clip(best_sim / 0.5, 0, 1) controls explore/exploit balance:
      lambda ~ 0  (no good guess yet)  -> diversity: pick word most unlike tried words
      lambda -> 1 (score >= 0.5 found) -> MMR over centroid-similar candidates

    Candidate pool expands as the game grows longer so we can reach words that
    are not in the top-500 GloVe neighbors of the centroid.

    Last-mile mode: when best_raw_distance <= 5 (answer is very close), we scan
    a wider pool combining centroid direction and the best-word's own neighbors,
    preventing the solver from getting stuck 1-2 ranks away from the answer.
    """

    def __init__(self, vocab: List[str], emb_matrix: np.ndarray) -> None:
        self.vocab = vocab
        self.emb = emb_matrix.astype(np.float32)
        self.n = len(vocab)
        self.best_idx: Optional[int] = None
        self.best_sim: float = -float("inf")
        self.best_raw_distance: float = float("inf")
        self.best_raw_idx: Optional[int] = None
        self.history: List[Tuple[int, float]] = []
        self._top_k: List[Tuple[float, int]] = []  # (sim, idx) sorted desc

    def update(self, idx: int, sim: float, raw_distance: float = float("inf")) -> None:
        if not np.isfinite(sim):
            sim = 0.0
        if sim > self.best_sim:
            self.best_sim = sim
            self.best_idx = idx
        if np.isfinite(raw_distance) and raw_distance >= 0 and raw_distance < self.best_raw_distance:
            self.best_raw_distance = raw_distance
            self.best_raw_idx = idx
        existing = {i for _, i in self._top_k}
        if idx not in existing:
            self._top_k.append((sim, idx))
        else:
            self._top_k = [(max(s, sim) if i == idx else s, i) for s, i in self._top_k]
        self._top_k.sort(reverse=True)
        self._top_k = self._top_k[:TOP_K_BEST]
        self.history.append((idx, sim))

    def propose_next(
        self,
        excluded: Set[int],
        corrected_centroid: Optional[np.ndarray] = None,
    ) -> int:
        lam = self._lambda()

        # When signal is weak (lam < 0.5), full-vocab diversity is better than
        # following a poorly-estimated centroid.
        if corrected_centroid is None or lam < 0.5:
            pool = [i for i in range(self.n) if i not in excluded]
            if not pool:
                raise RuntimeError("No candidates available")
            return self._diverse_pick(pool)

        actual_guesses = len(self.history)

        # Last-mile mode: when we're very close (best raw rank <= 5), the centroid
        # alone may not surface the exact answer due to GloVe/Contexto mismatch.
        # Scan a wide pool combining centroid direction + best-word's GloVe neighbors.
        if self.best_raw_distance <= 5 and self.best_raw_idx is not None:
            best_emb = self.emb[self.best_raw_idx]
            combined = 0.5 * (self.emb @ corrected_centroid) + 0.5 * (self.emb @ best_emb)
            k = min(20000, self.n)
            top_pos = np.argpartition(-combined, k - 1)[:k]
            pool = [int(i) for i in top_pos if i not in excluded]
            if not pool:
                pool = [i for i in range(self.n) if i not in excluded]
            if not pool:
                raise RuntimeError("No candidates available")
            pool_arr = np.array(pool, dtype=np.int32)
            return int(pool_arr[np.argmax(combined[pool_arr])])

        # Strong signal (lam >= 0.5): exploit the centroid with a pool that
        # grows as the game lengthens, so we don't get permanently stuck after
        # exhausting the top-500 GloVe neighbors.
        if actual_guesses <= 100:
            k_candidates = 500
        elif actual_guesses <= 400:
            k_candidates = 3000
        elif actual_guesses <= 800:
            k_candidates = 15000
        else:
            k_candidates = self.n  # full vocabulary

        sims = self.emb @ corrected_centroid
        k = min(k_candidates, self.n)
        top_pos = np.argpartition(-sims, k - 1)[:k]
        pool = [int(i) for i in top_pos if i not in excluded]
        if not pool:
            pool = [i for i in range(self.n) if i not in excluded]
        if not pool:
            raise RuntimeError("No candidates available")

        pool_arr = np.array(pool, dtype=np.int32)
        relevance = self.emb[pool_arr] @ corrected_centroid
        redundancy = self._redundancy(pool_arr)
        scores = lam * relevance - (1.0 - lam) * redundancy
        return int(pool_arr[np.argmax(scores)])

    def _lambda(self) -> float:
        if self.best_sim <= -float("inf"):
            return 0.0
        return float(np.clip(max(0.0, self.best_sim) / 0.5, 0.0, 1.0))

    def _redundancy(self, pool_arr: np.ndarray) -> np.ndarray:
        if not self._top_k:
            return np.zeros(len(pool_arr), dtype=np.float32)
        top5 = [i for _, i in self._top_k[:5]]
        mean_dir = self.emb[top5].mean(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(mean_dir))
        if nrm < 1e-12:
            return np.zeros(len(pool_arr), dtype=np.float32)
        mean_dir /= nrm
        return (self.emb[pool_arr] @ mean_dir).astype(np.float32)

    def _diverse_pick(self, pool: List[int]) -> int:
        if not self.history:
            return random.choice(pool)
        tried = [idx for idx, _ in self.history]
        mean_dir = self.emb[tried].mean(axis=0).astype(np.float32)
        nrm = float(np.linalg.norm(mean_dir))
        if nrm < 1e-12:
            return random.choice(pool)
        mean_dir /= nrm
        pool_arr = np.array(pool, dtype=np.int32)
        return int(pool_arr[np.argmax(-(self.emb[pool_arr] @ mean_dir))])

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .config import (
    EXPLOIT_BASE,
    INITIAL_TEMPERATURE,
    MIN_TEMPERATURE,
    NEIGHBOR_K,
    TEMPERATURE_DECAY,
    UCB_ALPHA,
    KMEANS_K,
    KMEANS_RANDOM_STATE,
    CLUSTER_SAMPLE_K,
)


class HybridSolver:
    def __init__(
        self,
        vocab: List[str],
        emb_matrix: np.ndarray,
        neighbor_k: int = NEIGHBOR_K,
        ucb_alpha: float = UCB_ALPHA,
        initial_temp: float = INITIAL_TEMPERATURE,
        exploit_base: float = EXPLOIT_BASE,
    ):
        self.vocab = vocab
        self.emb = emb_matrix.astype(np.float32)
        self.n = len(vocab)
        self.best_idx: Optional[int] = None
        self.best_sim = -np.inf
        self.history: List[Tuple[int, float]] = []
        self.neighbor_k = neighbor_k
        self.ucb_alpha = ucb_alpha
        self.temperature = initial_temp
        self.exploit_base = exploit_base

        self._neighbor_cache: Dict[int, np.ndarray] = {}

        # --- Cluster index for exploration ---
        self.kmeans = KMeans(
            n_clusters=min(KMEANS_K, max(2, min(2000, self.n // 50))),
            n_init=3,
            random_state=KMEANS_RANDOM_STATE,
            verbose=0,
        )
        self.cluster_id = self.kmeans.fit_predict(self.emb)
        self.centroids = self.kmeans.cluster_centers_.astype(np.float32)
        self.K = self.centroids.shape[0]
        self.cluster_members: List[np.ndarray] = [
            np.where(self.cluster_id == k)[0] for k in range(self.K)
        ]

        # Cluster stats: reward & pulls
        self.cluster_stats: Dict[int, Tuple[float, int]] = {k: (0.0, 0) for k in range(self.K)}

    def update(self, idx: int, sim: float):
        if not np.isfinite(sim):
            sim = 0.0
        prev_best = self.best_sim
        if sim > self.best_sim:
            self.best_sim = sim
            self.best_idx = idx
        improvement = max(0.0, sim - prev_best)

        # Update cluster stats
        cid = int(self.cluster_id[idx])
        tot, cnt = self.cluster_stats[cid]
        self.cluster_stats[cid] = (tot + improvement, cnt + 1)

        self.history.append((idx, sim))
        self._cooldown()

    def propose_next(self, excluded: Set[int]) -> int:
        exploit_pool = self._exploit_pool(excluded)
        explore_pool = self._explore_pool(excluded)

        use_exploit = random.random() < self._exploit_bias()
        if use_exploit and len(exploit_pool) > 0:
            pool = exploit_pool
        elif len(explore_pool) > 0:
            pool = explore_pool
        else:
            pool = [i for i in range(self.n) if i not in excluded]

        if not pool:
            raise RuntimeError("No candidates available to propose")

        if self.best_idx is not None:
            sims = self.emb[pool] @ self.emb[self.best_idx]
            logits = sims.astype(np.float64) / max(1e-6, self.temperature)
            logits -= logits.max()
            probs = np.exp(logits)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                return random.choice(pool)
            probs /= probs_sum
            return int(np.random.choice(pool, p=probs))
        else:
            return random.choice(pool)

    def _exploit_pool(self, excluded: Set[int]) -> List[int]:
        if self.best_idx is None:
            return []
        neigh = self._get_neighbors(self.best_idx)
        return [i for i in neigh if i not in excluded]

    def _explore_pool(self, excluded: Set[int]) -> List[int]:
        total_pulls = sum(cnt for _, cnt in self.cluster_stats.values()) + 1
        best_ucb = -1.0
        best_cluster = None
        for cid, (tot, cnt) in self.cluster_stats.items():
            avg = (tot / cnt) if cnt > 0 else 0.0
            ucb = avg + self.ucb_alpha * math.sqrt(max(0.0, math.log(total_pulls)) / (cnt + 1))
            if ucb > best_ucb:
                best_ucb = ucb
                best_cluster = cid
        if best_cluster is None:
            return []
        members = self.cluster_members[best_cluster]
        if members.size == 0:
            return []
        m = min(CLUSTER_SAMPLE_K, members.size)
        c = self.centroids[best_cluster]
        sims = self.emb[members] @ c
        order = np.argsort(-sims)[:m]
        cand = members[order]
        return [int(i) for i in cand if i not in excluded]

    def _get_neighbors(self, idx: int) -> np.ndarray:
        if idx in self._neighbor_cache:
            return self._neighbor_cache[idx]
        sims = self.emb @ self.emb[idx]
        k = min(self.neighbor_k, self.n)
        top_k_idx = np.argpartition(-sims, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(-sims[top_k_idx])]
        self._neighbor_cache[idx] = top_k_idx
        return top_k_idx

    def _cooldown(self):
        self.temperature = max(MIN_TEMPERATURE, self.temperature * TEMPERATURE_DECAY)

    def _exploit_bias(self) -> float:
        base = self.exploit_base
        if self.best_idx is None:
            return base
        neigh = self._get_neighbors(self.best_idx)
        if neigh.size < 2:
            return base
        sims = self.emb[neigh] @ self.emb[self.best_idx]
        sims_sorted = np.sort(sims)[::-1]
        top1 = sims_sorted[0]
        tail = sims_sorted[1:11] if sims_sorted.size > 10 else sims_sorted[1:]
        if tail.size == 0:
            return min(0.95, base + 0.1)
        gap = top1 - float(np.mean(tail))
        bump = max(0.0, min(0.25, gap * 0.5))
        return max(0.0, min(0.98, base + bump))

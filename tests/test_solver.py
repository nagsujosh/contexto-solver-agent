import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contexto_solver.solver import HybridSolver


def _solver(n=20, d=8, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return HybridSolver([f"w{i}" for i in range(n)], emb), emb


def test_propose_returns_valid_index():
    s, _ = _solver()
    assert 0 <= s.propose_next(set()) < 20


def test_propose_never_returns_excluded():
    s, _ = _solver()
    excluded = {0, 1, 2, 3}
    for _ in range(30):
        assert s.propose_next(excluded) not in excluded


def test_high_score_shifts_selection_toward_centroid():
    s, emb = _solver(n=50, d=8)
    s.update(10, sim=0.8)          # lam = clip(0.8/0.5, 0, 1) = 1.0
    centroid = emb[10].copy()
    idx = s.propose_next({10}, corrected_centroid=centroid)
    chosen_sim = float(centroid @ emb[idx])
    mean_sim = float((emb @ centroid).mean())
    assert chosen_sim >= mean_sim


def test_no_score_triggers_diversity_mode():
    s, _ = _solver(n=50, d=8)
    s.update(0, sim=0.0)
    # lam=0 -> diversity; must not raise and must not return excluded
    idx = s.propose_next({0}, corrected_centroid=None)
    assert idx != 0


def test_update_tracks_best():
    s, _ = _solver()
    s.update(5, 0.3); s.update(7, 0.9); s.update(2, 0.4)
    assert s.best_idx == 7
    assert abs(s.best_sim - 0.9) < 1e-6


def test_top_k_sorted_and_bounded():
    s, _ = _solver(n=20)
    for i in range(10):
        s.update(i, float(i) * 0.1)
    sims = [sim for sim, _ in s._top_k]
    assert sims == sorted(sims, reverse=True)
    assert len(s._top_k) <= 5


def test_diversity_when_weak_signal():
    """When lam < 0.5 (weak signal), use full-vocab diversity regardless of excluded count."""
    rng = np.random.default_rng(42)
    n = 300
    d = 8
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    s = HybridSolver([f"w{i}" for i in range(n)], emb)
    # Simulate weak signal: best_sim = 0.2, lam = 0.4 < 0.5
    s.update(0, sim=0.2)
    excluded = set(range(1, 160))
    idx = s.propose_next(excluded, corrected_centroid=emb[0].copy())
    assert idx not in excluded
    assert idx != 0


def test_raw_distance_tracking():
    """update() tracks best_raw_distance and best_raw_idx correctly."""
    s, _ = _solver()
    s.update(3, sim=0.4, raw_distance=500.0)
    s.update(7, sim=0.3, raw_distance=50.0)
    s.update(2, sim=0.6, raw_distance=200.0)
    assert s.best_raw_idx == 7
    assert s.best_raw_distance == 50.0
    assert s.best_idx == 2  # best_sim still tracks score, not distance


def test_last_mile_uses_best_word_neighborhood():
    """When best_raw_distance <= 5, propose_next uses combined centroid+best-word pool."""
    rng = np.random.default_rng(99)
    n, d = 200, 8
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    s = HybridSolver([f"w{i}" for i in range(n)], emb)
    # Trigger last-mile: high sim (lam=1) and close raw distance
    s.update(10, sim=0.9, raw_distance=2.0)
    centroid = emb[10].copy()
    idx = s.propose_next({10}, corrected_centroid=centroid)
    assert idx != 10
    # Result must be in the top GloVe neighbors of word 10 (or centroid)
    combined = 0.5 * (emb @ centroid) + 0.5 * (emb @ emb[10])
    combined[10] = -1.0
    top20 = set(np.argsort(-combined)[:20])
    assert idx in top20, f"last-mile picked {idx} which is not in top-20 combined neighbors"

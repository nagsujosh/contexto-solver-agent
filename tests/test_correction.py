import json, os, tempfile
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contexto_solver.correction import CorrectionStore


def _tmp():
    d = tempfile.mkdtemp()
    return (os.path.join(d, "bias.npy"),
            os.path.join(d, "matrix.npy"),
            os.path.join(d, "meta.json"))


def _vec(d=8, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def test_correct_passthrough_with_no_data():
    bp, mp, mf = _tmp()
    s = CorrectionStore(bp, mp, mf, min_games_for_matrix=20)
    v = _vec()
    np.testing.assert_allclose(s.correct(v), v, atol=1e-5)


def test_correct_none_returns_none():
    bp, mp, mf = _tmp()
    s = CorrectionStore(bp, mp, mf)
    assert s.correct(None) is None


def test_bias_shifts_centroid_toward_target():
    bp, mp, mf = _tmp()
    s = CorrectionStore(bp, mp, mf, min_games_for_matrix=100)
    centroid = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    target   = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(5):
        s.record_game(centroid.copy(), target.copy())
    corrected = s.correct(centroid)
    assert corrected[1] > centroid[1]


def test_game_count_persists():
    bp, mp, mf = _tmp()
    s1 = CorrectionStore(bp, mp, mf, min_games_for_matrix=100)
    s1.record_game(_vec(), _vec(seed=1))
    s1.record_game(_vec(), _vec(seed=2))
    s2 = CorrectionStore(bp, mp, mf, min_games_for_matrix=100)
    assert s2.game_count == 2


def test_matrix_upgrade_at_threshold():
    bp, mp, mf = _tmp()
    d = 8
    s = CorrectionStore(bp, mp, mf, min_games_for_matrix=3)
    rng = np.random.default_rng(7)
    for _ in range(3):
        c = rng.standard_normal(d).astype(np.float32); c /= np.linalg.norm(c)
        t = rng.standard_normal(d).astype(np.float32); t /= np.linalg.norm(t)
        s.record_game(c, t)
    assert s.has_matrix
    result = s.correct(_vec(d=d))
    assert result is not None
    assert abs(np.linalg.norm(result) - 1.0) < 1e-4


def test_ridge_matrix_well_conditioned():
    """Identity-regularized ridge must stay well-conditioned when n_samples << dim."""
    bp, mp, mf = _tmp()
    d = 300
    # n_samples << d — the underdetermined regime that caused ~90° distortion with plain lstsq
    n = 10
    s = CorrectionStore(bp, mp, mf, min_games_for_matrix=n)
    rng = np.random.default_rng(42)
    for _ in range(n):
        c = rng.standard_normal(d).astype(np.float32); c /= np.linalg.norm(c)
        t = rng.standard_normal(d).astype(np.float32); t /= np.linalg.norm(t)
        s.record_game(c, t)
    assert s.has_matrix
    M = s._matrix.astype(np.float64)
    sv = np.linalg.svd(M, compute_uv=False)
    cond = sv[0] / sv[-1]
    assert cond < 1000, f"Matrix condition number {cond:.2e} too large (was ~4e10 without ridge)"
    # Correction must not wildly distort a random unit vector
    v = rng.standard_normal(d).astype(np.float32); v /= np.linalg.norm(v)
    corrected = s.correct(v)
    angle = float(np.degrees(np.arccos(np.clip(float(np.dot(v, corrected)), -1, 1))))
    assert angle < 30, f"Correction distorted vector by {angle:.1f}° (should be <30°)"

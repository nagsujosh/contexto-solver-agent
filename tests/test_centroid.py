import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contexto_solver.centroid import RankWeightedCentroid


def _emb(n=10, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def test_none_before_update():
    assert RankWeightedCentroid(_emb()).get() is None


def test_single_answer_word_gives_its_embedding():
    emb = _emb()
    c = RankWeightedCentroid(emb)
    c.update(idx=3, raw_distance=1.0)
    np.testing.assert_allclose(c.get(), emb[3], atol=1e-5)


def test_close_word_dominates_far_word():
    emb = _emb(n=5, d=4)
    c = RankWeightedCentroid(emb)
    c.update(idx=0, raw_distance=1000.0)
    c.update(idx=1, raw_distance=1.0)
    centroid = c.get()
    assert float(centroid @ emb[1]) > float(centroid @ emb[0])


def test_centroid_is_unit_norm():
    emb = _emb()
    c = RankWeightedCentroid(emb)
    for i in range(5):
        c.update(idx=i, raw_distance=float(i + 1) * 10)
    assert abs(np.linalg.norm(c.get()) - 1.0) < 1e-5


def test_inverse_distance_weights():
    # distance=1 -> weight 1.0, distance=2 -> weight 0.5
    # centroid = normalize(1.0*emb[0] + 0.5*emb[1])
    emb = _emb(n=2, d=4)
    c = RankWeightedCentroid(emb)
    c.update(0, raw_distance=1.0)
    c.update(1, raw_distance=2.0)
    raw = 1.0 * emb[0].astype(np.float64) + 0.5 * emb[1].astype(np.float64)
    raw /= np.linalg.norm(raw)
    np.testing.assert_allclose(c.get(), raw.astype(np.float32), atol=1e-5)

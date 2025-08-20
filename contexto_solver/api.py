import math
import time
from functools import lru_cache
from typing import Callable, Set, Tuple, List

import requests
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import API_BASE, LANG, RATE_LIMIT_SLEEP


class DistanceScoreMapper:
    """Maps raw 'distance' to a [0,1] heat score. Learns a monotone mapping on the fly."""
    def __init__(self, max_observed_distance: float = 5000.0, fit_min_samples: int = 40):
        self.max_dist = max_observed_distance
        self.fit_min_samples = fit_min_samples
        self._dist_buf = []
        self._score_buf = []
        self._iso = None  # IsotonicRegression, decreasing in distance

    def _fallback(self, distance: float) -> float:
        if distance <= 1:
            return 1.0
        s = 1.0 - math.log(max(distance, 1.0)) / math.log(self.max_dist)
        return float(np.clip(s, 0.0, 1.0))

    def add_observation(self, distance: float, mapped_score: float) -> None:
        if not np.isfinite(distance):
            return
        s = float(np.clip(mapped_score, 0.0, 1.0))
        if not np.isfinite(s):
            return
        self._dist_buf.append(float(distance))
        self._score_buf.append(s)
        if self._iso is None and len(self._dist_buf) >= self.fit_min_samples:
            x = np.asarray(self._dist_buf)
            y = np.asarray(self._score_buf)
            self._iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
            try:
                self._iso.fit(x, y)
            except Exception:
                self._iso = None  # stay safe

    def score(self, distance: float) -> float:
        if not np.isfinite(distance):
            return 0.0
        if self._iso is not None:
            try:
                y = float(self._iso.predict([distance])[0])
                if np.isfinite(y):
                    return float(np.clip(y, 0.0, 1.0))
            except Exception:
                pass
        return self._fallback(distance)


class ContextoAPI:
    def __init__(self, game_id: int, language: str = "en", base_url: str = "https://api.contexto.me"):
        self.game_id = game_id
        self.language = language
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.mapper = DistanceScoreMapper()
        self._last_time = 0.0

    @lru_cache(maxsize=8192)
    def query(self, word: str) -> Tuple[float, dict]:
        """
        Query the Contexto API for a single word.

        Returns:
          (mapped_score, raw_dict)
            - mapped_score \in [0,1], monotone decreasing in distance
            - raw_dict contains the API fields (e.g., distance, lemma, word; may include rank/correct on some deployments)

        Notes:
          - 404 is returned via raw_dict {"404": True}
          - A distance of 0 or 1 can indicate the true answer, depending on the server. Do not rely on an exact value here;
            use the robust detection in `make_api_feedback_fn`.
        """
        now = time.time()
        # Simple client-side rate limit
        from .config import RATE_LIMIT_SLEEP  # keep consistent with your config
        if now - self._last_time < RATE_LIMIT_SLEEP:
            time.sleep(RATE_LIMIT_SLEEP)

        url = f"{self.base_url}/machado/{self.language}/game/{self.game_id}/{word}"
        try:
            resp = self.session.get(url, timeout=5)
            self._last_time = time.time()
            if resp.status_code == 404:
                return 0.0, {"404": True}
            resp.raise_for_status()
            data = resp.json()
            distance = data.get("distance", float("inf"))
            mapped = self.mapper.score(distance)
            self.mapper.add_observation(distance, mapped)
            return float(mapped), data
        except Exception as e:
            print(f"[WARN] API error for '{word}': {e}")
            time.sleep(0.5)
            return 0.0, {}


def make_api_feedback_fn(api: "ContextoAPI", bad_words: Set[str]):
    """
    Returns a function(word) -> (score: float, state: dict)

    Behavior:
      - Calls the Contexto API via `api.query(word)` which returns (mapped_score, raw_dict).
      - Marks words that 404 or map to non-finite/zero scores as "bad" (added to `bad_words`).
      - Detects an exact hit robustly:
          * Many Contexto deployments return distance == 0 for the answer.
          * Some return distance == 1 for the answer.
        We therefore treat any finite numeric distance <= 1 as an exact hit.
      - Also honors optional API fields if present:
          * `correct` / `is_answer` truthy
          * `rank == 1`

    Returns:
      - score: float in [0, 1] (fallback-safe)
      - state: dict with key 'hit' (bool)
    """
    state = {"hit": False}

    def fn(word: str) -> Tuple[float, dict]:
        # Early-out on previously marked bad tokens
        if word in bad_words:
            return 0.0, state

        score, raw = api.query(word)

        # Sanitize score
        if not np.isfinite(score):
            score = 0.0

        # Mark bad words: explicit 404 or effectively-useless scores
        if raw.get("404"):
            bad_words.add(word)
        if score <= 0.0:
            bad_words.add(word)

        # --- Robust exact-hit detection ---
        # Prefer explicit flags if available
        try:
            # Flags some deployments may expose
            flag_correct = bool(raw.get("correct")) or bool(raw.get("is_answer"))
            rank = raw.get("rank", None)
            rank_is_one = isinstance(rank, (int, float)) and int(rank) == 1
        except Exception:
            flag_correct = False
            rank_is_one = False

        # Fallback to distance threshold (covers distance==0 and distance==1 cases)
        dist = raw.get("distance", None)
        dist_hit = False
        if isinstance(dist, (int, float)):
            try:
                dist_hit = float(dist) <= 1.0
            except Exception:
                dist_hit = False

        if flag_correct or rank_is_one or dist_hit:
            state["hit"] = True
            print("[SUCCESS-DETECTED] Exact hit. Halting after this step.")

        print(f"Guess '{word}': raw={raw} mapped_score={score:.4f}")
        return float(score), state

    return fn
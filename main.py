import sqlite3
import numpy as np
import random
import math
import time
import json
import os
from typing import List, Tuple, Set, Callable, Optional
from functools import lru_cache
from datetime import datetime

import requests
from sentence_transformers import SentenceTransformer
from wordfreq import top_n_list

# ---------- Configurable constants ----------
DB_PATH = "contexto_vocab.db"
GAME_ID_FILE = "last_game_id.txt"  # persists the last used ID
RESULTS_DIR = "results"
VOCAB_SIZE = 40000
SEED_COUNT = 5
NEIGHBOR_K = 100
INITIAL_TEMPERATURE = 1.0
TEMPERATURE_DECAY = 0.95  # per step
MIN_TEMPERATURE = 0.2
EXPLOIT_BASE = 0.6  # will adapt upward
UCB_ALPHA = 1.0
MAX_ITERS = 500
API_BASE = "https://api.contexto.me"
LANG = "en"
RATE_LIMIT_SLEEP = 0.1  # polite throttle
BAD_WORDS_CACHE = "bad_words.json"  # store permanently rejected words

# ---------- Utilities ----------

def load_last_game_id() -> int:
    if os.path.exists(GAME_ID_FILE):
        with open(GAME_ID_FILE, "r") as f:
            try:
                return int(f.read().strip())
            except:
                return 0
    return 0

def bump_and_store_game_id() -> int:
    last = load_last_game_id()
    new = last + 1
    with open(GAME_ID_FILE, "w") as f:
        f.write(str(new))
    return new

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ---------- Embedding model ----------

class EmbeddingModelHF:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# ---------- Vocabulary persistence ----------

class VocabDB:
    def __init__(self, path: str = DB_PATH):
        self.conn = sqlite3.connect(path)
        self._init_schema()
    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            embedding BLOB
        );
        """)
        self.conn.commit()
    def upsert_word(self, word: str, embedding: np.ndarray):
        cur = self.conn.cursor()
        emb_bytes = embedding.astype(np.float32).tobytes()
        cur.execute("""
        INSERT INTO vocab(word, embedding)
        VALUES(?, ?)
        ON CONFLICT(word) DO UPDATE SET embedding=excluded.embedding;
        """, (word, emb_bytes))
        self.conn.commit()
    def get_all(self) -> Tuple[List[str], np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT word, embedding FROM vocab;")
        rows = cur.fetchall()
        words, mats = [], []
        for word, blob in rows:
            arr = np.frombuffer(blob, dtype=np.float32)
            words.append(word)
            mats.append(arr)
        if not mats:
            return [], np.zeros((0, 0))
        mat = np.vstack(mats)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mat = mat / norms
        return words, mat
    def close(self):
        self.conn.close()

# ---------- API feedback ----------

class DistanceScoreMapper:
    def __init__(self, max_observed_distance: float = 5000.0):
        self.max_dist = max_observed_distance
    def score(self, distance: float) -> float:
        if distance <= 1:
            return 1.0
        s = 1.0 - math.log(distance) / math.log(self.max_dist)
        return max(0.0, min(1.0, s))

class ContextoAPI:
    def __init__(self, game_id: int, language: str = LANG, base_url: str = API_BASE):
        self.game_id = game_id
        self.language = language
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.mapper = DistanceScoreMapper()
        self._last_time = 0.0
    @lru_cache(maxsize=8192)
    def query(self, word: str) -> Tuple[float, dict]:
        now = time.time()
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
            sc = self.mapper.score(distance)
            return sc, data
        except Exception as e:
            # network/transient error: warn and return low score
            print(f"[WARN] API error for '{word}': {e}")
            time.sleep(0.5)
            return 0.0, {}

def make_api_feedback_fn(api: ContextoAPI, bad_words: Set[str]) -> Callable[[str], float]:
    def fn(word: str) -> float:
        if word in bad_words:
            return 0.0
        score, raw = api.query(word)
        if raw.get("404"):
            bad_words.add(word)
        print(f"Guess '{word}': raw={raw} mapped_score={score:.4f}")
        return score
    return fn

# ---------- Hybrid solver with adaptive temperature & explore/exploit ----------

class HybridSolver:
    def __init__(self, vocab: List[str], emb_matrix: np.ndarray,
                 neighbor_k: int = NEIGHBOR_K, ucb_alpha: float = UCB_ALPHA,
                 initial_temp: float = INITIAL_TEMPERATURE, exploit_base: float = EXPLOIT_BASE):
        self.vocab = vocab
        self.emb = emb_matrix  # normalized
        self.n = len(vocab)
        self.best_idx: Optional[int] = None
        self.best_sim = -np.inf
        self.history: List[Tuple[int, float]] = []
        self.neighbor_k = neighbor_k
        self.ucb_alpha = ucb_alpha
        self.temperature = initial_temp
        self.exploit_base = exploit_base  # base probability to exploit
        sims = self.emb @ self.emb.T
        self.neighbors = [np.argsort(-sims[i])[: neighbor_k] for i in range(self.n)]
        self.cluster_stats = {}  # center_idx -> (total_improve, count)

    def update(self, idx: int, sim: float):
        previous_best = self.best_sim
        if sim > self.best_sim:
            self.best_sim = sim
            self.best_idx = idx
        improvement = max(0.0, sim - previous_best)
        tot, cnt = self.cluster_stats.get(idx, (0.0, 0))
        self.cluster_stats[idx] = (tot + improvement, cnt + 1)
        self.history.append((idx, sim))
        # decay temperature slowly as confidence grows
        self.temperature = max(
            MIN_TEMPERATURE, self.temperature * TEMPERATURE_DECAY
        )

    def _exploit_pool(self, excluded: Set[int]) -> List[int]:
        if self.best_idx is None:
            return []
        return [i for i in self.neighbors[self.best_idx] if i not in excluded]

    def _explore_pool(self) -> List[int]:
        total = sum(cnt for _, cnt in self.cluster_stats.values()) + 1
        best_ucb = -1
        best_center = None
        for center, (tot, cnt) in self.cluster_stats.items():
            avg = tot / cnt if cnt > 0 else 0.0
            ucb = avg + self.ucb_alpha * math.sqrt(math.log(total) / (cnt + 1))
            if ucb > best_ucb:
                best_ucb = ucb
                best_center = center
        if best_center is not None:
            return self.neighbors[best_center]
        return []

    def propose_next(self, excluded: Set[int]) -> int:
        exploit_pool = self._exploit_pool(excluded)
        explore_pool = self._explore_pool()

        # Normalize types: make sure both are lists of ints
        if isinstance(exploit_pool, np.ndarray):
            exploit_pool = exploit_pool.tolist()
        if isinstance(explore_pool, np.ndarray):
            explore_pool = explore_pool.tolist()

        candidate_set = set()

        # Decide whether to exploit or explore (adaptive)
        use_exploit = random.random() < (self.exploit_base + (self.best_sim if self.best_sim > 0 else 0) * 0.2)

        if use_exploit and exploit_pool:
            candidate_set.update(exploit_pool)
        elif explore_pool:
            candidate_set.update(explore_pool)

        # Fallback to any non-excluded if empty
        if not candidate_set:
            candidate_set = {i for i in range(self.n) if i not in excluded}
        else:
            candidate_set = {i for i in candidate_set if i not in excluded}
            if not candidate_set:
                candidate_set = {i for i in range(self.n) if i not in excluded}

        pool = list(candidate_set)
        if self.best_idx is not None:
            sims = self.emb[pool] @ self.emb[self.best_idx]
            # sims is an array; pick highest
            choice = pool[int(np.argmax(sims))]
        else:
            choice = random.choice(pool)
        return choice


# ---------- Vocabulary ingestion ----------

def build_or_refresh_vocab(db: VocabDB, embedder: EmbeddingModelHF, size: int):
    print(f"[INFO] Ingesting top {size} words (may take a moment)...")
    words = top_n_list("en", size)
    batch_size = 1024
    for i in range(0, len(words), batch_size):
        chunk = words[i : i + batch_size]
        embs = embedder.encode_batch(chunk)
        for w, emb in zip(chunk, embs):
            db.upsert_word(w, emb.astype(np.float32))
        print(f"  indexed {i + len(chunk):,}/{len(words):,}")
    print("[INFO] Vocabulary ingestion complete.")

# ---------- Persistence for bad words ----------

def load_bad_words() -> Set[str]:
    if os.path.exists(BAD_WORDS_CACHE):
        try:
            with open(BAD_WORDS_CACHE, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_bad_words(bad: Set[str]):
    with open(BAD_WORDS_CACHE, "w") as f:
        json.dump(sorted(list(bad)), f)

# ---------- Run / driver logic ----------

def play_game_and_record():
    ensure_dir(RESULTS_DIR)
    # Load current game ID (will increment only on success)
    current_game_id = load_last_game_id()
    game_id = current_game_id + 1  # Next game to attempt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"game_{game_id}_{timestamp}.jsonl")

    # Setup
    embedder = EmbeddingModelHF()
    db = VocabDB(DB_PATH)
    # Only build vocabulary the first time; if existing, skip heavy re-ingest
    vocab, emb_matrix = db.get_all()
    if len(vocab) < VOCAB_SIZE:
        build_or_refresh_vocab(db, embedder, VOCAB_SIZE)
        vocab, emb_matrix = db.get_all()
    print(f"[INFO] Loaded vocab size={len(vocab)} emb_dim={emb_matrix.shape[1]}")

    api = ContextoAPI(game_id=game_id, language=LANG, base_url=API_BASE)
    bad_words = load_bad_words()
    feedback_fn = make_api_feedback_fn(api, bad_words)
    solver = HybridSolver(vocab, emb_matrix)

    excluded = set()
    trajectory = []
    start_time = time.time()
    last_best = -1.0
    stagnant = 0

    # smart seed: choose SEED_COUNT words spread out (max-min)
    seeds = []
    if len(vocab) >= SEED_COUNT:
        # pick first seed randomly, then iteratively max-distance
        first = random.randrange(len(vocab))
        seeds.append(first)
        for _ in range(SEED_COUNT - 1):
            remaining = [i for i in range(len(vocab)) if i not in seeds]
            # choose the one maximizing min distance to current seeds
            dists = []
            for cand in remaining:
                sim_to_seeds = [float(emb_matrix[cand] @ emb_matrix[s]) for s in seeds]
                min_sim = min(sim_to_seeds)
                dists.append((cand, min_sim))
            # pick candidate with smallest similarity (i.e., most distant)
            next_idx = min(dists, key=lambda x: x[1])[0]
            seeds.append(next_idx)
    else:
        seeds = [0]

    # seed guesses
    for idx in seeds:
        if idx in excluded:
            continue
        word = vocab[idx]
        score = feedback_fn(word)
        solver.update(idx, score)
        excluded.add(idx)
        trajectory.append({"guess": word, "score": score, "best_so_far": vocab[solver.best_idx]})
        if score >= 0.999:
            break

    for step in range(1, MAX_ITERS + 1):
        if solver.best_sim >= 0.999:
            break
        guess_idx = solver.propose_next(excluded)
        word = vocab[guess_idx]
        score = feedback_fn(word)
        solver.update(guess_idx, score)
        excluded.add(guess_idx)
        trajectory.append({"guess": word, "score": score, "best_so_far": vocab[solver.best_idx]})

        print(f"[STEP {step}] guess='{word}' score={score:.4f} best='{vocab[solver.best_idx]}'={solver.best_sim:.4f}")

        # early stop
        if score >= 0.999:
            print("[SUCCESS] reached perfect/near-perfect.")
            break

        # stagnation detection, diversification
        if solver.best_sim <= last_best + 1e-5:
            stagnant += 1
        else:
            stagnant = 0
            last_best = solver.best_sim

        if stagnant >= 5:
            if solver.best_idx is not None:
                sims_to_best = emb_matrix @ emb_matrix[solver.best_idx]
                far_idx = int(np.argmin(sims_to_best))
                if far_idx not in excluded:
                    print("[DIVERSIFY] using distant word:", vocab[far_idx])
                    w2 = vocab[far_idx]
                    sc2 = feedback_fn(w2)
                    solver.update(far_idx, sc2)
                    excluded.add(far_idx)
                    trajectory.append({"guess": w2, "score": sc2, "best_so_far": vocab[solver.best_idx]})
                    stagnant = 0

    duration = time.time() - start_time
    is_successful = solver.best_sim >= 0.999
    
    final = {"game_id": game_id,
             "timestamp": timestamp,
             "duration_seconds": duration,
             "best_word": vocab[solver.best_idx] if solver.best_idx is not None else None,
             "best_score": solver.best_sim,
             "trajectory": trajectory,
             "successful": is_successful}

    # persist
    with open(result_file, "a") as f:
        f.write(json.dumps(final) + "\n")
    print(f"[DONE] Results appended to {result_file}")

    # Only increment game ID if successful
    if is_successful:
        with open(GAME_ID_FILE, "w") as f:
            f.write(str(game_id))
        print(f"[SUCCESS] Game #{game_id} completed successfully! Game ID incremented.")
    else:
        print(f"[PARTIAL] Game #{game_id} ended without perfect score ({solver.best_sim:.4f}). Game ID not incremented.")

    # persist bad words
    save_bad_words(bad_words)
    db.close()

if __name__ == "__main__":
    play_game_and_record()

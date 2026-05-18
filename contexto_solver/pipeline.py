import json
import re
import time
from typing import Optional

import numpy as np

from .api import ContextoAPI, make_api_feedback_fn
from .centroid import RankWeightedCentroid
from .config import (
    API_BASE,
    CORRECTION_BIAS_FILE,
    CORRECTION_MATRIX_FILE,
    CORRECTION_META_FILE,
    CORRECTION_MIN_GAMES,
    LANG,
    MAX_GUESSES_PER_GAME,
    PROBE_WORDS,
    VOCAB_SIZE,
)
from .correction import CorrectionStore
from .embedding import GloveEmbedding
from .io import (
    ensure_dir,
    load_bad_words,
    load_current_game_id,
    now_timestamp,
    result_filepath,
    save_bad_words,
    set_current_game_id,
    set_last_successful_game_id,
)
from .solver import HybridSolver

_WORD_OK_RE = re.compile(r"^[a-z]{2,20}$")


def _valid_vocab_mask(words):
    return np.array([bool(_WORD_OK_RE.match(w.lower())) for w in words], dtype=bool)


def _prepare_vocab(embedder: GloveEmbedding):
    vocab, X = embedder.vocab_and_matrix()
    if VOCAB_SIZE and len(vocab) > VOCAB_SIZE:
        vocab = vocab[:VOCAB_SIZE]
        X = X[:VOCAB_SIZE]
    return vocab, X.astype(np.float32)



def play_game_and_record() -> None:
    ensure_dir("results")

    game_id = load_current_game_id() + 1
    timestamp = now_timestamp()
    result_file = result_filepath(game_id, timestamp)

    embedder = GloveEmbedding()
    vocab, emb_matrix = _prepare_vocab(embedder)

    mask = _valid_vocab_mask(vocab)
    if not mask.any():
        raise RuntimeError("All vocab filtered out by validity mask.")
    vocab = [w for w, m in zip(vocab, mask) if m]
    emb_matrix = emb_matrix[mask]
    print(f"[INFO] vocab={len(vocab)} dim={emb_matrix.shape[1]}")

    api = ContextoAPI(game_id=game_id, language=LANG, base_url=API_BASE)
    bad_words = load_bad_words()
    feedback_fn = make_api_feedback_fn(api, bad_words)

    solver = HybridSolver(vocab, emb_matrix)
    centroid_est = RankWeightedCentroid(emb_matrix)
    correction = CorrectionStore(
        CORRECTION_BIAS_FILE, CORRECTION_MATRIX_FILE, CORRECTION_META_FILE, CORRECTION_MIN_GAMES
    )

    excluded: set = {i for i, w in enumerate(vocab) if w in bad_words}

    # Semantic probes: play diverse anchor words first to rapidly identify
    # the answer's semantic domain before adaptive search starts.
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    probe_queue: list = [
        word_to_idx[w] for w in PROBE_WORDS
        if w in word_to_idx and word_to_idx[w] not in excluded
    ]

    trajectory = []
    start_time = time.time()
    hit_seen = False
    pre_answer_centroid: Optional[np.ndarray] = None
    seen_distances: set = set()  # track raw_distances seen this game

    step = 0
    while not hit_seen and len(excluded) < len(vocab) and step < MAX_GUESSES_PER_GAME:
        step += 1

        if probe_queue:
            guess_idx = probe_queue.pop(0)
        else:
            corrected = correction.correct(centroid_est.get())
            guess_idx = solver.propose_next(excluded, corrected_centroid=corrected)
        word = vocab[guess_idx]

        score, st = feedback_fn(word)
        raw_dist = st.get("raw_distance", float("inf"))
        excluded.add(guess_idx)
        hit_seen = hit_seen or st.get("hit", False)

        # Deduplicate inflected forms: same raw_distance means same lemma,
        # so no new information — skip trajectory but still exclude from vocab.
        if np.isfinite(raw_dist) and raw_dist > 1:
            dist_key = round(raw_dist)
            if dist_key in seen_distances:
                step -= 1  # don't count wasted guess
                continue
            seen_distances.add(dist_key)

        solver.update(guess_idx, score, raw_distance=raw_dist)
        centroid_est.update(guess_idx, raw_dist)

        trajectory.append({
            "iter": step,
            "guess": word,
            "score": float(score),
            "raw_distance": raw_dist if np.isfinite(raw_dist) else -1,
            "best_so_far": vocab[solver.best_idx] if solver.best_idx is not None else None,
        })

        lam = solver._lambda()
        print(
            f"[{step}] '{word}' dist={raw_dist:.0f} score={score:.3f}"
            f" best='{vocab[solver.best_idx] if solver.best_idx is not None else '?'}'"
            f"={solver.best_sim:.3f} lam={lam:.2f}"
        )

        if hit_seen:
            print("[SUCCESS] exact hit.")
            break

        # Keep a rolling snapshot of centroid before each new guess (training signal)
        if centroid_est.get() is not None:
            pre_answer_centroid = centroid_est.get().copy()

    duration = time.time() - start_time
    is_successful = bool(hit_seen)
    capped = step >= MAX_GUESSES_PER_GAME and not is_successful
    if capped:
        print(f"[CAP] game #{game_id}: hit {MAX_GUESSES_PER_GAME}-guess limit. Recording as PARTIAL.")

    final = {
        "game_id": game_id,
        "timestamp": timestamp,
        "duration_seconds": float(duration),
        "best_word": vocab[solver.best_idx] if solver.best_idx is not None else None,
        "best_score": float(solver.best_sim),
        "trajectory": trajectory,
        "successful": is_successful,
    }
    with open(result_file, "a") as f:
        f.write(json.dumps(final) + "\n")
    print(f"[DONE] {result_file}")

    save_bad_words(bad_words)
    set_current_game_id(game_id)

    if is_successful:
        set_last_successful_game_id(game_id)
        if pre_answer_centroid is not None and solver.best_idx is not None:
            correction.record_game(pre_answer_centroid, emb_matrix[solver.best_idx])
            print(f"[CORRECTION] game #{game_id} recorded. total={correction.game_count}")
        print(f"[SUCCESS] game #{game_id} solved in {step} guesses.")
    else:
        print(f"[PARTIAL] game #{game_id}: {step} guesses, best={solver.best_sim:.4f}")

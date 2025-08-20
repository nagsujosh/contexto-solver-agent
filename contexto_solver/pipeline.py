import json
import random
import time
import re
from typing import List

import numpy as np

from .api import ContextoAPI, make_api_feedback_fn
from .config import (
    API_BASE,
    LANG,
    NEIGHBOR_K,
    SEED_COUNT,
    UCB_ALPHA,
    VOCAB_SIZE,
    EARLY_PROBES,
    EARLY_PROBE_TURNS,
)
from .embedding import GloveEmbedding
from .io import (
    ensure_dir,
    load_bad_words,
    load_current_game_id,
    set_current_game_id,
    set_last_successful_game_id,
    now_timestamp,
    result_filepath,
    save_bad_words,
)
from .solver import HybridSolver

_WORD_OK_RE = re.compile(r"^[a-z]{2,20}$")  # only ascii letters, len 2..20


def _valid_vocab_mask(words):
    mask = []
    for w in words:
        ok = bool(_WORD_OK_RE.match(w.lower()))
        mask.append(ok)
    return np.array(mask, dtype=bool)


def _pick_seed_indices_farthest(emb_matrix: np.ndarray, count: int) -> List[int]:
    n = emb_matrix.shape[0]
    if n == 0:
        return []
    first = random.randrange(n)
    seeds = [first]
    # greedy max-min coverage
    min_sim = emb_matrix @ emb_matrix[first]
    for _ in range(count - 1):
        idx = int(np.argmin(min_sim))
        seeds.append(idx)
        min_sim = np.minimum(min_sim, emb_matrix @ emb_matrix[idx])
    return seeds


def _prepare_vocab(embedder: GloveEmbedding):
    vocab, X = embedder.vocab_and_matrix()
    if VOCAB_SIZE and len(vocab) > VOCAB_SIZE:
        vocab = vocab[:VOCAB_SIZE]
        X = X[:VOCAB_SIZE]
    X = X.astype(np.float32)
    return vocab, X


def play_game_and_record():
    ensure_dir("results")

    current_game_id = load_current_game_id()
    game_id = current_game_id + 1
    timestamp = now_timestamp()
    result_file = result_filepath(game_id, timestamp)

    embedder = GloveEmbedding()
    vocab, emb_matrix = _prepare_vocab(embedder)

    # Filter out non-words / noisy tokens up front
    mask = _valid_vocab_mask(vocab)
    if not mask.any():
        raise RuntimeError("All vocab filtered out by validity mask.")
    vocab = [w for w, m in zip(vocab, mask) if m]
    emb_matrix = emb_matrix[mask]

    if not vocab or emb_matrix.size == 0:
        raise RuntimeError("Vocabulary is empty after filtering. Check embeddings and filters.")
    print(f"[INFO] Loaded vocab size={len(vocab)} emb_dim={emb_matrix.shape[1]}")

    api = ContextoAPI(game_id=game_id, language=LANG, base_url=API_BASE)
    bad_words = load_bad_words()
    feedback_fn = make_api_feedback_fn(api, bad_words)
    solver = HybridSolver(vocab, emb_matrix, neighbor_k=NEIGHBOR_K, ucb_alpha=UCB_ALPHA)

    excluded = {i for i, w in enumerate(vocab) if w in bad_words}
    trajectory = []
    start_time = time.time()
    last_best = -1.0
    stagnant = 0
    hit_seen = False

    # --- Early probes ---
    probe_words = [w for w in EARLY_PROBES if w in set(vocab)]
    random.shuffle(probe_words)
    probe_words = probe_words[: EARLY_PROBE_TURNS]

    # Greedy farthest seeds
    far_seeds = _pick_seed_indices_farthest(emb_matrix, SEED_COUNT)

    # Make seed index list (probes + far seeds)
    seed_idxs = []
    seen = set()
    for w in probe_words:
        i = vocab.index(w)
        if i not in excluded and i not in seen:
            seed_idxs.append(i)
            seen.add(i)
    for i in far_seeds:
        if i not in excluded and i not in seen:
            seed_idxs.append(i)
            seen.add(i)

    # Evaluate seeds
    for idx in seed_idxs:
        if len(trajectory) >= EARLY_PROBE_TURNS + SEED_COUNT:
            break
        word = vocab[idx]
        score, st = feedback_fn(word)
        solver.update(idx, score)
        excluded.add(idx)
        hit_seen = hit_seen or st.get("hit", False)
        trajectory.append({
            "iter": len(trajectory)+1,
            "mode": "seed",
            "guess": word,
            "score": float(score),
            "best_so_far": vocab[solver.best_idx] if solver.best_idx is not None else None
        })
        if st.get("hit", False):
            print("[SUCCESS] exact hit reported by API.")
            break

    step = 0
    # Main loop: run until hit OR vocab exhausted (no MAX_ITERS limit)
    while not hit_seen and len(excluded) < len(vocab):
        step += 1
        guess_idx = solver.propose_next(excluded)
        word = vocab[guess_idx]
        score, st = feedback_fn(word)
        solver.update(guess_idx, score)
        excluded.add(guess_idx)
        hit_seen = hit_seen or st.get("hit", False)
        trajectory.append({
            "iter": len(trajectory)+1,
            "mode": "main",
            "guess": word,
            "score": float(score),
            "best_so_far": vocab[solver.best_idx] if solver.best_idx is not None else None
        })

        print(
            f"[STEP {step}] guess='{word}' score={score:.4f} best='{vocab[solver.best_idx] if solver.best_idx is not None else None}'={solver.best_sim:.4f}"
        )

        if hit_seen:
            print("[SUCCESS] exact hit reported by API.")
            break

        # Stagnation tracking
        if solver.best_sim <= last_best + 1e-5:
            stagnant += 1
        else:
            stagnant = 0
            last_best = solver.best_sim

        # Diversify via farthest heuristic (safe: vocab already clean)
        if stagnant >= 5 and solver.best_idx is not None and len(excluded) < len(vocab):
            sims_to_best = emb_matrix @ emb_matrix[solver.best_idx]
            far_idx = int(np.argmin(sims_to_best))
            if far_idx not in excluded:
                print("[DIVERSIFY] using distant word:", vocab[far_idx])
                w2 = vocab[far_idx]
                sc2, st2 = feedback_fn(w2)
                solver.update(far_idx, sc2)
                excluded.add(far_idx)
                hit_seen = hit_seen or st2.get("hit", False)
                trajectory.append({
                    "iter": len(trajectory)+1,
                    "mode": "jump",
                    "guess": w2,
                    "score": float(sc2),
                    "best_so_far": vocab[solver.best_idx] if solver.best_idx is not None else None
                })
                stagnant = 0
                if hit_seen:
                    print("[SUCCESS] exact hit reported by API.")
                    break

    duration = time.time() - start_time
    is_successful = bool(hit_seen)

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
    print(f"[DONE] Results appended to {result_file}")

    # Persist bad words
    save_bad_words(bad_words)

    # Update game ID tracking files
    set_current_game_id(game_id)
    if is_successful:
        set_last_successful_game_id(game_id)
        print(f"[SUCCESS] Game #{game_id} completed successfully! Game ID incremented.")
    else:
        print(
            f"[PARTIAL] Game #{game_id} ended without perfect score ({solver.best_sim:.4f}). Current game ID updated, last successful unchanged."
        )

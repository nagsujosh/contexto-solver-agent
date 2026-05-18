# Contexto Solver Agent

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/math-NumPy%2FSciPy-blue.svg)
![AI](https://img.shields.io/badge/embeddings-GloVe_6B_300d-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

A fully automated solver for the **[Contexto](https://contexto.me)** daily word game. Given a game ID, the solver queries the Contexto API and converges on the answer using GloVe 6B 300d embeddings, an adaptive rank-weighted centroid, MMR-based word selection, and a cross-game correction store — reaching the answer in a **median of ~60 guesses** with a **100% success rate** across 145 recorded games.

---

## How It Works

Contexto returns a semantic **distance rank** for every guessed word (rank 0 = the answer). The solver's job is to find rank 0 efficiently by estimating where the answer sits in GloVe embedding space and selecting guesses that maximize information gain.

### 1. Semantic Probes

At the start of each game, 15 curated anchor words spanning distinct semantic domains (animals, food, music, buildings, water, sport, money, medicine, books, machines, religion, geography, people, geology, color) are guessed first. Within 2–3 guesses the solver typically identifies the answer's domain and shifts into exploitation mode.

### 2. Rank-Weighted Centroid

After each guess, the solver updates its estimate of the answer's direction in embedding space:

$$w_i = \frac{1}{\max(1,\, d_i)}, \qquad \hat{c} = \frac{\sum_i w_i x_i}{\left\|\sum_i w_i x_i\right\|}$$

where $d_i$ is the raw Contexto distance for word $i$. Words close to the answer receive weight $\approx 1$; words far away receive weight $\approx 0$, so the centroid converges toward the answer regardless of irrelevant guesses.

### 3. Adaptive MMR Selection

The next guess is chosen via **Maximal Marginal Relevance**, balancing exploitation (following the centroid) against diversity (avoiding already-explored regions):

$$\text{score}(w) = \lambda \cdot \text{sim}(w,\, \hat{c}) - (1-\lambda) \cdot \text{sim}(w,\, \bar{x}_{\text{top5}})$$

where the first term is centroid relevance and the second is a redundancy penalty (mean similarity to the top-5 tried words).

The trade-off parameter $\lambda$ is driven by the best mapped score seen so far:

$$\lambda = \text{clip}\left(\frac{s_{\text{best}}}{0.5},\ 0,\ 1\right)$$

- **λ = 0**: no useful signal yet — pick the word most dissimilar to all tried words (full diversity)
- **λ = 1**: strong signal — pick the word most similar to the centroid (full exploitation)

The score mapper converts raw Contexto rank to a $[0,1]$ score using a log-scale fallback over the full 70 000-word distance range, refined by an isotonic regression fitted on in-game observations. This means a word at rank 3 000 already pushes $\lambda$ to ~0.56, enabling exploitation well before reaching a top-100 word.

### 4. Progressive Candidate Pool

The set of candidate words passed to MMR grows as the game lengthens, so the solver does not get permanently stuck after exhausting the nearest GloVe neighbors:

| Guesses elapsed | Candidate pool |
|----------------|----------------|
| ≤ 100 | Top-500 GloVe neighbors of centroid |
| 101–400 | Top-3 000 |
| 401–800 | Top-15 000 |
| > 800 | Full vocabulary |

### 5. Last-Mile Combined Search

When the best rank reached is ≤ 5 (within one or two steps of the answer), the solver switches to a combined similarity query:

$$\text{score}(w) = 0.5 \cdot \text{sim}(w,\, \hat{c}) + 0.5 \cdot \text{sim}(w,\, x_{\text{best}})$$

over the top-20 000 candidates. This handles the case where the answer is not the closest GloVe neighbor of the centroid but is a close GloVe neighbor of the best-ranked word found so far.

### 6. Cross-Game Correction Store

The solver learns from each completed game to correct the systematic offset between the rank-weighted centroid and the true answer direction.

**Phase 1 (fewer than 20 completed games)** — exponential moving average bias:

$$\text{bias} \leftarrow 0.7 \cdot \text{bias} + 0.3 \cdot (t - \hat{c})$$

Applied with strength $\alpha = \min(0.4,\, N/50)$ where $N$ is the number of games recorded.

**Phase 2 (≥ 20 games)** — identity-regularized ridge regression correction matrix:

$$M = \left(X^\top X + \lambda I\right)^{-1}\!\left(X^\top Y + \lambda I\right), \quad \lambda = 1.0$$

where rows of $X$ are final centroid estimates and rows of $Y$ are the true answer embeddings from completed games. Regularizing toward $\mathbf{I}$ (not toward $\mathbf{0}$) ensures that directions where training data is sparse map to themselves rather than to noise. With plain least-squares on the underdetermined 300×300 system (fewer than 300 training games), the condition number reaches ~4×10¹⁰ and distorts centroids by ~87°; with $\lambda=1$ the condition number stays ≈4 and distortion ≈11°.

---

## Performance

Across 145 recorded games (100% success on the most recent 25):

| Metric | Value |
|--------|-------|
| Success rate | 100% |
| Median guesses | ~62 |
| Average guesses | ~97 |
| Best game | 26 guesses |
| Worst game (recent) | 431 guesses |

Remaining hard cases occur when the answer is semantically distant from all 15 probes in GloVe space and the Contexto model places it in a different cluster than GloVe does.

> See [RESULTS.md](RESULTS.md) for the full per-game breakdown and rolling average.

---

## Project Structure

```
contexto-solver-agent/
├── contexto_solver/
│   ├── api.py          # Contexto HTTP client + distance→score mapper (isotonic regression)
│   ├── centroid.py     # Rank-weighted centroid estimator
│   ├── config.py       # All tunables: vocab size, probe words, correction thresholds, etc.
│   ├── correction.py   # Cross-game correction store (EMA bias → ridge matrix)
│   ├── embedding.py    # GloVe loader, PCA whitening, unit normalization
│   ├── io.py           # Filesystem helpers (bad-word cache, game-ID tracking, JSONL)
│   ├── pipeline.py     # Game loop: probes → adaptive MMR → correction update
│   └── solver.py       # HybridSolver: MMR scoring, progressive pool, last-mile search
├── analyze_results.py  # Terminal + RESULTS.md report generator
├── automation_script.py # Batch runner (--start N --end M)
├── main.py             # Single-game entry point
├── tests/              # 20 unit tests covering centroid, correction, and solver
├── results/            # Per-game JSONL traces (one file per run)
├── RESULTS.md          # Auto-generated analysis report
└── centroid_{bias,ch,th,matrix,meta}.npy/.json  # Persisted cross-game learning state
```

---

## Quick Start

```bash
git clone <repo>
cd contexto-solver-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download GloVe 6B 300d and place as glove.6B.300d.txt in the repo root.
# Source: https://nlp.stanford.edu/data/glove.6B.zip

python main.py                                      # Play today's game
python automation_script.py                         # Play today's game + update RESULTS.md
python automation_script.py --start 1180 --end 1204 # Batch replay by game ID
python analyze_results.py                           # Regenerate RESULTS.md from all traces
```

---

## Configuration

All tunables live in `contexto_solver/config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `VOCAB_SIZE` | 80 000 | GloVe words loaded (sorted by frequency) |
| `N_PROBES` | 15 | Semantic anchor words played at game start |
| `PROBE_WORDS` | see config | The 15 anchor words spanning semantic domains |
| `TOP_K_BEST` | 5 | Top-k tried words used for redundancy penalty |
| `CORRECTION_MIN_GAMES` | 20 | Games before upgrading EMA bias → ridge matrix |
| `MAX_GUESSES_PER_GAME` | 2 000 | Hard cap per game |
| `RATE_LIMIT_SLEEP` | 0.1 s | Client-side rate limiting |
| `EMB_CENTER` | True | Subtract mean before PCA whitening |
| `EMB_REMOVE_TOP_K` | 3 | Remove top-K principal directions (isotropy) |

---

## Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

20 tests covering: centroid estimator, correction store (EMA, ridge conditioning), and solver (MMR scoring, diversity mode, raw-distance tracking, last-mile search).

---

## License

MIT License (see `LICENSE`).

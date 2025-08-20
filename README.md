# Contexto Solver Agent

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![GitHub Actions](https://img.shields.io/badge/automation-GitHub_Actions-orange.svg)
![AI](https://img.shields.io/badge/AI-GloVe_6B_300d-purple.svg)
![Success Rate](https://img.shields.io/badge/success_rate-auto-green.svg)
![Games Played](https://img.shields.io/badge/games_played-auto-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

A fully automated solver for the **Contexto** daily word game. The agent combines a static distributional semantic space (GloVe 6B 300d) with an **exploration–exploitation controller** that alternates between local neighborhood search and UCB-guided cluster exploration. The system runs headlessly via GitHub Actions, persists full trajectories, and updates human-readable reports.

---

## 1. Abstract

Let the hidden target word be $w$. Contexto returns a **semantic distance** $d(w)$ for any guess $w$. We treat distances as **monotone signals** and map them to a bounded score $s(w)\in[0,1]$, then solve a sequential decision problem over a discrete vocabulary $V$ embedded in $\mathbb{R}^D$. At each round $t$, we select a candidate $w_t\in V\setminus H_t$ (where $H_t$ is the set of tried indices), observe $s_t$, update local/global statistics, and continue until an **exact hit**. Exactness is detected robustly by **$ \text{distance} \le 1$** (covers deployments that report 0 or 1 for the answer).

---

## 2. System Overview

```
             +-------------------+
             |  GloVe 6B 300d    |
             |  loader/whitener  |
             +---------+---------+
                       |
                 (V × D matrix, unit-norm, optional fp16)
                       |
+----------------------+-----------------------+
|                                              |
|      HybridSolver (controller)               |
|  - best-so-far index, temp τ                 |
|  - neighbor cache (top-K cosine)             |
|  - KMeans clusters + UCB over clusters       |
|  - proposal distribution (exploit/explore)   |
|                                              |
+----------------------+-----------------------+
                       |
                   candidate
                       |
              +--------v---------+
              |  Contexto API    |
              | /distance query  |
              +--------+---------+
                       |
              distance, metadata
                       |
              +--------v---------+
              | Distance→Score   |
              |  mapper          |
              |  (log prior +    |
              |  isotonic fit)   |
              +--------+---------+
                       |
                  s ∈ [0,1]
                       |
                +------v-------+
                |   Pipeline   |
                |  (orches.)   |
                +------+-------+
                       |
           JSONL trace, RESULTS.md, README stats
```

---

## 3. Embedding Space

### 3.1 Loading and Normalization

* Source: **GloVe 6B 300d** (`glove.6B.300d.txt`).
* For each word $w$ with raw vector $\tilde{x}_w\in\mathbb{R}^{300}$, we apply:

  * Optional mean-centering: $x'_w = \tilde{x}_w - \mu$.
  * Optional removal of top $K$ principal directions:

    $$
      X' = X - (X V_K) V_K^\top,\ \ \ V_K\in\mathbb{R}^{D\times K}.
    $$
  * Unit-normalization: $x_w = \frac{x'_w}{\|x'_w\|_2}$.
* Store as `float16` optionally to reduce memory.
* Valid vocabulary filter: ASCII letters, length in $[2,20]$ only.

**Similarity** is cosine via dot product: $\text{sim}(i,j) = x_i^\top x_j$.

### 3.2 Complexity and Memory

* Embedding matrix $X\in\mathbb{R}^{|V|\times D}$.
* Memory footprint $\approx |V|\cdot D \cdot \text{dtype}$.
* Neighborhood queries use `argpartition` for **top-K** similarity in $O(|V|)$ time per query.

---

## 4. Distance → Score Calibration

Contexto returns a nonnegative `distance`. We map this to a score $s \in [0,1]$ that is **monotone decreasing** in distance.

### 4.1 Log Prior (fallback)

For maximum observed distance hyperparameter $D_{\max}$,

$$
  s_\text{log}(d) = \mathrm{clip}\left(1 - \frac{\log(\max(d,1))}{\log(D_{\max})},\ 0,\ 1 \right),
$$

with the convention $s( d\le 1 ) = 1$.

### 4.2 On-the-fly Isotonic Regression

We maintain a buffer $\{(d_i, s_i)\}$ and fit a **decreasing isotonic** model $\hat{s}(d)\in[0,1]$ once enough samples accumulate. Prediction uses $\hat{s}$, with fallback to $s_\text{log}$ if the fit fails.

> Note: the buffer currently uses the mapped scores; the monotone fit refines the discretization while preserving monotonicity.

---

## 5. Controller: Hybrid Exploration–Exploitation

### 5.1 State

* `best_idx` and `best_sim`: index/value of the current best candidate in embedding space.
* `history`: list of $(\text{idx}, s)$.
* `temperature` $\tau$ with exponential decay: $\tau \leftarrow \max(\tau_{\min}, \tau \cdot \gamma)$.
* **Neighbor cache** for fast local proposals around `best_idx`.
* **KMeans** partitioning of $X$ into $K$ clusters with centers $c_k$; we maintain per-cluster statistics.

### 5.2 Cluster-Level UCB

For cluster $k$, maintain $(R_k, N_k)$ where $R_k$ aggregates **improvements** over `best_sim` and $N_k$ counts pulls. Let $T = \sum_k N_k + 1$.

$$
\text{UCB}_k \ = \ \underbrace{\frac{R_k}{\max(1,N_k)}}_{\text{exploitation}} \ + \ \alpha \sqrt{\frac{\log T}{\max(1,N_k)}} \quad (\alpha>0).
$$

We select the **max-UCB cluster** for exploration; within that cluster we rank members by $x_i^\top c_k$ and sample up to `CLUSTER_SAMPLE_K` viable candidates.

### 5.3 Local Exploitation Pool

Given `best_idx = b`, we precompute the top-`NEIGHBOR_K` nearest neighbors by cosine to form a local pool $\mathcal{N}(b)$.

### 5.4 Exploit vs Explore Decision

We form a probability of **exploitation** based on a base rate and the **gap** between the nearest neighbor and the tail of the local similarity distribution:

$$
p_\text{exploit} \ = \ \mathrm{clip}\big( \text{BASE} + \beta \cdot \max(0, \text{top1} - \overline{\text{tail}}),\ 0,\ p_{\max} \big).
$$

* `BASE` = `EXPLOIT_BASE`.
* The tail is the mean of neighbors 2–11 when available.
* This encourages exploitation when the local manifold is **peaky**.

### 5.5 Proposal Distribution

* If exploiting: candidate pool $P = \mathcal{N}(b)\setminus H$.
* Else exploring: candidate pool from the best-UCB cluster.
* If both empty: use the remaining vocabulary.
* With `best_idx` known, we sample with **Boltzmann weights** by similarity to the current best:

$$
\Pr(i \mid P, b) \ \propto \ \exp\Big( \frac{x_i^\top x_b}{\tau} \Big), \quad i \in P.
$$

If `best_idx` is `None`, pick uniformly at random.

### 5.6 Stagnation Handling

We track lack of improvement in `best_sim`. After $\ge 5$ stagnant steps, we **jump** to the farthest word (minimal cosine) from the current best to escape local basins.

---

## 6. Orchestration and Termination

### 6.1 Early Phase

* **Early probes**: a fixed set of high-level category words (e.g., `animal`, `city`, `food`, …) seeded if present in the vocab, limited by `EARLY_PROBE_TURNS`.
* **Diverse seeds**: greedy **max–min** coverage in cosine space to pick far-apart initial points.

### 6.2 Main Loop

For $t=1,2,\dots$

1. Propose next candidate index $i_t$.
2. Query API → raw `distance` and metadata.
3. Map to $s_t\in[0,1]$.
4. **Exact-hit detection**:

   * If `distance` ≤ 1 → **halt** (true solution).
   * Optional flags if present (`correct`, `is_answer`, or `rank==1`) are also treated as hits.
5. Update:

   * `best_idx`, `best_sim` if $s_t$ improves the best.
   * Cluster stats $(R_k, N_k)$ using **improvement**.
   * Temperature decay.
   * Mark 404/zero-score tokens as **bad** (blacklist).
6. Persist step to trajectory.

**Stop conditions**

* Exact hit (preferred)
* Exhausted vocabulary

---

## 7. Robust API Handling

* **Rate limiting**: minimal sleep between requests (`RATE_LIMIT_SLEEP`).
* **Caching**: LRU cache on `query(word)` to avoid duplicate HTTP calls.
* **404 handling**: words returning 404 or non-finite/zero scores are added to `bad_words.json` and ignored subsequently.
* **Answer detection**: treat `distance ≤ 1` as exact, covering deployments that emit 0 or 1.

---

## 8. I/O Contracts

### 8.1 Inputs

* `glove.6B.300d.txt` in repository root.
* Optional state files:

  * `current_game_id.txt`, `last_successful_game_id.txt`
  * `bad_words.json`

### 8.2 Outputs

* `results/game_<id>_<timestamp>.jsonl` (one JSON object per run):

  ```json
  {
    "game_id": 7,
    "timestamp": "YYYYMMDD_HHMMSS",
    "duration_seconds": 4.7,
    "best_word": "neighborhood",
    "best_score": 1.0,
    "trajectory": [
      { "iter": 1, "mode": "seed", "guess": "animal", "score": 0.03, "best_so_far": "animal" },
      ...
    ],
    "successful": true
  }
  ```
* `RESULTS.md`: aggregate stats + recent trajectories.
* `README.md`: badges and headline stats updated by automation.

---

## 9. Algorithms (concise pseudocode)

```text
LOAD GloVe embeddings → X ∈ ℝ^{|V|×D}, normalize, optional whitening
FILTER vocab by regex ⇒ V
INIT HybridSolver(V, X)
INIT ContextoAPI(game_id, lang)

# Seeds
S ← early_probes ∩ V  ∪  farthest_points(X, count=SEED_COUNT)
FOR i ∈ S:
  s_i, raw ← API.query(V[i]); s_i ← map_distance(raw.distance)
  update_solver(i, s_i); persist
  IF hit(raw): HALT

# Main loop
excluded ← {indices in S}
WHILE not hit:
  i ← solver.propose_next(excluded)
  s_i, raw ← API.query(V[i]); s_i ← map_distance(raw.distance)
  update_solver(i, s_i); excluded ← excluded ∪ {i}; persist
  IF hit(raw): HALT
  IF stagnation ≥ 5:
    j ← argmin_k (X[k]^T X[best]) over k ∉ excluded
    do one jump step on j
```

---

## 10. Computational Considerations

* **Neighbor computation**: each call uses one matrix–vector dot $X\cdot x_b$ with cost $O(|V|\cdot D)$, followed by `argpartition` $O(|V|)$.
* **KMeans**: one-time cost roughly $O(|V|\cdot K\cdot D \cdot \text{iters})$. Defaults keep $K$ moderate (e.g., 512).
* **End-to-end**: dominated by API latency and matrix–vector operations.

---

## 11. Configuration (key hyperparameters)

Defined in `contexto_solver/config.py` (values shown reflect the current solver defaults in this repository):

| Parameter             | Meaning                                 | Default                   |
| --------------------- | --------------------------------------- | ------------------------- |
| `VOCAB_SIZE`          | Max vocabulary rows loaded from GloVe   | 80,000                    |
| `SEED_COUNT`          | Number of farthest-point seeds          | 5                         |
| `NEIGHBOR_K`          | Size of local neighbor pool             | 64                        |
| `INITIAL_TEMPERATURE` | τ at t=0                                | 0.9                       |
| `TEMPERATURE_DECAY`   | τ ← τ·γ per step                        | 0.97                      |
| `MIN_TEMPERATURE`     | Lower bound on τ                        | 0.25                      |
| `EXPLOIT_BASE`        | Base exploitation probability           | 0.55                      |
| `UCB_ALPHA`           | UCB exploration weight α                | 1.3                       |
| `KMEANS_K`            | Cluster count                           | 512                       |
| `CLUSTER_SAMPLE_K`    | Max members sampled from chosen cluster | 512                       |
| `EMB_CENTER`          | Mean-center before whitening            | True                      |
| `EMB_REMOVE_TOP_K`    | Remove top-K PCs                        | 3                         |
| `EMB_FP16`            | Store embeddings as float16             | True                      |
| `RATE_LIMIT_SLEEP`    | Client-side inter-request sleep         | 0.1 s                     |
| `API_BASE`            | Contexto API root                       | `https://api.contexto.me` |
| `LANG`                | Language code                           | `en`                      |

> If your local `config.py` differs, that file is the source of truth.

---

## 12. Usage

### 12.1 Quick Start (local)

```bash
git clone https://github.com/nagsujosh/contexto-solver-agent.git
cd contexto-solver-agent
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Place GloVe 6B 300d at repo root (glove.6B.300d.txt)
# https://nlp.stanford.edu/data/glove.6B.zip

python main.py                   # Play one game (increments game id)
python automation_script.py      # Aggregate results and update docs
```

### 12.2 GitHub Actions (headless)

See `.github/workflows/daily-contexto.yml`. The workflow runs on a cron schedule, executes the solver, regenerates `RESULTS.md`, updates README badges, commits, and pushes.

---

## 13. Reproducibility

The run is **stochastic** due to:

* Random farthest-point start (`first = random.randrange(n)`).
* Bernoulli exploit/explore decision.
* Boltzmann sampling inside pools.

For reproducible debugging, set deterministic seeds early in your entrypoint:

```python
import os, random, numpy as np
random.seed(42); np.random.seed(42); os.environ["PYTHONHASHSEED"]="42"
```

---

## 14. Evaluation Metrics

* **Success rate**: fraction of games with an exact hit (distance ≤ 1).
* **Average guesses**: $\frac{1}{N}\sum \text{len(trajectory)}$.
* **Average duration**: wall-clock seconds per game (dominated by HTTP).
* **Latest success**: most recent solved game metadata.

Automation computes and writes these into `RESULTS.md` and patches README badges.

---

## 15. Failure Modes and Defenses

* **API variants**: some servers return `distance=0` for the answer, others `1`. We treat **distance ≤ 1** as exact.
* **404 / OOV**: bad tokens cached in `bad_words.json` to avoid re-querying.
* **Local optima**: stagnation-triggered **farthest jump** prevents getting stuck near suboptimal manifolds.
* **Vocabulary noise**: regex filter drops many non-words; OOV embeddings become zeros and are excluded.

---

## 16. Extensibility

* **Better calibrators**: replace isotonic with spline/Platt or learn a global calibration from historical games.
* **Language models**: re-rank local pools with a lightweight LM priors conditioned on topological cues.
* **Adaptive K**: anneal `NEIGHBOR_K` as a function of τ and recent improvements.
* **Multi-armed bandits**: treat semantically coherent **topics** as arms via dynamic clustering.

---

## 17. Project Structure

```
contexto-solver-agent/
├── contexto_solver/
│   ├── api.py          # HTTP client; distance→score mapping; robust hit detect
│   ├── config.py       # Tunables and file paths
│   ├── embedding.py    # GloVe loader, whitening, normalization
│   ├── io.py           # IDs, bad words, filesystem utilities
│   ├── pipeline.py     # Orchestrates seeds → main loop; persists trace
│   └── solver.py       # Hybrid controller (neighbors + cluster UCB)
├── automation_script.py # Generates RESULTS.md; patches README stats; batch mode
├── main.py              # One-game entrypoint
├── results/             # JSONL outputs
├── RESULTS.md           # Generated report
└── README.md
```

---

## 18. License

MIT License (see `LICENSE`).
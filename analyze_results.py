#!/usr/bin/env python3
"""
Analysis of Contexto solver performance across all recorded games.

Produces both a terminal report and updates RESULTS.md with detailed statistics.
"""

import json
import glob
import math
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    # Sort files so later timestamps (retried games) appear last and win dedup.
    files = sorted(glob.glob(f"{results_dir}/game_*.jsonl"))
    by_game: Dict[int, Dict[str, Any]] = {}
    for path in files:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        gid = r.get("game_id", -1)
                        by_game[gid] = r  # later file wins (latest retry)
                    except json.JSONDecodeError:
                        pass
    results = list(by_game.values())
    results.sort(key=lambda r: r.get("game_id", 0))
    return results


# ── Statistics helpers ────────────────────────────────────────────────────────

def guess_count(r: Dict[str, Any]) -> int:
    return len(r.get("trajectory", []))


def is_success(r: Dict[str, Any]) -> bool:
    return bool(r.get("successful", r.get("best_score", 0) >= 0.999))


def best_rank(r: Dict[str, Any]) -> Optional[int]:
    """Return the best (lowest) raw_distance reached in a game, or None."""
    traj = r.get("trajectory", [])
    dists = [
        int(s["raw_distance"])
        for s in traj
        if isinstance(s.get("raw_distance"), (int, float))
        and math.isfinite(s["raw_distance"])
        and s["raw_distance"] >= 0
    ]
    return min(dists) if dists else None


def compute_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    guesses = [guess_count(r) for r in results]
    durations = [r.get("duration_seconds", 0) for r in results]
    successes = [is_success(r) for r in results]
    return {
        "n": len(results),
        "success_rate": sum(successes) / len(successes) * 100 if successes else 0,
        "avg_guesses": statistics.mean(guesses) if guesses else 0,
        "median_guesses": statistics.median(guesses) if guesses else 0,
        "stdev_guesses": statistics.stdev(guesses) if len(guesses) > 1 else 0,
        "min_guesses": min(guesses) if guesses else 0,
        "max_guesses": max(guesses) if guesses else 0,
        "avg_duration": statistics.mean(durations) if durations else 0,
    }


def rolling_avg(results: List[Dict[str, Any]], window: int = 10) -> List[Tuple[int, float]]:
    """Return (game_id, rolling_avg_guesses) over a sliding window."""
    out = []
    for i in range(len(results)):
        start = max(0, i - window + 1)
        chunk = results[start : i + 1]
        avg = statistics.mean(guess_count(r) for r in chunk)
        out.append((results[i]["game_id"], avg))
    return out


# ── Terminal report ───────────────────────────────────────────────────────────

def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def print_report(results: List[Dict[str, Any]]) -> None:
    print(_hr("═"))
    print("  CONTEXTO SOLVER — PERFORMANCE ANALYSIS")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(_hr("═"))

    if not results:
        print("  No results found.")
        return

    # ── Overall stats ──────────────────────────────────────────────────────
    stats = compute_stats(results)
    print(f"\nOVERALL")
    print(_hr())
    print(f"  Games played   : {stats['n']}")
    print(f"  Success rate   : {stats['success_rate']:.1f}%")
    print(f"  Guesses  avg   : {stats['avg_guesses']:.1f}")
    print(f"           median: {stats['median_guesses']:.1f}")
    print(f"           stdev : {stats['stdev_guesses']:.1f}")
    print(f"           min   : {stats['min_guesses']}")
    print(f"           max   : {stats['max_guesses']}")
    print(f"  Duration avg   : {stats['avg_duration']:.1f}s")

    # ── Rolling average ────────────────────────────────────────────────────
    print(f"\nROLLING AVG (window=10 games)")
    print(_hr())
    rolling = rolling_avg(results, window=10)
    step = max(1, len(rolling) // 20)
    for i, (gid, avg) in enumerate(rolling):
        if i % step == 0 or i == len(rolling) - 1:
            bar_len = min(50, int(avg / 50))
            bar = "█" * bar_len
            print(f"  game {gid:5d}: {avg:6.1f} guesses  {bar}")

    # ── Per-game table ────────────────────────────────────────────────────
    print(f"\nPER-GAME TABLE")
    print(_hr())
    print(f"  {'ID':>5}  {'Word':<20}  {'Guesses':>7}  {'Dur(s)':>6}  {'BestRank':>8}  Status")
    print(f"  {'─'*5}  {'─'*20}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}")
    for r in results:
        gid = r.get("game_id", "?")
        word = str(r.get("best_word", "?"))[:20]
        guesses = guess_count(r)
        dur = r.get("duration_seconds", 0)
        rank = best_rank(r)
        rank_str = str(rank) if rank is not None else "?"
        status = "OK" if is_success(r) else "MISS"
        print(f"  {gid:>5}  {word:<20}  {guesses:>7}  {dur:>6.1f}  {rank_str:>8}  {status}")

    print(f"\n{_hr('═')}")


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}m {s % 60:.0f}s"


def generate_markdown(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    a = lines.append

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    a("# Contexto Solver — Results & Analysis")
    a("")
    a(f"> Last updated: {now_str}")
    a("")

    if not results:
        a("No games recorded yet.")
        return "\n".join(lines)

    # ── Overall summary ────────────────────────────────────────────────────
    stats = compute_stats(results)
    a("## Overall Performance")
    a("")
    a("| Metric | Value |")
    a("|--------|-------|")
    a(f"| Games played | {stats['n']} |")
    a(f"| Success rate | {stats['success_rate']:.1f}% |")
    a(f"| Avg guesses | {stats['avg_guesses']:.1f} |")
    a(f"| Median guesses | {stats['median_guesses']:.1f} |")
    a(f"| Std dev | {stats['stdev_guesses']:.1f} |")
    a(f"| Min guesses | {stats['min_guesses']} |")
    a(f"| Max guesses | {stats['max_guesses']} |")
    a(f"| Avg duration | {_fmt_dur(stats['avg_duration'])} |")
    a("")

    # ── Guess count distribution ───────────────────────────────────────────
    a("## Guess Count Distribution")
    a("")
    a("Breakdown of games by difficulty (number of guesses required).")
    a("")
    buckets = [
        ("≤ 50",    lambda g: g <= 50),
        ("51–100",  lambda g: 50 < g <= 100),
        ("101–200", lambda g: 100 < g <= 200),
        ("201–400", lambda g: 200 < g <= 400),
        ("401–999", lambda g: 400 < g <= 999),
        ("1000+",   lambda g: g >= 1000),
    ]
    all_guesses = [guess_count(r) for r in results]
    a("| Range | Count | % |")
    a("|-------|-------|---|")
    for label, pred in buckets:
        cnt = sum(1 for g in all_guesses if pred(g))
        pct = cnt / len(all_guesses) * 100
        a(f"| {label} | {cnt} | {pct:.1f}% |")
    a("")

    # ── Rolling-average section ────────────────────────────────────────────
    a("## Learning Curve (Rolling 10-game Average)")
    a("")
    a("Shows whether the cross-game correction store improves performance over time.")
    a("")
    a("| Game # | Rolling Avg Guesses |")
    a("|--------|---------------------|")
    rolling = rolling_avg(results, window=10)
    step = max(1, len(rolling) // 30)
    for i, (gid, avg) in enumerate(rolling):
        if i % step == 0 or i == len(rolling) - 1:
            bar_len = min(40, int(avg / 30))
            bar = "▓" * bar_len
            a(f"| {gid} | {avg:.1f} {bar} |")
    a("")

    # ── Correction store progress ─────────────────────────────────────────
    try:
        with open("centroid_meta.json") as fh:
            meta = json.load(fh)
        game_count = meta.get("game_count", 0)
        a("## Cross-Game Correction Store")
        a("")
        a(f"- Games recorded: **{game_count}**")
        a(f"- Matrix correction active: **{'yes' if game_count >= 20 else f'no — needs {20 - game_count} more games'}**")
        a("")
    except FileNotFoundError:
        pass

    # ── Full per-game table ────────────────────────────────────────────────
    a("## Full Game Log")
    a("")
    a("| # | Game ID | Word | Guesses | Duration | Best Rank | Status |")
    a("|---|---------|------|---------|----------|-----------|--------|")

    for i, r in enumerate(results, 1):
        gid = r.get("game_id", "?")
        word = r.get("best_word", "?")
        guesses = guess_count(r)
        dur = _fmt_dur(r.get("duration_seconds", 0))
        rank = best_rank(r)
        rank_str = str(rank) if rank is not None else "?"
        status = "✓" if is_success(r) else "✗"
        a(f"| {i} | {gid} | {word} | {guesses} | {dur} | {rank_str} | {status} |")

    a("")

    # ── Key findings ──────────────────────────────────────────────────────
    a("## Key Findings")
    a("")
    a("- **Semantic probes** (15 curated anchor words played at game start) identify the answer's domain")
    a("  within 2–3 guesses, pushing λ into exploitation range immediately instead of after 50–100 guesses.")
    a("- **Score range 70K**: the fallback log-scale formula uses max_dist=70,000, so any word at rank")
    a("  ≤3,000 already drives λ above 0.5 (exploitation mode). Narrower ranges left λ near zero for")
    a("  most of the game, causing unnecessary diversity sweeps.")
    a("- **Progressive candidate pool**: grows 500→3,000→15,000→full vocab as guess count increases,")
    a("  ensuring the solver cannot get permanently stuck in a local GloVe neighborhood.")
    a("- **Last-mile combined search**: when best raw_distance ≤ 5, the candidate pool switches to the")
    a("  top-20,000 by `0.5·sim(w, centroid) + 0.5·sim(w, best_word)`. This finds answers that are")
    a("  close to the best-seen word in GloVe space but not close to the centroid (e.g. 'bookstore').")
    a("- **Identity-regularized ridge correction matrix**: plain least-squares on a 300×300 system with")
    a("  fewer than 300 training samples leaves 225+ null-space directions unconstrained. Without")
    a("  regularization the condition number reaches ~4×10¹⁰ and centroids are distorted by ~87°.")
    a("  Regularizing toward the identity (`M = (XᵀX + λI)⁻¹(XᵀY + λI)`) keeps condition ≈4 and")
    a("  distortion ≈11°.")
    a("- **Remaining bottleneck**: GloVe vs. Contexto embedding mismatch — long games occur when the")
    a("  answer sits in a different semantic cluster in Contexto's model than in GloVe 6B.")
    a("")

    a("## Recent Games (last 20)")
    a("")
    for r in reversed(results[-20:]):
        gid = r.get("game_id", "?")
        word = r.get("best_word", "?")
        traj = r.get("trajectory", [])
        dur = _fmt_dur(r.get("duration_seconds", 0))
        ts = r.get("timestamp", "")
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
        except Exception:
            dt = ts
        status = "SUCCESS" if is_success(r) else "PARTIAL"

        a(f"### Game #{gid} — {word} ({dt})")
        a(f"**{status}** · {len(traj)} guesses · {dur}")
        a("")
        if traj:
            a("<details><summary>Guess trajectory</summary>")
            a("")
            if len(traj) <= 12:
                for s in traj:
                    raw_d = s.get("raw_distance", -1)
                    d_str = f"rank {int(raw_d)}" if raw_d > 0 else "HIT"
                    a(f"- {s['iter']:3d}. **{s['guess']}** — {d_str} (score {s['score']:.3f})")
            else:
                for s in traj[:6]:
                    raw_d = s.get("raw_distance", -1)
                    d_str = f"rank {int(raw_d)}" if raw_d > 0 else "HIT"
                    a(f"- {s['iter']:3d}. **{s['guess']}** — {d_str} (score {s['score']:.3f})")
                a(f"- *(… {len(traj) - 12} guesses …)*")
                for s in traj[-6:]:
                    raw_d = s.get("raw_distance", -1)
                    d_str = f"rank {int(raw_d)}" if raw_d > 0 else "HIT"
                    a(f"- {s['iter']:3d}. **{s['guess']}** — {d_str} (score {s['score']:.3f})")
            a("")
            a("</details>")
        a("")

    a("---")
    a(f"*Auto-generated by `analyze_results.py` · {now_str}*")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Contexto solver results")
    parser.add_argument("--no-md", action="store_true", help="Skip writing RESULTS.md")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    print_report(results)

    if not args.no_md:
        md = generate_markdown(results)
        with open("RESULTS.md", "w") as fh:
            fh.write(md)
        print(f"\n  RESULTS.md updated ({len(md)} chars, {len(results)} games).")


if __name__ == "__main__":
    main()

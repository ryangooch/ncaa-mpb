#!/usr/bin/env python3
"""
What-If Bracket Simulator — Monte Carlo NCAA tournament simulation.

Lets users pick a focus stat from the Torvik game stats DB that shifts
win probabilities, then simulates the bracket thousands of times to see
how outcomes change.

Usage:
    uv run python whatif.py [--bracket PATH] [--db PATH] [--sims N] [--seed N]
"""

import argparse
import difflib
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import unquote

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from bracket import load_bracket

# ---------------------------------------------------------------------------
# Name mapping: Massey → Torvik
# ---------------------------------------------------------------------------

# Hardcoded mapping for teams whose names differ between sources.
# Update this dict when the Massey CSV or Torvik DB team names change.
# Set a value to None to explicitly mark a team as having no Torvik match.
MASSEY_TO_TORVIK = {
    "St Mary's CA": "Saint Mary's",
    "NC State": "N.C. State",
    "Miami FL": "Miami FL",
    "Iowa St": "Iowa St.",
    "Michigan St": "Michigan St.",
    "Ohio St": "Ohio St.",
    "Utah St": "Utah St.",
    "McNeese St": "McNeese St.",
    "Kennesaw St": "Kennesaw St.",
    "St. John's": "St. John%27s",
    "McNeese": "McNeese St.",
    "N Dakota St": "North Dakota St.",
    "Iowa State": "Iowa St.",
    "Long Island": "LIU",
}


def _build_name_map(massey_names: list[str], torvik_names: list[str],
                    console: Console | None = None) -> dict:
    """Build Massey → Torvik name mapping using hardcoded + exact matching.

    Fuzzy matching uses a high cutoff (0.85) to avoid false positives like
    Drake→Duke. Unmatched teams are reported so the user can add them to
    MASSEY_TO_TORVIK.
    """
    mapping = {}
    torvik_decoded = {unquote(t): t for t in torvik_names}
    decoded_list = list(torvik_decoded.keys())

    for mname in massey_names:
        ms = mname.strip()

        # 1. Hardcoded mapping
        if ms in MASSEY_TO_TORVIK:
            candidate = MASSEY_TO_TORVIK[ms]
            if candidate is None:
                mapping[ms] = None
                continue
            if candidate in torvik_decoded:
                mapping[ms] = torvik_decoded[candidate]
                continue
            # Candidate not found in DB — try raw form too
            if candidate in torvik_names:
                mapping[ms] = candidate
                continue

        # 2. Exact match (decoded Torvik name)
        if ms in torvik_decoded:
            mapping[ms] = torvik_decoded[ms]
            continue

        # 3. Fuzzy match with high cutoff to avoid false positives
        matches = difflib.get_close_matches(ms, decoded_list, n=1, cutoff=0.85)
        if matches:
            matched_decoded = matches[0]
            mapping[ms] = torvik_decoded[matched_decoded]
            if console:
                console.print(f"  [dim]Fuzzy: {ms} → {matched_decoded}[/dim]")
            continue

        mapping[ms] = None

    return mapping


# ---------------------------------------------------------------------------
# Seed-based priors for unmatched teams
# ---------------------------------------------------------------------------

# Typical adj_margin by seed (approximate historical averages)
SEED_ADJ_MARGIN = {
    1: 25.0, 2: 20.0, 3: 17.0, 4: 14.0,
    5: 12.0, 6: 10.0, 7: 8.5, 8: 7.0,
    9: 5.5, 10: 4.5, 11: 3.5, 12: 2.5,
    13: 1.0, 14: -1.0, 15: -3.0, 16: -6.0,
}


def _seed_prior_stats(team_name: str, seed: int) -> "TeamStats":
    """Create a TeamStats with seed-based priors for unmatched teams."""
    margin = SEED_ADJ_MARGIN.get(seed, 0.0)
    return TeamStats(
        team=team_name,
        seed=seed,
        adj_margin=margin,
        eff_margin=margin * 0.8,
        experience=SEED_EXPERIENCE.get(seed, 1.0),
    )


# ---------------------------------------------------------------------------
# Team stats from Torvik DB
# ---------------------------------------------------------------------------

@dataclass
class TeamStats:
    """Season averages and derived stats for a single team."""
    team: str
    seed: int = 0
    region: int = -1
    has_data: bool = False
    # Season averages
    avg_adj_o: float = 0.0
    avg_adj_d: float = 0.0
    avg_off_eff: float = 0.0
    avg_def_eff: float = 0.0
    avg_off_to_pct: float = 0.0
    avg_def_to_pct: float = 0.0
    avg_off_or_pct: float = 0.0
    avg_def_or_pct: float = 0.0
    avg_off_ftr: float = 0.0
    avg_def_ftr: float = 0.0
    avg_wab: float = 0.0
    # Derived
    adj_margin: float = 0.0
    eff_margin: float = 0.0
    three_pt_pct: float = 0.0
    three_pt_variance: float = 0.0
    late_season_trend: float = 0.0
    turnover_diff: float = 0.0
    rebound_diff: float = 0.0
    ftr_diff: float = 0.0
    wab: float = 0.0
    # Cinderella
    experience: float = 0.0
    cinderella_score: float = 0.0
    # Style profile
    pace: float = 0.0           # off_eff + def_eff proxy
    inside_rate: float = 0.5    # 2pa / (2pa + 3pa)
    three_pt_rate: float = 0.5  # 3pa / (2pa + 3pa)
    off_reb_rate: float = 0.0   # avg_off_or_pct
    to_force_rate: float = 0.0  # avg_def_to_pct
    ft_rate: float = 0.0        # avg_off_ftr
    def_three_pt_pct: float = 0.0  # opponent 3pt%
    def_inside_rate: float = 0.5   # opponent 2pa / (2pa + 3pa)


# Seed-based experience priors (years of experience typical per seed)
SEED_EXPERIENCE = {
    1: 1.8, 2: 1.7, 3: 1.6, 4: 1.5,
    5: 1.4, 6: 1.4, 7: 1.3, 8: 1.3,
    9: 1.2, 10: 1.2, 11: 1.1, 12: 1.1,
    13: 1.0, 14: 1.0, 15: 0.9, 16: 0.8,
}


def load_team_stats(db_path: str, torvik_name: str) -> TeamStats | None:
    """Load and compute stats for a single team from the Torvik DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM games WHERE team = ? ORDER BY id", (torvik_name,)
    ).fetchall()

    if not rows:
        conn.close()
        return None

    ts = TeamStats(team=torvik_name, has_data=True)

    n = len(rows)
    ts.avg_adj_o = sum(r["adj_o"] or 0 for r in rows) / n
    ts.avg_adj_d = sum(r["adj_d"] or 0 for r in rows) / n
    ts.avg_off_eff = sum(r["off_eff"] or 0 for r in rows) / n
    ts.avg_def_eff = sum(r["def_eff"] or 0 for r in rows) / n
    ts.avg_off_to_pct = sum(r["off_to_pct"] or 0 for r in rows) / n
    ts.avg_def_to_pct = sum(r["def_to_pct"] or 0 for r in rows) / n
    ts.avg_off_or_pct = sum(r["off_or_pct"] or 0 for r in rows) / n
    ts.avg_def_or_pct = sum(r["def_or_pct"] or 0 for r in rows) / n
    ts.avg_off_ftr = sum(r["off_ftr"] or 0 for r in rows) / n
    ts.avg_def_ftr = sum(r["def_ftr"] or 0 for r in rows) / n
    ts.avg_wab = sum(r["wab"] or 0 for r in rows) / n

    ts.adj_margin = ts.avg_adj_o - ts.avg_adj_d
    ts.eff_margin = ts.avg_off_eff - ts.avg_def_eff
    ts.turnover_diff = ts.avg_def_to_pct - ts.avg_off_to_pct
    ts.rebound_diff = ts.avg_off_or_pct - ts.avg_def_or_pct
    ts.ftr_diff = ts.avg_off_ftr - ts.avg_def_ftr
    ts.wab = ts.avg_wab

    game_3p_pcts = []
    for r in rows:
        att = r["off_3pa"]
        made = r["off_3pm"]
        if att and att > 0:
            game_3p_pcts.append(made / att)
    if game_3p_pcts:
        ts.three_pt_pct = statistics.mean(game_3p_pcts)
        ts.three_pt_variance = statistics.stdev(game_3p_pcts) if len(game_3p_pcts) > 1 else 0.0

    # Style profile
    ts.pace = ts.avg_off_eff + ts.avg_def_eff
    total_off_2pa = sum(r["off_2pa"] or 0 for r in rows)
    total_off_3pa = sum(r["off_3pa"] or 0 for r in rows)
    total_att = total_off_2pa + total_off_3pa
    if total_att > 0:
        ts.inside_rate = total_off_2pa / total_att
        ts.three_pt_rate = total_off_3pa / total_att
    ts.off_reb_rate = ts.avg_off_or_pct
    ts.to_force_rate = ts.avg_def_to_pct
    ts.ft_rate = ts.avg_off_ftr

    # Defensive style
    total_def_2pa = sum(r["def_2pa"] or 0 for r in rows)
    total_def_3pa = sum(r["def_3pa"] or 0 for r in rows)
    total_def_att = total_def_2pa + total_def_3pa
    if total_def_att > 0:
        ts.def_inside_rate = total_def_2pa / total_def_att
    def_3p_pcts = []
    for r in rows:
        datt = r["def_3pa"]
        dmade = r["def_3pm"]
        if datt and datt > 0:
            def_3p_pcts.append(dmade / datt)
    if def_3p_pcts:
        ts.def_three_pt_pct = statistics.mean(def_3p_pcts)

    full_margins = [(r["off_eff"] or 0) - (r["def_eff"] or 0) for r in rows]
    full_avg = sum(full_margins) / len(full_margins) if full_margins else 0
    last_10 = full_margins[-10:]
    last_10_avg = sum(last_10) / len(last_10) if last_10 else 0
    ts.late_season_trend = last_10_avg - full_avg

    # Experience — pull from team_season_stats; will fall back to seed prior later
    exp_row = conn.execute(
        "SELECT experience FROM team_season_stats WHERE team = ?", (torvik_name,)
    ).fetchone()
    if exp_row and exp_row["experience"] is not None:
        ts.experience = exp_row["experience"]
    # (seed-based fallback applied in main() after seed is assigned)

    conn.close()
    return ts


# ---------------------------------------------------------------------------
# Focus stats definition
# ---------------------------------------------------------------------------

FOCUS_STATS = {
    "1": ("adj_margin", "Adjusted Efficiency Margin (AdjO - AdjD)"),
    "2": ("eff_margin", "Raw Efficiency Margin"),
    "3": ("three_pt_pct", "Three-Point Shooting %"),
    "4": ("three_pt_variance", "Three-Point Variance (get hot factor)"),
    "5": ("late_season_trend", "Late-Season Trend (peaking teams)"),
    "6": ("turnover_diff", "Turnover Differential (force TOs - commit TOs)"),
    "7": ("rebound_diff", "Rebounding Differential (OFF reb% - DEF reb%)"),
    "8": ("ftr_diff", "Free Throw Rate Differential"),
    "9": ("wab", "Wins Above Bubble"),
    "10": ("cinderella_score", "Cinderella Potential (composite upset score)"),
}


# ---------------------------------------------------------------------------
# Win probability model
# ---------------------------------------------------------------------------

# Historical seed-vs-seed win rates for round 1
SEED_WIN_RATES = {
    (1, 16): 0.985, (2, 15): 0.935, (3, 14): 0.855, (4, 13): 0.790,
    (5, 12): 0.645, (6, 11): 0.625, (7, 10): 0.605, (8, 9): 0.510,
}


def _logistic(x: float) -> float:
    """Logistic function mapping efficiency diff to win probability."""
    return 1.0 / (1.0 + math.exp(-0.15 * x))


def win_probability(
    team_a: TeamStats,
    team_b: TeamStats,
    round_num: int,
    focus_stat: str | None,
    z_scores: dict[str, dict[str, float]],
    is_3pt_focus: bool = False,
) -> float:
    """Compute win probability for team_a vs team_b.

    round_num: 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=Championship
    """
    # Round 1 between standard seed matchups: use historical rates
    if round_num == 1 and team_a.seed > 0 and team_b.seed > 0:
        s_lo = min(team_a.seed, team_b.seed)
        s_hi = max(team_a.seed, team_b.seed)
        if (s_lo, s_hi) in SEED_WIN_RATES:
            base_p = SEED_WIN_RATES[(s_lo, s_hi)]
            if team_a.seed > team_b.seed:
                base_p = 1.0 - base_p
            # Blend with logistic if both have real data (70% historical, 30% data)
            if team_a.has_data and team_b.has_data:
                data_p = _logistic(team_a.adj_margin - team_b.adj_margin)
                base_p = 0.7 * base_p + 0.3 * data_p
        else:
            base_p = _logistic(team_a.adj_margin - team_b.adj_margin)
    else:
        base_p = _logistic(team_a.adj_margin - team_b.adj_margin)

    # Focus stat shift
    if focus_stat:
        z_a = z_scores.get(team_a.team, {}).get(focus_stat, 0.0)
        z_b = z_scores.get(team_b.team, {}).get(focus_stat, 0.0)
        shift_weight = 0.1
        shift = shift_weight * (z_a - z_b)
        shift = max(-0.25, min(0.25, shift))

        # 3-point variance special: high variance slightly boosts underdog
        if is_3pt_focus and team_a.seed > team_b.seed:
            var_boost = 0.05 * z_a
            shift += max(0, min(0.05, var_boost))
        elif is_3pt_focus and team_b.seed > team_a.seed:
            var_boost = 0.05 * z_b
            shift -= max(0, min(0.05, var_boost))

        base_p += shift

    return max(0.02, min(0.98, base_p))


# ---------------------------------------------------------------------------
# Style mismatch & composite model
# ---------------------------------------------------------------------------

def style_mismatch_bonus(underdog: TeamStats, favorite: TeamStats) -> float:
    """Compute a mismatch bonus (0 to ~0.08) for the underdog.

    Looks at whether the underdog's offensive strengths exploit the
    favorite's defensive weaknesses, which is how real upsets happen.
    """
    bonus = 0.0

    # Underdog shoots lots of 3s and favorite gives up high 3pt%
    if underdog.three_pt_rate > 0.4 and favorite.def_three_pt_pct > 0.34:
        bonus += 2*0.025

    # Underdog forces turnovers and favorite is turnover-prone
    if underdog.to_force_rate > 18.0 and favorite.avg_off_to_pct > 17.0:
        bonus += 2*0.02

    # Underdog crashes boards and favorite weak on def rebounding
    if underdog.off_reb_rate > 30.0 and favorite.avg_def_or_pct > 28.0:
        bonus += 2*0.015

    # Pace mismatch: slow underdog vs fast favorite (tempo control)
    pace_diff = favorite.pace - underdog.pace
    if pace_diff > 15:
        bonus += 2*0.02

    return min(bonus, 0.08)


# Composite signal weights for bracket filling
_COMPOSITE_WEIGHTS = {
    "adj_margin":        0.30,
    "three_pt_variance": 0.12,
    "three_pt_pct":      0.08,
    "late_season_trend": 0.10,
    "turnover_diff":     0.05,
    "experience":        0.08,
    "cinderella_score":  0.07,
    "wab":               0.05,
    "rebound_diff":      0.05,
    "ftr_diff":          0.03,
}
# Remaining 0.07 comes from style mismatch (applied separately)


def composite_win_probability(
    team_a: TeamStats,
    team_b: TeamStats,
    round_num: int,
    z_scores: dict[str, dict[str, float]],
) -> float:
    """Compute win probability using all signals blended together.

    Uses the same base (historical + logistic) as the original model,
    then applies a composite shift from all weighted signals plus a
    style mismatch bonus for the underdog.
    """
    # Base probability (same as win_probability)
    if round_num == 1 and team_a.seed > 0 and team_b.seed > 0:
        s_lo = min(team_a.seed, team_b.seed)
        s_hi = max(team_a.seed, team_b.seed)
        if (s_lo, s_hi) in SEED_WIN_RATES:
            base_p = SEED_WIN_RATES[(s_lo, s_hi)]
            if team_a.seed > team_b.seed:
                base_p = 1.0 - base_p
            if team_a.has_data and team_b.has_data:
                data_p = _logistic(team_a.adj_margin - team_b.adj_margin)
                base_p = 0.65 * base_p + 0.35 * data_p
        else:
            base_p = _logistic(team_a.adj_margin - team_b.adj_margin)
    else:
        base_p = _logistic(team_a.adj_margin - team_b.adj_margin)

    # Composite z-score shift from all signals
    z_a = z_scores.get(team_a.team, {})
    z_b = z_scores.get(team_b.team, {})
    shift = 0.0
    for stat, weight in _COMPOSITE_WEIGHTS.items():
        diff = z_a.get(stat, 0.0) - z_b.get(stat, 0.0)
        shift += weight * 0.05 * diff  # 0.05 scales z-diff to probability space

    # 3pt variance underdog boost (high variance = can get hot)
    if team_a.seed > team_b.seed:
        var_boost = 0.02 * z_a.get("three_pt_variance", 0.0)
        shift += max(0, min(0.04, var_boost))
    elif team_b.seed > team_a.seed:
        var_boost = 0.02 * z_b.get("three_pt_variance", 0.0)
        shift -= max(0, min(0.04, var_boost))

    shift = max(-0.20, min(0.20, shift))
    base_p += shift

    # Style mismatch: boost for the underdog
    if team_a.seed > team_b.seed and team_a.has_data and team_b.has_data:
        base_p += style_mismatch_bonus(team_a, team_b)
    elif team_b.seed > team_a.seed and team_a.has_data and team_b.has_data:
        base_p -= style_mismatch_bonus(team_b, team_a)

    return max(0.02, min(0.98, base_p))


# Cinderella score weights
_CIND_WEIGHTS = {
    "three_pt_variance": 0.35,
    "experience": 0.30,
    "three_pt_pct": 0.20,
    "wab": 0.15,
}


def compute_cinderella_score(
    team_stats: TeamStats, z_scores: dict[str, dict[str, float]]
) -> float:
    """Weighted z-score sum for Cinderella profile.  Raw value — caller rescales."""
    team_z = z_scores.get(team_stats.team, {})
    return sum(w * team_z.get(stat, 0.0) for stat, w in _CIND_WEIGHTS.items())


def compute_z_scores(all_stats: dict[str, TeamStats]) -> dict[str, dict[str, float]]:
    """Compute z-scores for each focus stat across all teams.

    Includes `experience` (used in the cinderella score) and the composite
    `cinderella_score` itself.  The cinderella_score z-scores are computed in
    a second pass after the component z-scores exist, then rescaled to 0–100
    and stored on each TeamStats object.
    """
    # All individual stats to z-score (includes experience and all focus stats)
    stat_names = [s[0] for s in FOCUS_STATS.values()] + ["experience"]
    # deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for s in stat_names:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    z_scores: dict[str, dict[str, float]] = {}

    for stat_name in ordered:
        if stat_name == "cinderella_score":
            continue  # computed in second pass
        values = [getattr(ts, stat_name, 0.0) for ts in all_stats.values()]
        if not values:
            continue
        mean = sum(values) / len(values)
        std = statistics.stdev(values) if len(values) > 1 else 1.0
        if std == 0:
            std = 1.0
        for ts in all_stats.values():
            z_scores.setdefault(ts.team, {})[stat_name] = (
                (getattr(ts, stat_name, 0.0) - mean) / std
            )

    # Second pass: compute raw cinderella scores, rescale to 0–100, store on TeamStats
    raw_scores = {
        ts.team: compute_cinderella_score(ts, z_scores) for ts in all_stats.values()
    }
    raw_vals = list(raw_scores.values())
    raw_min = min(raw_vals) if raw_vals else 0.0
    raw_max = max(raw_vals) if raw_vals else 1.0
    raw_range = raw_max - raw_min if raw_max != raw_min else 1.0

    for ts in all_stats.values():
        ts.cinderella_score = 100.0 * (raw_scores[ts.team] - raw_min) / raw_range

    # Now z-score the cinderella_score itself for use as a focus stat
    cind_values = [ts.cinderella_score for ts in all_stats.values()]
    cind_mean = sum(cind_values) / len(cind_values) if cind_values else 0.0
    cind_std = statistics.stdev(cind_values) if len(cind_values) > 1 else 1.0
    if cind_std == 0:
        cind_std = 1.0
    for ts in all_stats.values():
        z_scores.setdefault(ts.team, {})["cinderella_score"] = (
            (ts.cinderella_score - cind_mean) / cind_std
        )

    return z_scores


# ---------------------------------------------------------------------------
# Monte Carlo engine
# ---------------------------------------------------------------------------

class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def simulate_bracket(
    regions: dict,
    first_four: list,
    team_lookup: dict[str, TeamStats],
    focus_stat: str | None,
    z_scores: dict[str, dict[str, float]],
    n_sims: int = 10_000,
    rng_seed: int | None = None,
    console: Console | None = None,
) -> dict:
    """Run Monte Carlo bracket simulation.

    team_lookup maps whitespace-stripped Massey names to TeamStats objects.
    """
    rng = np.random.default_rng(rng_seed)
    is_3pt = focus_stat == "three_pt_variance"

    advancement: dict[str, list[int]] = defaultdict(lambda: [0] * 7)
    champion_counts: dict[str, int] = defaultdict(int)

    def get_stats(team_name: str) -> TeamStats:
        name = team_name.strip()
        if name in team_lookup:
            return team_lookup[name]
        return TeamStats(team=name)

    def sim_game(a_name: str, b_name: str, round_num: int) -> str:
        stats_a = get_stats(a_name)
        stats_b = get_stats(b_name)
        p = win_probability(stats_a, stats_b, round_num, focus_stat, z_scores, is_3pt)
        return a_name if rng.random() < p else b_name

    progress = None
    if console:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        )

    ctx = progress if progress else _nullcontext()
    with ctx:
        task_id = progress.add_task("Simulating brackets...", total=n_sims) if progress else None

        for _ in range(n_sims):
            # Resolve First Four
            ff_winners = []
            for t1, t2, seed in first_four:
                winner = sim_game(t1, t2, 1)
                ff_winners.append(winner)

            ff_idx = 0
            region_winners = []
            for r_idx in range(4):
                region = regions[r_idx]
                matchups = []
                for s in [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]:
                    team = region.get(s, "TBD")
                    if team == "Play-in":
                        if ff_idx < len(ff_winners):
                            team = ff_winners[ff_idx]
                            ff_idx += 1
                    matchups.append(team)

                # R64 (8 games)
                r64_w = []
                for i in range(0, 16, 2):
                    w = sim_game(matchups[i], matchups[i + 1], 1)
                    advancement[w][0] += 1
                    r64_w.append(w)

                # R32 (4 games)
                r32_w = []
                for i in range(0, 8, 2):
                    w = sim_game(r64_w[i], r64_w[i + 1], 2)
                    advancement[w][1] += 1
                    r32_w.append(w)

                # Sweet 16 (2 games)
                s16_w = []
                for i in range(0, 4, 2):
                    w = sim_game(r32_w[i], r32_w[i + 1], 3)
                    advancement[w][2] += 1
                    s16_w.append(w)

                # Elite 8
                e8_w = sim_game(s16_w[0], s16_w[1], 4)
                advancement[e8_w][3] += 1
                region_winners.append(e8_w)

            # Final Four
            ff1 = sim_game(region_winners[0], region_winners[1], 5)
            advancement[ff1][4] += 1
            ff2 = sim_game(region_winners[2], region_winners[3], 5)
            advancement[ff2][4] += 1

            # Championship
            champ = sim_game(ff1, ff2, 6)
            advancement[champ][5] += 1
            champion_counts[champ] += 1

            if progress and task_id is not None:
                progress.update(task_id, advance=1)

    return {
        "advancement": dict(advancement),
        "champion_counts": dict(champion_counts),
        "n_sims": n_sims,
    }


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

REGION_NAMES = ["SOUTH", "WEST", "EAST", "MIDWEST"]
MATCHUP_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# ---------------------------------------------------------------------------
# Bracket filler — single deterministic bracket with smart upsets
# ---------------------------------------------------------------------------

# Historical average upsets per round (seeds 1-16 matchups in R64)
# Roughly 5-6 upsets per tournament in R64, ~3 in R32
_TARGET_R64_UPSETS = 5
_UPSET_THRESHOLD = 0.22  # if underdog has >= 22% chance, eligible for upset


def fill_bracket(
    regions: dict,
    first_four: list,
    team_lookup: dict[str, TeamStats],
    z_scores: dict[str, dict[str, float]],
    rng_seed: int | None = None,
    console: Console | None = None,
) -> dict:
    """Generate a single filled bracket using the composite model.

    Instead of pure Monte Carlo, this uses composite probabilities to pick
    winners, injecting a realistic number of upsets based on cinderella
    scores and style mismatches. Returns a nested dict of results by round.
    """
    rng = np.random.default_rng(rng_seed)

    def get_stats(team_name: str) -> TeamStats:
        name = team_name.strip()
        if name in team_lookup:
            return team_lookup[name]
        return TeamStats(team=name)

    def pick_winner(a_name: str, b_name: str, round_num: int) -> tuple[str, float]:
        """Pick a winner using composite model with probabilistic flip.

        Uses the composite probability as a weighted coin flip so that
        weight changes actually shift outcomes. Returns (winner, p_a).
        """
        stats_a = get_stats(a_name)
        stats_b = get_stats(b_name)
        p_a = composite_win_probability(stats_a, stats_b, round_num, z_scores)
        winner = a_name if rng.random() < p_a else b_name
        return winner, p_a

    def get_seed(team_name: str) -> int:
        ts = get_stats(team_name)
        return ts.seed if ts.seed > 0 else 16

    # Resolve First Four
    ff_winners = []
    for t1, t2, seed in first_four:
        winner, _ = pick_winner(t1, t2, 1)
        ff_winners.append(winner)

    bracket = {"first_four": list(ff_winners), "regions": {}, "final_four": [], "championship": None}

    # Phase 1: build all R64 matchups across regions and collect upset info
    all_r64_games = []
    ff_idx = 0  # global index into ff_winners, shared across regions
    for r_idx in range(4):
        region = regions[r_idx]

        # Build matchup list (same logic as simulate_bracket)
        matchups = []
        for s in MATCHUP_ORDER:
            team = region.get(s, "TBD")
            if team == "Play-in":
                if ff_idx < len(ff_winners):
                    team = ff_winners[ff_idx]
                    ff_idx += 1
            matchups.append(team)

        for i in range(0, 16, 2):
            a, b = matchups[i], matchups[i + 1]
            stats_a, stats_b = get_stats(a), get_stats(b)
            p_a = composite_win_probability(stats_a, stats_b, 1, z_scores)

            seed_a, seed_b = get_seed(a), get_seed(b)
            if seed_a < seed_b:
                fav, dog, dog_p = a, b, 1.0 - p_a
                dog_cinderella = stats_b.cinderella_score
            elif seed_b < seed_a:
                fav, dog, dog_p = b, a, p_a
                dog_cinderella = stats_a.cinderella_score
            else:
                fav, dog, dog_p = (a, b, 1.0 - p_a) if p_a >= 0.5 else (b, a, p_a)
                dog_cinderella = 0

            all_r64_games.append({
                "a": a, "b": b, "fav": fav, "dog": dog,
                "dog_p": dog_p, "dog_cinderella": dog_cinderella,
                "region": r_idx,
            })

    # Phase 2: decide which R64 games are upsets
    upset_candidates = [g for g in all_r64_games if g["dog_p"] >= _UPSET_THRESHOLD]
    for g in upset_candidates:
        g["upset_score"] = g["dog_p"] * 0.6 + (g["dog_cinderella"] / 100.0) * 0.4
    upset_candidates.sort(key=lambda g: g["upset_score"], reverse=True)

    n_upsets = _TARGET_R64_UPSETS + int(rng.choice([-1, 0, 0, 1]))
    n_upsets = max(3, min(7, n_upsets))

    upset_set = set()
    for g in upset_candidates[:n_upsets]:
        if rng.random() < g["upset_score"] + 0.3:
            upset_set.add(id(g))
    # Backfill if we didn't get enough
    for g in upset_candidates[n_upsets:n_upsets + 3]:
        if len(upset_set) < 3 and rng.random() < g["dog_p"]:
            upset_set.add(id(g))

    # Mark winners on all R64 games
    for g in all_r64_games:
        g["winner"] = g["dog"] if id(g) in upset_set else g["fav"]
        g["upset"] = id(g) in upset_set and get_seed(g["dog"]) > get_seed(g["fav"])

    # Phase 3: fill regions R32 through E8
    region_winners = []
    game_idx = 0
    for r_idx in range(4):
        bracket["regions"][r_idx] = {"r64": [], "r32": [], "s16": [], "e8": None}
        region_games = all_r64_games[game_idx:game_idx + 8]
        game_idx += 8

        r64_winners = []
        for g in region_games:
            r64_winners.append(g["winner"])
            bracket["regions"][r_idx]["r64"].append(g)

        # R32
        r32_winners = []
        for i in range(0, 8, 2):
            a, b = r64_winners[i], r64_winners[i + 1]
            winner, _ = pick_winner(a, b, 2)
            r32_winners.append(winner)
            bracket["regions"][r_idx]["r32"].append({"winner": winner, "a": a, "b": b})

        # S16
        s16_winners = []
        for i in range(0, 4, 2):
            a, b = r32_winners[i], r32_winners[i + 1]
            winner, _ = pick_winner(a, b, 3)
            s16_winners.append(winner)
            bracket["regions"][r_idx]["s16"].append({"winner": winner, "a": a, "b": b})

        # E8
        e8_winner, _ = pick_winner(s16_winners[0], s16_winners[1], 4)
        bracket["regions"][r_idx]["e8"] = {
            "winner": e8_winner, "a": s16_winners[0], "b": s16_winners[1]
        }
        region_winners.append(e8_winner)

    # Final Four
    ff1_winner, _ = pick_winner(region_winners[0], region_winners[1], 5)
    ff2_winner, _ = pick_winner(region_winners[2], region_winners[3], 5)
    bracket["final_four"] = [
        {"winner": ff1_winner, "a": region_winners[0], "b": region_winners[1]},
        {"winner": ff2_winner, "a": region_winners[2], "b": region_winners[3]},
    ]

    # Championship
    champ, _ = pick_winner(ff1_winner, ff2_winner, 6)
    bracket["championship"] = {"winner": champ, "a": ff1_winner, "b": ff2_winner}

    return bracket


def display_filled_bracket(bracket: dict, regions: dict, first_four: list,
                           team_lookup: dict[str, TeamStats], console: Console):
    """Display the filled bracket as a rich table."""
    console.print("\n[bold underline]FILLED BRACKET[/bold underline]\n")

    total_upsets = 0

    for r_idx in range(4):
        region_data = bracket["regions"][r_idx]
        console.print(f"[bold cyan]── {REGION_NAMES[r_idx]} ──[/bold cyan]")

        # R64
        for game in region_data["r64"]:
            a, b, winner = game["a"], game["b"], game["winner"]
            ts_a, ts_b = team_lookup.get(a.strip()), team_lookup.get(b.strip())
            seed_a = ts_a.seed if ts_a else "?"
            seed_b = ts_b.seed if ts_b else "?"
            upset = game.get("upset", False)
            if upset:
                total_upsets += 1
                marker = " [bold red]UPSET[/bold red]"
            else:
                marker = ""
            style = "[bold green]" if not upset else "[bold yellow]"
            console.print(
                f"  ({seed_a:>2}) {_s(a):18s} vs ({seed_b:>2}) {_s(b):18s}"
                f"  → {style}{_s(winner)}[/]{marker}"
            )

        # R32
        console.print(f"  [dim]Round of 32:[/dim]")
        for game in region_data["r32"]:
            w = game["winner"]
            ts = team_lookup.get(w.strip())
            seed = ts.seed if ts else "?"
            console.print(f"    [bold]({seed:>2}) {_s(w)}[/bold]")

        # S16
        console.print(f"  [dim]Sweet 16:[/dim]")
        for game in region_data["s16"]:
            w = game["winner"]
            ts = team_lookup.get(w.strip())
            seed = ts.seed if ts else "?"
            console.print(f"    [bold]({seed:>2}) {_s(w)}[/bold]")

        # E8
        e8 = region_data["e8"]
        ts = team_lookup.get(e8["winner"].strip())
        seed = ts.seed if ts else "?"
        console.print(f"  [dim]Elite 8 → [/dim][bold magenta]({seed:>2}) {_s(e8['winner'])}[/bold magenta]")
        console.print()

    # Final Four
    console.print("[bold cyan]── FINAL FOUR ──[/bold cyan]")
    for game in bracket["final_four"]:
        a, b, w = game["a"], game["b"], game["winner"]
        ts_a = team_lookup.get(a.strip())
        ts_b = team_lookup.get(b.strip())
        sa = ts_a.seed if ts_a else "?"
        sb = ts_b.seed if ts_b else "?"
        sw = team_lookup.get(w.strip())
        seed_w = sw.seed if sw else "?"
        console.print(
            f"  ({sa:>2}) {_s(a):18s} vs ({sb:>2}) {_s(b):18s}"
            f"  → [bold magenta]({seed_w}) {_s(w)}[/bold magenta]"
        )

    # Championship
    champ = bracket["championship"]
    a, b, w = champ["a"], champ["b"], champ["winner"]
    ts_w = team_lookup.get(w.strip())
    seed_w = ts_w.seed if ts_w else "?"
    console.print(f"\n[bold cyan]── CHAMPIONSHIP ──[/bold cyan]")
    console.print(f"  {_s(a)} vs {_s(b)}")
    console.print(f"\n  [bold yellow]🏆 CHAMPION: ({seed_w}) {_s(w)}[/bold yellow]")
    console.print(f"\n  [dim]R64 upsets: {total_upsets}[/dim]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _pct_style(pct: float) -> str:
    if pct >= 75:
        return "[green]"
    elif pct >= 50:
        return "[yellow]"
    else:
        return "[red]"


def _s(name: str) -> str:
    """Strip whitespace from team name for display."""
    return name.strip()


def display_bracket(
    results: dict,
    regions: dict,
    first_four: list,
    console: Console,
):
    """Display bracket-style view showing most likely winners per matchup."""
    n = results["n_sims"]
    adv = results["advancement"]

    console.print("\n[bold underline]BRACKET — Most Likely Outcomes[/bold underline]\n")

    for r_idx in range(4):
        region = regions[r_idx]
        console.print(f"[bold cyan]── {REGION_NAMES[r_idx]} ──[/bold cyan]")

        # Build matchup pairs in bracket order
        teams_in_order = []
        for s in MATCHUP_ORDER:
            team = region.get(s, "TBD")
            teams_in_order.append((s, team))

        # R64 matchups
        r64_pairs = []
        for i in range(0, 16, 2):
            s1, t1 = teams_in_order[i]
            s2, t2 = teams_in_order[i + 1]
            c1 = adv.get(t1, [0] * 7)[0]
            c2 = adv.get(t2, [0] * 7)[0]
            p1 = 100 * c1 / n if n > 0 else 0
            p2 = 100 * c2 / n if n > 0 else 0
            winner = t1 if c1 >= c2 else t2
            w_seed = s1 if c1 >= c2 else s2
            w_pct = max(p1, p2)
            r64_pairs.append((winner, w_seed, w_pct))

            style = _pct_style(w_pct)
            console.print(
                f"  ({s1:>2}) {_s(t1):18s} vs ({s2:>2}) {_s(t2):18s}"
                f"  → {style}{_s(winner)} ({w_pct:.0f}%)[/]"
            )

        # R32
        r32_pairs = []
        for i in range(0, 8, 2):
            w1, ws1, _ = r64_pairs[i]
            w2, ws2, _ = r64_pairs[i + 1]
            c1 = adv.get(w1, [0] * 7)[1]
            c2 = adv.get(w2, [0] * 7)[1]
            winner = w1 if c1 >= c2 else w2
            w_seed = ws1 if c1 >= c2 else ws2
            w_pct = 100 * max(c1, c2) / n if n > 0 else 0
            r32_pairs.append((winner, w_seed, w_pct))

        console.print(f"  [dim]Round of 32:[/dim]")
        for w, ws, wp in r32_pairs:
            style = _pct_style(wp)
            console.print(f"    {style}({ws:>2}) {_s(w)} ({wp:.0f}%)[/]")

        # S16
        s16_pairs = []
        for i in range(0, 4, 2):
            w1, ws1, _ = r32_pairs[i]
            w2, ws2, _ = r32_pairs[i + 1]
            c1 = adv.get(w1, [0] * 7)[2]
            c2 = adv.get(w2, [0] * 7)[2]
            winner = w1 if c1 >= c2 else w2
            w_seed = ws1 if c1 >= c2 else ws2
            w_pct = 100 * max(c1, c2) / n if n > 0 else 0
            s16_pairs.append((winner, w_seed, w_pct))

        console.print(f"  [dim]Sweet 16:[/dim]")
        for w, ws, wp in s16_pairs:
            style = _pct_style(wp)
            console.print(f"    {style}({ws:>2}) {_s(w)} ({wp:.0f}%)[/]")

        # E8
        w1, ws1, _ = s16_pairs[0]
        w2, ws2, _ = s16_pairs[1]
        c1 = adv.get(w1, [0] * 7)[3]
        c2 = adv.get(w2, [0] * 7)[3]
        winner = w1 if c1 >= c2 else w2
        w_seed = ws1 if c1 >= c2 else ws2
        w_pct = 100 * max(c1, c2) / n if n > 0 else 0
        style = _pct_style(w_pct)
        console.print(f"  [dim]Elite 8 → [/dim]{style}({w_seed:>2}) {_s(winner)} ({w_pct:.0f}%)[/]")
        console.print()


def display_results(
    results: dict,
    regions: dict,
    first_four: list,
    focus_stat: str | None,
    focus_label: str,
    baseline_results: dict | None,
    console: Console,
    team_lookup: dict | None = None,
    market_odds: tuple | None = None,
):
    """Display full simulation results."""
    n = results["n_sims"]
    adv = results["advancement"]
    champs = results["champion_counts"]

    # Bracket view first
    display_bracket(results, regions, first_four, console)

    # Region tables with all teams
    ff_display_idx = 0
    for r_idx in range(4):
        region = regions[r_idx]
        table = Table(
            title=f"[bold]{REGION_NAMES[r_idx]} Region[/bold]",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Seed", justify="right", width=4)
        table.add_column("Team", width=20)
        table.add_column("R64 %", justify="right", width=7)
        table.add_column("R32 %", justify="right", width=7)
        table.add_column("S16 %", justify="right", width=7)
        table.add_column("E8 %", justify="right", width=7)

        for seed in MATCHUP_ORDER:
            team = region.get(seed, "TBD")
            if team == "Play-in":
                # Show play-in teams — consume next matching first-four pair
                if ff_display_idx < len(first_four):
                    t1, t2, ff_seed = first_four[ff_display_idx]
                    ff_display_idx += 1
                    for ff_team in (t1, t2):
                        counts = adv.get(ff_team, [0] * 7)
                        pcts = [100 * c / n for c in counts[:4]]
                        row = [f"{ff_seed}*", _s(ff_team)]
                        for p in pcts:
                            row.append(f"{_pct_style(p)}{p:5.1f}%[/]")
                        table.add_row(*row)
                continue
            counts = adv.get(team, [0] * 7)
            pcts = [100 * c / n for c in counts[:4]]
            row = [str(seed), _s(team)]
            for p in pcts:
                row.append(f"{_pct_style(p)}{p:5.1f}%[/]")
            table.add_row(*row)

        console.print(table)
        console.print()

    # Final Four / Championship odds
    ff_table = Table(
        title="[bold]Final Four & Championship Odds[/bold]",
        show_header=True,
        header_style="bold magenta",
    )
    ff_table.add_column("Team", width=20)
    ff_table.add_column("Final Four %", justify="right", width=12)
    ff_table.add_column("Finals %", justify="right", width=12)
    ff_table.add_column("Champion %", justify="right", width=12)

    sorted_teams = sorted(champs.items(), key=lambda x: x[1], reverse=True)
    for team, count in sorted_teams[:16]:
        team_counts = adv.get(team, [0] * 7)
        f4_pct = 100 * team_counts[4] / n
        final_pct = 100 * team_counts[5] / n
        champ_pct = 100 * count / n
        row = [_s(team)]
        for p in [f4_pct, final_pct, champ_pct]:
            row.append(f"{_pct_style(p)}{p:5.1f}%[/]")
        ff_table.add_row(*row)

    console.print(ff_table)
    console.print()

    # Cinderella watch
    cinderellas = []
    for team, counts in adv.items():
        seed = _find_seed(team, regions, first_four)
        if seed is not None and seed >= 10:
            r32_pct = 100 * counts[1] / n
            s16_pct = 100 * counts[2] / n
            e8_pct = 100 * counts[3] / n
            if r32_pct > 5 or s16_pct > 1:
                ts = (team_lookup or {}).get(team.strip())
                c_score = round(ts.cinderella_score) if ts else 0
                cinderellas.append((_s(team), seed, r32_pct, s16_pct, e8_pct, c_score))

    if cinderellas:
        cind_table = Table(
            title="[bold]Cinderella Watch (Seed >= 10)[/bold]",
            show_header=True,
            header_style="bold yellow",
        )
        cind_table.add_column("Seed", justify="right", width=4)
        cind_table.add_column("Team", width=20)
        cind_table.add_column("C-Score", justify="right", width=7)
        cind_table.add_column("R32 %", justify="right", width=8)
        cind_table.add_column("S16 %", justify="right", width=8)
        cind_table.add_column("E8 %", justify="right", width=8)

        cinderellas.sort(key=lambda x: x[5], reverse=True)
        for team, seed, r32_p, s16_p, e8_p, c_score in cinderellas:
            cind_table.add_row(
                str(seed), team, str(c_score),
                f"{r32_p:5.1f}%", f"{s16_p:5.1f}%", f"{e8_p:5.1f}%",
            )
        console.print(cind_table)
        console.print()

    # Impact summary
    if baseline_results and focus_stat:
        _display_impact(results, baseline_results, focus_label, console)

    # Market comparison
    if market_odds:
        _display_market_comparison(results, regions, first_four, market_odds, console)


def _display_market_comparison(results, regions, first_four, market_odds, console):
    """Show Monte Carlo championship % vs market-implied % with edge detection."""
    futures_map, matchup_map = market_odds

    n = results["n_sims"]
    champs = results["champion_counts"]

    if futures_map:
        table = Table(
            title="[bold]Model vs Market — Championship Odds[/bold]",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Team", width=20)
        table.add_column("Seed", justify="right", width=4)
        table.add_column("Model %", justify="right", width=9)
        table.add_column("Market %", justify="right", width=9)
        table.add_column("Edge", justify="right", width=8)
        table.add_column("Best Odds", justify="right", width=10)

        rows = []
        for team_name in set(list(champs.keys()) + list(futures_map.keys())):
            model_pct = 100 * champs.get(team_name, 0) / n
            fut = futures_map.get(team_name)
            if fut is None and model_pct < 0.1:
                continue
            market_pct = 100 * fut.implied_prob if fut else 0.0
            edge = model_pct - market_pct
            seed = _find_seed(team_name, regions, first_four)
            best = f"+{fut.best_price}" if fut and fut.best_price > 0 else (str(fut.best_price) if fut else "")
            rows.append((team_name, seed or 0, model_pct, market_pct, edge, best))

        rows.sort(key=lambda r: abs(r[4]), reverse=True)
        for team_name, seed, model_pct, market_pct, edge, best in rows[:20]:
            sign = "+" if edge > 0 else ""
            color = "[green]" if edge > 1.0 else ("[red]" if edge < -1.0 else "[dim]")
            table.add_row(
                _s(team_name),
                str(seed) if seed else "",
                f"{model_pct:5.1f}%",
                f"{market_pct:5.1f}%" if market_pct > 0 else "[dim]—[/dim]",
                f"{color}{sign}{edge:4.1f}%[/]",
                best,
            )

        console.print(table)
        console.print()

        # Value picks summary
        value_picks = [(t, s, m, mk, e) for t, s, m, mk, e, _ in rows if e > 1.0]
        if value_picks:
            console.print("[bold green]Value Picks[/bold green] (model significantly higher than market):")
            for team, seed, model_pct, market_pct, edge in value_picks[:5]:
                console.print(
                    f"  ({seed:>2}) {_s(team):18s}  model {model_pct:.1f}% vs market {market_pct:.1f}%"
                    f"  [green]+{edge:.1f}% edge[/green]"
                )
            console.print()

        fades = [(t, s, m, mk, e) for t, s, m, mk, e, _ in rows if e < -1.0]
        if fades:
            console.print("[bold red]Market Favorites to Fade[/bold red] (model significantly lower than market):")
            for team, seed, model_pct, market_pct, edge in fades[:5]:
                console.print(
                    f"  ({seed:>2}) {_s(team):18s}  model {model_pct:.1f}% vs market {market_pct:.1f}%"
                    f"  [red]{edge:.1f}% edge[/red]"
                )
            console.print()

    if matchup_map:
        # Find upcoming matchups where both teams are in our bracket
        bracket_matchups = [
            m for m in matchup_map
            if m.home_bracket and m.away_bracket
        ]
        if bracket_matchups:
            table = Table(
                title="[bold]Matchup Odds — Model vs Market[/bold]",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Matchup", width=35)
            table.add_column("Model", justify="right", width=8)
            table.add_column("Market", justify="right", width=8)
            table.add_column("Edge", justify="right", width=8)
            table.add_column("Spread", justify="right", width=7)

            adv = results["advancement"]
            for m in bracket_matchups:
                home = m.home_bracket or _s(m.home_team)
                away = m.away_bracket or _s(m.away_team)
                # Model R64 win rate for the home team
                home_wins = adv.get(home, [0] * 7)[0]
                away_wins = adv.get(away, [0] * 7)[0]
                total_wins = home_wins + away_wins
                if total_wins > 0:
                    model_home_pct = 100 * home_wins / total_wins
                else:
                    model_home_pct = 50.0
                market_home_pct = 100 * m.home_h2h_prob
                edge = model_home_pct - market_home_pct

                sign = "+" if edge > 0 else ""
                color = "[green]" if abs(edge) > 5 else "[dim]"

                # Show the favored team's perspective
                if model_home_pct >= 50:
                    label = f"{_s(home)} vs {_s(away)}"
                    model_str = f"{model_home_pct:.0f}%"
                    market_str = f"{market_home_pct:.0f}%"
                else:
                    label = f"{_s(away)} vs {_s(home)}"
                    model_str = f"{100 - model_home_pct:.0f}%"
                    market_str = f"{100 - market_home_pct:.0f}%"
                    edge = -edge

                sign = "+" if edge > 0 else ""
                color = "[green]" if edge > 5 else ("[red]" if edge < -5 else "[dim]")

                table.add_row(
                    label,
                    model_str,
                    market_str,
                    f"{color}{sign}{edge:.0f}%[/]",
                    f"{m.spread:+.1f}" if m.spread else "",
                )

            console.print(table)
            console.print()


def _find_seed(team: str, regions: dict, first_four: list) -> int | None:
    """Find a team's seed from regions or first four."""
    for r_idx in range(4):
        for s, t in regions[r_idx].items():
            if t == team:
                return s
    for t1, t2, seed in first_four:
        if t1 == team or t2 == team:
            return seed
    return None


def _display_impact(results, baseline_results, focus_label, console):
    """Show how the focus stat shifted championship odds vs baseline."""
    n = results["n_sims"]
    base_n = baseline_results["n_sims"]
    champs = results["champion_counts"]
    base_champs = baseline_results["champion_counts"]

    impact_table = Table(
        title=f"[bold]Impact of Focus: {focus_label}[/bold]",
        show_header=True,
        header_style="bold blue",
    )
    impact_table.add_column("Team", width=20)
    impact_table.add_column("Baseline %", justify="right", width=12)
    impact_table.add_column("Focus %", justify="right", width=10)
    impact_table.add_column("Shift", justify="right", width=8)

    all_teams = set(list(champs.keys()) + list(base_champs.keys()))
    shifts = []
    for team in all_teams:
        base_pct = 100 * base_champs.get(team, 0) / base_n
        focus_pct = 100 * champs.get(team, 0) / n
        shifts.append((_s(team), base_pct, focus_pct, focus_pct - base_pct))

    shifts.sort(key=lambda x: abs(x[3]), reverse=True)
    for team, base_pct, focus_pct, delta in shifts[:10]:
        sign = "+" if delta > 0 else ""
        color = "[green]" if delta > 0 else "[red]"
        impact_table.add_row(
            team, f"{base_pct:5.1f}%", f"{focus_pct:5.1f}%",
            f"{color}{sign}{delta:4.1f}%[/]",
        )
    console.print(impact_table)
    console.print()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="What-If Bracket Simulator")
    parser.add_argument("--bracket", default="bracket.csv", help="Committee bracket CSV path")
    parser.add_argument("--db", default="torvik_game_stats.db", help="Torvik game stats DB path")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--odds", action="store_true", help="Show cached betting odds comparison")
    parser.add_argument("--fetch-live-odds", action="store_true", help="Fetch fresh odds from API (implies --odds)")
    parser.add_argument("--injuries", action="store_true", help="Apply injury adjustments to team strength")
    parser.add_argument("--injury-db", default="injuries.db", help="Injury database path")
    parser.add_argument("--fill-bracket", action="store_true",
                        help="Generate a single filled bracket using composite model with smart upsets")
    args = parser.parse_args()

    if args.fetch_live_odds:
        args.odds = True

    console = Console()
    console.print("[bold]Loading bracket data...[/bold]")

    # Load bracket from committee bracket CSV
    regions, first_four = load_bracket(args.bracket)

    # Collect all bracket team names
    bracket_names = []
    for r_idx in range(4):
        for seed, team in regions[r_idx].items():
            if team != "Play-in":
                bracket_names.append(team)
    for t1, t2, _ in first_four:
        bracket_names.extend([t1, t2])

    # Get Torvik team names from DB
    conn = sqlite3.connect(args.db)
    torvik_names = [
        r[0] for r in conn.execute("SELECT DISTINCT team FROM games ORDER BY team").fetchall()
    ]
    conn.close()

    # Build name mapping (bracket CSV names → Torvik DB names)
    name_map = _build_name_map(bracket_names, torvik_names, console)

    matched = sum(1 for v in name_map.values() if v is not None)
    total = len(name_map)
    console.print(f"Matched [green]{matched}[/green]/{total} teams between bracket and Torvik DB")

    unmatched = [k for k, v in name_map.items() if v is None]
    if unmatched:
        console.print(f"[yellow]Unmatched ({len(unmatched)}):[/yellow] {', '.join(unmatched)}")
        console.print("[dim]Add missing mappings to MASSEY_TO_TORVIK in whatif.py[/dim]")

    # Load team stats from DB
    console.print("[bold]Loading team stats from Torvik DB...[/bold]")
    all_stats: dict[str, TeamStats] = {}
    for torvik_name in torvik_names:
        ts = load_team_stats(args.db, torvik_name)
        if ts:
            all_stats[torvik_name] = ts

    # Build team_lookup: bracket name → TeamStats
    # For matched teams: use real DB stats
    # For unmatched teams: use seed-based priors
    team_lookup: dict[str, TeamStats] = {}
    for r_idx in range(4):
        for seed, team in regions[r_idx].items():
            ms = team.strip()
            if team == "Play-in":
                continue
            torvik = name_map.get(ms)
            if torvik and torvik in all_stats:
                stats = all_stats[torvik]
                stats.seed = seed
                stats.region = r_idx
                team_lookup[ms] = stats
            else:
                team_lookup[ms] = _seed_prior_stats(ms, seed)

    # Also add first-four teams
    for t1, t2, seed in first_four:
        for ff_team in (t1, t2):
            ms = ff_team.strip()
            if ms not in team_lookup:
                torvik = name_map.get(ms)
                if torvik and torvik in all_stats:
                    stats = all_stats[torvik]
                    stats.seed = seed
                    team_lookup[ms] = stats
                else:
                    team_lookup[ms] = _seed_prior_stats(ms, seed)

    # Apply seed-based experience prior for teams that have no scraped value
    for ts in team_lookup.values():
        if ts.experience == 0.0 and ts.seed > 0:
            ts.experience = SEED_EXPERIENCE.get(ts.seed, 1.0)

    # Compute z-scores across all teams in the lookup
    z_scores = compute_z_scores(team_lookup)

    data_teams = sum(1 for ts in team_lookup.values() if ts.has_data)
    console.print(
        f"[green]{data_teams} teams with Torvik data, "
        f"{len(team_lookup) - data_teams} using seed priors[/green]\n"
    )

    # Apply injury adjustments if requested
    if args.injuries:
        try:
            from injuries import init_db as inj_init_db, get_all_team_impacts, display_team_impacts
            inj_conn = inj_init_db(args.injury_db)
            impacts = get_all_team_impacts(inj_conn, list(team_lookup.keys()))
            impacted_count = sum(1 for m in impacts.values() if m < 1.0)
            if impacted_count:
                console.print(f"[bold]Applying injury adjustments to {impacted_count} teams...[/bold]")
                display_team_impacts(impacts, console)
                console.print()
                for team_name, mult in impacts.items():
                    if mult < 1.0 and team_name in team_lookup:
                        ts = team_lookup[team_name]
                        ts.adj_margin *= mult
                        ts.eff_margin *= mult
            else:
                console.print("[green]No bracket teams with injuries.[/green]\n")
            inj_conn.close()
        except Exception as exc:
            console.print(f"[yellow]Could not load injuries: {exc}[/yellow]\n")

    # Load market odds if requested
    market_odds = None
    if args.odds:
        try:
            from odds import (
                fetch_futures, fetch_matchup_odds,
                load_cached_futures, load_cached_matchups,
                _fmt_timestamp,
            )
            bracket_name_set = set(bracket_names)

            if args.fetch_live_odds:
                console.print("[bold]Fetching live betting odds...[/bold]")
                futures = fetch_futures(bracket_name_set)
                matchups = fetch_matchup_odds(bracket_name_set)
                console.print("[green]Odds fetched and cached.[/green]")
            else:
                console.print("[bold]Loading cached betting odds...[/bold]")
                futures, f_ts = load_cached_futures()
                matchups, m_ts = load_cached_matchups()
                if not futures and not matchups:
                    console.print(
                        "[yellow]No cached odds found. "
                        "Run with --fetch-live-odds first.[/yellow]\n"
                    )
                else:
                    ts = f_ts or m_ts
                    console.print(f"[dim]Odds cached at {_fmt_timestamp(ts)}[/dim]")

            # Build futures lookup: bracket name → FuturesOdds
            futures_map = {}
            for f in futures:
                if f.team_bracket and f.team_bracket in bracket_name_set:
                    futures_map[f.team_bracket] = f

            if futures or matchups:
                matched_f = len(futures_map)
                console.print(
                    f"[green]Matched {matched_f}/{len(futures)} futures, "
                    f"{len(matchups)} matchups loaded[/green]\n"
                )
                market_odds = (futures_map, matchups)
        except Exception as exc:
            console.print(f"[yellow]Could not load odds: {exc}[/yellow]\n")

    # Fill bracket mode: generate single bracket and exit
    if args.fill_bracket:
        console.print("[bold]Generating filled bracket (composite model + smart upsets)...[/bold]\n")
        filled = fill_bracket(
            regions, first_four, team_lookup, z_scores,
            rng_seed=args.seed, console=console,
        )
        display_filled_bracket(filled, regions, first_four, team_lookup, console)
        return

    # Run baseline simulation
    console.print("[bold]Running baseline simulation (no focus stat)...[/bold]")
    baseline = simulate_bracket(
        regions, first_four, team_lookup,
        focus_stat=None, z_scores=z_scores,
        n_sims=args.sims, rng_seed=args.seed, console=console,
    )

    # Interactive loop
    while True:
        console.print("\n[bold cyan]Select a focus stat to shift win probabilities:[/bold cyan]")
        for key, (_, label) in FOCUS_STATS.items():
            console.print(f"  [bold]{key}[/bold]. {label}")
        console.print("  [bold]0[/bold]. Baseline (no focus stat)")
        console.print("  [bold]q[/bold]. Quit")

        choice = Prompt.ask(
            "\n[bold]Choose",
            choices=list(FOCUS_STATS.keys()) + ["0", "q"],
        )

        if choice == "q":
            console.print("[bold]Goodbye![/bold]")
            break

        if choice == "0":
            focus = None
            label = "Baseline"
        else:
            focus, label = FOCUS_STATS[choice]

        console.print(f"\n[bold]Simulating with focus: {label}...[/bold]")

        results = simulate_bracket(
            regions, first_four, team_lookup,
            focus_stat=focus, z_scores=z_scores,
            n_sims=args.sims, rng_seed=args.seed, console=console,
        )

        display_results(
            results, regions, first_four,
            focus, label,
            baseline_results=baseline if focus else None,
            console=console,
            team_lookup=team_lookup,
            market_odds=market_odds,
        )


if __name__ == "__main__":
    main()

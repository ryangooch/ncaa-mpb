# ncaab-mpb

This is a Monte Carlo NCAA tournament bracket simulator.
As a user, just pick a stat you think matters, and the simulator will
shift win probabilities accordingly across thousands of bracket simulations.
Then, you can see how your chosen stat changes outcomes.

## Kudos

Massive kudos to statisticians and developers who maintain websites for
college basketball analytics.
This is a toy/learning project that relies on data from various websites
and which is inspired by over a decade of my own nerding out about basketball
by reading folks like KenPom, Bart Torvik, and many more.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Data

The simulator needs two data files:

- **`bracket.csv`** — this year's tournament bracket
    - it's a CSV with columns `team,seed,region,first_four_opponent`
    - region is an index (0=South, 1=West, 2=East, 3=Midwest) or a name
    - First Four teams list each other in the `first_four_opponent` column; regular teams leave it blank.
- **`torvik_game_stats.db`** — SQLite database of game-level Torvik stats.
    - scraped via `scrape_torvik.py`
    - contains a `games` table with per-game efficiency, shooting, turnover, rebounding, and other stats.

## Running the what-if tool

```
uv run python whatif.py
```

This loads the bracket and data, runs a baseline simulation, then drops you
into an interactive CLI menu where you pick a focus stat:

```
1. Adjusted Efficiency Margin (AdjO - AdjD)
2. Raw Efficiency Margin
3. Three-Point Shooting %
4. Three-Point Variance (get hot factor)
5. Late-Season Trend (peaking teams)
6. Turnover Differential (force TOs - commit TOs)
7. Rebounding Differential (OFF reb% - DEF reb%)
8. Free Throw Rate Differential
9. Wins Above Bubble
0. Baseline (no focus stat)
q. Quit
```

Selecting a stat re-runs the simulation with z-score-based probability shifts
for that stat, then prints:

- A bracket view with the most likely winner of each matchup
- Per-region advancement odds (R64 through E8)
- Final Four and championship odds for the top contenders
- A Cinderella watch for seeds 10+
- An impact table showing how championship odds shifted vs the baseline

### Options

| Flag | Default | Description |
|---|---|---|
| `--bracket` | `bracket.csv` | Path to bracket CSV |
| `--db` | `torvik_game_stats.db` | Path to Torvik SQLite DB |
| `--sims` | `10000` | Number of Monte Carlo simulations |
| `--seed` | random | RNG seed for reproducible results |

Example with more sims and a fixed seed:

```
uv run python whatif.py --sims 50000 --seed 42
```

## How it works

Win probabilities come from a logistic model on adjusted efficiency margin,
blended with historical seed-matchup win rates in the first round (70/30 split).
When you pick a focus stat, each team's z-score for that stat shifts their win
probability by up to +/-15%.
Three-point variance gets a small extra underdog boost to model the "getting hot"
effect.

Teams in the bracket that can't be matched to the Torvik DB fall back to
seed-based priors (historical average adjusted margin for that seed line).

## Name matching

The bracket CSV uses Massey-style names, and the Torvik DB uses its own naming.
The `MASSEY_TO_TORVIK` dict in `whatif.py` handles known mismatches.
If you see unmatched teams in the output, add the mapping there.


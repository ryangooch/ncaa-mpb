#!/usr/bin/env python3
"""
Fetch NCAA basketball odds from The Odds API.

Provides head-to-head moneylines, spreads, and championship futures
for comparison against Monte Carlo bracket simulations.

Results are cached in SQLite so repeated runs don't burn API credits.
Use --fetch to pull fresh odds from the API.

Requires THE_ODDS_API_KEY in .env or environment (only for fetching).

Usage:
    uv run python odds.py                    # show cached odds
    uv run python odds.py --fetch            # fetch fresh + display
    uv run python odds.py --fetch --futures  # fetch + show futures only
"""

import json
import os
import sqlite3
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_ncaab"
FUTURES_KEY = "basketball_ncaab_championship_winner"
ODDS_DB_PATH = "odds_cache.db"


def _get_api_key() -> str:
    """Load API key from environment or .env file."""
    key = os.environ.get("THE_ODDS_API_KEY")
    if key:
        return key
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("THE_ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "THE_ODDS_API_KEY not found. Set it in .env or as an environment variable."
    )


def _api_get(url: str) -> dict | list:
    """Make a GET request and return parsed JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None:
            print(f"  [API quota] used: {used}, remaining: {remaining}")
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Name mapping: Odds API full names → bracket.csv short names
# ---------------------------------------------------------------------------

# The Odds API uses full mascot names (e.g., "Duke Blue Devils").
# bracket.csv uses short names (e.g., "Duke").
# This mapping handles non-obvious cases; the rest are matched by prefix.
ODDS_TO_BRACKET = {
    "UConn Huskies": "Connecticut",
    "Iowa State Cyclones": "Iowa State",
    "Michigan St Spartans": "Michigan St",
    "NC State Wolfpack": "NC State",
    "Saint Mary's Gaels": "Saint Mary's CA",
    "Miami Hurricanes": "Miami FL",
    "Miami (OH) RedHawks": "Miami OH",
    "Ohio State Buckeyes": "Ohio St",
    "Utah State Aggies": "Utah St",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "LIU Sharks": "Long Island",
    "McNeese Cowboys": "McNeese",
    "Kennesaw St Owls": "Kennesaw St",
    "Pennsylvania Quakers": "Penn",
    "Cal Baptist Lancers": "Cal Baptist",
    "Queens University Royals": "Queens",
    "North Dakota St Bison": "N Dakota St",
    "Prairie View Panthers": "Prairie View A&M",
    "Tennessee St Tigers": "Tennessee St",
    "Wright St Raiders": "Wright St",
    "St. John's Red Storm": "St. John's",
    "Northern Iowa Panthers": "Northern Iowa",
    "South Florida Bulls": "South Florida",
    "Texas A&M Aggies": "Texas A&M",
    "North Carolina Tar Heels": "North Carolina",
}


def _strip_mascot(full_name: str) -> str:
    """Try to extract the school name by dropping the last word (mascot).

    Works for simple cases like 'Duke Blue Devils' → 'Duke'.
    Multi-word mascots (e.g., 'Red Storm') need the explicit mapping above.
    """
    # First check explicit mapping
    if full_name in ODDS_TO_BRACKET:
        return ODDS_TO_BRACKET[full_name]

    # Try dropping last word(s) and matching
    parts = full_name.split()
    # Try dropping 1 word, then 2 words
    for drop in (1, 2):
        if len(parts) > drop:
            candidate = " ".join(parts[:-drop])
            # Return the candidate — caller will try to match it
            if drop == 1:
                first_try = candidate
            else:
                return first_try  # prefer dropping just 1 word

    return full_name  # fallback


def map_odds_name(full_name: str, bracket_names: set[str]) -> str | None:
    """Map an Odds API team name to a bracket.csv team name.

    Returns None if no match found.
    """
    # Explicit mapping
    if full_name in ODDS_TO_BRACKET:
        mapped = ODDS_TO_BRACKET[full_name]
        if mapped in bracket_names:
            return mapped
        return mapped  # return anyway for reporting

    # Try dropping mascot words
    parts = full_name.split()
    for drop in (1, 2, 3):
        if len(parts) > drop:
            candidate = " ".join(parts[:-drop])
            if candidate in bracket_names:
                return candidate

    return None


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------

def american_to_implied(price: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if price > 0:
        return 100.0 / (price + 100.0)
    else:
        return abs(price) / (abs(price) + 100.0)


def american_to_decimal(price: int) -> float:
    """Convert American odds to decimal odds."""
    if price > 0:
        return (price / 100.0) + 1.0
    else:
        return (100.0 / abs(price)) + 1.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MatchupOdds:
    """Odds for a single head-to-head matchup."""
    event_id: str
    commence_time: str
    home_team: str  # Odds API full name
    away_team: str  # Odds API full name
    home_bracket: str | None = None  # bracket.csv name
    away_bracket: str | None = None  # bracket.csv name
    # Consensus (average across bookmakers)
    home_h2h_prob: float = 0.0
    away_h2h_prob: float = 0.0
    spread: float = 0.0  # home team spread (negative = favored)
    n_books: int = 0


@dataclass
class FuturesOdds:
    """Championship futures odds for a single team."""
    team_full: str  # Odds API full name
    team_bracket: str | None = None  # bracket.csv name
    implied_prob: float = 0.0  # consensus implied probability
    best_price: int = 0  # best American odds available
    avg_price: float = 0.0  # average American odds
    n_books: int = 0


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at  TEXT NOT NULL,
    kind        TEXT NOT NULL  -- 'matchups' or 'futures'
);

CREATE TABLE IF NOT EXISTS cached_matchups (
    snapshot_id     INTEGER NOT NULL REFERENCES odds_snapshots(id),
    event_id        TEXT,
    commence_time   TEXT,
    home_team       TEXT,
    away_team       TEXT,
    home_bracket    TEXT,
    away_bracket    TEXT,
    home_h2h_prob   REAL,
    away_h2h_prob   REAL,
    spread          REAL,
    n_books         INTEGER
);

CREATE TABLE IF NOT EXISTS cached_futures (
    snapshot_id     INTEGER NOT NULL REFERENCES odds_snapshots(id),
    team_full       TEXT,
    team_bracket    TEXT,
    implied_prob    REAL,
    best_price      INTEGER,
    avg_price       REAL,
    n_books         INTEGER
);
"""


def _init_cache(db_path: str = ODDS_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_TABLES)
    return conn


def save_matchups(matchups: list[MatchupOdds], db_path: str = ODDS_DB_PATH) -> int:
    """Save matchup odds to cache. Returns snapshot id."""
    conn = _init_cache(db_path)
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO odds_snapshots (fetched_at, kind) VALUES (?, ?)",
        (now, "matchups"),
    )
    snap_id = cur.lastrowid
    conn.executemany(
        """INSERT INTO cached_matchups
           (snapshot_id, event_id, commence_time, home_team, away_team,
            home_bracket, away_bracket, home_h2h_prob, away_h2h_prob, spread, n_books)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (snap_id, m.event_id, m.commence_time, m.home_team, m.away_team,
             m.home_bracket, m.away_bracket, m.home_h2h_prob, m.away_h2h_prob,
             m.spread, m.n_books)
            for m in matchups
        ],
    )
    conn.commit()
    conn.close()
    return snap_id


def save_futures(futures: list[FuturesOdds], db_path: str = ODDS_DB_PATH) -> int:
    """Save futures odds to cache. Returns snapshot id."""
    conn = _init_cache(db_path)
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO odds_snapshots (fetched_at, kind) VALUES (?, ?)",
        (now, "futures"),
    )
    snap_id = cur.lastrowid
    conn.executemany(
        """INSERT INTO cached_futures
           (snapshot_id, team_full, team_bracket, implied_prob, best_price, avg_price, n_books)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (snap_id, f.team_full, f.team_bracket, f.implied_prob,
             f.best_price, f.avg_price, f.n_books)
            for f in futures
        ],
    )
    conn.commit()
    conn.close()
    return snap_id


def load_cached_matchups(db_path: str = ODDS_DB_PATH) -> tuple[list[MatchupOdds], str | None]:
    """Load most recent cached matchup odds. Returns (matchups, fetched_at)."""
    conn = _init_cache(db_path)
    snap = conn.execute(
        "SELECT id, fetched_at FROM odds_snapshots WHERE kind = 'matchups' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if snap is None:
        conn.close()
        return [], None

    rows = conn.execute(
        "SELECT * FROM cached_matchups WHERE snapshot_id = ?", (snap["id"],)
    ).fetchall()
    conn.close()

    matchups = [
        MatchupOdds(
            event_id=r["event_id"],
            commence_time=r["commence_time"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            home_bracket=r["home_bracket"],
            away_bracket=r["away_bracket"],
            home_h2h_prob=r["home_h2h_prob"],
            away_h2h_prob=r["away_h2h_prob"],
            spread=r["spread"],
            n_books=r["n_books"],
        )
        for r in rows
    ]
    return matchups, snap["fetched_at"]


def load_cached_futures(db_path: str = ODDS_DB_PATH) -> tuple[list[FuturesOdds], str | None]:
    """Load most recent cached futures odds. Returns (futures, fetched_at)."""
    conn = _init_cache(db_path)
    snap = conn.execute(
        "SELECT id, fetched_at FROM odds_snapshots WHERE kind = 'futures' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if snap is None:
        conn.close()
        return [], None

    rows = conn.execute(
        "SELECT * FROM cached_futures WHERE snapshot_id = ?", (snap["id"],)
    ).fetchall()
    conn.close()

    futures = [
        FuturesOdds(
            team_full=r["team_full"],
            team_bracket=r["team_bracket"],
            implied_prob=r["implied_prob"],
            best_price=r["best_price"],
            avg_price=r["avg_price"],
            n_books=r["n_books"],
        )
        for r in rows
    ]
    return futures, snap["fetched_at"]


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def fetch_matchup_odds(bracket_names: set[str] | None = None) -> list[MatchupOdds]:
    """Fetch head-to-head and spread odds for upcoming NCAAB games.

    Costs 2 credits (h2h + spreads, 1 region).
    """
    api_key = _get_api_key()
    url = (
        f"{BASE_URL}/sports/{SPORT_KEY}/odds"
        f"?apiKey={api_key}&regions=us&markets=h2h,spreads"
        f"&oddsFormat=american"
    )

    print("Fetching matchup odds...")
    data = _api_get(url)

    results = []
    for event in data:
        odds = MatchupOdds(
            event_id=event["id"],
            commence_time=event["commence_time"],
            home_team=event["home_team"],
            away_team=event["away_team"],
        )

        if bracket_names:
            odds.home_bracket = map_odds_name(event["home_team"], bracket_names)
            odds.away_bracket = map_odds_name(event["away_team"], bracket_names)

        # Average across bookmakers
        h2h_home_probs = []
        h2h_away_probs = []
        spreads = []

        for bk in event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "h2h":
                    for outcome in mkt["outcomes"]:
                        prob = american_to_implied(outcome["price"])
                        if outcome["name"] == event["home_team"]:
                            h2h_home_probs.append(prob)
                        else:
                            h2h_away_probs.append(prob)
                elif mkt["key"] == "spreads":
                    for outcome in mkt["outcomes"]:
                        if outcome["name"] == event["home_team"]:
                            spreads.append(outcome.get("point", 0))

        if h2h_home_probs:
            # Remove vig by normalizing
            raw_home = sum(h2h_home_probs) / len(h2h_home_probs)
            raw_away = sum(h2h_away_probs) / len(h2h_away_probs) if h2h_away_probs else 1 - raw_home
            total = raw_home + raw_away
            odds.home_h2h_prob = raw_home / total
            odds.away_h2h_prob = raw_away / total
            odds.n_books = len(h2h_home_probs)

        if spreads:
            odds.spread = sum(spreads) / len(spreads)

        results.append(odds)

    save_matchups(results)
    return results


def fetch_futures(bracket_names: set[str] | None = None) -> list[FuturesOdds]:
    """Fetch championship futures odds.

    Costs 1 credit (outrights, 1 region).
    """
    api_key = _get_api_key()
    url = (
        f"{BASE_URL}/sports/{FUTURES_KEY}/odds"
        f"?apiKey={api_key}&regions=us&markets=outrights"
        f"&oddsFormat=american"
    )

    print("Fetching championship futures...")
    data = _api_get(url)

    # Collect odds per team across bookmakers
    team_prices: dict[str, list[int]] = {}
    for event in data:
        for bk in event.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] == "outrights":
                    for outcome in mkt["outcomes"]:
                        name = outcome["name"]
                        price = outcome["price"]
                        team_prices.setdefault(name, []).append(price)

    results = []
    for team_full, prices in team_prices.items():
        fut = FuturesOdds(
            team_full=team_full,
            best_price=max(prices),  # best odds for the bettor
            avg_price=sum(prices) / len(prices),
            n_books=len(prices),
            implied_prob=american_to_implied(
                round(sum(prices) / len(prices))
            ),
        )
        if bracket_names:
            fut.team_bracket = map_odds_name(team_full, bracket_names)
        results.append(fut)

    # Sort by implied probability descending
    results.sort(key=lambda f: f.implied_prob, reverse=True)

    save_futures(results)
    return results


# ---------------------------------------------------------------------------
# Display (standalone mode)
# ---------------------------------------------------------------------------

def _fmt_prob(p: float) -> str:
    return f"{100 * p:.1f}%"


def _fmt_american(price: int | float) -> str:
    p = round(price)
    return f"+{p}" if p > 0 else str(p)


def _fmt_timestamp(iso_str: str) -> str:
    """Format an ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return iso_str or "unknown"


def main():
    import argparse
    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser(description="NCAA basketball odds (cached + live)")
    parser.add_argument("--matchups", action="store_true", help="Show matchup odds")
    parser.add_argument("--futures", action="store_true", help="Show championship futures")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh odds from API (default: use cache)")
    parser.add_argument("--bracket", default="bracket.csv", help="Bracket CSV for name mapping")
    args = parser.parse_args()

    if not args.matchups and not args.futures:
        args.matchups = True
        args.futures = True

    console = Console()

    # Load bracket names for mapping
    bracket_names: set[str] = set()
    try:
        import csv
        with open(args.bracket, newline="") as f:
            for row in csv.DictReader(f):
                bracket_names.add(row["team"].strip())
    except FileNotFoundError:
        console.print("[yellow]bracket.csv not found, skipping name mapping[/yellow]")

    if args.matchups:
        if args.fetch:
            matchups = fetch_matchup_odds(bracket_names)
            console.print("[green]Matchup odds fetched and cached.[/green]")
        else:
            matchups, fetched_at = load_cached_matchups()
            if not matchups:
                console.print("[yellow]No cached matchup odds. Run with --fetch first.[/yellow]")
            else:
                console.print(f"[dim]Using cached matchup odds from {_fmt_timestamp(fetched_at)}[/dim]")

        # Filter to bracket teams only
        bracket_matchups = [
            m for m in matchups
            if m.home_bracket or m.away_bracket
        ]

        table = Table(title="[bold]Tournament Matchup Odds[/bold]", show_header=True,
                      header_style="bold cyan")
        table.add_column("Away", width=22)
        table.add_column("Win%", justify="right", width=7)
        table.add_column("Home", width=22)
        table.add_column("Win%", justify="right", width=7)
        table.add_column("Spread", justify="right", width=7)
        table.add_column("Books", justify="right", width=5)

        for m in bracket_matchups:
            away_name = m.away_bracket or m.away_team
            home_name = m.home_bracket or m.home_team
            table.add_row(
                away_name,
                _fmt_prob(m.away_h2h_prob),
                home_name,
                _fmt_prob(m.home_h2h_prob),
                f"{m.spread:+.1f}" if m.spread else "",
                str(m.n_books),
            )

        if bracket_matchups:
            console.print(table)
        else:
            console.print("[yellow]No upcoming tournament matchups found with odds.[/yellow]")

        # Show non-bracket games too
        other = [m for m in matchups if not m.home_bracket and not m.away_bracket]
        if other:
            console.print(f"\n[dim]{len(other)} non-tournament games also available[/dim]")

    if args.futures:
        if args.fetch:
            futures = fetch_futures(bracket_names)
            console.print("[green]Futures odds fetched and cached.[/green]")
        else:
            futures, fetched_at = load_cached_futures()
            if not futures:
                console.print("[yellow]No cached futures odds. Run with --fetch first.[/yellow]")
                futures = []
            else:
                console.print(f"[dim]Using cached futures from {_fmt_timestamp(fetched_at)}[/dim]")

        if futures:
            table = Table(title="\n[bold]Championship Futures[/bold]", show_header=True,
                          header_style="bold magenta")
            table.add_column("Team", width=22)
            table.add_column("Implied %", justify="right", width=10)
            table.add_column("Best Odds", justify="right", width=10)
            table.add_column("Avg Odds", justify="right", width=10)
            table.add_column("Books", justify="right", width=5)

            for f in futures:
                name = f.team_bracket or f.team_full
                table.add_row(
                    name,
                    _fmt_prob(f.implied_prob),
                    _fmt_american(f.best_price),
                    _fmt_american(f.avg_price),
                    str(f.n_books),
                )

            console.print(table)


if __name__ == "__main__":
    main()

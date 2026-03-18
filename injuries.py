#!/usr/bin/env python3
"""
Scrape NCAA basketball injury data from RotoWire and store in SQLite.

The RotoWire injury report provides a JSON API with player injury status
for college basketball. This module fetches that data, stores it locally,
and provides query functions for the whatif simulator.

Usage:
    uv run python injuries.py                  # Fetch if cache is stale (>6h), else use cache
    uv run python injuries.py --refresh        # Force fetch regardless of cache age
    uv run python injuries.py --no-fetch       # Never fetch; use cached data only
    uv run python injuries.py --max-age 12     # Treat cache as fresh for up to 12 hours
    uv run python injuries.py --show           # Display injury table
    uv run python injuries.py --team Duke      # Show injuries for a team
    uv run python injuries.py --status Out     # Filter by status
"""

import argparse
import json
import sqlite3
import urllib.request
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INJURY_API_URL = (
    "https://www.rotowire.com/cbasketball/tables/injury-report.php"
    "?pos=ALL&conf=ALL&team=ALL"
)
DB_PATH = "injuries.db"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS injuries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER,
    firstname   TEXT,
    lastname    TEXT,
    player      TEXT NOT NULL,
    team        TEXT NOT NULL,
    position    TEXT,
    injury      TEXT,
    status      TEXT,
    fetched_at  TEXT NOT NULL,
    UNIQUE(player, team, fetched_at)
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team);
"""


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_TABLE)
    conn.execute(CREATE_INDEX)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

DEFAULT_MAX_AGE_HOURS = 6


def get_cache_age_hours(conn: sqlite3.Connection) -> float | None:
    """Return how many hours ago the last fetch occurred, or None if no data."""
    row = conn.execute("SELECT MAX(fetched_at) AS ts FROM injuries").fetchone()
    if not row or not row["ts"]:
        return None
    last = datetime.strptime(row["ts"], "%Y-%m-%d %H:%M")
    return (datetime.now() - last).total_seconds() / 3600


def is_cache_fresh(conn: sqlite3.Connection, max_age_hours: float = DEFAULT_MAX_AGE_HOURS) -> bool:
    """Return True if cached data exists and is younger than max_age_hours."""
    age = get_cache_age_hours(conn)
    return age is not None and age < max_age_hours


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_injuries() -> list[dict]:
    """Fetch injury report JSON from RotoWire. Returns list of player dicts."""
    req = urllib.request.Request(INJURY_API_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data


def store_injuries(conn: sqlite3.Connection, injuries: list[dict]) -> int:
    """Insert injury records into the database. Returns count inserted."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sql = """
    INSERT OR IGNORE INTO injuries
        (player_id, firstname, lastname, player, team, position, injury, status, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (
            int(p.get("ID", 0)) or None,
            p.get("firstname", ""),
            p.get("lastname", ""),
            p.get("player", "").strip(),
            p.get("team", "").strip(),
            p.get("position", "").strip(),
            p.get("injury", "").strip(),
            p.get("status", "").strip(),
            now,
        )
        for p in injuries
        if p.get("player") and p.get("team")
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

# Map RotoWire team names → bracket CSV names (Massey format).
# Only teams that differ need entries here.
ROTOWIRE_TO_BRACKET = {
    "NC State": "NC State",
    "St. John's": "St. John's",
    "Iowa State": "Iowa State",
    "Michigan State": "Michigan St",
    "Ohio State": "Ohio St",
    "Utah State": "Utah St",
    "McNeese State": "McNeese",
    "North Dakota State": "N Dakota St",
    "Kennesaw State": "Kennesaw St",
    "Saint Mary's": "Saint Mary's CA",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Texas A&M": "Texas A&M",
    "Prairie View": "Prairie View A&M",
    "LIU Brooklyn": "Long Island",
    "LIU": "Long Island",
}


def _normalize_team(rotowire_name: str) -> str:
    """Convert a RotoWire team name to the bracket CSV name."""
    return ROTOWIRE_TO_BRACKET.get(rotowire_name, rotowire_name)


@dataclass
class PlayerInjury:
    player: str
    team: str           # RotoWire team name
    bracket_team: str   # bracket CSV team name
    position: str
    injury: str
    status: str         # "Out", "Out For Season", "Game Time Decision"


# Status severity for sorting and impact weighting
STATUS_SEVERITY = {
    "Out For Season": 3,
    "Out": 2,
    "Game Time Decision": 1,
}


def get_latest_injuries(
    conn: sqlite3.Connection,
    team: str | None = None,
    status: str | None = None,
) -> list[PlayerInjury]:
    """Get the most recent injury snapshot, optionally filtered."""
    # Find the latest fetch timestamp
    row = conn.execute("SELECT MAX(fetched_at) AS ts FROM injuries").fetchone()
    if not row or not row["ts"]:
        return []
    latest = row["ts"]

    sql = "SELECT * FROM injuries WHERE fetched_at = ?"
    params: list = [latest]

    if team:
        sql += " AND team LIKE ?"
        params.append(f"%{team}%")
    if status:
        sql += " AND status = ?"
        params.append(status)

    sql += " ORDER BY team, status, player"
    rows = conn.execute(sql, params).fetchall()

    return [
        PlayerInjury(
            player=r["player"],
            team=r["team"],
            bracket_team=_normalize_team(r["team"]),
            position=r["position"],
            injury=r["injury"],
            status=r["status"],
        )
        for r in rows
    ]


def get_team_injury_impact(
    conn: sqlite3.Connection, bracket_team: str
) -> float:
    """Compute a penalty multiplier (0.0 to 1.0) for a team based on injuries.

    Returns a value that can be used to scale down a team's adj_margin.
    1.0 = no injured players, lower = more impacted.

    This is a simple fallback used when player stats are unavailable.
    The whatif.py simulator uses the more accurate PPG-relative penalty.

    Heuristic: each player "Out" costs ~2% of team strength,
    "Out For Season" costs ~2.5%, "Game Time Decision" costs ~0.5%.
    Capped at 15% total penalty.
    """
    injuries = get_latest_injuries(conn, team=bracket_team)
    if not injuries:
        # Also try the bracket name directly since team names may not match
        return 1.0

    penalty = 0.0
    for inj in injuries:
        if inj.status == "Out For Season":
            penalty += 0.025
        elif inj.status == "Out":
            penalty += 0.020
        elif inj.status == "Game Time Decision":
            penalty += 0.005

    penalty = min(penalty, 0.15)
    return 1.0 - penalty


def get_all_team_impacts(
    conn: sqlite3.Connection, bracket_teams: list[str]
) -> dict[str, float]:
    """Get injury impact multipliers for all bracket teams.

    Returns dict mapping bracket team name → multiplier (1.0 = healthy).

    This is a simple fallback used when player stats are unavailable.
    The whatif.py simulator uses the more accurate PPG-relative penalty.
    """
    # Build reverse lookup: bracket_name → rotowire search terms
    all_injuries = get_latest_injuries(conn)
    if not all_injuries:
        return {t: 1.0 for t in bracket_teams}

    # Group by bracket team name
    team_injuries: dict[str, list[PlayerInjury]] = {}
    for inj in all_injuries:
        team_injuries.setdefault(inj.bracket_team, []).append(inj)

    result = {}
    for bt in bracket_teams:
        injuries = team_injuries.get(bt, [])
        penalty = 0.0
        for inj in injuries:
            if inj.status == "Out For Season":
                penalty += 0.025
            elif inj.status == "Out":
                penalty += 0.020
            elif inj.status == "Game Time Decision":
                penalty += 0.005
        penalty = min(penalty, 0.15)
        result[bt] = 1.0 - penalty

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_injuries(injuries: list[PlayerInjury], console: Console):
    """Pretty-print injury list."""
    if not injuries:
        console.print("[dim]No injuries found.[/dim]")
        return

    table = Table(title="NCAA Basketball Injury Report", show_header=True,
                  header_style="bold cyan")
    table.add_column("Team", width=22)
    table.add_column("Player", width=22)
    table.add_column("Pos", width=4)
    table.add_column("Injury", width=18)
    table.add_column("Status", width=20)

    for inj in injuries:
        status_color = {
            "Out For Season": "[red]",
            "Out": "[yellow]",
            "Game Time Decision": "[green]",
        }.get(inj.status, "[dim]")

        table.add_row(
            inj.team,
            inj.player,
            inj.position,
            inj.injury,
            f"{status_color}{inj.status}[/]",
        )

    console.print(table)


def display_team_impacts(
    impacts: dict[str, float], console: Console
):
    """Show teams with injury penalties."""
    impacted = {t: m for t, m in sorted(impacts.items()) if m < 1.0}
    if not impacted:
        console.print("[green]No bracket teams with significant injuries.[/green]")
        return

    table = Table(title="Injury Impact on Bracket Teams", show_header=True,
                  header_style="bold yellow")
    table.add_column("Team", width=22)
    table.add_column("Health %", justify="right", width=10)
    table.add_column("Penalty", justify="right", width=10)

    for team, mult in sorted(impacted.items(), key=lambda x: x[1]):
        pct = mult * 100
        pen = (1.0 - mult) * 100
        color = "[red]" if pen >= 5 else "[yellow]"
        table.add_row(team, f"{pct:.1f}%", f"{color}-{pen:.1f}%[/]")

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="NCAA Basketball Injury Scraper")
    parser.add_argument("--db", default=DB_PATH, help=f"SQLite database path (default: {DB_PATH})")
    parser.add_argument("--show", action="store_true", help="Display injuries after fetching")
    parser.add_argument("--team", type=str, default=None, help="Filter by team name")
    parser.add_argument("--status", type=str, default=None,
                        help="Filter by status (Out, 'Out For Season', 'Game Time Decision')")
    parser.add_argument("--impacts", action="store_true",
                        help="Show injury impact on bracket teams")

    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument("--refresh", action="store_true",
                             help="Force fetch new data even if cache is fresh")
    fetch_group.add_argument("--no-fetch", action="store_true",
                             help="Never fetch; use cached data only")
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_HOURS, metavar="HOURS",
                        help=f"Cache TTL in hours (default: {DEFAULT_MAX_AGE_HOURS})")
    args = parser.parse_args()

    console = Console()
    conn = init_db(args.db)

    should_fetch = False
    if args.refresh:
        should_fetch = True
    elif not args.no_fetch:
        if is_cache_fresh(conn, args.max_age):
            age = get_cache_age_hours(conn)
            console.print(f"[dim]Cache is fresh ({age:.1f}h old). Use --refresh to force update.[/dim]")
        else:
            should_fetch = True

    if should_fetch:
        console.print("[bold]Fetching injury data from RotoWire...[/bold]")
        try:
            data = fetch_injuries()
            count = store_injuries(conn, data)
            console.print(f"[green]Stored {count} injury records.[/green]")
        except Exception as exc:
            console.print(f"[red]Error fetching injuries: {exc}[/red]")
            console.print("[dim]Falling back to cached data.[/dim]")

    injuries = get_latest_injuries(conn, team=args.team, status=args.status)

    if args.show or args.team or args.status:
        display_injuries(injuries, console)

    if args.impacts:
        # Load bracket teams
        from bracket import load_bracket
        regions, first_four = load_bracket("bracket.csv")
        bracket_teams = []
        for r_idx in range(4):
            for seed, team in regions[r_idx].items():
                if team != "Play-in":
                    bracket_teams.append(team)
        for t1, t2, _ in first_four:
            bracket_teams.extend([t1, t2])

        impacts = get_all_team_impacts(conn, bracket_teams)
        display_team_impacts(impacts, console)

    if not (args.show or args.team or args.status or args.impacts):
        console.print(f"[dim]{len(injuries)} injuries in database. "
                      f"Use --show to display, --impacts for bracket impact.[/dim]")

    conn.close()


if __name__ == "__main__":
    main()

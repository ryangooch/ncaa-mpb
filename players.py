#!/usr/bin/env python3
"""
Scrape player contribution data from barttorvik.com team pages.

Data is fetched from the "Player Stats" table on each team's page
(e.g. barttorvik.com/team.php?team=SMU&year=2026) using the same
Selenium/Playwright browser helper as scrape_torvik.py.

Stores per-player stats in SQLite (torvik_game_stats.db) and provides
analysis functions for:
  - Contribution lost due to injury
  - Team star-power concentration (HHI)
  - "Stud" detection (high minutes + high usage)
  - Cross-referencing with RotoWire injury reports

Usage:
    uv run python players.py                        # fetch all bracket teams (cache-aware)
    uv run python players.py --refresh              # force re-fetch
    uv run python players.py --no-fetch             # cache only
    uv run python players.py --show --team Duke     # display player stats for a team
    uv run python players.py --cross-ref            # injury impact w/ real contribution %
    uv run python players.py --studs                # list stud players across bracket teams
    uv run python players.py --concentration        # team minute concentration (HHI)
    uv run python players.py --team "N.C. State"    # single team (Torvik name)
"""

import argparse
import difflib
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime

from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table

from scrape_torvik import get_page, TORVIK_TEAM_URL, REQUEST_DELAY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = "torvik_game_stats.db"
YEAR = 2026
DEFAULT_MAX_AGE_HOURS = 24.0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS player_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    team         TEXT    NOT NULL,
    year         INTEGER NOT NULL,
    player       TEXT    NOT NULL,
    num          TEXT,
    yr_class     TEXT,
    height       TEXT,
    gp           INTEGER,
    min_pct      REAL,
    ortg         REAL,
    usg_pct      REAL,
    efg_pct      REAL,
    ts_pct       REAL,
    orb_pct      REAL,
    drb_pct      REAL,
    ast_pct      REAL,
    to_pct       REAL,
    blk_pct      REAL,
    stl_pct      REAL,
    ftr          REAL,
    two_pt_pct   REAL,
    three_pt_pct REAL,
    ft_pct       REAL,
    bpm          REAL,
    ppg          REAL,
    fetched_at   TEXT NOT NULL,
    UNIQUE(team, year, player)
);
"""
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_player_stats_team ON player_stats(team, year);
"""


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def get_cache_age_hours(conn: sqlite3.Connection, team: str, year: int = YEAR) -> float | None:
    """Return age in hours of most recent player_stats for a team, or None."""
    row = conn.execute(
        "SELECT MAX(fetched_at) AS ts FROM player_stats WHERE team = ? AND year = ?",
        (team, year),
    ).fetchone()
    if not row or not row["ts"]:
        return None
    last = datetime.strptime(row["ts"], "%Y-%m-%d %H:%M")
    return (datetime.now() - last).total_seconds() / 3600


def get_global_cache_age_hours(conn: sqlite3.Connection, year: int = YEAR) -> float | None:
    """Return age in hours of most recent player_stats snapshot, or None."""
    row = conn.execute(
        "SELECT MAX(fetched_at) AS ts FROM player_stats WHERE year = ?", (year,)
    ).fetchone()
    if not row or not row["ts"]:
        return None
    last = datetime.strptime(row["ts"], "%Y-%m-%d %H:%M")
    return (datetime.now() - last).total_seconds() / 3600


def is_team_cache_fresh(
    conn: sqlite3.Connection,
    team: str,
    year: int = YEAR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> bool:
    age = get_cache_age_hours(conn, team, year)
    return age is not None and age < max_age_hours


def is_cache_fresh(
    conn: sqlite3.Connection,
    year: int = YEAR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> bool:
    age = get_global_cache_age_hours(conn, year)
    return age is not None and age < max_age_hours


# ---------------------------------------------------------------------------
# Fetching & parsing from team pages
# ---------------------------------------------------------------------------


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _find_player_stats_table(soup: BeautifulSoup):
    """Locate the Player Stats table on a Torvik team page."""
    # Look for a heading containing "Player Stats" then grab the next table
    for tag in soup.find_all(["h2", "h3", "h4", "b", "strong"]):
        if "player stats" in tag.get_text(strip=True).lower():
            table = tag.find_next("table")
            if table:
                return table

    # Fallback: table whose own text contains "Player Stats"
    for table in soup.find_all("table"):
        # Check first row / caption
        first_text = ""
        cap = table.find("caption")
        if cap:
            first_text = cap.get_text(strip=True).lower()
        if not first_text:
            first_row = table.find("tr")
            if first_row:
                first_text = first_row.get_text(strip=True).lower()
        if "player stats" in first_text:
            return table

    # Fallback: look for a table whose header row contains Min% and Usg
    for table in soup.find_all("table"):
        ths = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        header_str = " ".join(ths)
        if "min" in header_str and "usg" in header_str:
            return table
    return None


def _build_col_map(header_row) -> dict[str, int]:
    """
    Build header-name → data-cell-index mapping, expanding colspans.

    Torvik's header has an empty spacer <th> after the Player colspan=3 that
    doesn't produce a <td> in data rows, so we skip empty-name headers.
    Shooting columns (colspan=2) expand to made-att + pct cells.
    """
    col_map: dict[str, int] = {}
    data_idx = 0
    for cell in header_row.find_all(["th", "td"]):
        name = cell.get_text(strip=True).lower().rstrip("%").strip()
        span = int(cell.get("colspan", 1))
        # Skip empty spacer headers (they don't produce data cells)
        if not name:
            continue
        if name not in col_map:
            col_map[name] = data_idx
        data_idx += span
    return col_map


def parse_player_stats_table(html: str, team: str, year: int = YEAR) -> list[dict]:
    """
    Parse the Player Stats table from a Torvik team page.

    Data row layout (confirmed via inspection):
      [0]  Rk              [11] PRPG! (≈PPG)   [22] OR%
      [1]  Pick            [12] D-PRPG          [23] DR%
      [2]  Class/Num       [13] BPM             [24] Ast%
      [3]  Height          [14] OBPM            [25] TO%
      [4]  Player Name     [15] DBPM            [26] A/TO
      [5]  247 Composite   [16] ORtg            [27] Blk%
      [6]  Team            [17] D-Rtg           [28] Stl%
      [7]  Conf            [18] Usg%            [29] FTR
      [8]  G               [19] eFG%            [30] FC/40
      [9]  Role            [20] TS%             [31-32] Dunks m-a, pct
      [10] Min%            [21] ...             [33-34] Close2 m-a, pct
                                                [35-36] Far2 m-a, pct
                                                [37-38] FT m-a, pct
                                                [39-40] 2P m-a, pct
                                                [41] 3PR  [42] 3P/100
                                                [43-44] 3P m-a, pct

    Returns a list of dicts ready for DB insertion.
    """
    soup = BeautifulSoup(html, "lxml")
    table = _find_player_stats_table(soup)
    if table is None:
        print(f"  WARNING: could not find player stats table for {team}")
        return []

    header_row = table.find("tr")
    if not header_row:
        return []

    col_map = _build_col_map(header_row)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    records = []

    for row in table.find_all("tr")[1:]:  # skip header
        cells = row.find_all(["td", "th"])
        if len(cells) < 20:
            continue
        raw = [c.get_text(strip=True) for c in cells]

        # Player name is at index 4 (within the Player colspan=3 span)
        player_name = raw[4] if len(raw) > 4 else ""
        if not player_name or player_name.lower() in ("player", "total", "totals"):
            continue

        # Parse class and jersey number from e.g. "#2Sr", "#0Jr"
        class_num = raw[2] if len(raw) > 2 else ""
        height = raw[3] if len(raw) > 3 else ""
        yr_class = None
        num = None
        m = re.match(r"#?(\d+)\s*(Fr|So|Jr|Sr)", class_num, re.IGNORECASE)
        if m:
            num = m.group(1)
            yr_class = m.group(2)

        def _val(idx: int) -> float | None:
            if idx >= len(raw):
                return None
            return _safe_float(raw[idx])

        records.append({
            "team":         team,
            "year":         year,
            "player":       player_name,
            "num":          num,
            "yr_class":     yr_class,
            "height":       height if height else None,
            "gp":           int(_val(8)) if _val(8) is not None else None,
            "min_pct":      _val(10),
            "ortg":         _val(16),
            "usg_pct":      _val(18),
            "efg_pct":      _val(19),
            "ts_pct":       _val(20),
            "orb_pct":      _val(22),
            "drb_pct":      _val(23),
            "ast_pct":      _val(24),
            "to_pct":       _val(25),
            "blk_pct":      _val(27),
            "stl_pct":      _val(28),
            "ftr":          _val(29),
            "two_pt_pct":   _safe_float(raw[39]) if len(raw) > 39 else None,   # 2P pct
            "three_pt_pct": _safe_float(raw[43]) if len(raw) > 43 else None,   # 3P pct
            "ft_pct":       _safe_float(raw[37]) if len(raw) > 37 else None,   # FT pct
            "bpm":          _val(13),
            "ppg":          _val(11),  # PRPG!
            "fetched_at":   now,
        })
    return records


def fetch_team_players(team: str, year: int = YEAR) -> list[dict]:
    """Fetch and parse player stats for a single team from its Torvik page."""
    url_team = team.replace(" ", "+")
    url = TORVIK_TEAM_URL.format(team=url_team, year=year)
    html = get_page(url)
    return parse_player_stats_table(html, team, year)


# ---------------------------------------------------------------------------
# Database writes
# ---------------------------------------------------------------------------

_INSERT_SQL = """
INSERT OR REPLACE INTO player_stats (
    team, year, player, num, yr_class, height, gp,
    min_pct, ortg, usg_pct, efg_pct, ts_pct,
    orb_pct, drb_pct, ast_pct, to_pct, blk_pct, stl_pct,
    ftr, two_pt_pct, three_pt_pct, ft_pct, bpm, ppg, fetched_at
) VALUES (
    :team, :year, :player, :num, :yr_class, :height, :gp,
    :min_pct, :ortg, :usg_pct, :efg_pct, :ts_pct,
    :orb_pct, :drb_pct, :ast_pct, :to_pct, :blk_pct, :stl_pct,
    :ftr, :two_pt_pct, :three_pt_pct, :ft_pct, :bpm, :ppg, :fetched_at
)
"""


def store_players(conn: sqlite3.Connection, records: list[dict]) -> int:
    conn.executemany(_INSERT_SQL, records)
    conn.commit()
    return len(records)


# ---------------------------------------------------------------------------
# Scraping orchestrator
# ---------------------------------------------------------------------------


def fetch_and_store_team(
    team: str,
    conn: sqlite3.Connection,
    year: int = YEAR,
) -> int:
    """Fetch player stats for one team and store in DB. Returns record count."""
    records = fetch_team_players(team, year)
    return store_players(conn, records)


def fetch_and_store_all(
    teams: list[str],
    conn: sqlite3.Connection,
    year: int = YEAR,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    force: bool = False,
    console: Console | None = None,
) -> int:
    """
    Fetch player stats for all given teams and store in DB.
    Skips teams whose cache is still fresh unless force=True.
    Returns total number of records stored.
    """
    total = 0
    fetched = 0
    for i, team in enumerate(teams, 1):
        if not force and is_team_cache_fresh(conn, team, year, max_age_hours):
            if console:
                console.print(f"  [{i}/{len(teams)}] {team} … [dim]cached[/dim]")
            continue
        if console:
            console.print(f"  [{i}/{len(teams)}] {team} … ", end="")
        try:
            n = fetch_and_store_team(team, conn, year)
            total += n
            fetched += 1
            if console:
                console.print(f"[green]{n} players[/green]")
        except Exception as exc:
            if console:
                console.print(f"[red]ERROR: {exc}[/red]")
        if fetched > 0 and i < len(teams):
            time.sleep(random.uniform(*REQUEST_DELAY))
    return total


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


@dataclass
class PlayerStat:
    team: str
    player: str
    yr_class: str | None
    height: str | None
    gp: int | None
    min_pct: float | None
    ortg: float | None
    usg_pct: float | None
    efg_pct: float | None
    ts_pct: float | None
    blk_pct: float | None
    stl_pct: float | None
    bpm: float | None
    ppg: float | None


def get_player_stats(
    conn: sqlite3.Connection,
    team: str,
    year: int = YEAR,
) -> list[PlayerStat]:
    """Return all player stats for a team, sorted by min_pct descending."""
    rows = conn.execute(
        "SELECT * FROM player_stats WHERE team = ? AND year = ? ORDER BY min_pct DESC",
        (team, year),
    ).fetchall()
    return [
        PlayerStat(
            team=r["team"],
            player=r["player"],
            yr_class=r["yr_class"],
            height=r["height"],
            gp=r["gp"],
            min_pct=r["min_pct"],
            ortg=r["ortg"],
            usg_pct=r["usg_pct"],
            efg_pct=r["efg_pct"],
            ts_pct=r["ts_pct"],
            blk_pct=r["blk_pct"],
            stl_pct=r["stl_pct"],
            bpm=r["bpm"],
            ppg=r["ppg"],
        )
        for r in rows
    ]


def get_all_teams_in_db(conn: sqlite3.Connection, year: int = YEAR) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT team FROM player_stats WHERE year = ? ORDER BY team",
        (year,),
    ).fetchall()
    return [r["team"] for r in rows]


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def contribution_hhi(players: list[PlayerStat]) -> float:
    """
    Herfindahl-Hirschman Index of minute concentration (0-1).

    ~0.10 = balanced  |  ~0.20+ = star-dependent  |  ~0.40+ = extreme
    """
    vals = [p.min_pct for p in players if (p.min_pct or 0) > 0]
    if not vals:
        return 0.0
    total = sum(vals)
    return sum((v / total) ** 2 for v in vals) if total else 0.0


def get_studs(
    players: list[PlayerStat],
    min_pct_threshold: float = 22.0,
    usg_threshold: float = 25.0,
) -> list[PlayerStat]:
    """
    A 'stud' has heavy minutes (≥ min_pct_threshold%) AND
    high usage (≥ usg_threshold%). Sorted by usage desc.
    """
    studs = [
        p for p in players
        if (p.min_pct or 0) >= min_pct_threshold
        and (p.usg_pct or 0) >= usg_threshold
    ]
    studs.sort(key=lambda p: p.usg_pct or 0, reverse=True)
    return studs


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    name = re.sub(r"['\-\.\,]", "", name.lower())
    return " ".join(name.split())


def match_injured_to_roster(
    injured_names: list[str],
    roster: list[PlayerStat],
    cutoff: float = 0.75,
) -> list[tuple[str, PlayerStat]]:
    """Fuzzy-match injury names to roster. Returns (injury_name, PlayerStat) pairs."""
    norm_roster = {normalize_name(p.player): p for p in roster}
    matches = []
    matched_norms = set()

    for inj_name in injured_names:
        norm = normalize_name(inj_name)
        if norm in norm_roster:
            matches.append((inj_name, norm_roster[norm]))
            matched_norms.add(norm)
        else:
            close = difflib.get_close_matches(norm, norm_roster.keys(), n=1, cutoff=cutoff)
            if close and close[0] not in matched_norms:
                matches.append((inj_name, norm_roster[close[0]]))
                matched_norms.add(close[0])

    return matches


def get_contribution_lost(
    roster: list[PlayerStat],
    injured_names: list[str],
    cutoff: float = 0.75,
) -> dict:
    """
    Given a roster and injured player names, compute:
      total_min_pct_lost  — sum of min_pct for matched injured players
      matched             — [(injury_name, PlayerStat), ...]
      unmatched           — injury names with no roster match
    """
    matched = match_injured_to_roster(injured_names, roster, cutoff)
    matched_set = {normalize_name(n) for n, _ in matched}
    unmatched = [n for n in injured_names if normalize_name(n) not in matched_set]
    return {
        "total_min_pct_lost": sum(p.min_pct or 0 for _, p in matched),
        "matched": matched,
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# Team name helpers (bracket/Massey → Torvik)
# ---------------------------------------------------------------------------

MASSEY_TO_TORVIK = {
    "St Mary's CA":   "Saint Mary's",
    "Saint Mary's CA":"Saint Mary's",
    "NC State":       "N.C. State",
    "Miami FL":       "Miami FL",
    "Iowa St":        "Iowa St.",
    "Michigan St":    "Michigan St.",
    "Ohio St":        "Ohio St.",
    "Utah St":        "Utah St.",
    "McNeese St":     "McNeese St.",
    "Kennesaw St":    "Kennesaw St.",
    "St. John's":     "St. John's",
    "McNeese":        "McNeese St.",
    "N Dakota St":    "North Dakota St.",
    "Iowa State":     "Iowa St.",
    "Long Island":    "LIU",
}


def bracket_to_torvik(bracket_name: str) -> str:
    return MASSEY_TO_TORVIK.get(bracket_name, bracket_name)


def load_bracket_teams(csv_path: str = "bracket.csv") -> list[str]:
    """Return Torvik team names for all bracket teams."""
    from bracket import load_bracket
    regions, first_four = load_bracket(csv_path)
    teams = []
    for r_idx in range(4):
        for seed, team in regions[r_idx].items():
            if team != "Play-in":
                teams.append(bracket_to_torvik(team))
    for t1, t2, _ in first_four:
        teams.extend([bracket_to_torvik(t1), bracket_to_torvik(t2)])
    return sorted(set(teams))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_players(players: list[PlayerStat], team: str, console: Console):
    if not players:
        console.print(f"[yellow]No player data for {team}.[/yellow]")
        return

    table = Table(
        title=f"[bold]{team}[/bold] — Player Contributions",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Player", width=22)
    table.add_column("Cl", width=4)
    table.add_column("Ht", width=5)
    table.add_column("G", justify="right", width=3)
    table.add_column("Min%", justify="right", width=6)
    table.add_column("ORtg", justify="right", width=6)
    table.add_column("Usg%", justify="right", width=6)
    table.add_column("eFG%", justify="right", width=6)
    table.add_column("TS%", justify="right", width=6)
    table.add_column("BPM", justify="right", width=6)
    table.add_column("PPG", justify="right", width=6)

    for p in players:
        table.add_row(
            p.player,
            p.yr_class or "",
            p.height or "",
            str(p.gp) if p.gp else "",
            f"{p.min_pct:.1f}" if p.min_pct is not None else "",
            f"{p.ortg:.0f}" if p.ortg is not None else "",
            f"{p.usg_pct:.1f}" if p.usg_pct is not None else "",
            f"{p.efg_pct:.1f}" if p.efg_pct is not None else "",
            f"{p.ts_pct:.1f}" if p.ts_pct is not None else "",
            f"{p.bpm:+.2f}" if p.bpm is not None else "",
            f"{p.ppg:.1f}" if p.ppg is not None else "",
        )

    hhi = contribution_hhi(players)
    studs = get_studs(players)
    stud_names = ", ".join(p.player for p in studs) if studs else "none"
    console.print(table)
    console.print(
        f"  HHI concentration: [bold]{hhi:.3f}[/bold]  |  "
        f"Stud(s): [bold yellow]{stud_names}[/bold yellow]"
    )


def display_concentration(
    teams: list[str], conn: sqlite3.Connection, year: int, console: Console
):
    table = Table(
        title="Team Minute Concentration (HHI)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Team", width=26)
    table.add_column("HHI", justify="right", width=7)
    table.add_column("Top Player", width=22)
    table.add_column("Min%", justify="right", width=6)
    table.add_column("Usg%", justify="right", width=6)
    table.add_column("Stud?", width=6)

    rows_data = []
    for team in teams:
        players = get_player_stats(conn, team, year)
        if not players:
            continue
        hhi = contribution_hhi(players)
        top = players[0]
        studs = get_studs(players)
        rows_data.append((team, hhi, top, bool(studs)))

    rows_data.sort(key=lambda x: x[1], reverse=True)

    for team, hhi, top, has_stud in rows_data:
        color = "[red]" if hhi >= 0.25 else "[yellow]" if hhi >= 0.18 else "[green]"
        stud_flag = "[bold yellow]YES[/bold yellow]" if has_stud else "[dim]no[/dim]"
        table.add_row(
            team,
            f"{color}{hhi:.3f}[/]",
            top.player,
            f"{top.min_pct:.1f}" if top.min_pct is not None else "",
            f"{top.usg_pct:.1f}" if top.usg_pct is not None else "",
            stud_flag,
        )

    console.print(table)


def display_studs(
    teams: list[str], conn: sqlite3.Connection, year: int, console: Console
):
    table = Table(
        title="Stud Players (Min% >= 22 & Usg% >= 25)",
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Team", width=24)
    table.add_column("Player", width=22)
    table.add_column("Cl", width=4)
    table.add_column("Min%", justify="right", width=6)
    table.add_column("Usg%", justify="right", width=6)
    table.add_column("ORtg", justify="right", width=6)
    table.add_column("BPM", justify="right", width=7)
    table.add_column("PPG", justify="right", width=6)

    found_any = False
    for team in sorted(teams):
        players = get_player_stats(conn, team, year)
        for p in get_studs(players):
            found_any = True
            table.add_row(
                team,
                p.player,
                p.yr_class or "",
                f"{p.min_pct:.1f}" if p.min_pct is not None else "",
                f"{p.usg_pct:.1f}" if p.usg_pct is not None else "",
                f"{p.ortg:.0f}" if p.ortg is not None else "",
                f"{p.bpm:+.2f}" if p.bpm is not None else "",
                f"{p.ppg:.1f}" if p.ppg is not None else "",
            )

    if found_any:
        console.print(table)
    else:
        console.print("[yellow]No stud players found (adjust thresholds with source).[/yellow]")


def display_cross_ref(
    teams: list[str],
    conn: sqlite3.Connection,
    year: int,
    inj_conn: sqlite3.Connection,
    console: Console,
):
    """Cross-reference injury reports with player contribution data."""
    from injuries import get_latest_injuries

    all_injuries = get_latest_injuries(inj_conn)
    if not all_injuries:
        console.print("[yellow]No injury data. Run: uv run python injuries.py[/yellow]")
        return

    # Group by bracket team name
    team_inj_map: dict[str, list] = {}
    for inj in all_injuries:
        team_inj_map.setdefault(inj.bracket_team, []).append(inj)

    table = Table(
        title="Injury Impact (Player Contribution)",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Team", width=24)
    table.add_column("Injured Player", width=22)
    table.add_column("Status", width=20)
    table.add_column("Min%", justify="right", width=6)
    table.add_column("Usg%", justify="right", width=6)
    table.add_column("BPM", justify="right", width=7)
    table.add_column("PPG", justify="right", width=6)

    team_totals = []

    for torvik_team in teams:
        roster = get_player_stats(conn, torvik_team, year)
        if not roster:
            continue

        # Find matching bracket name
        bracket_name = next(
            (k for k, v in MASSEY_TO_TORVIK.items() if v == torvik_team),
            torvik_team,
        )

        injuries = (
            team_inj_map.get(bracket_name, [])
            or team_inj_map.get(torvik_team, [])
        )
        if not injuries:
            continue

        result = get_contribution_lost(roster, [inj.player for inj in injuries])
        if not result["matched"]:
            continue

        inj_status = {normalize_name(inj.player): inj.status for inj in injuries}
        team_totals.append(
            (torvik_team, result["total_min_pct_lost"], result["matched"], inj_status)
        )

    team_totals.sort(key=lambda x: x[1], reverse=True)

    for torvik_team, total_lost, matched, inj_status in team_totals:
        for inj_name, player in matched:
            status = inj_status.get(normalize_name(inj_name), "")
            sc = {"Out For Season": "[red]", "Out": "[yellow]", "Game Time Decision": "[green]"}.get(
                status, "[dim]"
            )
            table.add_row(
                torvik_team,
                player.player,
                f"{sc}{status}[/]",
                f"{player.min_pct:.1f}" if player.min_pct is not None else "?",
                f"{player.usg_pct:.1f}" if player.usg_pct is not None else "?",
                f"{player.bpm:+.2f}" if player.bpm is not None else "?",
                f"{player.ppg:.1f}" if player.ppg is not None else "?",
            )
        table.add_row(
            f"[bold]{torvik_team}[/bold]",
            "[bold]TOTAL MIN% LOST[/bold]",
            "",
            f"[bold]{total_lost:.1f}%[/bold]",
            "", "", "",
        )

    if team_totals:
        console.print(table)
    else:
        console.print("[green]No bracket teams with matched injured players.[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Torvik player contribution scraper & analyzer"
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--bracket", default="bracket.csv")
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--team", type=str, default=None,
                        help="Single team (Torvik name, e.g. 'N.C. State')")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--studs", action="store_true")
    parser.add_argument("--concentration", action="store_true")
    parser.add_argument("--cross-ref", action="store_true")
    parser.add_argument("--injuries-db", default="injuries.db")

    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument("--refresh", action="store_true",
                             help="Force re-fetch even if cache is fresh")
    fetch_group.add_argument("--no-fetch", action="store_true",
                             help="Never fetch; use cached data only")
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_HOURS)
    args = parser.parse_args()

    console = Console()
    conn = init_db(args.db)

    # Determine scope
    if args.team:
        display_teams = [args.team]
    else:
        try:
            display_teams = load_bracket_teams(args.bracket)
        except FileNotFoundError:
            console.print(
                "[red]bracket.csv not found. Use --team to specify a single team.[/red]"
            )
            conn.close()
            return

    # Fetch
    if not args.no_fetch:
        if args.refresh or not is_cache_fresh(conn, args.year, args.max_age):
            console.print("[bold]Fetching player stats from barttorvik.com...[/bold]")
            try:
                n = fetch_and_store_all(
                    display_teams, conn, args.year,
                    max_age_hours=args.max_age, force=args.refresh,
                    console=console,
                )
                console.print(f"[green]Stored {n} player records.[/green]")
            except Exception as exc:
                console.print(f"[red]Error fetching: {exc}[/red]")
                console.print("[dim]Falling back to cached data.[/dim]")
        else:
            age = get_global_cache_age_hours(conn, args.year)
            console.print(
                f"[dim]Cache is fresh ({age:.1f}h old). Use --refresh to force update.[/dim]"
            )

    # Display
    if args.show or args.team:
        for team in display_teams:
            players = get_player_stats(conn, team, args.year)
            display_players(players, team, console)

    if args.concentration:
        all_db_teams = get_all_teams_in_db(conn, args.year)
        display_concentration(all_db_teams, conn, args.year, console)

    if args.studs:
        all_db_teams = get_all_teams_in_db(conn, args.year)
        display_studs(all_db_teams, conn, args.year, console)

    if args.cross_ref:
        import sqlite3 as _sqlite3
        inj_conn = _sqlite3.connect(args.injuries_db)
        inj_conn.row_factory = _sqlite3.Row
        all_db_teams = get_all_teams_in_db(conn, args.year)
        display_cross_ref(all_db_teams, conn, args.year, inj_conn, console)
        inj_conn.close()

    if not any([args.show, args.team, args.concentration, args.studs, args.cross_ref]):
        total = conn.execute(
            "SELECT COUNT(*) FROM player_stats WHERE year = ?", (args.year,)
        ).fetchone()[0]
        n_teams = len(get_all_teams_in_db(conn, args.year))
        console.print(
            f"[dim]{total} players across {n_teams} teams in DB. "
            f"Use --show, --studs, --concentration, or --cross-ref.[/dim]"
        )

    conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scrape game-by-game statistics from barttorvik.com for NCAA tournament teams.

Stores offense and defense stats per game in a SQLite database, splitting
compound columns (2P, 3P) into made/attempted for downstream analytics
like three-point variance and late-season strength.
"""

import sqlite3
import time
import random
import re
import sys

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TORVIK_TEAM_URL = "https://barttorvik.com/team.php?team={team}&year={year}"
TORVIK_TEAMS_URL = "https://barttorvik.com/tourneytime.php?year={year}"
DB_PATH = "torvik_game_stats.db"
YEAR = 2026
REQUEST_DELAY = (2.5, 5.0)  # seconds between requests (min, max)

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

CREATE_TEAM_SEASON_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS team_season_stats (
    team        TEXT    NOT NULL,
    year        INTEGER NOT NULL,
    experience  REAL,
    PRIMARY KEY (team, year)
);
"""

CREATE_GAMES_TABLE = """
CREATE TABLE IF NOT EXISTS games (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team        TEXT    NOT NULL,
    date        TEXT,
    location    TEXT,
    opponent    TEXT,
    result      TEXT,
    score       TEXT,
    record      TEXT,
    wab         REAL,
    adj_o       REAL,
    adj_d       REAL,
    -- Offense
    off_eff     REAL,
    off_efg_pct REAL,
    off_to_pct  REAL,
    off_or_pct  REAL,
    off_ftr     REAL,
    off_2pm     INTEGER,
    off_2pa     INTEGER,
    off_3pm     INTEGER,
    off_3pa     INTEGER,
    -- Defense
    def_eff     REAL,
    def_efg_pct REAL,
    def_to_pct  REAL,
    def_or_pct  REAL,
    def_ftr     REAL,
    def_2pm     INTEGER,
    def_2pa     INTEGER,
    def_3pm     INTEGER,
    def_3pa     INTEGER,
    -- Misc
    game_score  INTEGER,
    plus_minus  REAL,
    year        INTEGER NOT NULL,
    UNIQUE(team, date, opponent, year)
);
"""


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_GAMES_TABLE)
    conn.execute(CREATE_TEAM_SEASON_STATS_TABLE)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Browser helpers — try Selenium first, fall back to Playwright
# ---------------------------------------------------------------------------

def _get_page_selenium(url):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    browser = webdriver.Chrome(options=opts)
    try:
        browser.get(url)
        WebDriverWait(browser, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
        )
        time.sleep(2)
        return browser.page_source
    finally:
        browser.quit()


def _get_page_playwright(url):
    from playwright.sync_api import sync_playwright

    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=ua)
        page = ctx.new_page()
        page.goto(url, wait_until="load", timeout=30000)
        # Wait for JS-rendered tables to appear in the DOM
        page.wait_for_selector("table", timeout=30000)
        page.wait_for_timeout(2000)
        html = page.content()
        browser.close()
        return html


def get_page(url, retries: int = 3):
    """Fetch fully-rendered HTML from *url*, trying Selenium then Playwright.

    Retries up to *retries* times on failure, alternating between backends.
    """
    backends = [_get_page_selenium, _get_page_playwright]
    last_exc = None
    for attempt in range(retries):
        for backend in backends:
            try:
                return backend(url)
            except Exception as exc:
                last_exc = exc
                continue
        # Exponential backoff before retrying the full cycle
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _split_made_attempted(text):
    """Parse '12-38' into (12, 38). Returns (None, None) on failure."""
    text = text.strip()
    m = re.match(r"(\d+)-(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _safe_float(text):
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def _safe_int(text):
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return None


def parse_team_page(html, team_name, year=YEAR):
    """
    Parse the Schedule / Results table from a Torvik team page.

    Returns a list of dicts ready for DB insertion.

    Column layout (30 td cells per game row):
      [0]  Date (link)          [15] off eFG%
      [1]  Location (H/A/N)     [16] off TO%
      [2]  Team rank             [17] off OR%
      [3]  Opp rank / quad       [18] off FTR
      [4]  (empty / conf flag)   [19] off 2P (made-att)
      [5]  Opponent (link)       [20] off 3P (made-att)
      [6]  Opp short name (link) [21] def EFF
      [7]  Result (link)         [22] def eFG%
      [8]  Line                  [23] def TO%
      [9]  Record                [24] def OR%
      [10] Conf record / empty   [25] def FTR
      [11] WAB                   [26] def 2P (made-att)
      [12] AdjO                  [27] def 3P (made-att)
      [13] AdjD                  [28] G-Sc
      [14] off EFF               [29] +/-
    """
    soup = BeautifulSoup(html, "lxml")

    # Find the schedule table — contains "Schedule / Results" header
    tables = soup.find_all("table")
    schedule_table = None
    for table in tables:
        header_text = table.get_text(strip=True).lower()
        if "date" in header_text and "opponent" in header_text:
            schedule_table = table
            break

    if schedule_table is None:
        print(f"  WARNING: could not find schedule table for {team_name}")
        return []

    rows = schedule_table.find_all("tr")
    games = []

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 28:
            continue  # skip header / separator rows

        raw = [c.get_text(strip=True) for c in cells]

        try:
            game = _parse_row(raw, cells, team_name, year)
            if game:
                games.append(game)
        except Exception as exc:
            print(f"  Skipping row for {team_name}: {exc}")
            continue

    return games


def _clean_date(raw_date):
    """Fix doubled date text like 'Mon 11-0311-03' → 'Mon 11-03'."""
    m = re.match(r"^([A-Za-z]+\s+\d{1,2}-\d{2})", raw_date)
    return m.group(1) if m else raw_date


def _parse_result(text):
    """Parse 'W, 118-102' or 'L, 67-66' into (result_char, full_text)."""
    text = text.strip()
    m = re.match(r"([WL]),?\s*(.*)", text)
    if m:
        return m.group(1), text
    return "", text


def _parse_row(raw, cells, team_name, year):
    """Parse a single game row using fixed column indices. Returns a dict or None."""
    # Validate that expected made-att columns look right
    if not re.match(r"^\d+-\d+$", raw[19].strip()):
        return None

    date_text = _clean_date(raw[0])
    location = raw[1].strip() if raw[1].strip() in ("H", "A", "N") else ""

    # Opponent: use the full-name link at index 5
    opponent = cells[5].find("a")
    opponent = opponent.get_text(strip=True) if opponent else raw[5]

    result_char, score = _parse_result(raw[7])
    record = raw[9].strip()

    off_2pm, off_2pa = _split_made_attempted(raw[19])
    off_3pm, off_3pa = _split_made_attempted(raw[20])
    def_2pm, def_2pa = _split_made_attempted(raw[26])
    def_3pm, def_3pa = _split_made_attempted(raw[27])

    return {
        "team": team_name,
        "date": date_text,
        "location": location,
        "opponent": opponent,
        "result": result_char,
        "score": score,
        "record": record,
        "wab": _safe_float(raw[11]),
        "adj_o": _safe_float(raw[12]),
        "adj_d": _safe_float(raw[13]),
        "off_eff": _safe_float(raw[14]),
        "off_efg_pct": _safe_float(raw[15]),
        "off_to_pct": _safe_float(raw[16]),
        "off_or_pct": _safe_float(raw[17]),
        "off_ftr": _safe_float(raw[18]),
        "off_2pm": off_2pm,
        "off_2pa": off_2pa,
        "off_3pm": off_3pm,
        "off_3pa": off_3pa,
        "def_eff": _safe_float(raw[21]),
        "def_efg_pct": _safe_float(raw[22]),
        "def_to_pct": _safe_float(raw[23]),
        "def_or_pct": _safe_float(raw[24]),
        "def_ftr": _safe_float(raw[25]),
        "def_2pm": def_2pm,
        "def_2pa": def_2pa,
        "def_3pm": def_3pm,
        "def_3pa": def_3pa,
        "game_score": _safe_int(raw[28]) if len(raw) > 28 else None,
        "plus_minus": _safe_float(raw[29]) if len(raw) > 29 else None,
        "year": year,
    }


# ---------------------------------------------------------------------------
# Team list
# ---------------------------------------------------------------------------

def get_tourney_teams(year=YEAR):
    """
    Scrape the list of tournament teams from Torvik's bracket page.
    Falls back to scraping the main team list page.
    Returns list of team name strings as used in Torvik URLs.
    """
    try:
        url = TORVIK_TEAMS_URL.format(year=year)
        html = get_page(url)
        soup = BeautifulSoup(html, "lxml")

        teams = set()
        # Look for team links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"team\.php\?team=([^&]+)", href)
            if m:
                team_name = m.group(1).replace("+", " ")
                teams.add(team_name)

        if teams:
            print(f"Found {len(teams)} teams from Torvik bracket page")
            return sorted(teams)
    except Exception as exc:
        print(f"Could not fetch bracket page: {exc}")

    # Fallback: use the main rankings page
    print("Falling back to main ratings page for team list …")
    url = f"https://barttorvik.com/trank.php?year={year}"
    html = get_page(url)
    soup = BeautifulSoup(html, "lxml")

    teams = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"team\.php\?team=([^&]+)", href)
        if m:
            team_name = m.group(1).replace("+", " ")
            teams.add(team_name)

    print(f"Found {len(teams)} teams from ratings page")
    return sorted(teams)


# ---------------------------------------------------------------------------
# Experience rating
# ---------------------------------------------------------------------------

TORVIK_TEAM_TABLES_URL = "https://barttorvik.com/team-tables_each.php?year={year}"


def scrape_all_experience(year: int = YEAR, db_path: str = DB_PATH) -> dict[str, float]:
    """Fetch experience ratings for all teams from Torvik's team-tables page.

    Returns a dict mapping team name → experience value, and upserts all
    results into team_season_stats.  One page load covers every team.
    """
    url = TORVIK_TEAM_TABLES_URL.format(year=year)
    print(f"Fetching experience data from {url} …")
    html = get_page(url)
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table")
    if table is None:
        print("  WARNING: could not find team-tables table")
        return {}

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    if "Exp." not in headers:
        print("  WARNING: 'Exp.' column not found in team-tables page")
        return {}

    exp_idx = headers.index("Exp.")
    # 'Team' appears twice (first and last); use the first occurrence
    team_idx = headers.index("Team")

    results = {}
    conn = sqlite3.connect(db_path)
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(exp_idx, team_idx):
            continue
        team_name = cells[team_idx].get_text(strip=True)
        exp_val = _safe_float(cells[exp_idx].get_text(strip=True))
        if not team_name or exp_val is None:
            continue
        results[team_name] = exp_val
        conn.execute(
            "INSERT OR REPLACE INTO team_season_stats (team, year, experience) VALUES (?, ?, ?)",
            (team_name, year, exp_val),
        )
    conn.commit()
    conn.close()
    return results


def scrape_experience(team: str, year: int = YEAR, db_path: str = DB_PATH) -> float | None:
    """Fetch the experience rating for a single team (used during full scrape)."""
    all_exp = scrape_all_experience(year, db_path)
    return all_exp.get(team)


# ---------------------------------------------------------------------------
# Scraping orchestrator
# ---------------------------------------------------------------------------

def insert_games(conn, games):
    """Insert game records, skipping duplicates."""
    sql = """
    INSERT OR IGNORE INTO games (
        team, date, location, opponent, result, score, record,
        wab, adj_o, adj_d,
        off_eff, off_efg_pct, off_to_pct, off_or_pct, off_ftr,
        off_2pm, off_2pa, off_3pm, off_3pa,
        def_eff, def_efg_pct, def_to_pct, def_or_pct, def_ftr,
        def_2pm, def_2pa, def_3pm, def_3pa,
        game_score, plus_minus, year
    ) VALUES (
        :team, :date, :location, :opponent, :result, :score, :record,
        :wab, :adj_o, :adj_d,
        :off_eff, :off_efg_pct, :off_to_pct, :off_or_pct, :off_ftr,
        :off_2pm, :off_2pa, :off_3pm, :off_3pa,
        :def_eff, :def_efg_pct, :def_to_pct, :def_or_pct, :def_ftr,
        :def_2pm, :def_2pa, :def_3pm, :def_3pa,
        :game_score, :plus_minus, :year
    )
    """
    conn.executemany(sql, games)
    conn.commit()


def scrape_experience_only(teams=None, year=YEAR, db_path=DB_PATH):
    """
    Fetch only experience ratings without re-scraping game data.

    Pulls all teams from Torvik's team-tables page in a single request, then
    optionally filters to the requested team list.  Much faster than a full
    scrape — one page load vs. one per team.
    """
    init_db(db_path)

    all_exp = scrape_all_experience(year, db_path)

    if teams is not None:
        # Report only the requested subset; DB already has all teams
        for team in teams:
            exp = all_exp.get(team)
            if exp is not None:
                print(f"  {team}: exp={exp:.3f}")
            else:
                print(f"  {team}: not found (team name may differ from Torvik's)")
    else:
        print(f"\nSaved experience for {len(all_exp)} teams.")

    print(f"Done. Experience ratings saved to {db_path}")


def scrape_all_teams(teams=None, year=YEAR, db_path=DB_PATH):
    """
    Scrape game-by-game stats for each team and store in SQLite.

    Parameters
    ----------
    teams : list[str] or None
        Team names in Torvik URL format. If None, fetches tourney teams.
    year : int
        Season year.
    db_path : str
        Path to SQLite database file.
    """
    conn = init_db(db_path)

    if teams is None:
        teams = get_tourney_teams(year)

    # Fetch all experience ratings in one request up front
    print("Fetching experience ratings …")
    all_exp = scrape_all_experience(year, db_path)

    total = len(teams)
    print(f"\nScraping {total} teams for {year} season …\n")

    for i, team in enumerate(teams, 1):
        url_team = team.replace(" ", "+")
        url = TORVIK_TEAM_URL.format(team=url_team, year=year)
        print(f"[{i}/{total}] {team} … ", end="", flush=True)

        try:
            html = get_page(url)
            games = parse_team_page(html, team, year)
            insert_games(conn, games)
            exp = all_exp.get(team)
            exp_str = f", exp={exp:.2f}" if exp is not None else ""
            print(f"{len(games)} games{exp_str}")
        except Exception as exc:
            print(f"ERROR: {exc}")

        # Polite delay between requests
        delay = random.uniform(*REQUEST_DELAY)
        time.sleep(delay)

    conn.close()
    print(f"\nDone. Data saved to {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Bart Torvik game-by-game stats into SQLite"
    )
    parser.add_argument(
        "--year", type=int, default=YEAR,
        help=f"Season year (default: {YEAR})"
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"SQLite database path (default: {DB_PATH})"
    )
    parser.add_argument(
        "--teams", nargs="+", default=None,
        help="Specific team names to scrape (Torvik URL format, e.g. 'Ohio St.')"
    )
    parser.add_argument(
        "--delay-min", type=float, default=REQUEST_DELAY[0],
        help=f"Min delay between requests in seconds (default: {REQUEST_DELAY[0]})"
    )
    parser.add_argument(
        "--delay-max", type=float, default=REQUEST_DELAY[1],
        help=f"Max delay between requests in seconds (default: {REQUEST_DELAY[1]})"
    )
    parser.add_argument(
        "--experience-only", action="store_true",
        help="Only scrape experience ratings (skips game data); much faster"
    )
    args = parser.parse_args()

    REQUEST_DELAY = (args.delay_min, args.delay_max)

    if args.experience_only:
        scrape_experience_only(teams=args.teams, year=args.year, db_path=args.db)
    else:
        scrape_all_teams(teams=args.teams, year=args.year, db_path=args.db)

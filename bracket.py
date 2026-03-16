#!/usr/bin/env python3
"""
Load the NCAA committee bracket from a CSV file.

Expected CSV format:
    team,seed,region[,first_four_opponent]

Columns:
    team   – team name
    seed   – integer seed (1-16)
    region – region index 0-3 (or name: South, West, East, Midwest)

First Four: two teams sharing the same seed+region slot should both have
the `first_four_opponent` column filled with each other's name, e.g.:

    Texas,11,2,Xavier
    Xavier,11,2,Texas

Teams without a play-in game leave that column blank.
"""

import csv
from collections import defaultdict


REGION_NAMES = ["SOUTH", "WEST", "EAST", "MIDWEST"]
_REGION_LOOKUP = {name: i for i, name in enumerate(REGION_NAMES)}
_REGION_LOOKUP.update({name.title(): i for i, name in enumerate(REGION_NAMES)})


def load_bracket(csv_path: str) -> tuple[dict, list]:
    """Load bracket from CSV.

    Returns
    -------
    regions : dict[int, dict[int, str]]
        {region_idx: {seed: team_name, ...}}
    first_four : list[tuple[str, str, int]]
        [(team_a, team_b, seed), ...]
    """
    regions: dict[int, dict[int, str]] = {i: {} for i in range(4)}
    ff_pairs: dict[tuple[int, int], list[str]] = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row["team"].strip()
            seed = int(row["seed"])
            region_raw = row["region"].strip()

            # Accept index or name
            if region_raw.isdigit():
                region_idx = int(region_raw)
            else:
                region_idx = _REGION_LOOKUP.get(region_raw)
                if region_idx is None:
                    raise ValueError(f"Unknown region: {region_raw!r}")

            ff_opp = row.get("first_four_opponent", "").strip()

            if ff_opp:
                key = (region_idx, seed)
                if team not in ff_pairs[key]:
                    ff_pairs[key].append(team)
                regions[region_idx][seed] = "Play-in"
            else:
                regions[region_idx][seed] = team

    first_four = []
    for (region_idx, seed), teams in ff_pairs.items():
        if len(teams) == 2:
            first_four.append((teams[0], teams[1], seed))
        else:
            raise ValueError(
                f"Expected 2 first-four teams for region {region_idx} seed {seed}, "
                f"got {len(teams)}: {teams}"
            )

    return regions, first_four

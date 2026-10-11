"""Fit points_model.json: how many points FPL players actually score, given FPL's own
projection for them, from past seasons of real gameweek data.

Data: vaastav/Fantasy-Premier-League on GitHub (per-player, per-gameweek history with FPL's
projection `xP` alongside the points scored). Clone it, then:

    python tools/fit_points_model.py /path/to/Fantasy-Premier-League/data 2023-24 2024-25 2025-26

Each player-match is keyed by position and his "adjusted projection"
    0.86 x xP x fixture factor (FIXTURE_FACTOR in monitor.py)
and the model stores, per position and projection bucket:
  - how often such a player didn't play at all,
  - the histogram of points when he did (for a match not yet started),
  - the histogram of his "returns" (points beyond appearance, clean sheet, goals-conceded
    deductions and bonus), used for the unplayed part of a match in progress.
Gameweeks where the dataset has few projections are skipped (2025-26 is patchy).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PTS_MIN, PTS_MAX = -4, 30
EDGES = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.5, 6.5, 8.0]
MIN_BUCKET = 60
FF = {  # same as monitor.FIXTURE_FACTOR, by position label
    "GK": (0.996, 0.996, 0.874, 0.773, 0.700), "DEF": (1.038, 1.038, 0.876, 0.653, 0.501),
    "MID": (1.060, 1.060, 0.985, 0.846, 0.733), "FWD": (1.024, 1.024, 0.949, 0.772, 0.582),
}
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}


def bucket(e):
    for i, ub in enumerate(EDGES):
        if e < ub:
            return i
    return len(EDGES)


def load(data_dir, season):
    df = pd.read_csv(f"{data_dir}/{season}/gws/merged_gw.csv", low_memory=False)
    fx = pd.read_csv(f"{data_dir}/{season}/fixtures.csv")[["id", "team_h_difficulty", "team_a_difficulty"]]
    df = df.merge(fx.rename(columns={"id": "fixture"}), on="fixture", how="left")
    df["position"] = df.position.replace({"GKP": "GK"})
    df = df[df.position.isin(list(FF))].copy()
    fdr = np.where(df.was_home, df.team_h_difficulty, df.team_a_difficulty)
    df["e"] = [0.86 * x * FF[p][int(min(5, max(1, d if d == d else 3))) - 1]
               for x, p, d in zip(df.xP, df.position, fdr)]
    # Keep only gameweeks the dataset actually has projections for.
    covered = df.groupby("GW").xP.transform(lambda x: (x > 0).sum()) >= 200
    df = df[covered]
    played = df.minutes > 0
    appearance = np.where(df.minutes >= 60, 2, np.where(played, 1, 0))
    cs = df.clean_sheets * df.position.map(CS_POINTS)
    conceded_pen = np.where(df.position.isin(["GK", "DEF"]), df.goals_conceded // 2, 0)
    df["returns"] = df.total_points - appearance - cs + conceded_pen - df.bonus
    return df


def hist(values):
    return np.bincount(np.clip(values, PTS_MIN, PTS_MAX) - PTS_MIN, minlength=PTS_MAX - PTS_MIN + 1).tolist()


def fit(df):
    out = {}
    for pos in FF:
        d = df[df.position == pos]
        b = np.array([bucket(x) for x in d.e])
        rows = []
        for i in range(len(EDGES) + 1):
            s = d[b == i]
            played = s[s.minutes > 0]
            rows.append({"n": len(s), "dnp": len(s) - len(played), "sum_e": float(s.e.sum()),
                         "hist": hist(played.total_points.values.astype(int)),
                         "returns": hist(played.returns.values.astype(int))})
        for i in range(1, len(rows)):  # pool thin buckets with the one below
            if rows[i]["n"] < MIN_BUCKET:
                a, p = rows[i], rows[i - 1]
                rows[i] = {"n": a["n"] + p["n"], "dnp": a["dnp"] + p["dnp"], "sum_e": a["sum_e"] + p["sum_e"],
                           "hist": [x + y for x, y in zip(a["hist"], p["hist"])],
                           "returns": [x + y for x, y in zip(a["returns"], p["returns"])]}
        for r in rows:
            r["mean_e"] = round(r.pop("sum_e") / max(1, r["n"]), 3)
        x, y = d.e.values, d.total_points.values
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
        out[pos] = {"slope": round(slope, 3), "buckets": rows}
    return out


def conceded_by_difficulty(data_dir, seasons):
    rows = []
    for s in seasons:
        f = pd.read_csv(f"{data_dir}/{s}/fixtures.csv")
        f = f[f.finished == True]  # noqa: E712
        rows += list(zip(f.team_h_difficulty, f.team_a_score)) + list(zip(f.team_a_difficulty, f.team_h_score))
    d = pd.DataFrame(rows, columns=["fdr", "conceded"])
    return {str(int(k)): round(float(v), 3) for k, v in d.groupby("fdr").conceded.mean().items()}


if __name__ == "__main__":
    data_dir, seasons = sys.argv[1], sys.argv[2:]
    df = pd.concat([load(data_dir, s) for s in seasons])
    model = {
        "source": "vaastav/Fantasy-Premier-League, seasons " + ", ".join(seasons),
        "player_matches": int(len(df)),
        "pts_min": PTS_MIN,
        "edges": EDGES,
        "conceded_per_match_by_difficulty": conceded_by_difficulty(data_dir, seasons),
        "by_position": fit(df),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "points_model.json")
    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"Wrote {path}: {len(df)} player-matches")

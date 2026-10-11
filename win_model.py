"""Chance to win each head-to-head, by simulating the rest of the gameweek.

Each player still to play gets a points total drawn from what real FPL players with the
same position and projection actually scored (points_model.json, fitted on past seasons
by tools/fit_points_model.py): mostly 1-3 points, sometimes nothing, now and then a
10-15 point haul. That lopsided spread is what makes comebacks possible, and a smooth bell
curve gets it wrong (it called matches 99%+ that the underdog went on to win or draw
about 1 time in 50).

Per player, by the state of each of his club's matches this gameweek:
  - not started: a full match drawn from the model, including the chance he doesn't play;
  - in progress, he's on the pitch: his further "returns" (goals, assists, defensive
    points) drawn from the model and scaled to the time left, the appearance point for
    reaching 60 minutes, and his clean sheet kept or lost at the rate clubs concede;
  - in progress, he hasn't come on: a small chance he still does;
  - over: his points as they stand.
Starters who don't play are replaced from the bench by FPL's automatic-substitution rules
(bench order, goalkeeper for goalkeeper, legal formation), in every simulated outcome.

Backtest (fit on earlier data, scored on the second half of 2025-26 under the current
rules): better scores than the old bell curve at every stage of a gameweek, and of the
matchups it called 99% or more, the underdog won or drew 1 in 397 (the old model: 9 in 434).
"""
import json
import math
import os

import numpy as np

SIMS = 20000
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "points_model.json")
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
FORMATION_MIN = {"DEF": 3, "MID": 2, "FWD": 1}
FORMATION_MAX = {"DEF": 5, "MID": 5, "FWD": 3}

_model = None
_cache = {}


def _load():
    global _model
    if _model is None:
        with open(_MODEL_PATH) as f:
            _model = json.load(f)
    return _model


def _bucket(e):
    for i, ub in enumerate(_load()["edges"]):
        if e < ub:
            return i
    return len(_load()["edges"])


def _dist(pos, e, kind):
    """(P(didn't play), points values, probabilities over those values when he did,
    within-bucket shift) for a position and adjusted projection."""
    key = (pos, _bucket(e), kind)
    if key not in _cache:
        m = _load()
        p = m["by_position"].get(pos) or m["by_position"]["MID"]
        b = p["buckets"][key[1]]
        h = np.array(b[kind], float)
        values = np.arange(len(h)) + m["pts_min"]
        _cache[key] = (b["dnp"] / b["n"], values, h / h.sum(), b["mean_e"], p["slope"])
    dnp, values, probs, mean_e, slope = _cache[key]
    return dnp, values, probs, (e - mean_e) * slope


def expected_points(pos, e):
    """Average points the model gives a full match with this projection (for display)."""
    dnp, values, probs, shift = _dist(pos, e, "hist")
    return (1 - dnp) * (float((values * probs).sum()) + shift)


def _conceded_rate(difficulty):
    table = _load()["conceded_per_match_by_difficulty"]
    return table.get(str(int(difficulty or 3)), 1.45)


def simulate_player(p, rng, n=SIMS):
    """Final points and whether he played, as arrays over n simulated outcomes.

    p: {"pos", "points", "minutes", "goals_conceded", "clean_sheet_points",
        "fixtures": [{"state": "not_started"|"live"|"done", "e", "left", "difficulty"}]}
    """
    pts = np.full(n, float(p["points"]))
    played = np.full(n, p["minutes"] > 0)
    stochastic = False
    for fx in p["fixtures"]:
        if fx["state"] == "not_started":
            stochastic = True
            dnp, values, probs, shift = _dist(p["pos"], fx["e"], "hist")
            plays = rng.random(n) >= dnp
            drawn = rng.choice(values, size=n, p=probs) + shift
            pts += np.where(plays, drawn, 0.0)
            played |= plays
        elif fx["state"] == "live":
            stochastic = True
            left = max(0.0, min(1.0, fx["left"]))
            _, values, probs, shift = _dist(p["pos"], fx["e"], "returns")
            returns = rng.choice(values, size=n, p=probs) + shift
            if p["minutes"] > 0:
                # Further returns, in proportion to the time left (all-or-nothing per outcome,
                # which keeps the occasional late haul).
                pts += np.where(rng.random(n) < left, returns, 0.0)
                # Appearance point for reaching 60 minutes, if there's time.
                if p["minutes"] < 60 and p["minutes"] + 90 * left >= 60:
                    pts += (rng.random(n) < 0.85).astype(float)
                # Clean sheet: kept, lost or still to earn, at the rate clubs concede.
                cs = CS_POINTS.get(p["pos"], 0)
                if cs and p.get("goals_conceded", 0) == 0:
                    keeps = rng.random(n) < math.exp(-_conceded_rate(fx.get("difficulty")) * left)
                    if p.get("clean_sheet_points", 0) > 0:
                        pts -= np.where(keeps, 0.0, p["clean_sheet_points"])
                    elif p["minutes"] + 90 * left >= 60:
                        pts += np.where(keeps & (rng.random(n) < 0.85), cs, 0.0)
            else:
                # Not on yet: a chance he comes off the bench.
                comes_on = rng.random(n) < 0.4 * left
                pts += np.where(comes_on, 1.0 + np.where(rng.random(n) < 0.25, returns, 0.0), 0.0)
                played |= comes_on
    return pts, played, stochastic


def simulate_team(starters, bench, rng, n=SIMS):
    """Final team totals over n outcomes, with automatic substitutions.
    starters/bench: lists of player dicts (see simulate_player), bench in FPL bench order."""
    any_stochastic = False
    s_out = []
    for p in starters:
        pts, played, st = simulate_player(p, rng, n)
        any_stochastic |= st
        s_out.append((p, pts, played))
    b_out = []
    for p in bench:
        pts, played, st = simulate_player(p, rng, n)
        b_out.append((p, pts, played, st))
    total = np.zeros(n)
    counts = {k: np.full(n, sum(1 for p in starters if p["pos"] == k)) for k in FORMATION_MIN}
    used = [np.zeros(n, bool) for _ in bench]
    for p, pts, played in s_out:
        total += np.where(played, pts, 0.0)
        need = ~played
        if not need.any():
            continue
        for j, (b, b_pts, b_played, b_st) in enumerate(b_out):
            if (b["pos"] == "GK") != (p["pos"] == "GK"):
                continue
            take = need & ~used[j] & b_played
            if p["pos"] != "GK" and b["pos"] != p["pos"]:
                after = {k: counts[k] - (k == p["pos"]) + (k == b["pos"]) for k in counts}
                legal = np.ones(n, bool)
                for k in counts:
                    legal &= (after[k] >= FORMATION_MIN[k]) & (after[k] <= FORMATION_MAX[k])
                take &= legal
            if not take.any():
                continue
            any_stochastic |= b_st
            total += np.where(take, b_pts, 0.0)
            used[j] |= take
            if p["pos"] != "GK":
                for k in counts:
                    counts[k] = counts[k] + np.where(take, int(k == b["pos"]) - int(k == p["pos"]), 0)
            need &= ~take
    return total, any_stochastic


def head_to_head(team_a, team_b, seed=0):
    """{"win_a", "draw", "win_b", "decided"} for two teams, each {"starters", "bench"}.
    Outcomes are seeded, so the same inputs always give the same odds."""
    rng = np.random.default_rng(seed)
    a, sa = simulate_team(team_a["starters"], team_a["bench"], rng)
    b, sb = simulate_team(team_b["starters"], team_b["bench"], rng)
    a, b = np.round(a), np.round(b)  # FPL scores are whole points
    return {
        "win_a": float(np.mean(a > b)),
        "draw": float(np.mean(a == b)),
        "win_b": float(np.mean(a < b)),
        "decided": not (sa or sb),
    }

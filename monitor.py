"""FPL Draft league dashboard: polls the Draft + classic FPL APIs for one or more
leagues, computes live scores (with provisional BPS-based bonus), tracks
roster/transaction changes, and serves a self-refreshing, switchable dashboard.
"""

import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request, send_from_directory

import fpl_api

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_F = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
STATE_DIR = os.path.join(DATA_DIR, "state")
LOG_DIR = os.path.join(DATA_DIR, "logs")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "monitor.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("monitor")

with open(CONFIG_F) as f:
    CONFIG = json.load(f)

LEAGUES = CONFIG["leagues"]
LEAGUES_BY_ID = {lg["league_id"]: lg for lg in LEAGUES}
IDLE_POLL_SECONDS = CONFIG.get("idle_poll_seconds", 900)
LIVE_POLL_SECONDS = CONFIG.get("live_poll_seconds", 75)
DASHBOARD_PORT = int(os.environ.get("PORT", CONFIG.get("dashboard_port", 8765)))
# Manual escape hatch for cases where the Draft API's own lineup data disagrees with
# what the official app actually shows (observed once so far — no corroborating
# transaction/ownership record on the API side, but the user confirmed it directly
# against the app). Keyed as league_id -> entry_id -> event -> [{out, in, note}, ...].
LINEUP_OVERRIDES = CONFIG.get("lineup_overrides", {})

_lock = threading.Lock()
_bootstrap_cache = {"draft": None, "classic": None, "fetched_at": 0}
BOOTSTRAP_TTL = 120  # player master data; refreshed frequently for near-live stats
_fixtures_cache = {"data": None, "fetched_at": 0}
FIXTURES_TTL = 1800  # full-season fixture list/difficulty ratings barely change; 30 min is plenty


def _state_path(league_id):
    return os.path.join(STATE_DIR, f"dashboard_state_{league_id}.json")


def _transactions_path(league_id):
    return os.path.join(STATE_DIR, f"transactions_{league_id}.json")


def _lineups_path(league_id):
    return os.path.join(STATE_DIR, f"lineups_{league_id}.json")


def _probability_history_path(league_id):
    return os.path.join(STATE_DIR, f"probability_history_{league_id}.json")


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"))  # compact: the page downloads this every minute
    os.replace(tmp, path)


def _get_bootstrap():
    now = time.time()
    if _bootstrap_cache["draft"] is None or now - _bootstrap_cache["fetched_at"] > BOOTSTRAP_TTL:
        log.info("Refreshing bootstrap-static caches")
        _bootstrap_cache["draft"] = fpl_api.draft_bootstrap_static()
        _bootstrap_cache["classic"] = fpl_api.classic_bootstrap_static()
        _bootstrap_cache["fetched_at"] = now
    return _bootstrap_cache["draft"], _bootstrap_cache["classic"]


def _get_all_fixtures():
    now = time.time()
    if _fixtures_cache["data"] is None or now - _fixtures_cache["fetched_at"] > FIXTURES_TTL:
        log.info("Refreshing full-season fixtures cache")
        _fixtures_cache["data"] = fpl_api.fixtures_all()
        _fixtures_cache["fetched_at"] = now
    return _fixtures_cache["data"]


UPCOMING_DIFFICULTY_WINDOW = 4  # unplayed fixtures kept per club: enough for three after the current gameweek


def _build_team_difficulty_index(all_fixtures, gw, teams_by_id):
    """Per team_id: the next UPCOMING_DIFFICULTY_WINDOW fixtures (from `gw` onward,
    inclusive — an unplayed fixture in the current gameweek still counts) with FPL's
    own 1-5 FDR rating from that team's perspective, plus a rounded average.
    """
    by_team = {}
    for f in sorted(all_fixtures, key=lambda f: (f.get("event") or 9999, f.get("id", 0))):
        event = f.get("event")
        if event is None or event < gw:
            continue
        if f.get("finished") or f.get("finished_provisional"):
            continue  # already played, so not part of the run of fixtures ahead
        for team_key, diff_key, opp_key, is_home in (
            ("team_h", "team_h_difficulty", "team_a", True),
            ("team_a", "team_a_difficulty", "team_h", False),
        ):
            team_id = f.get(team_key)
            if team_id is None:
                continue
            entries = by_team.setdefault(team_id, [])
            if len(entries) >= UPCOMING_DIFFICULTY_WINDOW:
                continue
            entries.append({
                "event": event,
                "opponent": teams_by_id.get(f.get(opp_key), {}).get("short_name", "?"),
                "is_home": is_home,
                "difficulty": f.get(diff_key),
            })
    index = {}
    for team_id, fixtures_list in by_team.items():
        vals = [f["difficulty"] for f in fixtures_list if f["difficulty"]]
        avg = round(sum(vals) / len(vals)) if vals else None
        index[team_id] = {"avg": avg, "fixtures": fixtures_list}
    return index


def _current_gameweek():
    game = fpl_api.game_status()
    gw = game.get("current_event") or game.get("next_event")
    is_live = game.get("current_event") is not None
    return gw, is_live, game


def _fixture_status(fixture):
    if fixture.get("finished"):
        return "finished"
    if fixture.get("finished_provisional"):
        return "finished_provisional"
    if fixture.get("started"):
        return "started"
    return "not_started"


# Fallback assumptions for a starting player's remaining points when we have no
# empirical per-gameweek average yet (e.g. pre-season, before any player has scored).
# Loosely typical for an FPL starter across a season: ~2-3 points mean, several points
# of spread (occasional hauls/blanks). Replaced by the real gameweek average once available.
DEFAULT_PLAYER_MEAN = 2.5
DEFAULT_PLAYER_STD = 3.0


def _win_probability(mean_a, var_a, mean_b, var_b):
    """P(team A's final score > team B's), modeling each team's remaining points as
    Normal(mean, var) and using that the difference of two independent normals is
    itself normal. A simple estimate, not a real predictive model."""
    diff_mean = mean_a - mean_b
    diff_var = var_a + var_b
    if diff_var <= 0:
        if diff_mean > 0:
            return 1.0
        if diff_mean < 0:
            return 0.0
        return 0.5
    z = diff_mean / math.sqrt(diff_var)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# Real-world summer transfer deadline window (2026-27 season) — players whose
# team_join_date falls in here show up in the "Transfer Deadline Moves" panel as a
# fixed snapshot of that window, not a rolling "recent joins" list.
TRANSFER_DEADLINE_WINDOW_START = "2026-08-26"
TRANSFER_DEADLINE_WINDOW_END = "2026-09-01"


def _compute_transfer_deadline_moves(classic_bootstrap, teams_by_id):
    """Real-world club-to-club transfers that landed in the deadline window, sourced
    from FPL's own per-player team_join_date — not to be confused with fantasy waiver/
    free-agent pickups (see moves_feed), this is actual football transfer activity.
    """
    moves = []
    for e in classic_bootstrap.get("elements", []):
        join_date = e.get("team_join_date")
        if not join_date or not (TRANSFER_DEADLINE_WINDOW_START <= join_date <= TRANSFER_DEADLINE_WINDOW_END):
            continue
        moves.append({
            "name": e.get("web_name", "?"),
            "team": teams_by_id.get(e.get("team"), {}).get("name", "?"),
            "join_date": join_date,
        })
    moves.sort(key=lambda m: m["join_date"], reverse=True)
    return moves


def _compute_h2h_standings(league, gw, team_scores, league_entry_id_to_entry_id):
    """Running H2H league table — win/draw/loss record, points for/against, and league
    points (3 for a win, 1 for a draw, standard classic-league scoring). Uses official
    results for finished gameweeks and our own live score as the projected result for
    whichever gameweek is currently in progress, so the table reflects "if this held
    right now" rather than waiting for FPL to mark the week finished.
    """
    stats = {
        e["entry_id"]: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "points_for": 0, "points_against": 0}
        for e in league["league_entries"]
    }
    for m in league.get("matches", []):
        event = m.get("event")
        if event is None or gw is None or event > gw:
            continue  # gameweek hasn't happened yet
        entry1 = league_entry_id_to_entry_id.get(m.get("league_entry_1"))
        entry2 = league_entry_id_to_entry_id.get(m.get("league_entry_2"))
        if entry1 not in stats or entry2 not in stats:
            continue
        if m.get("finished"):
            score1 = m.get("league_entry_1_points", 0)
            score2 = m.get("league_entry_2_points", 0)
        elif event == gw:
            score1 = team_scores.get(entry1, {}).get("total") or 0
            score2 = team_scores.get(entry2, {}).get("total") or 0
        else:
            continue  # a past gameweek FPL hasn't finished yet — no reliable score to use
        s1, s2 = stats[entry1], stats[entry2]
        s1["played"] += 1
        s2["played"] += 1
        s1["points_for"] += score1
        s1["points_against"] += score2
        s2["points_for"] += score2
        s2["points_against"] += score1
        if score1 > score2:
            s1["won"] += 1
            s2["lost"] += 1
        elif score2 > score1:
            s2["won"] += 1
            s1["lost"] += 1
        else:
            s1["drawn"] += 1
            s2["drawn"] += 1

    table = []
    for e in league["league_entries"]:
        entry_id = e["entry_id"]
        s = stats[entry_id]
        table.append({
            "entry_id": entry_id,
            "played": s["played"],
            "won": s["won"],
            "drawn": s["drawn"],
            "lost": s["lost"],
            "points_for": s["points_for"],
            "points_against": s["points_against"],
            "point_diff": s["points_for"] - s["points_against"],
            "league_points": s["won"] * 3 + s["drawn"],
        })
    # FPL Draft's own tiebreak is total points scored (points_for), not differential.
    table.sort(key=lambda r: (-r["league_points"], -r["points_for"]))
    for i, row in enumerate(table):
        row["rank"] = i + 1
    return table


KICKOFF_LOOKAHEAD = timedelta(minutes=90)

# Defensive Contribution scoring: FPL awards a flat 2 points once a player crosses a
# per-position action-count threshold (confirmed via game_config.scoring.defensive_contribution:
# DEF/MID/FWD=2, GKP=0). The threshold itself isn't exposed by the API — these are FPL's
# publicly documented values from the stat's introduction. Easy to correct here if they're wrong.
DEFCON_THRESHOLD_BY_POSITION = {1: None, 2: 10, 3: 12, 4: 12}  # element_type -> CBIT needed
DEFCON_POINTS = 2

# Real FPL scoring values by element_type (1=GKP, 2=DEF, 3=MID, 4=FWD), from
# game_config.scoring.{goals_scored,clean_sheets} — hardcoded like the DEFCON
# threshold above since these rarely change season-to-season.
GOAL_POINTS_BY_POSITION = {1: 10, 2: 6, 3: 5, 4: 4}
CLEAN_SHEET_POINTS_BY_POSITION = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_POINTS = 3
POSITION_LABEL = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Which fixture.stats identifiers to surface as match events, and how to label them.
EVENT_LABELS = {
    "goals_scored": ("⚽", "Goal"),
    "own_goals": ("⚽", "Own Goal"),
    "assists": ("🅰️", "Assist"),
    "yellow_cards": ("🟨", "Yellow Card"),
    "red_cards": ("🟥", "Red Card"),
    "penalties_missed": ("❌", "Penalty Missed"),
    "penalties_saved": ("🧤", "Penalty Saved"),
}


def _build_match_events(fx, classic_elements_by_id, teams_by_id):
    """Live goals/cards/etc per fixture, sourced straight from the `fixtures` endpoint's
    own stats — this updates faster than the per-player `event/live` feed used for
    scoring (observed lag of a minute or more between the two), so this is the quickest
    way to surface "what just happened" even before fantasy points catch up.
    """
    matches = []
    for f in fx:
        status = _fixture_status(f)
        if status == "not_started":
            continue
        events = []
        for stat in f.get("stats", []):
            identifier = stat.get("identifier")
            if identifier not in EVENT_LABELS:
                continue
            icon, label = EVENT_LABELS[identifier]
            for side in ("h", "a"):
                for entry in stat.get(side, []):
                    el = classic_elements_by_id.get(entry["element"], {})
                    events.append({
                        "side": side,
                        "icon": icon,
                        "label": label,
                        "player": el.get("web_name", "?"),
                        "value": entry.get("value", 1),
                    })
        matches.append({
            "fixture_id": f["id"],
            "status": status,
            "home_team": teams_by_id.get(f["team_h"], {}).get("short_name", "?"),
            "away_team": teams_by_id.get(f["team_a"], {}).get("short_name", "?"),
            "home_score": f.get("team_h_score"),
            "away_score": f.get("team_a_score"),
            "minutes": f.get("minutes", 0),
            "events": events,
        })
    return matches


FORMATION_MIN = {2: 3, 3: 2, 4: 1}  # element_type -> minimum on the pitch (DEF/MID/FWD)
FORMATION_MAX = {2: 5, 3: 5, 4: 3}  # element_type -> maximum on the pitch (DEF/MID/FWD)


def _apply_autosubs(starting, bench, live_stats_by_element, classic_elements_by_id, fixture_status_by_team):
    """Fill in for starting-XI players confirmed not to be playing (0 minutes, their real
    fixture already live/finished) using bench players in priority order — mirrors FPL's
    own autosub rule. Position 12 (bench GKP) only covers a benched starting GKP;
    positions 13-15 (outfield bench, in priority order) cover any other benched starter,
    but only if doing so keeps the squad in a legal formation (1 GK, 3-5 DEF, 2-5 MID,
    1-3 FWD) — e.g. if the starting XI has exactly 3 defenders and one doesn't play, only
    a defender can come on, no matter what the bench's priority order says, since bringing
    on anyone else would drop below the 3-defender minimum. A bench player only comes in
    once we know they actually played (minutes > 0) — no point swapping one zero for another.
    """
    def is_confirmed_out(pick):
        stats = live_stats_by_element.get(pick["element"], {})
        if stats.get("minutes", 0) > 0:
            return False
        team_id = classic_elements_by_id.get(pick["element"], {}).get("team")
        status = fixture_status_by_team.get(team_id, "not_started")
        return status in ("started", "finished_provisional", "finished")

    def has_played(pick):
        return live_stats_by_element.get(pick["element"], {}).get("minutes", 0) > 0

    def real_position(pick):
        return classic_elements_by_id.get(pick["element"], {}).get("element_type")

    gkp_bench = [p for p in bench if p.get("position") == 12]
    outfield_bench = sorted((p for p in bench if p.get("position", 0) > 12), key=lambda p: p["position"])

    result = list(starting)
    subs_made = []
    used_bench_ids = set()

    def formation_counts():
        counts = {2: 0, 3: 0, 4: 0}
        for p in result:
            pos = real_position(p)
            if pos in counts:
                counts[pos] += 1
        return counts

    for i, pick in enumerate(result):
        if not is_confirmed_out(pick):
            continue
        is_gkp = pick.get("position") == 1
        pool = gkp_bench if is_gkp else outfield_bench
        out_pos = real_position(pick)
        counts = formation_counts()
        for sub in pool:
            if sub["element"] in used_bench_ids or not has_played(sub):
                continue
            if not is_gkp:
                in_pos = real_position(sub)
                if in_pos != out_pos:
                    new_defmidfwd = dict(counts)
                    if out_pos in new_defmidfwd:
                        new_defmidfwd[out_pos] -= 1
                    if in_pos in new_defmidfwd:
                        new_defmidfwd[in_pos] += 1
                    if not all(FORMATION_MIN[p] <= new_defmidfwd[p] <= FORMATION_MAX[p] for p in (2, 3, 4)):
                        continue  # would break the squad's legal formation — try the next bench player
            result[i] = sub
            used_bench_ids.add(sub["element"])
            subs_made.append({"out": pick["element"], "in": sub["element"]})
            break
    return result, subs_made


def _season_ranks(classic_elements_by_id):
    """Season rank by total FPL points, overall and within each position (GK/DEF/MID/FWD).
    Standard competition ranking: players level on points share a rank (1, 2, 2, 4).
    Returns {element_id: (overall_rank, position_rank)} plus counts for context."""
    players = list(classic_elements_by_id.values())

    def rank_map(group):
        ordered = sorted(group, key=lambda e: -(e.get("total_points") or 0))
        ranks, prev_pts, prev_rank = {}, None, 0
        for i, e in enumerate(ordered, start=1):
            pts = e.get("total_points") or 0
            if pts != prev_pts:
                prev_rank, prev_pts = i, pts
            ranks[e["id"]] = prev_rank
        return ranks

    overall = rank_map(players)
    by_pos, pos_counts = {}, {}
    for pos in (1, 2, 3, 4):
        group = [e for e in players if e.get("element_type") == pos]
        pos_counts[pos] = len(group)
        by_pos.update(rank_map(group))
    return {eid: (overall[eid], by_pos.get(eid)) for eid in overall}, len(players), pos_counts


def _compute_live_scores(gw, entries_lineups, draft_elements_by_id, classic_elements_by_id, teams_by_id, difficulty_index):
    """Returns (team_scores, any_fixture_live, fast_poll_needed, match_events)."""
    season_ranks, total_players, pos_counts = _season_ranks(classic_elements_by_id)
    fx = fpl_api.fixtures(gw)
    match_events = _build_match_events(fx, classic_elements_by_id, teams_by_id)
    try:
        live = fpl_api.event_live(gw)
        live_stats_by_element = {row["id"]: row["stats"] for row in live["elements"]}
    except Exception as e:
        # Live per-player stats (goals/bps/minutes) can be transiently unavailable
        # (observed pre-season). Degrade gracefully rather than losing fixture-status
        # info (e.g. "pending") that doesn't depend on this endpoint at all.
        log.warning("event_live fetch failed for gw %s, proceeding without live stats: %s", gw, e)
        live_stats_by_element = {}

    fixture_status_by_team = {}
    fixture_minutes_by_team = {}
    # This gameweek's fixture(s) per club, kept even once played (the upcoming-difficulty
    # index drops finished matches), so the page can show it greyed out.
    current_fixtures_by_team = {}
    for f in fx:
        for team_key, diff_key, opp_key, is_home in (
            ("team_h", "team_h_difficulty", "team_a", True),
            ("team_a", "team_a_difficulty", "team_h", False),
        ):
            current_fixtures_by_team.setdefault(f[team_key], []).append({
                "event": gw,
                "opponent": teams_by_id.get(f.get(opp_key), {}).get("short_name", "?"),
                "is_home": is_home,
                "difficulty": f.get(diff_key),
                "status": _fixture_status(f),
            })
    for f in fx:
        status = _fixture_status(f)
        fixture_status_by_team[f["team_h"]] = status
        fixture_status_by_team[f["team_a"]] = status
        fixture_minutes_by_team[f["team_h"]] = f.get("minutes", 0)
        fixture_minutes_by_team[f["team_a"]] = f.get("minutes", 0)

    # any_live drives the "LIVE" badge shown to users — strictly "actually in progress",
    # so it correctly turns off the instant a match ends.
    any_live = any(_fixture_status(f) == "started" for f in fx)

    # fast_poll_needed drives polling *cadence* and is deliberately broader than any_live:
    # a fixture that just went "finished_provisional" still needs fast polling for a while
    # longer, because bonus/defcon/stats can keep being corrected until FPL marks it fully
    # "finished". Without this, the very poll that detects a match ending immediately drops
    # to the 15-minute idle interval — which is exactly backwards; that moment is when a lot
    # of stats are still actively settling (observed: left stale "live" data on screen for
    # ~15 minutes after a match ended, because the next poll was 15 minutes away).
    # Bounded to 3 hours post-kickoff, since finished_provisional can occasionally linger
    # for a long time before FPL fully finalizes it — don't poll aggressively forever.
    now = datetime.now(timezone.utc)
    still_finalizing = any(
        _fixture_status(f) == "finished_provisional"
        and f.get("kickoff_time")
        and (now - datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))) <= timedelta(hours=3)
        for f in fx
    )

    # FPL's public API has no "official lineup confirmed" signal (that's sourced from
    # club media/broadcasters, not exposed here) — the earliest we can detect anything
    # is `minutes` ticking up right at kickoff. So instead of waiting for a fixture to
    # actually start before switching to fast polling (which could leave us up to
    # IDLE_POLL_SECONDS late catching kickoff), treat an imminent kickoff the same as
    # "live" for polling-cadence purposes, so we're already on the fast interval by the
    # time there's anything to detect.
    near_kickoff = False
    for f in fx:
        if f.get("started") or not f.get("kickoff_time"):
            continue
        kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
        if timedelta(0) <= (kickoff - now) <= KICKOFF_LOOKAHEAD:
            near_kickoff = True
            break
    fast_poll_needed = any_live or still_finalizing or near_kickoff

    def _build_player_row(pick, is_bench, subbed_in_ids):
        eid = pick["element"]
        stats = live_stats_by_element.get(eid, {})
        # FPL's own live total_points already includes bonus — their own live BPS-rank
        # estimate while a fixture is in progress, official once it ends — so trust it
        # directly rather than layering our own separately-computed estimate on top,
        # which double-counted bonus once FPL started exposing it live mid-match.
        player_total = stats.get("total_points", 0)
        bonus_points = stats.get("bonus", 0)
        el = classic_elements_by_id.get(eid, {})
        team_id = el.get("team")
        position = el.get("element_type")
        status = fixture_status_by_team.get(team_id, "not_started")
        minutes = stats.get("minutes", 0)
        cbit = stats.get("defensive_contribution", 0)
        defcon_threshold = DEFCON_THRESHOLD_BY_POSITION.get(position)
        defcon_points = DEFCON_POINTS if defcon_threshold and cbit >= defcon_threshold else 0
        goals_scored = stats.get("goals_scored", 0)
        assists = stats.get("assists", 0)
        clean_sheets = stats.get("clean_sheets", 0)
        goals_points = goals_scored * GOAL_POINTS_BY_POSITION.get(position, 0)
        assists_points = assists * ASSIST_POINTS
        clean_sheet_points = clean_sheets * CLEAN_SHEET_POINTS_BY_POSITION.get(position, 0)
        fixture_minutes = fixture_minutes_by_team.get(team_id, 0)
        if minutes > 0 and status == "started":
            club_status = "live"
        elif minutes > 0:
            club_status = "finished"
        elif status in ("finished_provisional", "finished"):
            club_status = "benched"
        elif status == "started" and fixture_minutes >= 5:
            # A brief grace window right at kickoff — FPL's per-player minutes counter
            # can lag the fixture clock by a poll cycle or two, so a genuine starter can
            # transiently show 0 minutes the instant their fixture flips to "started".
            # Without this, every starter would flash "benched" for a few seconds at
            # kickoff before their first live stats land.
            club_status = "benched"
        else:
            club_status = "pending"
        row = {
            "element": eid,
            "name": classic_elements_by_id.get(eid, {}).get("web_name", "?"),
            "position": POSITION_LABEL.get(position, "?"),
            "is_bench": is_bench,
            "bonus_provisional": status == "started",
            "points": player_total,
            "fixture_status": status,
            "club_status": club_status,
            "minutes": minutes,
            "goals_scored": goals_scored,
            "assists": assists,
            "goals_points": goals_points,
            "assists_points": assists_points,
            "defensive_contribution": stats.get("defensive_contribution", 0),
            "defcon_points": defcon_points,
            "clean_sheets": clean_sheets,
            "clean_sheet_points": clean_sheet_points,
            "bps": stats.get("bps", 0),
            "bonus_points": bonus_points,
            "auto_sub": eid in subbed_in_ids,
            "difficulty": difficulty_index.get(team_id, {}),
            "current_fixtures": current_fixtures_by_team.get(team_id, []),
            "season_points": el.get("total_points"),
            "season_rank": season_ranks.get(eid, (None, None))[0],
            "season_pos_rank": season_ranks.get(eid, (None, None))[1],
            "season_rank_of": total_players,
            "season_pos_rank_of": pos_counts.get(position),
            # FPL's form (average points per match over the last 30 days) and availability flag.
            "form": float(el.get("form") or 0),
            "status": el.get("status", "a"),
            "chance_next": el.get("chance_of_playing_next_round"),
            "news": el.get("news") or "",
        }
        return row, player_total

    team_scores = {}
    for entry_id, picks in entries_lineups.items():
        total = 0
        player_rows = []
        effective_starting, subs_made = _apply_autosubs(
            picks.get("starting", []), picks.get("bench", []),
            live_stats_by_element, classic_elements_by_id, fixture_status_by_team,
        )
        subbed_in_ids = {s["in"] for s in subs_made}
        for pick in effective_starting:
            row, player_total = _build_player_row(pick, False, subbed_in_ids)
            total += player_total
            player_rows.append(row)
        # Bench players never auto-subbed in — shown for visibility (ESPN-style "BN"
        # section) but their points never count toward the team total.
        subbed_out_ids = {s["out"] for s in subs_made}
        for pick in picks.get("starting", []):
            if pick["element"] in subbed_out_ids:
                row, _ = _build_player_row(pick, True, subbed_in_ids)
                row["subbed_off"] = True
                player_rows.append(row)
        for pick in picks.get("bench", []):
            if pick["element"] in subbed_in_ids:
                continue
            row, _ = _build_player_row(pick, True, subbed_in_ids)
            player_rows.append(row)
        team_scores[entry_id] = {"total": total, "players": player_rows, "subs_made": subs_made}
    return team_scores, any_live, fast_poll_needed, match_events


def _apply_lineup_override(league_id, entry_id, gw, starting, bench):
    overrides = (
        LINEUP_OVERRIDES.get(str(league_id), {}).get(str(entry_id), {}).get(str(gw), [])
    )
    for ov in overrides:
        out_id, in_id = ov["out"], ov["in"]
        # Search both lists — an override can correct either a starting-XI mixup or
        # simply a wrong element id on the bench (e.g. two similarly-numbered players
        # the Draft API confused), so it should fix whichever list actually has it.
        for p in starting + bench:
            if p["element"] == out_id:
                p["element"] = in_id
                log.info(
                    "Applied lineup override for entry %s gw %s: %s -> %s (%s)",
                    entry_id, gw, out_id, in_id, ov.get("note", ""),
                )
                break
    return starting, bench


def _fetch_lineups(gw, entry_ids, league_id):
    lineups = {}
    for entry_id in entry_ids:
        try:
            resp = fpl_api.entry_event_picks(entry_id, gw)
        except Exception as e:
            log.warning("Failed to fetch picks for entry %s gw %s: %s", entry_id, gw, e)
            resp = None
        if not isinstance(resp, dict) or "picks" not in resp:
            lineups[entry_id] = {"starting": [], "bench": []}
            continue
        picks = resp["picks"]
        starting = [p for p in picks if p.get("position", 99) <= 11]
        bench = [p for p in picks if p.get("position", 99) > 11]
        starting, bench = _apply_lineup_override(league_id, entry_id, gw, starting, bench)
        lineups[entry_id] = {"starting": starting, "bench": bench}
    return lineups


MAX_PROBABILITY_HISTORY_POINTS = 400


def _update_probability_history(league_id, gw, current_matchups):
    """Append a timestamped win-probability snapshot for every team's H2H matchup, for
    the live odds-over-time chart. Resets when the gameweek changes. Skips writing a new
    point if nothing changed since the last one, so a long idle stretch (pre-kickoff, or
    between goals) doesn't bloat storage with identical repeats — the chart still draws
    correctly since points are keyed by real timestamp, not index.
    """
    path = _probability_history_path(league_id)
    history = _load_json(path, {"gw": None, "points": []})
    if history.get("gw") != gw:
        history = {"gw": gw, "points": []}
    snapshot_probs = {}
    for m in current_matchups:
        snapshot_probs[str(m["team1"]["entry_id"])] = m["team1_win_prob"]
        snapshot_probs[str(m["team2"]["entry_id"])] = m["team2_win_prob"]
    if history["points"] and history["points"][-1]["probs"] == snapshot_probs:
        return history["points"]
    history["points"].append({
        "t": datetime.now(timezone.utc).isoformat(),
        "probs": snapshot_probs,
    })
    history["points"] = history["points"][-MAX_PROBABILITY_HISTORY_POINTS:]
    _save_json(path, history)
    return history["points"]


POSITION_SHORT = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _premier_league_table(all_fixtures, teams_by_id):
    """The Premier League table worked out from finished fixtures (FPL doesn't publish
    one): points, then goal difference, then goals scored. Returns rows in order."""
    rec = {tid: {"club": t.get("short_name"), "team_id": tid, "played": 0, "pts": 0, "gf": 0, "ga": 0}
           for tid, t in teams_by_id.items()}
    for f in all_fixtures or []:
        if not (f.get("finished") or f.get("finished_provisional")):
            continue
        hs, as_ = f.get("team_h_score"), f.get("team_a_score")
        if hs is None or as_ is None:
            continue
        for tid, gf, ga in ((f["team_h"], hs, as_), (f["team_a"], as_, hs)):
            r = rec.get(tid)
            if not r:
                continue
            r["played"] += 1; r["gf"] += gf; r["ga"] += ga
            r["pts"] += 3 if gf > ga else 1 if gf == ga else 0
    rows = sorted(rec.values(), key=lambda r: (-r["pts"], -(r["gf"] - r["ga"]), -r["gf"], r["club"] or ""))
    for i, r in enumerate(rows, start=1):
        r["pos"] = i
        r["gd"] = r["gf"] - r["ga"]
    return rows


def _recent_minutes(last_done_gw, n=3):
    """Minutes per player (classic ids) in each of the last `n` finished gameweeks, from
    FPL's per-gameweek live stats: one request per gameweek, not one per player.
    Returns (gameweeks oldest first, {element_id: {gw: minutes}})."""
    gws = [g for g in range(last_done_gw - n + 1, last_done_gw + 1) if g >= 1]
    minutes = {}
    for g in gws:
        for row in (fpl_api.event_live(g) or {}).get("elements", []):
            minutes.setdefault(row["id"], {})[g] = (row.get("stats") or {}).get("minutes", 0)
    return gws, minutes


def _free_agents(league_id, draft_bootstrap, to_classic, difficulty_index, gw, teams_by_id, transactions, entries, pl_pos=None, recent=None):
    """Every unowned player in this league who has played this season, with the numbers
    managers use for waiver calls: form, points, starts, xGI, defensive contributions,
    penalty duty, availability, the next three fixtures, and who dropped him last.
    Stats come from the Draft API's own player list, so its ids line up with element-status."""
    statuses = (fpl_api.element_status(league_id) or {}).get("element_status", [])
    free = {s["element"]: s for s in statuses if not s.get("owner") and s.get("status") in ("a", "w")}
    entry_name = {e["entry_id"]: e["entry_name"] for e in entries}
    dropped = {}
    for t in sorted(transactions, key=lambda t: t.get("added", "")):
        if t.get("result") == "a" and t.get("element_out"):
            dropped[t["element_out"]] = (entry_name.get(t.get("entry")), t.get("added"))
    rows = []
    for e in draft_bootstrap["elements"]:
        if e["id"] not in free or not e.get("minutes"):
            continue
        cid = to_classic.get(e["id"], e["id"])
        fixtures = [f for f in (difficulty_index.get(e["team"], {}).get("fixtures") or []) if f["event"] > (gw or 0)][:3]
        by, at = dropped.get(cid, (None, None))
        recent_gws, recent_mins = recent or ([], {})
        rows.append({
            "id": cid,
            "name": e.get("web_name"),
            "pos": POSITION_SHORT.get(e.get("element_type")),
            "club": teams_by_id.get(e.get("team"), {}).get("short_name", "?"),
            "club_pos": (pl_pos or {}).get(e.get("team")),
            "form": float(e.get("form") or 0),
            "pts": e.get("total_points") or 0,
            "ppg": float(e.get("points_per_game") or 0),
            "starts": e.get("starts") or 0,
            "mins": e.get("minutes") or 0,
            "xgi": round(float(e.get("expected_goal_involvements") or 0), 2),
            "dc": e.get("defensive_contribution") or 0,
            "pens": e.get("penalties_order"),
            "status": e.get("status", "a"),
            "chance": e.get("chance_of_playing_next_round"),
            "news": e.get("news") or "",
            "on_waivers": free[e["id"]].get("status") == "w",
            "fixtures": fixtures,
            "dropped_by": by,
            "dropped_at": at,
            "recent_minutes": [recent_mins.get(cid, {}).get(g, 0) for g in recent_gws],
        })
    rows.sort(key=lambda r: (-r["form"], -r["pts"]))
    return rows


def _next_gameweek_info(draft_bootstrap, gw):
    """The coming gameweek's Draft deadlines (trades, waivers, lineups), for the
    Next gameweek section. None if the season is over."""
    events = draft_bootstrap.get("events") or {}
    data = events.get("data", []) if isinstance(events, dict) else events
    nxt = next((e for e in data if gw and e.get("id") == gw + 1), None)
    if not nxt:
        return None
    return {
        "event": nxt["id"],
        "deadline": nxt.get("deadline_time"),
        "waivers": nxt.get("waivers_time"),
        "trades": nxt.get("trades_time"),
    }


def _draft_to_classic_ids(draft_bootstrap, classic_bootstrap):
    """Draft and classic FPL number players independently: they agree for players present
    at launch, but anyone added later (ids 554+ in 26/27, 64 players as of GW5) gets a
    different id in each game. Every per-player number we show (live points, minutes,
    season totals, position, club) comes from the classic API, so translate Draft ids
    through the permanent player `code`, which both games share."""
    classic_id_by_code = {e["code"]: e["id"] for e in classic_bootstrap["elements"]}
    mapping = {}
    for e in draft_bootstrap["elements"]:
        cid = classic_id_by_code.get(e.get("code"))
        if cid is None:
            log.warning("Draft player %s (%s) has no classic match by code", e["id"], e.get("web_name"))
            cid = e["id"]
        mapping[e["id"]] = cid
    return mapping


def poll_once(league_cfg):
    league_id = league_cfg["league_id"]
    my_entry_id = league_cfg["my_entry_id"]

    draft_bootstrap, classic_bootstrap = _get_bootstrap()
    classic_elements_by_id = {e["id"]: e for e in classic_bootstrap["elements"]}
    draft_elements_by_id = {e["id"]: e for e in draft_bootstrap["elements"]}
    teams_by_id = {t["id"]: t for t in classic_bootstrap["teams"]}

    gw, is_live, game = _current_gameweek()
    league = fpl_api.league_details(league_id)
    entries = league["league_entries"]
    entry_ids = [e["entry_id"] for e in entries]

    transactions_resp = fpl_api.league_transactions(league_id)
    to_classic = _draft_to_classic_ids(draft_bootstrap, classic_bootstrap)
    transactions = [
        {**t,
         "element_in": to_classic.get(t.get("element_in"), t.get("element_in")),
         "element_out": to_classic.get(t.get("element_out"), t.get("element_out"))}
        for t in transactions_resp.get("transactions", [])
    ]

    lineups = _fetch_lineups(gw, entry_ids, league_id)
    for lineup in lineups.values():
        for pick in lineup["starting"] + lineup["bench"]:
            pick["element"] = to_classic.get(pick["element"], pick["element"])

    any_live = False
    fast_poll_needed = False
    team_scores = {}
    match_events = []
    difficulty_index = {}
    pl_table = []
    gameweek_status = None
    if gw:
        try:
            all_fixtures = _get_all_fixtures()
            difficulty_index = _build_team_difficulty_index(all_fixtures, gw, teams_by_id)
            pl_table = _premier_league_table(all_fixtures, teams_by_id)
            # How far through the gameweek we are: the page opens on Live from the first
            # kickoff until the last match is over, and on Matchups between gameweeks.
            gw_fx = [f for f in all_fixtures if f.get("event") == gw]
            gameweek_status = {
                "total": len(gw_fx),
                "started": sum(1 for f in gw_fx if f.get("started")),
                "finished": sum(1 for f in gw_fx if f.get("finished") or f.get("finished_provisional")),
            }
            team_scores, any_live, fast_poll_needed, match_events = _compute_live_scores(
                gw, lineups, draft_elements_by_id, classic_elements_by_id, teams_by_id, difficulty_index
            )
        except Exception as e:
            log.warning("Live score computation failed for league %s gw %s: %s", league_id, gw, e)

    recent_gws = []
    try:
        # Last three *finished* gameweeks: include the current one only once it is over.
        current_done = bool(gameweek_status and gameweek_status["total"]
                            and gameweek_status["finished"] >= gameweek_status["total"])
        recent = _recent_minutes((gw or 0) if current_done else (gw or 1) - 1)
        recent_gws = recent[0]
    except Exception as e:
        log.warning("Recent minutes failed for league %s: %s", league_id, e)
        recent = None
    try:
        free_agents = _free_agents(league_id, draft_bootstrap, to_classic, difficulty_index, gw,
                                   teams_by_id, transactions, entries,
                                   {r["team_id"]: r["pos"] for r in pl_table}, recent)
    except Exception as e:
        log.warning("Free agent list failed for league %s: %s", league_id, e)
        free_agents = []

    # Once FPL marks a manager's H2H match for this gameweek as finished, trust their own
    # official score over our live approximation (our auto-sub logic is simplified — no
    # formation-legality check — and provisional bonus is only ever an estimate). This is
    # the real reconciliation step: whatever our live math produced gets silently replaced
    # by the number FPL actually used to decide the match, so any drift self-corrects once
    # the gameweek is done rather than lingering as a wrong "final" score.
    finalized_entries = set()
    league_entry_id_to_entry_id = {e["id"]: e["entry_id"] for e in entries}
    for m in league.get("matches", []):
        if m.get("event") != gw or not m.get("finished"):
            continue
        for entry_key, points_key in (
            ("league_entry_1", "league_entry_1_points"), ("league_entry_2", "league_entry_2_points"),
        ):
            entry_id = league_entry_id_to_entry_id.get(m.get(entry_key))
            if entry_id is None:
                continue
            official_points = m.get(points_key)
            if official_points is None:
                continue
            prior = team_scores.setdefault(entry_id, {"total": None, "players": [], "subs_made": []})
            if prior["total"] != official_points:
                log.info(
                    "[league %s] Official score override for entry %s gw %s: %s -> %s",
                    league_id, entry_id, gw, prior["total"], official_points,
                )
            prior["total"] = official_points
            finalized_entries.add(entry_id)

    # Diff transactions against last-seen state for the "latest moves" feed.
    transactions_f = _transactions_path(league_id)
    prev_transactions = _load_json(transactions_f, [])
    prev_ids = {t["id"] for t in prev_transactions}
    new_transactions = [t for t in transactions if t["id"] not in prev_ids]
    if new_transactions:
        log.info("[league %s] Detected %d new transaction(s)", league_id, len(new_transactions))
    _save_json(transactions_f, transactions)

    lineups_f = _lineups_path(league_id)
    prev_lineups = _load_json(lineups_f, {})
    lineup_changes = []
    for entry_id_str, entry_lineup in lineups.items():
        key = str(entry_id_str)
        prev = prev_lineups.get(key)
        cur_ids = sorted(p["element"] for p in entry_lineup["starting"])
        if prev is not None and sorted(prev) != cur_ids and cur_ids:
            lineup_changes.append({"entry_id": entry_id_str, "gw": gw})
        if cur_ids:
            prev_lineups[key] = cur_ids
    _save_json(lineups_f, prev_lineups)

    # Global context: most recent gameweek with real average/highest score data.
    global_context = None
    for ev in reversed(classic_bootstrap.get("events", [])):
        if ev.get("average_entry_score"):
            global_context = {
                "gw": ev["id"],
                "average_entry_score": ev["average_entry_score"],
                "highest_score": ev.get("highest_score"),
            }
            break

    manager_by_entry = {e["entry_id"]: e for e in entries}

    RESULT_LABELS = {
        "a": {"status": "accepted", "label": "Accepted"},
        "di": {"status": "declined", "label": "Declined — player already claimed"},
        "do": {"status": "declined", "label": "Declined — drop no longer available"},
    }

    moves_feed = []
    for t in sorted(transactions, key=lambda t: t.get("added", ""), reverse=True)[:200]:
        mgr = manager_by_entry.get(t["entry"], {})
        result_info = RESULT_LABELS.get(t.get("result"), {"status": "unknown", "label": t.get("result")})
        moves_feed.append({
            "time": t.get("added"),
            "entry_id": t.get("entry"),
            "manager": mgr.get("entry_name", f"Entry {t.get('entry')}"),
            "kind": t.get("kind"),
            "result": t.get("result"),
            "status": result_info["status"],
            "status_label": result_info["label"],
            "element_in": classic_elements_by_id.get(t.get("element_in"), {}).get("web_name"),
            "element_out": classic_elements_by_id.get(t.get("element_out"), {}).get("web_name"),
        })

    def _team_summary(entry_id):
        all_players = team_scores.get(entry_id, {}).get("players", [])
        # Only the (post-autosub) starting XI ever counts toward the team's points —
        # bench players are included in `all_players` purely for BN-section visibility
        # and must never inflate these summary figures.
        players = [p for p in all_players if not p.get("is_bench")]
        # "played"/"club_benched"/"pending" describe real-world PL matchday status of the
        # manager's *fantasy starting XI* (not the Draft roster's own 11/4 split, which is
        # fixed at lineup-submission time and tells you nothing). Derived from each
        # player's own club_status so there's a single source of truth for what counts as
        # "benched" (including the brief post-kickoff grace window there).
        return {
            "fantasy_starters": len(lineups.get(entry_id, {}).get("starting", [])),
            "played": sum(1 for p in players if p["minutes"] > 0),
            "live_now": sum(1 for p in players if p["club_status"] == "live"),
            "club_benched": sum(1 for p in players if p["club_status"] == "benched"),
            "pending": sum(1 for p in players if p["club_status"] == "pending"),
            "goals_points": sum(p["goals_points"] for p in players),
            "assists_points": sum(p["assists_points"] for p in players),
            "defensive_contribution": sum(p["defensive_contribution"] for p in players),
            "defcon_points": sum(p["defcon_points"] for p in players),
            "clean_sheet_points": sum(p["clean_sheet_points"] for p in players),
            "bps": sum(p["bps"] for p in players),
            "bonus_points": sum(p["bonus_points"] for p in players),
        }

    # Win probability for this gameweek's H2H matchups: model each team's remaining
    # score as Normal(current_live_score + pending_players * mu, pending_players * sigma^2)
    # and estimate P(team A finishes higher) from the normal difference. mu/sigma come
    # from this season's actual gameweek average once we have one; otherwise fall back
    # to a generic assumption (see DEFAULT_PLAYER_MEAN/STD).
    if global_context and global_context.get("average_entry_score"):
        player_mean = global_context["average_entry_score"] / 11
    else:
        player_mean = DEFAULT_PLAYER_MEAN
    player_std = DEFAULT_PLAYER_STD

    summaries_by_entry = {e["entry_id"]: _team_summary(e["entry_id"]) for e in entries}

    current_matchups = []
    for m in league.get("matches", []):
        if m.get("event") != gw:
            continue
        entry1 = league_entry_id_to_entry_id.get(m.get("league_entry_1"))
        entry2 = league_entry_id_to_entry_id.get(m.get("league_entry_2"))
        if entry1 is None or entry2 is None:
            continue
        score1 = team_scores.get(entry1, {}).get("total") or 0
        score2 = team_scores.get(entry2, {}).get("total") or 0
        if m.get("finished"):
            # Deterministic once FPL has closed the book on this H2H match — use their
            # own final points and snap the probability to the actual result, rather
            # than dropping the matchup (which would freeze its last estimate forever
            # and make it vanish from the odds board/chart selector).
            score1 = m.get("league_entry_1_points", score1)
            score2 = m.get("league_entry_2_points", score2)
            prob1 = 1.0 if score1 > score2 else 0.0 if score1 < score2 else 0.5
        else:
            pending1 = summaries_by_entry.get(entry1, {}).get("pending", 0)
            pending2 = summaries_by_entry.get(entry2, {}).get("pending", 0)
            mean1, var1 = score1 + pending1 * player_mean, pending1 * player_std ** 2
            mean2, var2 = score2 + pending2 * player_mean, pending2 * player_std ** 2
            prob1 = _win_probability(mean1, var1, mean2, var2)
        current_matchups.append({
            "team1": {
                "entry_id": entry1, "name": manager_by_entry[entry1]["entry_name"],
                "score": score1, "is_me": entry1 == my_entry_id,
            },
            "team2": {
                "entry_id": entry2, "name": manager_by_entry[entry2]["entry_name"],
                "score": score2, "is_me": entry2 == my_entry_id,
            },
            "team1_win_prob": round(prob1, 3),
            "team2_win_prob": round(1 - prob1, 3),
            "involves_me": entry1 == my_entry_id or entry2 == my_entry_id,
        })

    probability_history_points = _update_probability_history(league_id, gw, current_matchups) if gw else []
    h2h_standings = _compute_h2h_standings(league, gw, team_scores, league_entry_id_to_entry_id)
    transfer_deadline_moves = _compute_transfer_deadline_moves(classic_bootstrap, teams_by_id)

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "league_id": league_id,
        "league_name": league["league"]["name"],
        "my_entry_id": my_entry_id,
        "gameweek": gw,
        "gameweek_live": any_live,
        "gameweek_status": gameweek_status,
        "next_gameweek": _next_gameweek_info(draft_bootstrap, gw),
        "managers": [
            {
                "entry_id": e["entry_id"],
                # The league's own id for this team: H2H fixtures in `matches` use it.
                "league_entry_id": e["id"],
                "name": e["entry_name"],
                # Public site: team name only, never the manager's real name.
                "manager": e["entry_name"],
                "is_me": e["entry_id"] == my_entry_id,
                "live_score": team_scores.get(e["entry_id"], {}).get("total"),
                "players": team_scores.get(e["entry_id"], {}).get("players", []),
                "subs_made": [
                    {
                        "out": classic_elements_by_id.get(s["out"], {}).get("web_name", "?"),
                        "in": classic_elements_by_id.get(s["in"], {}).get("web_name", "?"),
                    }
                    for s in team_scores.get(e["entry_id"], {}).get("subs_made", [])
                ],
                "score_finalized": e["entry_id"] in finalized_entries,
                "summary": summaries_by_entry[e["entry_id"]],
            }
            for e in entries
        ],
        "current_matchups": current_matchups,
        "h2h_standings": h2h_standings,
        "matches": league.get("matches", []),
        "match_events": match_events,
        "probability_history": probability_history_points,
        "free_agents": free_agents,
        "recent_gws": recent_gws,
        "pl_table": pl_table,
        "new_moves_count": len(new_transactions),
        "lineup_changes": lineup_changes,
        "global_context": global_context,
    }
    state_f = _state_path(league_id)
    with _lock:
        _save_json(state_f, state)
    log.info(
        "[league %s] Poll complete: gw=%s live=%s fast_poll=%s transactions=%d new=%d",
        league_id, gw, any_live, fast_poll_needed, len(transactions), len(new_transactions),
    )
    return fast_poll_needed


def poll_loop():
    while True:
        any_fast_poll_needed = False
        for league_cfg in LEAGUES:
            try:
                if poll_once(league_cfg):
                    any_fast_poll_needed = True
            except Exception:
                log.exception("Poll failed for league %s", league_cfg["league_id"])
        sleep_for = LIVE_POLL_SECONDS if any_fast_poll_needed else IDLE_POLL_SECONDS
        time.sleep(sleep_for)


app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/api/leagues")
def api_leagues():
    return jsonify([
        {"league_id": lg["league_id"], "label": lg["label"]} for lg in LEAGUES
    ])


@app.route("/api/state")
def api_state():
    league_id = request.args.get("league_id", type=int)
    if league_id not in LEAGUES_BY_ID:
        league_id = LEAGUES[0]["league_id"]
    with _lock:
        state = _load_json(_state_path(league_id), {})
    return jsonify(state)


_last_manual_refresh = {}
MANUAL_REFRESH_COOLDOWN = 5  # seconds; this dashboard is shared, guard against button-mashing


@app.route("/api/refresh")
def api_refresh():
    """Force an immediate poll for one league, rather than just returning whatever the
    background loop last cached (which, outside a live gameweek, could be up to
    IDLE_POLL_SECONDS old)."""
    league_id = request.args.get("league_id", type=int)
    if league_id not in LEAGUES_BY_ID:
        return jsonify({"error": "unknown league_id"}), 400
    now = time.time()
    if now - _last_manual_refresh.get(league_id, 0) >= MANUAL_REFRESH_COOLDOWN:
        _last_manual_refresh[league_id] = now
        try:
            poll_once(LEAGUES_BY_ID[league_id])
        except Exception:
            log.exception("Manual refresh failed for league %s", league_id)
    with _lock:
        state = _load_json(_state_path(league_id), {})
    return jsonify(state)


if __name__ == "__main__":
    once = "--once" in sys.argv
    if once:
        for lg in LEAGUES:
            poll_once(lg)
        print("Wrote state for all configured leagues")
    else:
        threading.Thread(target=poll_loop, daemon=True).start()
        app.run(host="0.0.0.0", port=DASHBOARD_PORT)

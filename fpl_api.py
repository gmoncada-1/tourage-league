"""Thin wrappers around the public FPL Draft and classic FPL APIs. No auth needed."""

import time

import requests

DRAFT_BASE = "https://draft.premierleague.com/api"
CLASSIC_BASE = "https://fantasy.premierleague.com/api"

_session = requests.Session()
_session.headers.update({
    # FPL rejects requests that don't look like a browser (403), which matters when
    # this runs on GitHub Actions instead of a home connection.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    "Accept": "application/json",
})


def _get(url, params=None, retries=3, backoff=2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


# --- Draft API ---

def league_details(league_id):
    return _get(f"{DRAFT_BASE}/league/{league_id}/details")


def element_status(league_id):
    return _get(f"{DRAFT_BASE}/league/{league_id}/element-status")


def entry_event_picks(entry_id, event):
    return _get(f"{DRAFT_BASE}/entry/{entry_id}/event/{event}")


def league_transactions(league_id):
    return _get(f"{DRAFT_BASE}/draft/league/{league_id}/transactions")


def game_status():
    return _get(f"{DRAFT_BASE}/game")


def draft_bootstrap_static():
    return _get(f"{DRAFT_BASE}/bootstrap-static")


# --- Classic API ---

def classic_bootstrap_static():
    return _get(f"{CLASSIC_BASE}/bootstrap-static/")


def fixtures(event):
    return _get(f"{CLASSIC_BASE}/fixtures/", params={"event": event})


def fixtures_all():
    """Full season fixture list (all gameweeks), including each fixture's
    team_h_difficulty/team_a_difficulty (FPL's own 1-5 FDR rating)."""
    return _get(f"{CLASSIC_BASE}/fixtures/")


def event_live(event):
    return _get(f"{CLASSIC_BASE}/event/{event}/live/")

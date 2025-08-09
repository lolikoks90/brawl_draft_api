from __future__ import annotations
import os, json, requests
from typing import Dict, Any, List

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BRAWLERS_URL = "https://api.brawlapi.com/v1/brawlers"
MAPS_URL = "https://api.brawlapi.com/v1/maps"
NOFF_TIERLIST = "https://www.noff.gg/brawl-stars/tier-list.json"

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def _get_json(url: str):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_brawlers(force: bool=False):
    p = _cache_path("brawlers.json")
    if (not force) and os.path.exists(p):
        return json.load(open(p))
    data = _get_json(BRAWLERS_URL)
    items = data.get("list") or data.get("items") or data
    json.dump(items, open(p,"w"))
    return items

def fetch_maps(force: bool=False):
    p = _cache_path("maps.json")
    if (not force) and os.path.exists(p):
        return json.load(open(p))
    data = _get_json(MAPS_URL)
    items = data.get("list") or data.get("items") or data
    json.dump(items, open(p,"w"))
    return items

def fetch_noff_tier(seed: float=1.0):
    try:
        data = _get_json(NOFF_TIERLIST)
        return {b["name"]: float(b.get("score", 7.0))*seed for b in data.get("brawlers",[])}
    except Exception:
        return {}

from __future__ import annotations
from typing import Dict, List, Tuple
import json, os

WEIGHTS = json.load(open(os.path.join(os.path.dirname(__file__), "weights.json")))
MATS = json.load(open(os.path.join(os.path.dirname(__file__), "matrices.json")))

ARCHETYPE_BY_NAME = {
    "Dynamike":"thrower","Barley":"thrower","Sprout":"thrower","Willow":"thrower","Tick":"thrower",
    "Bull":"tank","El Primo":"tank","Bibi":"tank","Frank":"tank","Hank":"tank","Darryl":"tank","Jacky":"tank",
    "Mortis":"assassin","Edgar":"assassin","Leon":"assassin","Fang":"assassin","Max":"assassin",
    "Piper":"sniper","Belle":"sniper","Byron":"sniper","Bea":"sniper","Mandy":"sniper","Colt":"sniper",
    "Pam":"healer","Poco":"healer","Byron":"healer","Amber":"dps","Spike":"dps","Griff":"dps","8-BIT":"dps",
    "Gale":"cc","Bo":"control","Sandy":"control","Gene":"utility","Gus":"utility","Tara":"utility","Buster":"utility","Sprout":"thrower"
}

def archetype_of(name: str) -> str:
    return ARCHETYPE_BY_NAME.get(name, "flex")

def base_value(name: str, meta_seed: Dict[str, float]) -> float:
    return float(meta_seed.get(name, 7.0))

def mode_fit(mode: str, tags: List[str]) -> float:
    mw = WEIGHTS["mode_weights"].get(mode, {})
    return sum(mw.get(t, 0.0) for t in tags)

def synergy_bonus(my_team: List[str], candidate: str) -> float:
    my_tags = [archetype_of(x) for x in my_team]
    cand_tag = archetype_of(candidate)
    bonus = WEIGHTS["ally_synergy_bonus"] * (1.0 if cand_tag not in my_tags else 0.0)
    sp = MATS.get("synergy_pairs",{}).get(candidate, [])
    bonus += 0.2 * sum(1 for a in my_team if a in sp or (a in MATS.get("synergy_pairs",{}) and candidate in MATS["synergy_pairs"][a]))
    return bonus

def anti_matchup_penalty(enemy_team: List[str], candidate: str, pick_slot: int) -> float:
    base = 0.0
    for e in enemy_team:
        if candidate in MATS.get("hard_counters",{}).get(e, []):
            base += 0.8
    if pick_slot <= 2:
        base *= (1.0 + WEIGHTS["pick_order"]["counter_risk_penalty"])
    return base

def redundancy_penalty(my_team: List[str], candidate: str) -> float:
    tags = [archetype_of(x) for x in my_team]
    t = archetype_of(candidate)
    return WEIGHTS["redundancy_penalty"].get(t, 0.0) if t in tags else 0.0

def first_pick_safety(candidate: str, enemy_team: List[str], is_first_pick: bool) -> float:
    if not is_first_pick: return 0.0
    common_threats = ["Mortis","Shelly","Piper","Griff","Spike"]
    safe = all(candidate not in MATS.get("hard_counters",{}).get(e, []) for e in enemy_team + common_threats)
    return WEIGHTS["pick_order"]["first_pick_safety"] if safe else 0.0

def score_candidate(candidate: str, mode: str, my_team: List[str], enemy_team: List[str], meta_seed: Dict[str, float], pick_slot: int, is_first_pick: bool) -> Tuple[float, Dict[str,float]]:
    base = base_value(candidate, meta_seed)
    tags = [archetype_of(candidate)]
    fit = mode_fit(mode, tags)
    syn = synergy_bonus(my_team, candidate)
    anti = anti_matchup_penalty(enemy_team, candidate, pick_slot)
    red = redundancy_penalty(my_team, candidate)
    fps = first_pick_safety(candidate, enemy_team, is_first_pick)
    total = base + fit + syn + fps - anti - red
    return total, {"base":base,"fit":fit,"synergy":syn,"first_pick_safety":fps,"anti":-anti,"redundancy":-red}

def recommend(all_brawlers: List[str], mode: str, my_team: List[str], enemy_team: List[str], unavailable: List[str], meta_seed: Dict[str, float], pick_slot: int=1, is_first_pick: bool=True, topk: int=3):
    pool = [b for b in all_brawlers if b not in unavailable]
    scored = []
    for b in pool:
        val, parts = score_candidate(b, mode, my_team, enemy_team, meta_seed, pick_slot, is_first_pick)
        why = []
        if parts["synergy"]>0: why.append("sinergia con tu comp")
        if parts["first_pick_safety"]>0: why.append("seguro para primer pick")
        if parts["anti"]<0: why.append("cuidado: puede ser contrarrestado")
        if parts["redundancy"]<0: why.append("evita redundancia")
        if not why: why.append("valor meta + ajuste de modo")
        scored.append({"brawler":b,"score":round(val,2),"why":"; ".join(why),"breakdown":parts})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:topk]

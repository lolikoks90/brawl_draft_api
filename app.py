from __future__ import annotations
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from data_providers.brawlapi import fetch_brawlers, fetch_maps, fetch_noff_tier
from engine.heuristic import recommend

app = Flask(__name__)
CORS(app)

BRAWLERS = [b.get("name") for b in fetch_brawlers(force=False)]
MAPS = fetch_maps(force=False)
META = fetch_noff_tier()

@app.get("/api/health")
def health(): return jsonify({"ok": True})

@app.get("/api/brawlers")
def brawlers(): return jsonify({"brawlers": BRAWLERS})

@app.get("/api/maps")
def maps():
    out = [{"name": m.get("name"), "mode": m.get("mode",{}).get("name","")} for m in MAPS if True]
    return jsonify({"maps": out})

@app.post("/api/recommend")
def api_recommend():
    data = request.get_json(silent=True) or request.form
    mode = data.get('mode','Hot Zone')
    _map = data.get('map','')
    mine = [x.strip() for x in (data.get('mine','') or '').split(',') if x.strip()]
    enemy = [x.strip() for x in (data.get('enemy','') or '').split(',') if x.strip()]
    bad = [x.strip() for x in (data.get('bad','') or '').split(',') if x.strip()]
    slot = int(data.get('slot',1) or 1)
    fp = str(data.get('fp','true')).lower()=='true'
    recs = recommend(BRAWLERS, mode, mine, enemy, bad, META, pick_slot=slot, is_first_pick=fp, topk=3)
    return jsonify({"mode": mode, "map": _map, "slot": slot, "first_pick": fp, "recommendations": recs})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)

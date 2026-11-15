"""Configuración del bot. Todo lo ajustable está aquí.

Coordenadas: fracciones (0..1) del área de juego, o baldosas de la arena (18 x 32,
fila 0 = fondo enemigo, fila 31 = nuestro fondo).
"""

CONFIG = {
    # --- Localización del juego ------------------------------------------------
    # Parte del título de la ventana del emulador (BlueStacks, LDPlayer, MuMu...).
    "window_title": "BlueStacks",
    # Si no se encuentra la ventana (o no usas Windows) se usa esta región absoluta.
    "fallback_region": {"left": 0, "top": 0, "width": 540, "height": 960},
    # Márgenes de la ventana que NO son juego (barra de título, barra lateral del emulador).
    "window_margins": {"top": 40, "bottom": 0, "left": 0, "right": 40},

    "target_fps": 30,

    # --- ROIs (fracciones del área de juego) --------------------------------------
    "elixir_roi": (0.285, 0.958, 0.975, 0.972),   # x1, y1, x2, y2
    "card_rois": [
        (0.235, 0.835, 0.410, 0.945),
        (0.420, 0.835, 0.595, 0.945),
        (0.605, 0.835, 0.780, 0.945),
        (0.790, 0.835, 0.965, 0.945),
    ],
    "arena_roi": (0.025, 0.075, 0.975, 0.795),
    "arena_tiles": (18, 32),
    "river_rows": (15, 16),

    # Torres en baldosas (col1, fila1, col2, fila2) con extremo exclusivo.
    "towers": {
        "enemy_left": (2, 5, 6, 9),
        "enemy_right": (12, 5, 16, 9),
        "enemy_king": (7, 1, 11, 5),
        "own_left": (2, 23, 6, 27),
        "own_right": (12, 23, 16, 27),
        "own_king": (7, 27, 11, 31),
    },
    # Una torre se da por destruida si su color de equipo cae por debajo de esta fracción
    # del valor medido al empezar la partida, durante `tower_confirm_s` segundos.
    "tower_destroyed_ratio": 0.35,
    "tower_confirm_s": 2.0,

    # --- Umbrales de color (OpenCV HSV: H 0-180, S 0-255, V 0-255) ----------------
    "elixir_hsv_low": (135, 90, 110),
    "elixir_hsv_high": (170, 255, 255),
    "enemy_hsv_ranges": [((0, 160, 150), (8, 255, 255)), ((170, 160, 150), (180, 255, 255))],
    "ally_hsv": ((95, 140, 140), (125, 255, 255)),
    "card_gray_saturation": 45,

    # --- Reconocimiento de cartas -------------------------------------------------
    "hash_match_distance": 10,
    "cards_db": "cards_db.json",

    # --- Detección por color (si no hay modelo YOLO) ------------------------------
    "threat_min_area": 12,
    "threat_heavy_area": 140,
    "threat_swarm_count": 4,

    # --- Detección de tropas con YOLO (ultralytics) -------------------------------
    "yolo": {
        "enabled": True,
        "model": "cr_yolo.pt",
        "base_weights": "yolo11n.pt",
        "conf": 0.40,
        "iou": 0.50,
        "imgsz": 640,
        "device": None,
        "every_n_frames": 2,
    },
    "dataset_dir": "dataset",
    "collect_interval_s": 1.5,

    # --- Ritmo / humanización -----------------------------------------------------
    "min_seconds_between_plays": 1.2,
    "click_jitter_px": 4,
    "click_delay_range": (0.05, 0.16),
    "move_duration_range": (0.04, 0.12),

    # =========================== FASE 1: autonomía ================================
    "screens": {
        # Botón "Batalla" del menú principal (amarillo) y "OK" de la pantalla final (azul).
        "battle_button": (0.50, 0.79),
        "battle_button_roi": (0.32, 0.75, 0.68, 0.83),
        "battle_button_hsv": ((18, 120, 170), (35, 255, 255)),
        "ok_button": (0.50, 0.90),
        "ok_button_roi": (0.38, 0.87, 0.62, 0.93),
        "ok_button_hsv": ((95, 120, 150), (115, 255, 255)),
        "min_button_fill": 0.35,
        "end_confirm_s": 1.5,        # s sin barra de elixir para dar la partida por terminada
        "unknown_action_after_s": 20,  # atascado en pantalla desconocida -> pulsar "atrás"
        "back_key": "esc",           # tecla "atrás" del emulador
        "pause_between_games_s": (3.0, 8.0),
        "max_games": 0,              # 0 = sin límite (también --max-games)
    },
    # Duración y multiplicador de elixir (segundos desde el inicio): (desde_s, multiplicador).
    "elixir_schedule": [(0, 1.0), (120, 2.0), (180, 2.0), (240, 3.0)],
    "elixir_regen_s": 2.8,
    "stats_file": "match_history.jsonl",

    "tactics": {
        "punish_opp_elixir_max": 3.0,   # atacar si el rival tiene <= este elixir estimado
        "punish_min_own_elixir": 5.0,
        "tank_push_elixir": 7.0,
        "tank_push_elixir_double": 6.0,
        "leak_elixir": 9.5,
        "spell_cluster_min": 3,
        "spell_cluster_radius": 2.5,    # baldosas
        "follow_tank_window_s": (2.5, 9.0),
    },

    # ====================== FASE 2: aprendizaje por refuerzo ======================
    "learner": {
        "enabled": True,
        "db": "learner_db.json",
        "ucb_c": 0.6,              # exploración (más alto = prueba más alternativas)
        "prior_weight": 3.0,       # cuánto pesa la prioridad de las reglas al principio
        "final_weight": 0.5,       # peso del resultado de la partida vs. recompensa local
        "local_window_s": 15.0,
        "wait_prior": 0.45,
        "wait_decision_interval_s": 3.0,
    },

    # ======================== FASE 3: aprendizaje por imitación ===================
    "imitation": {
        "enabled": True,
        "model": "policy.pt",
        "data_dir": "policy_data",
        "min_prob": 0.35,
        "video_fps": 10,
        "grid": (6, 8),           # rejilla gruesa del estado (columnas, filas)
        "tile_grid": (9, 16),     # rejilla de colocación (2x2 baldosas por celda)
        "auto_retrain_every_games": 10,
        "min_samples": 300,
        "epochs": 40,
        "source_weights": {"pro": 1.0, "win": 1.0, "draw": 0.3, "loss": 0.0},
        "min_battle_s": 60,
    },
}

# Mazo: nombre -> (coste, rol). Roles:
#   tank, tank_killer, splash, spell_small, spell_big, support, cycle, building
DECK = {
    "Giant": (5, "tank"),
    "Mini P.E.K.K.A": (4, "tank_killer"),
    "Musketeer": (4, "support"),
    "Valkyrie": (4, "splash"),
    "Fireball": (4, "spell_big"),
    "Zap": (2, "spell_small"),
    "Skeletons": (1, "cycle"),
    "Cannon": (3, "building"),
}

ROLES = ["tank", "tank_killer", "splash", "spell_small", "spell_big", "support", "cycle", "building"]
KINDS = ["heavy", "air", "swarm", "building", "light"]
KIND_PRIORITY = ["heavy", "building", "air", "swarm", "light"]

COUNTERS = {
    "heavy": ["tank_killer", "building", "cycle", "support"],
    "swarm": ["spell_small", "splash", "spell_big", "support"],
    "light": ["splash", "support", "cycle", "tank_killer", "building"],
    "air": ["support", "spell_big", "building", "spell_small"],
    "building": ["spell_big", "tank", "tank_killer"],
}

# Clase YOLO (normalizada) -> tipo. "spell" = hechizo (cuenta para el elixir rival),
# "ignore" = no es amenaza. Las clases que no estén aquí son "light".
TROOP_PROFILES = {
    "pekka": "heavy", "mega_knight": "heavy", "giant": "heavy", "golem": "heavy",
    "royal_giant": "heavy", "hog_rider": "heavy", "ram_rider": "heavy", "elixir_golem": "heavy",
    "electro_giant": "heavy", "goblin_giant": "heavy", "giant_skeleton": "heavy",
    "balloon": "air", "lava_hound": "air", "lava_pups": "air", "baby_dragon": "air",
    "minions": "air", "minion_horde": "air", "bats": "air", "inferno_dragon": "air",
    "mega_minion": "air", "skeleton_dragons": "air",
    "skeleton_army": "swarm", "goblin_gang": "swarm", "barbarians": "swarm", "skeletons": "swarm",
    "goblins": "swarm", "spear_goblins": "swarm", "guards": "swarm", "royal_recruits": "swarm",
    "x_bow": "building", "mortar": "building", "goblin_hut": "building", "furnace": "building",
    "tombstone": "building", "barbarian_hut": "building", "inferno_tower": "building",
    "cannon": "building", "tesla": "building", "bomb_tower": "building",
    "arrows": "spell", "fireball": "spell", "zap": "spell", "log": "spell", "poison": "spell",
    "rocket": "spell", "lightning": "spell", "earthquake": "spell", "tornado": "spell",
    "freeze": "spell", "rage": "spell", "graveyard": "spell", "goblin_barrel": "spell",
    "king_tower": "ignore", "princess_tower": "ignore", "clock": "ignore",
}

SPECIFIC_COUNTERS = {
    "pekka": ["tank_killer", "building", "cycle"],
    "mega_knight": ["tank_killer", "building", "support"],
    "hog_rider": ["building", "tank_killer", "cycle"],
    "balloon": ["support", "building"],
    "skeleton_army": ["spell_small", "splash"],
    "goblin_gang": ["spell_small", "splash"],
    "minion_horde": ["spell_big", "support"],
    "x_bow": ["spell_big", "tank"],
}

# Coste aproximado de las cartas del rival (cambia con los ajustes de balance del juego).
ENEMY_CARD_COSTS = {
    "pekka": 7, "mega_knight": 7, "giant": 5, "golem": 8, "royal_giant": 6, "hog_rider": 4,
    "ram_rider": 5, "elixir_golem": 3, "electro_giant": 7, "goblin_giant": 6, "giant_skeleton": 6,
    "balloon": 5, "lava_hound": 7, "baby_dragon": 4, "minions": 3, "minion_horde": 5, "bats": 2,
    "inferno_dragon": 4, "mega_minion": 3, "skeleton_dragons": 4, "skeleton_army": 3,
    "goblin_gang": 3, "barbarians": 5, "skeletons": 1, "goblins": 2, "spear_goblins": 2,
    "guards": 3, "royal_recruits": 7, "x_bow": 6, "mortar": 4, "goblin_hut": 5, "furnace": 4,
    "tombstone": 3, "barbarian_hut": 6, "inferno_tower": 5, "cannon": 3, "tesla": 4,
    "bomb_tower": 4, "arrows": 3, "fireball": 4, "zap": 2, "log": 2, "poison": 4, "rocket": 6,
    "lightning": 6, "earthquake": 3, "tornado": 3, "freeze": 4, "rage": 2, "graveyard": 5,
    "goblin_barrel": 3, "knight": 3, "archers": 3, "musketeer": 4, "valkyrie": 4, "wizard": 5,
    "witch": 5, "mini_pekka": 4, "prince": 5, "dark_prince": 4, "miner": 3, "princess": 3,
    "ice_spirit": 1, "fire_spirit": 1, "electro_spirit": 1, "ice_golem": 2, "hunter": 4,
    "executioner": 5, "bowler": 5, "electro_wizard": 4, "ice_wizard": 3, "magic_archer": 4,
    "bandit": 3, "royal_ghost": 3, "lumberjack": 4, "sparky": 6, "three_musketeers": 9,
    "bomber": 2, "firecracker": 3,
}
DEFAULT_ENEMY_COST = 3.5

# Unidades generadas por otras (no cuestan elixir al rival).
SPAWNS = {
    "goblin_hut": ["spear_goblins"], "barbarian_hut": ["barbarians"], "tombstone": ["skeletons"],
    "witch": ["skeletons"], "furnace": ["fire_spirit"], "graveyard": ["skeletons"],
    "lava_hound": ["lava_pups"], "golem": ["golemite"], "goblin_giant": ["spear_goblins"],
    "goblin_barrel": ["goblins"], "elixir_golem": ["elixir_golemite"],
}

HIGH_VALUE_TARGETS = {"musketeer", "wizard", "witch", "executioner", "three_musketeers",
                      "magic_archer", "electro_wizard", "princess", "sparky", "inferno_dragon"}

TEAM_PREFIXES = {"enemy_": "enemy", "red_": "enemy", "opp_": "enemy",
                 "ally_": "ally", "own_": "ally", "blue_": "ally"}

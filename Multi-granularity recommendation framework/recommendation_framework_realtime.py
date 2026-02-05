from __future__ import annotations

import json
import pickle
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np


# ---------------------------
# Defaults
# ---------------------------

# Temporary default location: central SG (Dhoby Ghaut-ish)
DEFAULT_LOCATION = {"latitude": 1.2996, "longitude": 103.8455}

# Progressive radius expansion (km). None = no limit.
RADIUS_TIERS_KM = [1.0, 3.0, 8.0, None]


# ---------------------------
# Taxonomy + Intent Logic
# ---------------------------

DATASET_CATEGORIES = {
    "atm", "attraction", "bakery", "bank", "bookstore", "cafe", "charging_station", "clinic",
    "clothing_store", "co_working", "convenience_store", "dentist", "department_store",
    "electronics_store", "fast_food", "gym", "hospital", "hostel", "hotel", "jewelry_store",
    "pharmacy", "restaurant", "shopping_mall", "supermarket", "university", "viewpoint", "yoga"
}

INTENT_KEYWORDS = {
    "shopping": ["shopping", "shop", "shops", "mall", "malls", "retail", "department", "boutique"],
    "supermarket": ["supermarket", "grocer", "grocery", "groceries", "fairprice", "giant", "ntuc", "cold storage"],
    "food": ["food", "eat", "eats", "dinner", "lunch", "breakfast", "restaurant", "hawker", "fast food"],
    "cafe": ["cafe", "cafes", "coffee", "brunch", "bakery", "dessert", "tea"],
    "nature": ["nature", "park", "parks", "garden", "beach", "hike", "hiking", "trail"],
    "culture": ["museum", "museums", "gallery", "galleries", "art", "history", "heritage", "exhibition"],
    "view": ["viewpoint", "views", "view", "skyline", "observation", "lookout"],
}

INTENT_CATEGORIES = {
    "shopping": {"shopping_mall", "department_store", "clothing_store", "electronics_store", "jewelry_store", "bookstore"},
    "supermarket": {"supermarket", "convenience_store"},
    "food": {"restaurant", "fast_food"},
    "cafe": {"cafe", "bakery"},
    "culture": {"attraction", "viewpoint", "bookstore"},
    "nature": {"attraction", "viewpoint"},
    "view": {"viewpoint"},
}

# Penalize drift categories based on PRIMARY intent only
INTENT_EXCLUDE = {
    "shopping": {"restaurant", "fast_food", "cafe", "bakery", "supermarket", "convenience_store"},
    "supermarket": {"shopping_mall", "department_store", "electronics_store", "clothing_store", "jewelry_store", "restaurant", "fast_food", "cafe", "bakery"},
    "cafe": {"shopping_mall", "department_store", "electronics_store", "clothing_store", "jewelry_store", "supermarket", "convenience_store", "restaurant", "fast_food"},
    "food": {"shopping_mall", "department_store", "electronics_store", "clothing_store", "jewelry_store", "supermarket", "convenience_store", "cafe", "bakery"},
    # others can be added if needed
}

CONTAINER_ONLY_INTENTS = {"shopping mall", "shopping malls", "mall", "malls"}


# ---------------------------
# Utilities
# ---------------------------

def _safe_lower(x: Any) -> str:
    return str(x).strip().lower() if x is not None else ""


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _safe_lower(text)))


def _detect_intents(prompt: str, interests: Optional[List[str]] = None) -> List[str]:
    """Return up to 2 intents based on prompt + interests (multi-intent)."""
    interests = interests or []
    text = f"{prompt} " + " ".join(interests)
    t = _safe_lower(text)
    toks = _tokenize(t)

    scores: Dict[str, int] = defaultdict(int)
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            kw_l = _safe_lower(kw)
            if " " in kw_l:
                if kw_l in t:
                    scores[intent] += 3
            else:
                if kw_l in toks:
                    scores[intent] += 2
                elif kw_l in t:
                    scores[intent] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, s in ranked if s > 0][:2]


def _primary_intent(prompt: str, interests: List[str]) -> Optional[str]:
    """
    Choose ONE dominant intent based on prompt.
    Shopping is last to prevent it overpowering more specific intents.
    """
    intents = _detect_intents(prompt, interests)
    if not intents:
        return None

    PRIORITY = [
        "supermarket",
        "cafe",
        "food",
        "culture",
        "nature",
        "view",
        "shopping",
    ]
    for p in PRIORITY:
        if p in intents:
            return p
    return intents[0]


def _is_container_intent(prompt: str) -> bool:
    p = _safe_lower(prompt)
    return any(phrase in p for phrase in CONTAINER_ONLY_INTENTS)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    if base_dir is None:
        return p.resolve()
    return (Path(base_dir) / p).resolve()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between points (km)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _distance_bonus_km(d_km: Optional[float]) -> float:
    """
    Smooth bonus: nearby places get a positive boost; far places ~0.
    Tuned to matter within ~0-5km.
    """
    if d_km is None:
        return 0.0
    return 0.35 * (1.0 / (1.0 + d_km))


# ---------------------------
# Data Structures
# ---------------------------

@dataclass
class POIDetails:
    poi_id: str
    name: str
    category: str
    level: str = "level_0"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: str = ""
    popularity: float = 0.0
    price: str = ""


# ---------------------------
# Framework
# ---------------------------

class MultiGranularityRecommendationFramework:
    VERSION = "realtime-final-location-v1"

    def __init__(
        self,
        embeddings_file: str,
        poi_tree_file: str,
        users_file: str,
        interactions_file: str,
        interaction_learning_file: Optional[str] = None,
        sources_dir: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parent

        self.embeddings_file = _resolve_path(embeddings_file, base_dir)
        self.poi_tree_file = _resolve_path(poi_tree_file, base_dir)
        self.users_file = _resolve_path(users_file, base_dir)
        self.interactions_file = _resolve_path(interactions_file, base_dir)
        self.interaction_learning_file = _resolve_path(interaction_learning_file, base_dir) if interaction_learning_file else None
        self.sources_dir = _resolve_path(sources_dir, base_dir) if sources_dir else None

        self.user_embeddings_mat: np.ndarray = np.zeros((0, 0), dtype=float)
        self.poi_embeddings_mat: np.ndarray = np.zeros((0, 0), dtype=float)

        self.user_id_to_idx: Dict[str, int] = {}
        self.poi_id_to_idx: Dict[str, int] = {}

        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.poi_details_by_id: Dict[str, POIDetails] = {}

        self._load_embeddings(self.embeddings_file)
        self._load_poi_tree_json(self.poi_tree_file)
        self._load_user_preferences_csv(self.users_file)

    # ---------------------------
    # Robust Embeddings Loader
    # ---------------------------

    def _load_embeddings(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"embeddings file not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        def _extract_vec(v):
            if isinstance(v, dict):
                v = v.get("embedding") or v.get("vector") or v.get("emb") or v.get("values")
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if not isinstance(v, (list, tuple)):
                return None

            if len(v) > 0 and isinstance(v[0], (list, tuple, np.ndarray)):
                if len(v) == 1:
                    v = v[0]
                else:
                    return None

            out = []
            for item in v:
                try:
                    out.append(float(item))
                except Exception:
                    return None
            return out

        def _finalize(vectors: List[List[float]], ids: List[str]):
            if not vectors:
                return np.zeros((0, 0), dtype=float), {}

            lengths = [len(v) for v in vectors]
            mode_len = max(set(lengths), key=lengths.count)

            cleaned_vecs = []
            cleaned_map: Dict[str, int] = {}
            for vid, vec in zip(ids, vectors):
                if len(vec) == mode_len:
                    cleaned_map[str(vid)] = len(cleaned_vecs)
                    cleaned_vecs.append(vec)

            if not cleaned_vecs:
                mode_len = max(lengths)
                cleaned_vecs = []
                cleaned_map = {}
                for vid, vec in zip(ids, vectors):
                    if not vec:
                        continue
                    vec2 = vec[:mode_len] + [0.0] * max(0, mode_len - len(vec))
                    cleaned_map[str(vid)] = len(cleaned_vecs)
                    cleaned_vecs.append(vec2)

            return np.array(cleaned_vecs, dtype=float), cleaned_map

        def _to_matrix_and_index(x, key_name: str):
            if x is None:
                return np.zeros((0, 0), dtype=float), {}

            if isinstance(x, np.ndarray):
                return np.array(x, dtype=float), {}

            if isinstance(x, list):
                if len(x) == 0:
                    return np.zeros((0, 0), dtype=float), {}

                if isinstance(x[0], (list, tuple, np.ndarray)):
                    vecs, ids = [], []
                    for i, v in enumerate(x):
                        vec = _extract_vec(v)
                        if vec is None:
                            continue
                        vecs.append(vec)
                        ids.append(str(i))
                    return _finalize(vecs, ids)

                if isinstance(x[0], dict):
                    vecs, ids = [], []
                    for row in x:
                        rid = row.get("id") or row.get("user_id") or row.get("poi_id") or row.get("uuid")
                        vec = _extract_vec(row)
                        if rid is None or vec is None:
                            continue
                        vecs.append(vec)
                        ids.append(str(rid))
                    return _finalize(vecs, ids)

            if isinstance(x, dict):
                vecs, ids = [], []
                for k, v in x.items():
                    vec = _extract_vec(v)
                    if vec is None:
                        continue
                    vecs.append(vec)
                    ids.append(str(k))
                return _finalize(vecs, ids)

            raise TypeError(f"Unsupported embedding format for {key_name}: {type(x)}")

        user_raw = obj.get("user_embeddings") or obj.get("user_embeddings_mat") or obj.get("users") or obj.get("user_vecs")
        poi_raw = obj.get("poi_embeddings") or obj.get("poi_embeddings_mat") or obj.get("pois") or obj.get("poi_vecs")

        user_mat, user_map = _to_matrix_and_index(user_raw, "user_embeddings")
        poi_mat, poi_map = _to_matrix_and_index(poi_raw, "poi_embeddings")

        self.user_id_to_idx = dict(obj.get("user_id_to_idx", user_map or {}))
        self.poi_id_to_idx = dict(obj.get("poi_id_to_idx", poi_map or {}))

        self.user_embeddings_mat = user_mat
        self.poi_embeddings_mat = poi_mat

        if self.user_embeddings_mat.size > 0:
            norms = np.linalg.norm(self.user_embeddings_mat, axis=1, keepdims=True) + 1e-12
            self.user_embeddings_mat = self.user_embeddings_mat / norms

        if self.poi_embeddings_mat.size > 0:
            norms = np.linalg.norm(self.poi_embeddings_mat, axis=1, keepdims=True) + 1e-12
            self.poi_embeddings_mat = self.poi_embeddings_mat / norms

    # ---------------------------
    # POI Tree Loader (ALL levels)
    # ---------------------------

    def _load_poi_tree_json(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"poi tree file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            tree_obj = json.load(f)

        self._load_poi_tree(tree_obj)

    def _load_poi_tree(self, tree_obj: Any):
        self.poi_details_by_id.clear()
        self.poi_id_to_idx = self.poi_id_to_idx or {}

        if not isinstance(tree_obj, dict):
            return

        levels = ["level_0", "level_1", "level_2", "level_3"]
        next_idx = max(self.poi_id_to_idx.values(), default=-1) + 1

        for lvl in levels:
            if lvl not in tree_obj or not isinstance(tree_obj[lvl], dict):
                continue

            for pid, node in tree_obj[lvl].items():
                pid = str(pid)
                if not isinstance(node, dict):
                    continue

                data = node.get("data") if isinstance(node.get("data"), dict) else {}
                name = str(node.get("name") or pid)
                cat = _safe_lower(data.get("category") or node.get("type") or "")

                lat = data.get("latitude")
                lon = data.get("longitude")

                det = POIDetails(
                    poi_id=pid,
                    name=name,
                    category=cat,
                    level=lvl,
                    latitude=float(lat) if lat not in (None, "") else None,
                    longitude=float(lon) if lon not in (None, "") else None,
                    region=str(data.get("region") or ""),
                    popularity=float(data.get("popularity") or 0.0),
                    price=str(data.get("price") or ""),
                )
                self.poi_details_by_id[pid] = det

                # containers may not have embeddings; safe to index anyway
                if pid not in self.poi_id_to_idx:
                    self.poi_id_to_idx[pid] = next_idx
                    next_idx += 1

    # ---------------------------
    # Users CSV loader (optional)
    # ---------------------------

    def _load_user_preferences_csv(self, path: Path):
        if not path.exists():
            return
        try:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("user_id") or row.get("userId") or "").strip()
                    if uid:
                        self.user_profiles[uid] = dict(row)
        except Exception:
            pass

    # ---------------------------
    # API helpers
    # ---------------------------

    def health(self) -> Dict[str, Any]:
        def _count(cat: str) -> int:
            return sum(1 for d in self.poi_details_by_id.values() if d.category == cat)

        return {
            "status": "healthy",
            "framework_loaded": True,
            "framework_version": self.VERSION,
            "total_users": len(self.user_id_to_idx),
            "total_pois_all_levels": len(self.poi_details_by_id),
            "shopping_mall_pois_loaded": _count("shopping_mall"),
            "cafe_pois_loaded": _count("cafe"),
            "supermarket_pois_loaded": _count("supermarket"),
            "user_embedding_dim": int(self.user_embeddings_mat.shape[1]) if self.user_embeddings_mat.size else 0,
            "poi_embedding_dim": int(self.poi_embeddings_mat.shape[1]) if self.poi_embeddings_mat.size else 0,
        }

    # ---------------------------
    # Onboarding
    # ---------------------------

    def add_user_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(profile.get("user_id", "")).strip()
        if not user_id:
            return {"status": "error", "message": "missing user_id"}

        if user_id in self.user_id_to_idx:
            # update profile
            self.user_profiles[user_id] = {**self.user_profiles.get(user_id, {}), **profile}
            return {"status": "exists", "user_id": user_id}

        # initialize a new embedding vector
        if self.user_embeddings_mat.size == 0:
            new_vec = np.random.normal(0, 0.01, size=(1, 64)).astype(float)
        else:
            mean_vec = np.mean(self.user_embeddings_mat, axis=0, keepdims=True)
            noise = np.random.normal(0, 0.005, size=mean_vec.shape)
            new_vec = mean_vec + noise

        new_vec = new_vec / (np.linalg.norm(new_vec, axis=1, keepdims=True) + 1e-12)

        if self.user_embeddings_mat.size == 0:
            self.user_embeddings_mat = new_vec
        else:
            self.user_embeddings_mat = np.vstack([self.user_embeddings_mat, new_vec])

        new_idx = int(self.user_embeddings_mat.shape[0] - 1)
        self.user_id_to_idx[user_id] = new_idx
        self.user_profiles[user_id] = profile

        return {"status": "created", "user_id": user_id, "user_index": new_idx}

    # ---------------------------
    # Allowed categories (PRIMARY intent locking)
    # ---------------------------

    def _intent_allowed_categories(self, user_profile: Dict[str, Any], prompt: str) -> List[str]:
        interests = user_profile.get("interests", []) or []

        # explicit malls/place intent
        if _is_container_intent(prompt):
            return ["shopping_mall"]

        primary = _primary_intent(prompt, interests)
        if not primary:
            return []

        allowed = INTENT_CATEGORIES.get(primary, set())
        return sorted([c for c in allowed if c in DATASET_CATEGORIES])

    # ---------------------------
    # Scoring
    # ---------------------------

    def _score_user_poi(
        self,
        uvec: np.ndarray,
        poi_id: str,
        det: POIDetails,
        allowed_categories: List[str],
        prompt: str,
        user_profile: Dict[str, Any],
    ) -> float:
        # embedding cosine similarity (if POI embedding exists)
        base = 0.0
        if self.poi_embeddings_mat.size > 0 and poi_id in self.poi_id_to_idx:
            pidx = self.poi_id_to_idx[poi_id]
            if 0 <= pidx < self.poi_embeddings_mat.shape[0]:
                base = _cosine(uvec, self.poi_embeddings_mat[pidx])

        # small popularity tie-breaker
        pop_bonus = (float(det.popularity) / 5.0) if det.popularity else 0.0
        score = float(base + 0.15 * pop_bonus)

        cat = _safe_lower(det.category)
        interests = user_profile.get("interests", []) or []
        primary = _primary_intent(prompt, interests)

        # allowed category boost/penalty
        if allowed_categories:
            allowed_set = set(allowed_categories)
            if cat in allowed_set:
                score += 0.55
            else:
                score -= 0.30

        # primary intent drift penalty (strong)
        if primary:
            excluded = INTENT_EXCLUDE.get(primary, set())
            if cat in excluded:
                score -= 0.85

        # explicit mall intent: force malls up
        if _is_container_intent(prompt):
            if cat == "shopping_mall":
                score += 0.85
            else:
                score -= 0.25

        return score

    # ---------------------------
    # Recommendation (nearby-first + radius expansion + bucketed outputs)
    # ---------------------------

    def recommend_multi_granularity(
        self,
        user_id: str,
        top_k: int = 10,
        prompt: str = "",
        use_intent_filter: bool = True,
        current_location: Optional[Dict[str, float]] = None,  # NEW
    ) -> Dict[str, List[Dict[str, Any]]]:

        user_id = str(user_id).strip()
        if user_id not in self.user_id_to_idx:
            raise ValueError(f"Unknown user_id: {user_id} (onboard first)")

        uidx = self.user_id_to_idx[user_id]
        uvec = self.user_embeddings_mat[uidx]

        user_profile = self.user_profiles.get(user_id, {}) or {}
        allowed_categories = self._intent_allowed_categories(user_profile, prompt) if use_intent_filter else []

        # Resolve location priority:
        # 1) request current_location
        # 2) profile last_location
        # 3) DEFAULT_LOCATION
        loc = current_location or user_profile.get("last_location") or DEFAULT_LOCATION
        try:
            u_lat = float(loc.get("latitude"))
            u_lon = float(loc.get("longitude"))
        except Exception:
            u_lat = float(DEFAULT_LOCATION["latitude"])
            u_lon = float(DEFAULT_LOCATION["longitude"])

        min_results = max(10, int(top_k))

        def score_with_filter(allowed: Optional[List[str]], max_radius_km: Optional[float]):
            out: List[Tuple[str, float]] = []
            allowed_set = set(allowed) if allowed else None

            for pid, det in self.poi_details_by_id.items():
                cat = _safe_lower(det.category)
                if allowed_set is not None and cat and cat not in allowed_set:
                    continue

                d_km: Optional[float] = None
                if det.latitude is not None and det.longitude is not None:
                    d_km = _haversine_km(u_lat, u_lon, float(det.latitude), float(det.longitude))
                    if max_radius_km is not None and d_km > max_radius_km:
                        continue

                base_score = self._score_user_poi(uvec, pid, det, allowed or [], prompt, user_profile)
                score = base_score + _distance_bonus_km(d_km)
                out.append((pid, score))

            return out

        # --- Pass 1: strict intent filter (with radius expansion) ---
        scored: List[Tuple[str, float]] = []
        if _is_container_intent(prompt):
            strict_allowed = ["shopping_mall"]
        else:
            strict_allowed = allowed_categories if allowed_categories else None

        for r in RADIUS_TIERS_KM:
            scored = score_with_filter(strict_allowed, r)
            if len(scored) >= min_results:
                break

        # --- Pass 2: widen categories (with radius expansion) ---
        if len(scored) < min_results:
            interests = user_profile.get("interests", []) or []
            primary = _primary_intent(prompt, interests)

            if _is_container_intent(prompt):
                widened = sorted(INTENT_CATEGORIES.get("shopping", {"shopping_mall"}))
            elif primary:
                widened = sorted(INTENT_CATEGORIES.get(primary, set()))
            else:
                widened = []

            if widened:
                for r in RADIUS_TIERS_KM:
                    scored = score_with_filter(widened, r)
                    if len(scored) >= min_results:
                        break
                allowed_categories = widened

        # --- Pass 3: full catalog (with radius expansion) ---
        if len(scored) < min_results:
            for r in RADIUS_TIERS_KM:
                scored = score_with_filter(None, r)
                if len(scored) >= min_results:
                    break

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(1, int(top_k))]

        # Build output bucketed by levels
        buckets: Dict[str, List[Dict[str, Any]]] = {"level_0": [], "level_1": [], "level_2": [], "level_3": []}

        for pid, s in top:
            det = self.poi_details_by_id.get(pid)
            lvl = det.level if det and det.level in buckets else "level_0"

            item = {
                "poi_id": pid,
                "name": det.name if det else pid,
                "score": float(s),
                "type": lvl.replace("_", " ").title(),
                "details": {
                    "name": det.name if det else pid,
                    "type": lvl.replace("_", " ").title(),
                    "category": det.category if det else "",
                    "price": det.price if det else "",
                    "popularity": det.popularity if det else 0,
                    "region": det.region if det else "",
                    "latitude": det.latitude if det else None,
                    "longitude": det.longitude if det else None,
                }
            }
            buckets[lvl].append(item)

        return buckets

import os
import re
import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

warnings.filterwarnings("ignore")


class MultiGranularityRecommendationFramework:
    """
    Multi-Granularity POI Recommendation System

    Supports:
    - Attribute-based (always)
    - Interaction-based (optional, if interaction_learning.pkl exists)

    Levels:
    - 0: POIs
    - 1: Containers
    - 2: Districts
    - 3: Regions
    """

    # ----------------------------
    # Init / Loaders
    # ----------------------------
    def __init__(
        self,
        embeddings_file: str = "embeddings.pkl",
        interaction_learning_file: str = "interaction_learning.pkl",
        poi_tree_file: str = "poi_tree.json",
        users_file: str = "users.csv",
        interactions_file: str = "user_poi_interactions.csv",
    ):
        print("=" * 70)
        print("INITIALIZING MULTI-GRANULARITY RECOMMENDATION FRAMEWORK")
        print("=" * 70)

        self.base_dir = Path(__file__).resolve().parent

        # Remember filenames for persistence (onboarding updates)
        self.embeddings_file = embeddings_file
        self.interaction_learning_file = interaction_learning_file
        self.poi_tree_file = poi_tree_file
        self.users_file = users_file
        self.interactions_file = interactions_file

        # ---- Load attribute embeddings ----
        print("\nLoading attribute-based embeddings...")
        emb_path = self._resolve_path(embeddings_file)
        with open(emb_path, "rb") as f:
            emb_data = pickle.load(f)

        # expected keys (best effort)
        self.user_embeddings = emb_data.get("user_embeddings", {})  # dict user_id -> vector/row
        self.poi_embeddings = emb_data.get("poi_embeddings", {})    # dict: level_0..level_3
        self.user_id_to_idx = emb_data.get("user_id_to_idx", {})    # dict user_id -> idx

        self.X = emb_data.get("X", None)
        self.X_A = emb_data.get("X_A", None)
        self.X_T = emb_data.get("X_T", None)

        # Optional encoders if already saved; if not, we refit from users_df.
        self.age_encoder = emb_data.get("age_encoder", None)
        self.interests_mlb = emb_data.get("interests_mlb", None)
        self.transport_mlb = emb_data.get("transport_mlb", None)
        self.price_encoder = emb_data.get("price_encoder", None)

        self.attribute_dim = int(self.X_A.shape[1]) if isinstance(self.X_A, np.ndarray) else None
        self.interaction_dim = int(self.X_T.shape[1]) if isinstance(self.X_T, np.ndarray) else 0

        # ---- Load POI tree ----
        print("Loading POI tree...")
        tree_path = self._resolve_path(poi_tree_file)
        with open(tree_path, "r", encoding="utf-8") as f:
            self.poi_tree = json.load(f)

        # ---- Load users + interactions ----
        print("Loading user profiles and interactions...")
        users_path = self._resolve_path(users_file)
        inter_path = self._resolve_path(interactions_file)
        self.users_df = pd.read_csv(users_path)
        self.interactions_df = pd.read_csv(inter_path)

        # ---- Build indices + history ----
        self._build_indices()

        # ---- Fit encoders if not present ----
        if any(x is None for x in [self.age_encoder, self.interests_mlb, self.transport_mlb, self.price_encoder]):
            self._fit_user_encoders()

        # ---- Try load interaction learner (optional) ----
        self._load_interaction_components()

        # weights
        self.alpha = 0.5
        self.beta = 0.3
        self.gamma = 0.2
        self.level_weights = {0: 0.6, 1: 0.25, 2: 0.10, 3: 0.05}

        print("\n" + "=" * 70)
        print("FRAMEWORK INITIALIZED")
        print("=" * 70)
        print(f"Users: {len(self.users_df)}")
        print(f"Interactions: {len(self.interactions_df)}")
        print(f"Interaction mode: {'ON' if self.has_interaction_model else 'OFF (attribute-only)'}")

    def _resolve_path(self, p: str) -> str:
        """Robust path resolver: supports running from different working dirs."""
        pth = Path(p)
        if pth.is_absolute() and pth.exists():
            return str(pth)

        candidates = [
            self.base_dir / p,
            self.base_dir.parent / p,
            Path.cwd() / p,
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        # search by filename in nearby repo dirs
        name = Path(p).name
        for root in [self.base_dir, self.base_dir.parent, Path.cwd()]:
            try:
                matches = list(root.rglob(name))
                if matches:
                    return str(matches[0])
            except Exception:
                pass

        # special: if embeddings.pkl missing, try any *embedding*.pkl
        if name.lower() == "embeddings.pkl":
            for root in [self.base_dir, self.base_dir.parent, Path.cwd()]:
                try:
                    cands = sorted([x for x in root.rglob("*.pkl") if "embedding" in x.name.lower()])
                    if cands:
                        return str(cands[0])
                except Exception:
                    pass

        # fallback to base_dir/p
        return str(self.base_dir / p)

    def _load_interaction_components(self) -> None:
        """Load interaction components if present; else run attribute-only mode."""
        self.has_interaction_model = False

        # defaults
        self.Theta_u = None
        self.A_l_p = None
        self.G_l = None
        self.P_l = None
        self.Q_l = None
        self.S_l = None
        self.U_l_g = None

        try:
            print("Loading interaction-based components...")
            path = self._resolve_path(self.interaction_learning_file)
            with open(path, "rb") as f:
                int_data = pickle.load(f)

            self.Theta_u = int_data.get("Theta_u")
            self.A_l_p = int_data.get("A_l_p")
            self.G_l = int_data.get("G_l")
            self.P_l = int_data.get("P_l")
            self.Q_l = int_data.get("Q_l")
            self.S_l = int_data.get("S_l")
            self.U_l_g = int_data.get("U_l_g")

            # Basic validation
            needed = [self.Q_l, self.S_l, self.U_l_g]
            self.has_interaction_model = all(x is not None for x in needed)
            if not self.has_interaction_model:
                print("⚠️ interaction_learning.pkl loaded but missing required keys — attribute-only mode.")
        except FileNotFoundError:
            print("⚠️ interaction_learning.pkl not found — attribute-only mode.")
        except Exception as e:
            print(f"⚠️ Could not load interaction components ({e}) — attribute-only mode.")

    # ----------------------------
    # Indices / History
    # ----------------------------
    def _build_indices(self):
        """Build lookup indices for fast access."""
        if "uudi" not in self.users_df.columns:
            raise ValueError("users.csv must contain column 'uudi'")

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users_df["uudi"])}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        # POI ids at each level from poi_embeddings
        self.poi_id_to_idx = {}
        self.idx_to_poi_id = {}
        for level in [0, 1, 2, 3]:
            level_key = f"level_{level}"
            if level_key not in self.poi_embeddings:
                # allow framework to run even if some levels missing
                self.poi_id_to_idx[level] = {}
                self.idx_to_poi_id[level] = {}
                continue

            poi_ids = self.poi_embeddings[level_key].get("poi_ids", [])
            self.poi_id_to_idx[level] = {pid: idx for idx, pid in enumerate(poi_ids)}
            self.idx_to_poi_id[level] = {idx: pid for pid, idx in self.poi_id_to_idx[level].items()}

        self._build_user_history()

    def _build_user_history(self):
        """Build visit history per level."""
        self.user_history = defaultdict(lambda: defaultdict(list))

        if not {"interaction_type", "user_id", "poi_id"}.issubset(set(self.interactions_df.columns)):
            return

        visits = self.interactions_df[self.interactions_df["interaction_type"] == "visit"]
        for _, row in visits.iterrows():
            user_id = row["user_id"]
            poi_id = row["poi_id"]

            if poi_id not in self.user_history[user_id][0]:
                self.user_history[user_id][0].append(poi_id)

            for level in [1, 2, 3]:
                parent_id = self._get_parent_at_level(poi_id, level)
                if parent_id and parent_id not in self.user_history[user_id][level]:
                    self.user_history[user_id][level].append(parent_id)

    def _get_parent_at_level(self, poi_id: str, target_level: int) -> Optional[str]:
        current_level = 0
        current_id = poi_id

        while current_level < target_level:
            level_key = f"level_{current_level}"
            if current_id in self.poi_tree.get(level_key, {}):
                parent = self.poi_tree[level_key][current_id].get("parent")
                if parent:
                    current_id = parent
                    current_level += 1
                else:
                    break
            else:
                break

        return current_id if current_level == target_level else None

    # ----------------------------
    # Real-time onboarding helpers
    # ----------------------------
    def _fit_user_encoders(self):
        """Fit encoders from current users_df."""
        df = self.users_df.copy()

        def split_list(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return []
            if isinstance(x, list):
                return x
            s = str(x)
            parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
            return parts

        for col in ["age_group", "interests", "transportation_modes", "price_sensitivity"]:
            if col not in df.columns:
                df[col] = "unknown"

        # AGE
        self.age_encoder = LabelEncoder()
        self.age_encoder.fit(df["age_group"].astype(str).fillna("unknown"))

        # PRICE
        self.price_encoder = LabelEncoder()
        self.price_encoder.fit(df["price_sensitivity"].astype(str).fillna("unknown"))

        # INTERESTS
        interests_lists = df["interests"].apply(split_list).tolist()
        self.interests_mlb = MultiLabelBinarizer()
        self.interests_mlb.fit(interests_lists)

        # TRANSPORT
        transport_lists = df["transportation_modes"].apply(split_list).tolist()
        self.transport_mlb = MultiLabelBinarizer()
        self.transport_mlb.fit(transport_lists)

    def _encode_user_attributes(
        self,
        age_group: str,
        interests: List[str],
        transportation_modes: List[str],
        price_sensitivity: str,
    ) -> np.ndarray:
        """Return 1 x attr_dim matrix."""
        if any(x is None for x in [self.age_encoder, self.price_encoder, self.interests_mlb, self.transport_mlb]):
            self._fit_user_encoders()

        interests = interests or []
        transportation_modes = transportation_modes or []

        # safe one-hot for LabelEncoder
        def one_hot(le: LabelEncoder, value: str) -> np.ndarray:
            classes = list(le.classes_)
            v = "unknown" if value is None else str(value)
            if v not in classes:
                v = "unknown" if "unknown" in classes else classes[0]
            idx = classes.index(v)
            out = np.zeros((1, len(classes)), dtype=float)
            out[0, idx] = 1.0
            return out

        age_vec = one_hot(self.age_encoder, age_group)
        price_vec = one_hot(self.price_encoder, price_sensitivity)

        interest_vec = self.interests_mlb.transform([interests]).astype(float)
        transport_vec = self.transport_mlb.transform([transportation_modes]).astype(float)

        x_a = np.hstack([age_vec, interest_vec, transport_vec, price_vec])
        return x_a

    def _persist_users_csv(self):
        path = self._resolve_path(self.users_file)
        self.users_df.to_csv(path, index=False)

    def _persist_embeddings_pkl(self):
        """Optional: persist updated encoders + embeddings so server restart keeps onboarded users."""
        path = self._resolve_path(self.embeddings_file)
        data = {
            "user_embeddings": self.user_embeddings,
            "poi_embeddings": self.poi_embeddings,
            "user_id_to_idx": self.user_id_to_idx,
            "X": self.X,
            "X_A": self.X_A,
            "X_T": self.X_T,
            # persist encoders for consistent onboarding
            "age_encoder": self.age_encoder,
            "interests_mlb": self.interests_mlb,
            "transport_mlb": self.transport_mlb,
            "price_encoder": self.price_encoder,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # ----------------------------
    # Real-time onboarding (MAIN)
    # ----------------------------
    def add_user_profile(
        self,
        user_id: str,
        age_group: str,
        interests: List[str],
        transportation_modes: List[str],
        price_sensitivity: str,
        persist: bool = True,
    ) -> Dict:
        """
        Real-time onboarding:
        - Encode attributes -> X_A
        - Cold-start X_T (zeros) if interaction dim exists
        - Update in-memory structures and persist to users.csv (+ embeddings.pkl optionally)
        """
        if user_id in self.user_id_to_idx:
            return {"status": "exists", "user_id": user_id}

        x_a = self._encode_user_attributes(age_group, interests, transportation_modes, price_sensitivity)

        # cold-start interaction vector (keep consistent dim if known)
        interaction_dim = self.interaction_dim if isinstance(self.interaction_dim, int) else 0
        x_t = np.zeros((1, interaction_dim), dtype=float)
        x = np.hstack([x_a, x_t]) if interaction_dim > 0 else x_a.copy()

        # update indices
        new_idx = len(self.user_id_to_idx)
        self.user_id_to_idx[user_id] = new_idx
        self.idx_to_user_id[new_idx] = user_id

        # update embedding dict
        self.user_embeddings[user_id] = x

        # update matrices (best effort)
        self.X_A = x_a if self.X_A is None else np.vstack([self.X_A, x_a])
        if interaction_dim > 0:
            self.X_T = x_t if self.X_T is None else np.vstack([self.X_T, x_t])
            self.X = x if self.X is None else np.vstack([self.X, x])
        else:
            self.X = x if self.X is None else np.vstack([self.X, x])

        # append to users_df (make sure these columns exist)
        for col in ["age_group", "interests", "transportation_modes", "price_sensitivity"]:
            if col not in self.users_df.columns:
                self.users_df[col] = ""

        new_row = {
            "uudi": user_id,
            "age_group": age_group,
            "interests": ";".join(interests or []),
            "transportation_modes": ";".join(transportation_modes or []),
            "price_sensitivity": price_sensitivity,
        }
        self.users_df = pd.concat([self.users_df, pd.DataFrame([new_row])], ignore_index=True)

        # re-fit encoders so new categories are included going forward
        self._fit_user_encoders()

        # rebuild user history container
        _ = self.user_history[user_id]  # create default structure

        if persist:
            self._persist_users_csv()
            try:
                self._persist_embeddings_pkl()
            except Exception as e:
                print(f"⚠️ Could not persist embeddings.pkl: {e}")

        return {
            "status": "created",
            "user_id": user_id,
            "new_index": int(new_idx),
            "attr_dim": int(x_a.shape[1]),
            "interaction_dim": int(interaction_dim),
            "total_dim": int(x.shape[1]),
            "mode": "interaction" if self.has_interaction_model else "attribute_only",
        }

    # ----------------------------
    # Scoring (safe)
    # ----------------------------
    def compute_feature_based_score(self, user_idx: int, poi_idx: int, level: int) -> float:
        """Safe: returns 0 if no interaction model or index out of bounds."""
        if not self.has_interaction_model or self.S_l is None:
            return 0.0
        level_key = f"level_{level}"
        mat = self.S_l.get(level_key)
        if mat is None:
            return 0.0
        if user_idx < 0 or user_idx >= mat.shape[0] or poi_idx < 0 or poi_idx >= mat.shape[1]:
            return 0.0
        return float(mat[user_idx, poi_idx])

    def compute_graph_based_score(self, user_idx: int, poi_idx: int, level: int) -> float:
        """Safe: returns 0 if no interaction model."""
        if not self.has_interaction_model or self.U_l_g is None or self.Q_l is None:
            return 0.0

        level_key = f"level_{level}"
        U = self.U_l_g.get(level_key)
        Q = self.Q_l.get(level_key)
        if U is None or Q is None:
            return 0.0
        if user_idx < 0 or user_idx >= U.shape[0] or poi_idx < 0 or poi_idx >= U.shape[1]:
            return 0.0

        U_g_up = U[user_idx, poi_idx]
        Q_p = Q[poi_idx]

        min_len = min(len(U_g_up), len(Q_p))
        U_g_up = U_g_up[:min_len]
        Q_p = Q_p[:min_len]

        if np.linalg.norm(U_g_up) == 0 or np.linalg.norm(Q_p) == 0:
            return 0.0

        score = np.dot(U_g_up, Q_p) / (np.linalg.norm(U_g_up) * np.linalg.norm(Q_p))
        return float(score)

    def compute_hierarchical_boost(self, poi_id: str, user_idx: int, level: int) -> float:
        """Hierarchical boost using children/parent."""
        if level >= 2:
            children = self.poi_tree.get(f"level_{level}", {}).get(poi_id, {}).get("children", [])
            if not children:
                return 0.0

            child_scores = []
            for child_id in children[:10]:
                child_level = level - 1
                if child_level in self.poi_id_to_idx and child_id in self.poi_id_to_idx[child_level]:
                    child_idx = self.poi_id_to_idx[child_level][child_id]
                    child_scores.append(self.compute_feature_based_score(user_idx, child_idx, child_level))
            return float(np.mean(child_scores)) if child_scores else 0.0

        # fine level: boost from parent
        poi_data = self.poi_tree.get(f"level_{level}", {}).get(poi_id)
        if not poi_data:
            return 0.0
        parent_id = poi_data.get("parent")
        if not parent_id:
            return 0.0

        parent_level = level + 1
        if parent_level in self.poi_id_to_idx and parent_id in self.poi_id_to_idx[parent_level]:
            parent_idx = self.poi_id_to_idx[parent_level][parent_id]
            return self.compute_feature_based_score(user_idx, parent_idx, parent_level)

        return 0.0

    def compute_interest_match(self, user_id: str, poi_id: str, level: int = 0) -> float:
        """Simple interest match based on category/characteristic."""
        if "interests" not in self.users_df.columns:
            return 0.0
        user_row = self.users_df[self.users_df["uudi"] == user_id]
        if user_row.empty:
            return 0.0
        user_row = user_row.iloc[0]

        try:
            user_interests = set([i.strip().lower() for i in str(user_row["interests"]).split(";") if i.strip()])
        except Exception:
            user_interests = set()

        if level != 0:
            return 0.5

        level_key = f"level_{level}"
        if poi_id not in self.poi_tree.get(level_key, {}):
            return 0.0

        poi_data = self.poi_tree[level_key][poi_id]
        poi_category = str(poi_data.get("data", {}).get("category", "")).lower()
        poi_chars = str(poi_data.get("data", {}).get("characteristic", "")).lower()

        matches = 0.0
        if "food" in user_interests and poi_category in ["restaurant", "cafe", "food_court"]:
            matches += 1.0
        if "shopping" in user_interests and poi_category in ["shopping_mall", "retail", "store"]:
            matches += 1.0
        if "movies" in user_interests and "cinema" in poi_category:
            matches += 2.0

        for interest in user_interests:
            if interest and interest in poi_chars:
                matches += 0.5

        return float(min(matches / 3.0, 1.0))

    def compute_distance_penalty(self, user_id: str, poi_id: str, level: int = 0) -> float:
        """Distance penalty from user's area_of_residence to POI coordinates (level 0 only)."""
        if level != 0:
            return 1.0
        if "area_of_residence" not in self.users_df.columns:
            return 0.5

        user_row = self.users_df[self.users_df["uudi"] == user_id]
        if user_row.empty:
            return 0.5
        user_row = user_row.iloc[0]

        area_coords = {
            "Jurong East": (1.3329, 103.7436),
            "Yishun": (1.4304, 103.8354),
            "Bishan": (1.3526, 103.8352),
            "Tampines": (1.3496, 103.9568),
            "Woodlands": (1.4382, 103.7891),
            "Ang Mo Kio": (1.3691, 103.8454),
            "Bedok": (1.3236, 103.9273),
            "Clementi": (1.3162, 103.7649),
            "Hougang": (1.3612, 103.8864),
            "Punggol": (1.4054, 103.9021),
            "Sengkang": (1.3868, 103.8914),
        }

        user_area = str(user_row.get("area_of_residence", ""))
        if user_area not in area_coords:
            return 0.5
        user_lat, user_lon = area_coords[user_area]

        level_key = f"level_{level}"
        poi_data = self.poi_tree.get(level_key, {}).get(poi_id)
        if not poi_data:
            return 0.5

        poi_spatial = poi_data.get("spatial")
        if isinstance(poi_spatial, str):
            try:
                poi_spatial = eval(poi_spatial)
            except Exception:
                poi_spatial = None

        if not poi_spatial or not isinstance(poi_spatial, (list, tuple)) or len(poi_spatial) < 2:
            return 0.5

        poi_lat, poi_lon = float(poi_spatial[0]), float(poi_spatial[1])
        distance = self._haversine_distance(user_lat, user_lon, poi_lat, poi_lon)

        user_transport = str(user_row.get("transportation_modes", "")).lower()
        if "car" in user_transport or "ride" in user_transport:
            max_dist = 15.0
        elif "mrt" in user_transport or "bus" in user_transport:
            max_dist = 8.0
        else:
            max_dist = 3.0

        return float(np.exp(-distance / max_dist))

    def compute_multi_granularity_score(
        self,
        user_id: str,
        poi_id: str,
        level: int = 0,
        use_hierarchical_boost: bool = True,
        use_graph_context: bool = True,
        use_distance_penalty: bool = True,
    ) -> float:
        """Safe scoring that works in attribute-only mode."""
        if user_id not in self.user_id_to_idx:
            return 0.0
        if level not in self.poi_id_to_idx or poi_id not in self.poi_id_to_idx[level]:
            return 0.0

        user_idx = self.user_id_to_idx[user_id]
        poi_idx = self.poi_id_to_idx[level][poi_id]

        feature_score = self.compute_feature_based_score(user_idx, poi_idx, level)

        graph_score = 0.0
        if use_graph_context and self.has_interaction_model:
            graph_score = self.compute_graph_based_score(user_idx, poi_idx, level)

        hier_score = 0.0
        if use_hierarchical_boost:
            hier_score = self.compute_hierarchical_boost(poi_id, user_idx, level)

        dist = 1.0
        if use_distance_penalty and level == 0:
            dist = self.compute_distance_penalty(user_id, poi_id, level)

        interest = self.compute_interest_match(user_id, poi_id, level)

        # If interaction model is off, rely on distance + interest (non-zero)
        if not self.has_interaction_model:
            dist = 1.0
            if use_distance_penalty and level == 0:
                dist = self.compute_distance_penalty(user_id, poi_id, level)

            interest = self.compute_interest_match(user_id, poi_id, level)

            # Add a small popularity component if available (optional)
            pop = 0.0
            level_key = f"level_{level}"
            poi_data = self.poi_tree.get(level_key, {}).get(poi_id, {})
            try:
                pop = float(poi_data.get("data", {}).get("popularity", 0))
            except:
                pop = 0.0

            # Normalize popularity roughly into 0-1 range
            pop_score = min(max(pop / 10.0, 0.0), 1.0)

            return float(0.45 * interest * 100 + 0.45 * dist * 100 + 0.10 * pop_score * 100)


        # Otherwise use mixed scoring (your original style)
        final = (
            0.3 * feature_score +
            0.2 * graph_score +
            0.1 * hier_score +
            0.3 * dist * 100 +
            0.1 * interest * 100
        )
        return float(final)

    # ----------------------------
    # Recommend
    # ----------------------------
    def recommend_at_level(
        self,
        user_id: str,
        level: int,
        top_k: int = 10,
        filter_visited: bool = True,
    ) -> List[Tuple[str, float, Dict]]:
        """Recommend at a single granularity level."""
        if user_id not in self.user_id_to_idx:
            return []

        level_key = f"level_{level}"
        all_poi_ids = list(self.poi_tree.get(level_key, {}).keys())
        visited = set(self.user_history.get(user_id, {}).get(level, []))

        scored = []
        for pid in all_poi_ids:
            if filter_visited and pid in visited:
                continue
            s = self.compute_multi_granularity_score(user_id, pid, level)
            scored.append((pid, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        out = []
        for pid, score in top:
            poi_data = self.poi_tree[level_key][pid]
            info = {"name": poi_data.get("name", pid), "type": f"Level {level}"}
            if level == 0:
                info.update({
                    "category": poi_data.get("data", {}).get("category", "N/A"),
                    "price": poi_data.get("data", {}).get("price", "N/A"),
                    "popularity": poi_data.get("data", {}).get("popularity", "N/A"),
                    "region": poi_data.get("data", {}).get("region", "N/A"),
                })
            out.append((pid, float(score), info))
        return out

    def recommend_multi_granularity(self, user_id, levels=[0,1,2,3], top_k_per_level=5, filter_visited=True, prompt: str = None, current_location=None
    ) -> Dict[int, List[Tuple[str, float, Dict]]]:
        """Recommend at multiple levels."""
        results = {}
        for lvl in levels:
            results[lvl] = self.recommend_at_level(
                user_id=user_id,
                level=lvl,
                top_k=top_k_per_level,
                filter_visited=filter_visited,
            )
        return results

    # ----------------------------
    # Utils
    # ----------------------------
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

import pandas as pd
import numpy as np
import json
import pickle
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import random
import warnings
warnings.filterwarnings('ignore')
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_LOCATION = {"latitude": 1.3521, "longitude": 103.8198}

DATASET_CATEGORIES = {
    "atm", "attraction", "bakery", "bank", "bookstore", "cafe", "charging_station", "clinic",
    "clothing_store", "co_working", "convenience_store", "dentist", "department_store",
    "electronics_store", "fast_food", "gym", "hospital", "hostel", "hotel", "jewelry_store",
    "pharmacy", "restaurant", "shopping_mall", "supermarket", "university", "viewpoint", "yoga"
}

INTENT_KEYWORDS = {
    "shopping": ["shopping", "shop", "shops", "mall", "malls", "retail", "department", "boutique"],
    "supermarket": ["supermarket", "grocer", "grocery", "groceries", "fairprice", "giant", "ntuc"],
    "food": ["food", "eat", "eats", "dinner", "lunch", "breakfast", "restaurant", "hawker", "fast food"],
    "cafe": ["cafe", "cafes", "coffee", "brunch", "bakery", "dessert", "tea"],
    "nature": ["nature", "park", "parks", "garden", "beach", "hike", "hiking", "trail"],
    "culture": ["museum", "museums", "gallery", "galleries", "art", "history", "heritage", "exhibition"],
    "view": ["viewpoint", "views", "view", "skyline", "observation", "lookout"],
    "gym": ["gym", "gyms", "fitness", "workout", "exercise", "training", "yoga", "pilates"],
}

INTENT_CATEGORIES = {
    "shopping": {"shopping_mall", "department_store", "clothing_store", "electronics_store", "jewelry_store", "bookstore"},
    "supermarket": {"supermarket", "convenience_store"},
    "food": {"restaurant", "fast_food"},
    "cafe": {"cafe", "bakery"},
    "culture": {"attraction", "viewpoint", "bookstore"},
    "nature": {"attraction", "viewpoint"},
    "view": {"viewpoint"},
    "gym": {"gym", "yoga"}, 
}

# Keywords that indicate exploratory intent ONLY if no category is mentioned
EXPLORATORY_ONLY_KEYWORDS = {
    "things to do", "explore", "exploring", "discover", "suggestions", 
    "what to do", "places to visit", "somewhere", "anywhere", "recommendations",
    "interests", "hobbies", "activities", "hangout", "chill"
}

@dataclass
class POIDetails:
    poi_id: str
    name: str
    category: str
    level: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: str = ""
    popularity: float = 0.0
    price: str = ""

def _safe_lower(x) -> str:
    return str(x).strip().lower() if x is not None else ""

def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", _safe_lower(text)))

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a)) if a <= 1 else 0.0

def _classify_prompt_type(prompt: str) -> str:
    """
    Classify prompt as 'exploratory' or 'category_based'.
    Priority: Category keywords > Exploratory keywords
    """
    if not prompt:
        return "exploratory"
    
    p_lower = _safe_lower(prompt)
    
    # CHECK FOR CATEGORY KEYWORDS FIRST (highest priority)
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in p_lower:
                return "category_based"
    
    # Only check for exploratory keywords if NO category keywords found
    for keyword in EXPLORATORY_ONLY_KEYWORDS:
        if keyword in p_lower:
            return "exploratory"
    
    # Default to exploratory if nothing specific found
    return "exploratory"

def _detect_primary_intent_strict(prompt: str) -> Optional[str]:
    """Detect ONLY the primary intent from prompt."""
    if not prompt:
        return None
    
    text = _safe_lower(prompt)
    toks = _tokenize(text)
    
    scores: Dict[str, int] = defaultdict(int)
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            kw_l = _safe_lower(kw)
            if " " in kw_l:
                if kw_l in text:
                    scores[intent] += 3
            else:
                if kw_l in toks:
                    scores[intent] += 2
                elif kw_l in text:
                    scores[intent] += 1
    
    if not scores:
        return None
    
    return max(scores.items(), key=lambda x: x[1])[0]

def _is_container_intent(prompt: str) -> bool:
    p = _safe_lower(prompt)
    return any(phrase in p for phrase in {"shopping mall", "shopping malls", "mall", "malls"})

def _explain_distance_score(dist_score: float, actual_km: Optional[float] = None) -> str:
    """Convert distance score to human-readable description."""
    if actual_km is not None:
        # Convert km to minutes walking (assuming 5km/h walking speed)
        walk_minutes = (actual_km / 5.0) * 60
        if walk_minutes < 5:
            return f"Very close ({actual_km:.2f}km, ~{int(walk_minutes)} min walk)"
        elif walk_minutes < 15:
            return f"Nearby ({actual_km:.2f}km, ~{int(walk_minutes)} min walk)"
        elif actual_km < 3:
            return f"Short drive away ({actual_km:.2f}km)"
        else:
            return f"{actual_km:.1f}km away"
    
    # Fallback to score-based
    if dist_score > 0.9:
        return "Very close to your location"
    elif dist_score > 0.7:
        return "Nearby"
    elif dist_score > 0.5:
        return "Moderate distance"
    else:
        return "Further away"

def _explain_preference_score(joint_score: float) -> str:
    """Convert joint embedding score to human-readable description."""
    if joint_score > 0.7:
        return "Strongly matches your preferences"
    elif joint_score > 0.5:
        return "Matches your interests"
    elif joint_score > 0.3:
        return "Somewhat aligned with your profile"
    else:
        return "New discovery for you"

def _explain_hierarchical_score(hier_score: float, level: int) -> str:
    """Convert hierarchical score to description."""
    if level == 0:
        if hier_score > 0.6:
            return "Located in an area you frequent"
        elif hier_score > 0.4:
            return "In a familiar neighborhood"
    else:
        if hier_score > 0.6:
            return "Contains popular destinations"
        elif hier_score > 0.4:
            return "Has well-rated spots"
    return ""

def _explain_intent_match(intent_boost: float, category: str, prompt_type: str) -> str:
    """Explain why this category was chosen."""
    if prompt_type == "category_based" and intent_boost > 0:
        return f"Matches your search for '{category}'"
    elif intent_boost > 0:
        return f"Fits your interest in {category}"
    return ""

# ==============================================================================
# MPR PIPELINE CLASS
# ==============================================================================

class MPR_Pipeline:
    def __init__(self,
                 joint_embeddings_file: str = '../Sources/Embeddings v3/joint_optimized_final.pkl',
                 poi_tree_file: str = '../Sources/Files/poi_tree.json',
                 users_file: str = '../Sources/Files/user_preferences.csv',
                 interactions_file: str = '../Sources/Files/user_poi_interactions.csv'):

        print("=" * 70)
        print("INITIALIZING MULTI-GRANULARITY RECOMMENDATION FRAMEWORK (MPR)")
        print("=" * 70)

        self.poi_details_by_id: Dict[str, POIDetails] = {}
        self.user_profiles: Dict[str, Dict] = {}
        self.poi_id_to_idx: Dict[int, Dict[str, int]] = {}
        self.idx_to_poi_id: Dict[int, Dict[int, str]] = {}
        
        print("\nLoading joint optimized embeddings...")
        with open(joint_embeddings_file, 'rb') as f:
            joint_data = pickle.load(f)

        self.U_u = joint_data['U_u']
        self.U_p_levels = joint_data['U_p_levels']
        self.joint_data = joint_data
        
        print(f"  User embeddings: {self.U_u.shape}")
        for i, emb in enumerate(self.U_p_levels):
            print(f"  Level {i} POI embeddings: {emb.shape}")

        print("\nLoading hierarchical POI tree...")
        with open(poi_tree_file, 'r', encoding='utf-8') as f:
            tree_obj = json.load(f)
            self._load_poi_tree(tree_obj)

        print("\nLoading user profiles...")
        self.users_df = pd.read_csv(users_file)
        self.user_id_col = 'uudi' if 'uudi' in self.users_df.columns else 'uuid'
        print(f"  Loaded {len(self.users_df)} users")

        self.interactions_df = pd.read_csv(interactions_file)
        
        self._build_indices()
        self._build_user_history()
        
        self._rng = random.Random(42)
        
        print("\n" + "=" * 70)
        print("INITIALIZATION COMPLETE")
        print("=" * 70)

    def _load_poi_tree(self, tree_obj: dict):
        if not isinstance(tree_obj, dict):
            raise ValueError("POI tree must be a dictionary")
            
        self.poi_tree = tree_obj
        levels = ["level_0", "level_1", "level_2", "level_3"]
        
        for lvl in levels:
            if lvl not in tree_obj:
                continue
                
            level_int = int(lvl.split('_')[1])
            self.poi_id_to_idx[level_int] = {}
            self.idx_to_poi_id[level_int] = {}
            
            for idx, (pid, node) in enumerate(tree_obj[lvl].items()):
                pid = str(pid)
                if not isinstance(node, dict):
                    continue
                
                self.poi_id_to_idx[level_int][pid] = idx
                self.idx_to_poi_id[level_int][idx] = pid
                
                data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
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

    def _build_indices(self):
        self.user_id_to_idx = {
            str(uid): idx for idx, uid in enumerate(self.users_df[self.user_id_col])
        }
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        
        self.poi_id_to_idx = {}
        self.idx_to_poi_id = {}

        for level in [0, 1, 2, 3]:
            level_key = f'level_{level}'
            poi_ids = list(self.poi_tree[level_key].keys())
            self.poi_id_to_idx[level] = {pid: idx for idx, pid in enumerate(poi_ids)}
            self.idx_to_poi_id[level] = {idx: pid for pid, idx in self.poi_id_to_idx[level].items()}

    def _build_user_history(self):
        self.user_history = defaultdict(lambda: defaultdict(list))
        
        visits = self.interactions_df[self.interactions_df['interaction_type'] == 'visit']
        
        for _, row in visits.iterrows():
            user_id = str(row['user_id'])
            poi_id = str(row['poi_id'])
            
            if poi_id not in self.user_history[user_id][0]:
                self.user_history[user_id][0].append(poi_id)
            
            current_id = poi_id
            for level in [1, 2, 3]:
                parent_id = self._get_parent_at_level(current_id, level)
                if parent_id and parent_id not in self.user_history[user_id][level]:
                    self.user_history[user_id][level].append(parent_id)
                current_id = parent_id

    def _get_parent_at_level(self, poi_id: str, target_level: int) -> Optional[str]:
        current_level = 0
        current_id = poi_id
        
        for lvl in [0, 1, 2, 3]:
            if current_id in self.poi_tree.get(f'level_{lvl}', {}):
                current_level = lvl
                break
        
        while current_level < target_level:
            level_key = f'level_{current_level}'
            node = self.poi_tree.get(level_key, {}).get(current_id, {})
            parent = node.get('parent')
            if not parent:
                return None
            current_id = parent
            current_level += 1
            
        return current_id if current_level == target_level else None

    def _get_children_at_level(self, poi_id: str, current_level: int) -> List[str]:
        level_key = f'level_{current_level}'
        node = self.poi_tree.get(level_key, {}).get(poi_id, {})
        return node.get('children', [])

    def _get_leaf_pois(self, poi_id: str, level: int, max_samples: int = 15) -> List[str]:
        if level == 0:
            return [poi_id]
        
        children = self._get_children_at_level(poi_id, level)
        if not children:
            return []
        
        if level == 1:
            if len(children) <= max_samples:
                return children
            return self._rng.sample(children, max_samples)
        
        samples_per_branch = max(1, max_samples // len(children))
        leaves = []
        shuffled_children = list(children)
        self._rng.shuffle(shuffled_children)
        
        for child_id in shuffled_children[:max_samples//2]:
            leaves.extend(self._get_leaf_pois(child_id, level - 1, samples_per_branch))
            if len(leaves) >= max_samples:
                break
        
        return leaves[:max_samples]

    def _get_centroid_from_tree(self, poi_id: str, level: int) -> Tuple[Optional[float], Optional[float]]:
        if level == 0:
            det = self.poi_details_by_id.get(poi_id)
            return (det.latitude, det.longitude) if det else (None, None)
        
        level_key = f"level_{level}"
        node = self.poi_tree.get(level_key, {}).get(poi_id, {})
        children = node.get('children', [])
        
        if not children:
            return None, None
        
        lats, lons = [], []
        child_level = level - 1
        
        for child_id in children[:10]:
            lat, lon = self._get_centroid_from_tree(child_id, child_level)
            if lat and lon:
                lats.append(lat)
                lons.append(lon)
        
        return (np.mean(lats), np.mean(lons)) if lats else (None, None)

    def _compute_distance(self, poi_id: str, level: int, current_location: Dict) -> float:
        """Compute distance from user location to POI. Returns km."""
        if not current_location:
            return float('inf')
        
        user_lat = current_location.get('latitude')
        user_lon = current_location.get('longitude')
        
        if user_lat is None or user_lon is None:
            return float('inf')
        
        poi_lat, poi_lon = None, None
        
        if level == 0:
            det = self.poi_details_by_id.get(poi_id)
            if det:
                poi_lat, poi_lon = det.latitude, det.longitude
        else:
            poi_lat, poi_lon = self._get_centroid_from_tree(poi_id, level)
        
        if poi_lat is None or poi_lon is None:
            return float('inf')
        
        return _haversine_km(user_lat, user_lon, poi_lat, poi_lon)

    def _fallback_distance_based(self, user_id: str, levels: List[int], 
                                   top_k_map: Dict[int, int], intent_data: Optional[Dict],
                                   current_location: Optional[Dict]) -> Dict[str, List[Dict]]:
        """
        FALLBACK MODE: When no interaction data exists or scores are too low.
        Simply returns closest POIs matching the category.
        """
        print("  [Fallback] Using distance-based ranking only...")
        
        results = {}
        primary_intent = intent_data.get('primary_intent') if intent_data else None
        
        for level in levels:
            level_key = f"level_{level}"
            level_top_k = top_k_map.get(level, 0)
            
            if level_top_k <= 0:
                results[level_key] = []
                continue
            
            # Get all POIs at this level
            candidates = list(self.idx_to_poi_id[level].values())
            
            # Score by distance only
            scored_items = []
            for poi_id in candidates:
                dist = self._compute_distance(poi_id, level, current_location)
                
                # Check category match if we have a specific intent
                category_match = 1.0
                if primary_intent and level == 0:
                    det = self.poi_details_by_id.get(poi_id)
                    if det:
                        allowed = INTENT_CATEGORIES.get(primary_intent, set())
                        if det.category.lower() not in allowed:
                            continue  # Skip if not matching category
                        category_match = 1.5  # Boost matching categories
                
                # Score: closer is better, category match boosts
                if dist < float('inf'):
                    # Convert distance to score (closer = higher score)
                    dist_score = max(0, 1.0 - (dist / 10.0))  # Normalize to 0-1, 10km max
                    final_score = dist_score * category_match
                    
                    scored_items.append((poi_id, final_score, dist))
            
            # Sort by score descending, then by distance ascending
            scored_items.sort(key=lambda x: (-x[1], x[2]))
            top_items = scored_items[:level_top_k]
            
            # Build results
            level_results = []
            for poi_id, score, dist in top_items:
                det = self.poi_details_by_id.get(poi_id, POIDetails(poi_id, poi_id, "", f"level_{level}"))
                level_key_node = f'level_{level}'
                node = self.poi_tree.get(level_key_node, {}).get(poi_id, {})
                
                info = {
                    'name': det.name,
                    'category': det.category,
                    'score_components': {
                        'final': round(score, 4),
                        'joint_embedding': 0.0,
                        'hierarchical': 0.0,
                        'spatial': round(max(0, 1.0 - (dist / 10.0)), 4),
                        'interest': 0.5,
                        'intent_match': 0.3 if primary_intent else None,
                        'distance_km': round(dist, 2)
                    }
                }
                
                if level == 0:
                    info['type'] = 'Individual POI'
                    info['price'] = det.price
                    info['popularity'] = det.popularity
                elif level == 1:
                    info['type'] = 'Container/Venue'
                    info['num_pois'] = len(node.get('children', []))
                elif level == 2:
                    info['type'] = 'District'
                    info['num_venues'] = len(node.get('children', []))
                else:
                    info['type'] = 'Region'
                    info['num_districts'] = len(node.get('children', []))
                
                level_results.append({
                    "poi_id": poi_id,
                    "name": info['name'],
                    "score": float(score),
                    "type": info['type'],
                    "details": info
                })
            
            results[level_key] = level_results
            print(f"  [Level {level}] Fallback results: {len(level_results)} (nearest matches)")
        
        return results

    def compute_joint_score(self, user_idx: int, poi_idx: int, level: int) -> float:
        if level >= len(self.U_p_levels) or user_idx >= self.U_u.shape[0] or poi_idx >= self.U_p_levels[level].shape[0]:
            return 0.0

        u_vec = self.U_u[user_idx]
        p_vec = self.U_p_levels[level][poi_idx]
        
        dot_product = np.dot(u_vec, p_vec)
        norm_u = np.linalg.norm(u_vec)
        norm_p = np.linalg.norm(p_vec)
        
        if norm_u == 0 or norm_p == 0:
            return 0.0
            
        return float(dot_product / (norm_u * norm_p))

    def compute_hierarchical_boost(self, poi_id: str, user_idx: int, level: int) -> float:
        if level < 2:
            level_key = f'level_{level}'
            node = self.poi_tree.get(level_key, {}).get(poi_id, {})
            parent_id = node.get('parent')
            
            if not parent_id or (level + 1) >= len(self.U_p_levels):
                return 0.5
                
            parent_level = level + 1
            if parent_id in self.poi_id_to_idx.get(parent_level, {}):
                parent_idx = self.poi_id_to_idx[parent_level][parent_id]
                return self.compute_joint_score(user_idx, parent_idx, parent_level)
            return 0.5
            
        else:
            level_key = f'level_{level}'
            node = self.poi_tree.get(level_key, {}).get(poi_id, {})
            children = node.get('children', [])
            
            if not children:
                return 0.5
                
            child_scores = []
            child_level = level - 1
            
            for child_id in children[:20]:
                if child_id in self.poi_id_to_idx.get(child_level, {}):
                    child_idx = self.poi_id_to_idx[child_level][child_id]
                    score = self.compute_joint_score(user_idx, child_idx, child_level)
                    child_scores.append(score)
            
            return np.mean(child_scores) if child_scores else 0.5

    def compute_distance_penalty(self, user_id: str, poi_id: str, level: int,
                                current_location: Optional[Dict] = None) -> float:
        """Returns decay factor 0-1 based on distance"""
        user_lat, user_lon = None, None
        
        if current_location:
            user_lat = current_location.get('latitude')
            user_lon = current_location.get('longitude')
        else:
            try:
                user_row = self.users_df[self.users_df[self.user_id_col] == user_id].iloc[0]
                area = user_row.get('area_of_residence', '')
                area_coords = {
                    'Jurong East': (1.3329, 103.7436), 'Yishun': (1.4304, 103.8354),
                    'Bishan': (1.3526, 103.8352), 'Tampines': (1.3496, 103.9568),
                    'Woodlands': (1.4382, 103.7891), 'Ang Mo Kio': (1.3691, 103.8454),
                    'Bedok': (1.3236, 103.9273), 'Clementi': (1.3162, 103.7649),
                    'Hougang': (1.3612, 103.8864), 'Punggol': (1.4054, 103.9021),
                    'Sengkang': (1.3868, 103.8914), 'Bukit Batok': (1.3590, 103.7637),
                    'Bukit Panjang': (1.3774, 103.7718), 'Choa Chu Kang': (1.3840, 103.7470),
                    'Pasir Ris': (1.3721, 103.9474), 'Sembawang': (1.4491, 103.8185),
                    'Serangoon': (1.3554, 103.8679), 'Toa Payoh': (1.3343, 103.8564),
                }
                if area in area_coords:
                    user_lat, user_lon = area_coords[area]
            except:
                pass
        
        if user_lat is None or user_lon is None:
            return 0.5
        
        poi_lat, poi_lon = None, None
        
        if level == 0:
            det = self.poi_details_by_id.get(poi_id)
            if det:
                poi_lat, poi_lon = det.latitude, det.longitude
        else:
            poi_lat, poi_lon = self._get_centroid_from_tree(poi_id, level)
        
        if poi_lat is None or poi_lon is None:
            return 0.5
        
        distance = _haversine_km(user_lat, user_lon, poi_lat, poi_lon)
        
        try:
            user_row = self.users_df[self.users_df[self.user_id_col] == user_id].iloc[0]
            transport = str(user_row.get('transportation_modes', '')).lower()
        except:
            transport = ''
        
        if 'car' in transport or 'ride' in transport:
            max_dist = 15.0
        elif 'mrt' in transport or 'bus' in transport:
            max_dist = 8.0
        else:
            max_dist = 3.0
            
        return float(np.exp(-distance / max_dist))

    def compute_interest_match(self, user_id: str, poi_id: str, level: int = 0) -> float:
        if level != 0:
            return 0.5
            
        try:
            user_row = self.users_df[self.users_df[self.user_id_col] == user_id].iloc[0]
            interests_raw = str(user_row['interests'])
            interests = set([i.strip().lower() for i in interests_raw.replace(',', ';').split(';') if i.strip()])
        except:
            return 0.5
        
        det = self.poi_details_by_id.get(poi_id)
        if not det:
            return 0.5
            
        score = 0.0
        cat = det.category.lower()

        interest_mapping = {
            # Food & Dining
            'food': {
                'primary': ['restaurant', 'fast_food', 'cafe', 'food_court', 'bakery'],
                'secondary': ['convenience_store', 'supermarket']  # Can buy food
            },
            'coffee': {
                'primary': ['cafe'],
                'secondary': ['restaurant']  # Some restaurants serve coffee
            },
            'healthy eating': {
                'primary': ['restaurant', 'cafe', 'food_court'],
                'secondary': ['supermarket', 'bakery']
            },
            'cooking': {
                'primary': ['supermarket'],
                'secondary': ['convenience_store', 'bakery']
            },
            
            # Shopping & Retail
            'shopping': {
                'primary': ['shopping_mall', 'department_store', 'clothing_store', 
                        'electronics_store', 'jewelry_store'],
                'secondary': ['bookstore', 'convenience_store', 'supermarket', 'bakery']
            },
            'books': {
                'primary': ['bookstore'],
                'secondary': ['shopping_mall']  # Bookstores in malls
            },
            'electronics': {
                'primary': ['electronics_store'],
                'secondary': ['shopping_mall', 'department_store']
            },
            'fashion': {
                'primary': ['clothing_store', 'jewelry_store'],
                'secondary': ['shopping_mall', 'department_store']
            },
            
            # Entertainment & Culture
            'movies': {
                'primary': ['attraction'],  # If attraction includes cinemas
                'secondary': ['shopping_mall']  # Cinemas often in malls
            },
            'culture': {
                'primary': ['attraction', 'viewpoint', 'bookstore', 'university'],
                'secondary': ['museum']  # If you add museums later
            },
            'nature': {
                'primary': ['attraction', 'viewpoint', 'park'],
                'secondary': ['viewpoint']
            },
            'hiking': {
                'primary': ['attraction', 'viewpoint'],
                'secondary': ['park']
            },
            'sightseeing': {
                'primary': ['attraction', 'viewpoint'],
                'secondary': ['hotel', 'hostel']  # Tourists stay here
            },
            
            # Health & Wellness
            'sports': {
                'primary': ['gym', 'yoga', 'attraction'],
                'secondary': ['park', 'viewpoint']
            },
            'gyms': {
                'primary': ['gym', 'yoga'],
                'secondary': ['attraction']  # Outdoor fitness areas
            },
            'fitness': {
                'primary': ['gym', 'yoga'],
                'secondary': ['park', 'attraction']
            },
            'wellness': {
                'primary': ['yoga', 'gym', 'clinic'],
                'secondary': ['pharmacy', 'hospital']
            },
            'health': {
                'primary': ['clinic', 'hospital', 'pharmacy', 'dentist'],
                'secondary': ['gym', 'yoga', 'supermarket']  # Health food/supplements
            },
            
            # Work & Business
            'work': {
                'primary': ['co_working'],
                'secondary': ['cafe', 'restaurant']  # Working from cafes
            },
            'business': {
                'primary': ['co_working', 'bank'],
                'secondary': ['cafe', 'restaurant', 'hotel']
            },
            'finance': {
                'primary': ['bank', 'atm'],
                'secondary': ['shopping_mall']  # Banks in malls
            },
            
            # Education
            'education': {
                'primary': ['university', 'bookstore'],
                'secondary': ['attraction', 'museum']
            },
            'university': {
                'primary': ['university'],
                'secondary': ['bookstore', 'cafe', 'restaurant']
            },
            
            # Services
            'medical': {
                'primary': ['hospital', 'clinic', 'pharmacy', 'dentist'],
                'secondary': ['supermarket', 'convenience_store']  # Pharmacies inside
            },
            'pharmacy': {
                'primary': ['pharmacy'],
                'secondary': ['hospital', 'clinic', 'supermarket']
            },
            
            # Travel & Accommodation
            'travel': {
                'primary': ['hotel', 'hostel', 'attraction', 'viewpoint'],
                'secondary': ['restaurant', 'cafe', 'shopping_mall']
            },
            'hotels': {
                'primary': ['hotel', 'hostel'],
                'secondary': ['restaurant', 'cafe']
            },
            'accommodation': {
                'primary': ['hotel', 'hostel'],
                'secondary': ['university']  # University housing
            },
            
            # Convenience
            'convenience': {
                'primary': ['convenience_store'],
                'secondary': ['supermarket', 'pharmacy']
            },
            'charging': {
                'primary': ['charging_station'],
                'secondary': ['shopping_mall', 'parking']  # If you have parking category
            },
            
            # Social
            'social': {
                'primary': ['cafe', 'restaurant', 'bar'],  # Add bar if you have it
                'secondary': ['shopping_mall', 'co_working']
            },
            'meeting': {
                'primary': ['co_working', 'cafe', 'restaurant'],
                'secondary': ['shopping_mall']
            }
        }

        for interest, mapping in interest_mapping.items():
            if interest in interests:
                if cat in mapping.get('primary', []):
                    score += 0.4 
                elif cat in mapping.get('secondary', []):
                    score += 0.2  
            
        return min(score, 1.0)

    def _compute_intent_match(self, poi_id: str, level: int, intent_data: Optional[Dict], 
                            strict_mode: bool = False) -> Tuple[bool, float]:
        """
        Determine if a POI matches the user intent.
        strict_mode: If True (Category-based), enforce strict primary-only filtering
        """
        if not intent_data:
            return True, 0.0
        
        prompt_type = intent_data.get('prompt_type', 'exploratory')
        primary_intent = intent_data.get('primary_intent')
        
        # EXPLORATORY MODE: No strict filtering
        if prompt_type == 'exploratory' or not strict_mode:
            return True, 0.0
        
        # CATEGORY-BASED MODE: Strict filtering by primary intent only
        if not primary_intent:
            return True, 0.0
        
        # LEVEL 0: Strict category matching
        if level == 0:
            det = self.poi_details_by_id.get(poi_id)
            if not det:
                return False, 0.0
            
            cat = det.category.lower()
            allowed = INTENT_CATEGORIES.get(primary_intent, set())
            
            if cat in allowed:
                return True, 0.30
            
            # Special case: shopping malls
            if primary_intent == 'shopping' and intent_data.get('is_mall_query'):
                parent_id = self._get_parent_at_level(poi_id, 1)
                if parent_id:
                    parent_det = self.poi_details_by_id.get(parent_id)
                    if parent_det and parent_det.category == 'shopping_mall':
                        return True, 0.15
            
            return False, 0.0
        
        # LEVEL 1+: Check containment
        elif level == 1:
            children = self._get_leaf_pois(poi_id, level, max_samples=10)
            if not children:
                return True, 0.0
            
            allowed = INTENT_CATEGORIES.get(primary_intent, set())
            matching = sum(1 for child_id in children 
                        if self.poi_details_by_id.get(child_id, POIDetails("", "", "")).category.lower() in allowed)
            
            ratio = matching / len(children) if children else 0
            if ratio >= 0.3:
                return True, 0.20 * ratio
            return False, 0.0
        
        else:
            sample_pois = self._get_leaf_pois(poi_id, level, max_samples=15)
            if not sample_pois:
                return True, 0.0
            
            allowed = INTENT_CATEGORIES.get(primary_intent, set())
            matching = sum(1 for child_id in sample_pois 
                        if self.poi_details_by_id.get(child_id, POIDetails("", "", "")).category.lower() in allowed)
            
            ratio = matching / len(sample_pois)
            if ratio > 0:
                return True, 0.10 + (0.15 * ratio)
            return False, 0.0

    def recommend_at_level(self, 
                    user_id: str, 
                    level: int, 
                    top_k: int = 10,
                    filter_visited: bool = True,
                    current_location: Optional[Dict] = None,
                    intent: Optional[Dict] = None,
                    strict_mode: bool = False) -> List[Tuple[str, float, Dict]]:
        """
        Generate top-k recommendations.
        """
        if user_id not in self.user_id_to_idx:
            return []
                
        user_idx = self.user_id_to_idx[user_id]
        
        if level not in self.idx_to_poi_id:
            return []
                
        candidates = list(self.idx_to_poi_id[level].values())
        visited = set(self.user_history.get(user_id, {}).get(level, []))
        
        prompt_type = intent.get('prompt_type', 'category_based') if intent else 'category_based'
        
        scored_items = []
        
        for poi_id in candidates:
            if filter_visited and poi_id in visited:
                continue

            should_include, intent_boost = self._compute_intent_match(
                poi_id, level, intent, strict_mode=strict_mode
            )
            
            if not should_include:
                continue
                
            poi_idx = self.poi_id_to_idx[level][poi_id]
            
            joint_score = self.compute_joint_score(user_idx, poi_idx, level)
            hier_score = self.compute_hierarchical_boost(poi_id, user_idx, level)
            dist_score = self.compute_distance_penalty(user_id, poi_id, level, current_location)
            interest_score = self.compute_interest_match(user_id, poi_id, level)
            
            # Now prompt_type is properly defined
            if prompt_type == "exploratory":
                # For "things to do near me": boost interest significantly
                final_score = (0.35 * joint_score + 
                            0.20 * hier_score + 
                            0.25 * dist_score + 
                            0.35 * interest_score +  # High weight on interests
                            0.00 * intent_boost)     # No prompt intent boost for exploratory
            else:
                # For category-based queries like "food near me"
                final_score = (0.30 * joint_score + 
                            0.20 * hier_score + 
                            0.30 * dist_score + 
                            0.20 * interest_score +
                            0.10 * intent_boost)
            
            if final_score > 0:
                components = {
                    'joint': joint_score,
                    'hier': hier_score,
                    'dist': dist_score,
                    'interest': interest_score,
                    'intent_boost': intent_boost
                }
                scored_items.append((poi_id, final_score, components))
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:top_k]
        
        # Build detailed results
        results = []
        for item in top_items:
            poi_id, score, comp = item
            det = self.poi_details_by_id.get(poi_id, POIDetails(poi_id, poi_id, "", f"level_{level}"))
            level_key = f'level_{level}'
            node = self.poi_tree.get(level_key, {}).get(poi_id, {})
            
            info = {
                'name': det.name if det else poi_id,
                'category': det.category if det else "",
                'score_components': {
                    'final': round(score, 4),
                    'joint_embedding': round(comp['joint'], 4),
                    'hierarchical': round(comp['hier'], 4),
                    'spatial': round(comp['dist'], 4),
                    'interest': round(comp['interest'], 4),
                    'intent_match': round(comp['intent_boost'], 4) if comp['intent_boost'] > 0 else None
                }
            }
            
            if level == 0:
                info['type'] = 'Individual POI'
                info['price'] = det.price if det else ""
                info['popularity'] = det.popularity if det else 0.0
            elif level == 1:
                info['type'] = 'Container/Venue'
                info['num_pois'] = len(node.get('children', [])) if isinstance(node, dict) else 0
            elif level == 2:
                info['type'] = 'District'
                info['num_venues'] = len(node.get('children', [])) if isinstance(node, dict) else 0
            else:
                info['type'] = 'Region'
                info['num_districts'] = len(node.get('children', [])) if isinstance(node, dict) else 0
                
            results.append((poi_id, score, info))
            
        return results
        
    def recommend_multi_granularity(self,
                            user_id: str,
                            levels: List[int] = [0, 1, 2, 3],
                            top_k = 5,
                            prompt: str = "",
                            current_location: Optional[Dict] = None,
                            filter_visited: bool = True) -> Dict[str, List[Dict]]:
        """
        Main MPR Method with fallback for cold start.
        """
        self.last_prompt_type = _classify_prompt_type(prompt)
        self.last_intent = _detect_primary_intent_strict(prompt) if self.last_prompt_type == 'category_based' else None

        if user_id not in self.user_id_to_idx:
            raise ValueError(f"Unknown user_id: {user_id}")
        
        # Handle top_k parameter
        if isinstance(top_k, int):
            top_k_map = {level: top_k for level in levels}
            user_provided_dict = False
        else:
            top_k_map = dict(top_k)
            user_provided_dict = True
            for level in levels:
                if level not in top_k_map:
                    top_k_map[level] = 0 
        
        # CLASSIFY PROMPT TYPE
        prompt_type = _classify_prompt_type(prompt)
        print(f"\n[Prompt Analysis] Type: {prompt_type.upper()} | Prompt: '{prompt}'")
        
        parsed_intent = None
        
        if prompt_type == 'category_based':
            primary = _detect_primary_intent_strict(prompt)
            
            if primary:
                parsed_intent = {
                    'prompt_type': 'category_based',
                    'primary_intent': primary,
                    'intents': [primary],
                    'raw_prompt': prompt,
                    'is_mall_query': _is_container_intent(prompt)
                }
                print(f"  [Strict Intent] Primary: {primary}")
            else:
                # No intent detected, treat as exploratory
                prompt_type = 'exploratory'
                parsed_intent = {
                    'prompt_type': 'exploratory',
                    'primary_intent': None,
                    'intents': [],
                    'raw_prompt': prompt
                }
                print(f"  [No strict intent detected] Switching to exploratory mode")
                    
        else:  # EXPLORATORY
            parsed_intent = {
                'prompt_type': 'exploratory',
                'primary_intent': None,
                'intents': [],
                'raw_prompt': prompt
            }
            print(f"  [Exploratory Mode] Diversity-focused")
            
            if not user_provided_dict:
                if 2 in top_k_map and top_k_map[2] == 0:
                    top_k_map[2] = 2
                if 3 in top_k_map and top_k_map[3] == 0:
                    top_k_map[3] = 1
        
        results = {}
        
        # Try normal MPR approach first
        for level in levels:
            level_key = f"level_{level}"
            level_top_k = top_k_map.get(level, 0)
            
            if level_top_k <= 0:
                results[level_key] = []
                continue
            
            strict_mode = (prompt_type == 'category_based')
            
            try:
                recs = self.recommend_at_level(
                    user_id=user_id,
                    level=level,
                    top_k=level_top_k,
                    filter_visited=filter_visited,
                    current_location=current_location,
                    intent=parsed_intent,
                    strict_mode=strict_mode
                )
                
                # Ensure recs is valid and truncate
                if not recs:
                    recs = []
                recs = recs[:level_top_k]
                
            except Exception as e:
                print(f"  [Level {level}] Error in recommendation: {e}")
                recs = []
            
            # Build level_results safely
            level_results = []
            for rec in recs:
                if len(rec) != 3:
                    continue
                poi_id, score, info = rec
                if not isinstance(info, dict):
                    continue
                
                # Safe dict access with defaults
                item = {
                    "poi_id": poi_id,
                    "name": info.get('name', poi_id),  # Fallback to poi_id if name missing
                    "score": float(score) if score else 0.0,
                    "type": info.get('type', 'Unknown'),
                    "details": info
                }
                level_results.append(item)
            
            results[level_key] = level_results
            
            # Logging
            if prompt_type == 'category_based':
                # FIX: Handle None values properly using 'or 0' 
                matches = sum(1 for r in level_results 
                            if (r['details'].get('score_components', {}).get('intent_match') or 0) > 0)
                print(f"  [Level {level}] Results: {len(level_results)}/{level_top_k} (matches: {matches})")
            else:
                print(f"  [Level {level}] Results: {len(level_results)}/{level_top_k}")
        
        # Check if we need fallback (cold start detection)
        total_results = sum(len(v) for v in results.values())
        avg_score = 0
        
        if total_results > 0:
            all_scores = [r['score'] for level in results.values() for r in level if 'score' in r]
            avg_score = np.mean(all_scores) if all_scores else 0
        
        # Trigger fallback conditions
        category_mismatch = False
        if prompt_type == 'category_based' and total_results > 0:
            intent_matches = sum(1 for level in results.values() for r in level 
                            if (((r.get('details') or {}).get('score_components') or {}).get('intent_match') or 0) > 0.2)
            if intent_matches == 0:
                category_mismatch = True
        
        # Fallback to distance-based if needed
        if total_results == 0 or avg_score < 0.15 or category_mismatch:
            print(f"\n[FALLBACK TRIGGERED] Low scores or no matches. Using distance-based fallback...")
            results = self._fallback_distance_based(
                user_id, levels, top_k_map, parsed_intent, current_location
            )
        
        return results

    def display_multi_granularity_recommendations(self, 
                                                recommendations: Dict[str, List[Dict]],
                                                show_explanations: bool = True):
        """
        Display recommendations with human-readable explanations.
        """
        level_names = {
            "level_0": "LEVEL 0: INDIVIDUAL POIs (Specific Places)",
            "level_1": "LEVEL 1: CONTAINERS/VENUES (Buildings/Streets)",
            "level_2": "LEVEL 2: DISTRICTS (Neighborhoods)",
            "level_3": "LEVEL 3: REGIONS (Planning Areas)"
        }
        
        print("\n" + "="*80)
        print("🎯 MULTI-GRANULARITY RECOMMENDATIONS")
        if hasattr(self, 'last_prompt_type') and self.last_prompt_type:
            print(f"   Query Type: {self.last_prompt_type.replace('_', ' ').title()}")
        print("="*80)
        
        for level_key in ["level_0", "level_1", "level_2", "level_3"]:
            if level_key not in recommendations:
                continue
                
            recs = recommendations[level_key]
            level_num = int(level_key.split("_")[1])
            
            print(f"\n{'─'*80}")
            print(f"📍 {level_names.get(level_key, level_key)}")
            print(f"{'─'*80}")
            
            if not recs:
                print("   No recommendations at this level")
                continue
            
            for rank, item in enumerate(recs, 1):
                name = item.get('name', 'Unknown')
                score = item.get('score', 0.0)
                details = item.get('details', {})
                category = details.get('category', 'N/A')
                
                # Main line with emoji based on level
                emoji = {0: "📌", 1: "🏢", 2: "🗺️", 3: "🌍"}.get(level_num, "•")
                print(f"\n{emoji} {rank}. {name}")
                
                # Category tag
                if category != 'N/A':
                    print(f"    Category: {category}")
                
                # Compact score display
                comp = details.get('score_components', {})
                brief_scores = []
                
                if comp.get('distance_km') is not None:
                    brief_scores.append(f"{comp['distance_km']:.2f}km away")
                elif comp.get('spatial', 0) > 0.9:
                    brief_scores.append("Very close")
                
                if comp.get('intent_match') and comp['intent_match'] > 0:
                    brief_scores.append("Category match ✓")
                
                if brief_scores:
                    print(f"    ({' | '.join(brief_scores)})")
                
                # Detailed explanation if requested
                if show_explanations:
                    explanation = self.generate_explanation_text(item, "", level_num)
                    # Indent the explanation
                    for line in explanation.split('\n'):
                        print(f"    {line}")
                
                # Level-specific details
                if level_key == "level_0":
                    if details.get('popularity'):
                        stars = "⭐" * int(min(details['popularity'], 5))
                        print(f"    Rating: {stars}")
                elif level_key == "level_1":
                    print(f"    Contains {details.get('num_pois', 0)} places inside")
                elif level_key == "level_2":
                    print(f"    Contains {details.get('num_venues', 0)} venues")
                    if details.get('region'):
                        print(f"    Part of: {details['region']}")
        
        print("\n" + "="*80)
        print("💡 Tip: These recommendations combine your preferences, location, and search intent")
        print("="*80)

    def explain_recommendation_simple(self, user_id: str, poi_id: str, level: int) -> str:
        """
        Quick one-sentence explanation for a specific recommendation.
        """
        if user_id not in self.user_id_to_idx or level not in self.poi_id_to_idx:
            return "Unable to explain this recommendation."
        
        user_idx = self.user_id_to_idx[user_id]
        poi_idx = self.poi_id_to_idx[level][poi_id]
        
        det = self.poi_details_by_id.get(poi_id, POIDetails(poi_id, poi_id, "", ""))
        
        # Calculate components
        joint = self.compute_joint_score(user_idx, poi_idx, level)
        dist = self.compute_distance_penalty(user_id, poi_id, level)
        
        parts = []
        
        # Build sentence
        if joint > 0.6:
            parts.append(f"{det.name} strongly matches your preferences")
        elif joint > 0.3:
            parts.append(f"{det.name} aligns with your interests")
        else:
            parts.append(f"{det.name} is a new discovery")
        
        if dist > 0.8:
            parts.append("and is very close to your location")
        elif dist > 0.5:
            parts.append("and is nearby")
        
        if self.last_intent and det.category.lower() in INTENT_CATEGORIES.get(self.last_intent, set()):
            parts.append(f", matching your search for {self.last_intent}")
        
        return " ".join(parts) + "."

    def folium_mapper(self, recommendations: Dict[str, List[Dict]], 
                    user_location: Optional[Dict] = None,
                    save_path: Optional[str] = None):
        center = user_location or DEFAULT_LOCATION
        min_lat, max_lat = 1.15, 1.47
        min_lon, max_lon = 103.60, 104.10

        m = folium.Map(
            location=[center["latitude"], center["longitude"]],
            zoom_start=12,
            tiles="cartodbpositron",
            max_bounds=True,
            min_lat=min_lat, 
            max_lat=max_lat,
            min_lon=min_lon, 
            max_lon=max_lon
        )
        
        if user_location:
            folium.CircleMarker(
                location=[user_location["latitude"], user_location["longitude"]],
                radius=8,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup="Your Location"
            ).add_to(m)
        else:
            folium.Marker(
                location=[DEFAULT_LOCATION["latitude"], DEFAULT_LOCATION["longitude"]],
                popup="Default (Singapore)",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        level_colors = {
            "level_0": {"color": "#2E86AB", "fill": "#A7C6DA", "radius": 6},
            "level_1": {"color": "#A23B72", "fill": "#F18FBB", "radius": 12},
            "level_2": {"color": "#F18F01", "fill": "#F9D78C", "radius": 20},
            "level_3": {"color": "#C73E1D", "fill": "#F4A688", "radius": 35}
        }
        
        level_names = {
            "level_0": "Individual POI",
            "level_1": "Container/Venue", 
            "level_2": "District",
            "level_3": "Region"
        }
        
        all_coords = []
        
        for level_key, recs in recommendations.items():
            if not recs:
                continue
                
            level_num = int(level_key.split("_")[1])
            style = level_colors.get(level_key, level_colors["level_0"])
            
            if level_key == "level_0":
                marker_cluster = MarkerCluster(name=f"Level {level_num}").add_to(m)
            
            for item in recs:
                details = item.get('details', {})
                lat = details.get('latitude')
                lon = details.get('longitude')
                
                if lat is None or lon is None:
                    lat, lon = self._get_centroid_from_tree(item['poi_id'], level_num)
                    if lat is None:
                        continue
                
                all_coords.append((float(lat), float(lon)))
                
                score = item.get('score', 0)
                name = item.get('name', 'Unknown')
                
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; width: 200px;">
                    <h4 style="margin: 0; color: {style['color']};">{name}</h4>
                    <p style="margin: 5px 0; font-size: 12px;">
                        <b>Type:</b> {level_names.get(level_key, 'Unknown')}<br>
                        <b>Score:</b> {score:.3f}<br>
                        <b>ID:</b> {item['poi_id'][:20]}...
                    </p>
                """
                
                if level_key == "level_0":
                    popup_html += f"""
                        <b>Category:</b> {details.get('category', 'N/A')}<br>
                        <b>Popularity:</b> {details.get('popularity', 'N/A')}<br>
                    """
                    if 'score_components' in details:
                        comp = details['score_components']
                        popup_html += f"""
                        <hr style="margin: 5px 0;">
                        <small>
                        Preference: {comp.get('joint_embedding', 0):.2f}<br>
                        Spatial: {comp.get('spatial', 0):.2f}
                        </small>
                        """
                elif level_key == "level_1":
                    popup_html += f"<b>Contains:</b> {details.get('num_pois', 0)} POIs<br>"
                elif level_key == "level_2":
                    popup_html += f"<b>Contains:</b> {details.get('num_venues', 0)} venues<br>"
                
                popup_html += "</div>"
                
                if level_key == "level_0":
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=style['radius'],
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=name,
                        color=style['color'],
                        fill=True,
                        fill_color=style['fill'],
                        fill_opacity=0.8,
                        weight=2
                    ).add_to(marker_cluster)
                    
                else:
                    folium.Circle(
                        location=[lat, lon],
                        radius=style['radius'] * 10,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{name} ({level_names[level_key]})",
                        color=style['color'],
                        fill=True,
                        fill_color=style['fill'],
                        fill_opacity=0.4,
                        weight=3,
                        dash_array='5, 5' if level_key == "level_3" else None
                    ).add_to(m)
        
        if all_coords:
            m.fit_bounds(
                [[min(c[0] for c in all_coords), min(c[1] for c in all_coords)],
                [max(c[0] for c in all_coords), max(c[1] for c in all_coords)]],
                padding=(50, 50)
            )
        
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
            print(f"Map saved to {save_path}")
        
        print(f"Plotted {len(all_coords)} POIs across {len([r for r in recommendations.values() if r])} levels")
        return m, all_coords

    def add_user_profile(self, profile: Dict[str, str]) -> Dict[str, str]:
        user_id = str(profile.get("user_id", "")).strip()
        if not user_id:
            return {"status": "error", "message": "missing user_id"}
            
        if user_id in self.user_id_to_idx:
            return {"status": "exists", "user_id": user_id}
            
        if self.U_u.shape[0] > 0:
            mean_vec = np.mean(self.U_u, axis=0, keepdims=True)
            noise = np.random.normal(0, 0.01, size=mean_vec.shape)
            new_vec = mean_vec + noise
        else:
            new_vec = np.random.normal(0, 0.1, size=(1, 64))
            
        new_vec = new_vec / (np.linalg.norm(new_vec, axis=1, keepdims=True) + 1e-12)
        
        self.U_u = np.vstack([self.U_u, new_vec])
        self.user_embeddings_mat = self.U_u
        
        new_idx = int(self.U_u.shape[0] - 1)
        self.user_id_to_idx[user_id] = new_idx
        self.idx_to_user_id[new_idx] = user_id
        self.user_profiles[user_id] = profile
        
        new_row = pd.DataFrame([profile])
        self.users_df = pd.concat([self.users_df, new_row], ignore_index=True)
        
        return {"status": "created", "user_id": user_id, "user_index": new_idx}
    
    def generate_explanation_text(self, item: Dict, user_id: str, level: int) -> str:
        """
        Generate a human-readable explanation for why this POI was recommended.
        """
        details = item.get('details', {})
        comp = details.get('score_components', {})
        name = item.get('name', 'Unknown')
        category = details.get('category', 'unknown')
        
        explanations = []
        
        # Title line
        explanations.append(f"📍 Why '{name}'?")
        
        # Distance explanation (most important for nearby searches)
        dist_km = comp.get('distance_km')
        spatial_score = comp.get('spatial', 0)
        
        if dist_km is not None:
            dist_desc = _explain_distance_score(spatial_score, dist_km)
            explanations.append(f"   • {dist_desc}")
        elif spatial_score > 0:
            dist_desc = _explain_distance_score(spatial_score)
            explanations.append(f"   • {dist_desc}")
        
        # Category/Intent match
        intent_match = comp.get('intent_match')
        if intent_match and intent_match > 0:
            if self.last_prompt_type == 'category_based' and self.last_intent:
                explanations.append(f"   • Matches your search for '{self.last_intent}' ({category})")
            else:
                explanations.append(f"   • Category: {category} (matches your interests)")
        
        # Preference match (if meaningful)
        joint_score = comp.get('joint_embedding', 0)
        if joint_score > 0.3:
            pref_desc = _explain_preference_score(joint_score)
            explanations.append(f"   • {pref_desc}")
        
        # Hierarchical context
        hier_score = comp.get('hierarchical', 0)
        if level > 0 and hier_score > 0.5:
            hier_desc = _explain_hierarchical_score(hier_score, level)
            if hier_desc:
                explanations.append(f"   • {hier_desc}")
        
        # Fallback indicator
        if comp.get('joint_embedding', 0) == 0 and comp.get('hierarchical', 0) == 0:
            explanations.append("   • Based on proximity to your location")
        
        # Popularity for Level 0
        if level == 0 and details.get('popularity'):
            pop = details['popularity']
            if pop >= 4:
                explanations.append(f"   • Highly rated/popular spot")
        
        return "\n".join(explanations)
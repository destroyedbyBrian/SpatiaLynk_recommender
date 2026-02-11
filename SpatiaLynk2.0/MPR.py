import pandas as pd
import numpy as np
import json
import pickle
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import random
import warnings
warnings.filterwarnings('ignore')
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

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
	if not prompt:
		return "exploratory"
	
	p_lower = _safe_lower(prompt)
	
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

class MPR_Pipeline:
	def __init__(self,
				joint_embeddings_file: str = '../Sources/Embeddings v3/joint_optimized_final.pkl',
				poi_tree_file: str = '../Sources/Files/poi_tree_with_uuids.json',
				users_file: str = '../Sources/Files/user_preferences.csv',
				interactions_file: str = '../Sources/Files/user_poi_interactions.csv'):

		print("=" * 70)
		print("INITIALIZING MULTI-GRANULARITY RECOMMENDATION FRAMEWORK (MPR)")
		print("=" * 70)

		self.poi_details_by_id: Dict[str, POIDetails] = {}
		self.user_profiles: Dict[str, Dict] = {}
		self.poi_id_to_idx: Dict[int, Dict[str, int]] = {}
		self.idx_to_poi_id: Dict[int, Dict[int, str]] = {}
		self._rng = random.Random(42)
		
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
		
		self._precompute_category_vectors()
		print("\n" + "=" * 70)
		print("INITIALIZATION COMPLETE")
		print("=" * 70)

	def _precompute_category_vectors(self):
		"""Calculates the average embedding vector for each category at Level 0."""
		print("Pre-computing category vectors for cold start...")
		self.category_vectors = defaultdict(list)
		
		# Loop through all Level 0 POIs
		for poi_id, idx in self.poi_id_to_idx.get(0, {}).items():
			det = self.poi_details_by_id.get(poi_id)
			if det and det.category:
				# Store the vector
				if idx < len(self.U_p_levels[0]):
					vec = self.U_p_levels[0][idx]
					self.category_vectors[det.category.lower()].append(vec)
		
		# Convert lists to mean vectors
		self.mean_category_vectors = {}
		for cat, vecs in self.category_vectors.items():
			if vecs:
				self.mean_category_vectors[cat] = np.mean(vecs, axis=0)

	def _load_poi_tree(self, tree_obj: dict):
		if not isinstance(tree_obj, dict):
			raise ValueError("POI tree must be a dictionary")
			
		self.poi_tree = tree_obj
		# Removed level_3 from processing
		levels = ["level_0", "level_1", "level_2"]
		
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

		# Removed level 3 from indices
		for level in [0, 1, 2]:
			level_key = f'level_{level}'
			poi_ids = list(self.poi_tree[level_key].keys())
			self.poi_id_to_idx[level] = {pid: idx for idx, pid in enumerate(poi_ids)}
			self.idx_to_poi_id[level] = {idx: pid for pid, idx in self.poi_id_to_idx[level].items()}

	def _build_user_history(self):
		self.user_history = defaultdict(lambda: defaultdict(list))
		self.user_ratings = defaultdict(dict) # user_id -> {poi_id: rating}
		self.category_affinity = defaultdict(Counter) # user_id -> Counter({'cafe': 5, 'gym': 2})
		
		# Process Visits
		visits = self.interactions_df[self.interactions_df['interaction_type'] == 'visit']
		for _, row in visits.iterrows():
			uid, pid = str(row['user_id']), str(row['poi_id'])
			self.user_history[uid][0].append(pid)
			
			# Update Category Affinity Count
			det = self.poi_details_by_id.get(pid)
			if det:
				self.category_affinity[uid][det.category] += 1

		# Process Ratings
		ratings = self.interactions_df[self.interactions_df['interaction_type'] == 'rating']
		for _, row in ratings.iterrows():
			uid, pid = str(row['user_id']), str(row['poi_id'])
			score = float(row.get('rating_value', 0))
			self.user_ratings[uid][pid] = score
			
			# Weight affinity by rating (Poor rating = subtract, High rating = double add)
			det = self.poi_details_by_id.get(pid)
			if det:
				if score >= 4:
					self.category_affinity[uid][det.category] += 2 # Strong boost
				elif score <= 2:
					self.category_affinity[uid][det.category] -= 2 # Penalize

	def _get_parent_at_level(self, poi_id: str, target_level: int) -> Optional[str]:
		current_level = 0
		current_id = poi_id
		
		for lvl in [0, 1, 2]:
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
				'secondary': ['convenience_store', 'supermarket']
			},
			'coffee': {
				'primary': ['cafe'],
				'secondary': ['restaurant']
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
				'secondary': ['shopping_mall']
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
				'primary': ['attraction'],
				'secondary': ['shopping_mall']
			},
			'culture': {
				'primary': ['attraction', 'viewpoint', 'bookstore', 'university'],
				'secondary': ['museum']
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
				'secondary': ['hotel', 'hostel']
			},
			
			# Health & Wellness
			'sports': {
				'primary': ['gym', 'yoga', 'attraction'],
				'secondary': ['park', 'viewpoint']
			},
			'gyms': {
				'primary': ['gym', 'yoga'],
				'secondary': ['attraction']
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
				'secondary': ['gym', 'yoga', 'supermarket']
			},
			
			# Work & Business
			'work': {
				'primary': ['co_working'],
				'secondary': ['cafe', 'restaurant']
			},
			'business': {
				'primary': ['co_working', 'bank'],
				'secondary': ['cafe', 'restaurant', 'hotel']
			},
			'finance': {
				'primary': ['bank', 'atm'],
				'secondary': ['shopping_mall']
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
				'secondary': ['supermarket', 'convenience_store']
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
				'secondary': ['university']
			},
			
			# Convenience
			'convenience': {
				'primary': ['convenience_store'],
				'secondary': ['supermarket', 'pharmacy']
			},
			'charging': {
				'primary': ['charging_station'],
				'secondary': ['shopping_mall']
			},
			
			# Social
			'social': {
				'primary': ['cafe', 'restaurant'],
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
		if prompt_type == 'exploratory':
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

	def recommend_at_level(self, user_id: str, level: int, top_k: int, 
						  current_location: Dict, intent: Optional[Dict],
						  filter_visited: bool = True, strict_mode: bool = False) -> List:
		"""
		Generate recommendations for a specific hierarchy level.
		
		Args:
			user_id: User identifier
			level: Hierarchy level (0, 1, or 2)
			top_k: Number of recommendations to return
			current_location: Dict with 'latitude' and 'longitude'
			intent: Intent data from query parsing
			filter_visited: Whether to exclude previously visited POIs
			strict_mode: Whether to enforce strict category filtering
		"""
		if user_id not in self.user_id_to_idx: 
			return []
		
		user_idx = self.user_id_to_idx[user_id]
		
		# 1. Determine User Maturity (Alpha Decay)
		# Count explicit interactions (visits + ratings)
		history_count = len(self.user_history.get(user_id, {}).get(0, [])) + \
						len(self.user_ratings.get(user_id, {}))
		
		# Alpha: How much to rely on "Static Interests" vs "Observed Behavior"
		# New users: alpha ~ 1.0 (Rely on what they said)
		# Old users: alpha ~ 0.0 (Rely on what they do)
		alpha = max(0.0, 1.0 - (history_count / 10.0)) 

		candidates = list(self.idx_to_poi_id[level].values())
		scored_items = []
		
		for poi_id in candidates:
			# Skip visited (Standard filter) if enabled
			if filter_visited and poi_id in self.user_history.get(user_id, {}).get(level, []): 
				continue

			# --- CALCULATE COMPONENTS ---
			
			# A. Model Score (The Embedding Dot Product)
			poi_idx = self.poi_id_to_idx[level][poi_id]
			joint_score = self.compute_joint_score(user_idx, poi_idx, level)
			
			# B. Spatial Score
			dist_score = self.compute_distance_penalty(user_id, poi_id, level, current_location)
			
			# C. Personalization Score (Weighted Mix)
			# Static Interest (from onboarding/profile)
			static_interest = self.compute_interest_match(user_id, poi_id, level)
			
			# Dynamic Behavior (Visits/Ratings)
			dynamic_affinity = self.compute_interaction_boost(user_id, poi_id, level)
			
			# Blended Personalization Score
			personalization_score = (alpha * static_interest) + ((1.0 - alpha) * dynamic_affinity)

			# --- FINAL FORMULA ---
			final_score = (
				0.35 * joint_score + 
				0.25 * dist_score + 
				0.40 * personalization_score
			)
			
			# Intent Boost (Search Query)
			intent_match_score = 0.0
			if intent:
				matches_intent, intent_boost = self._compute_intent_match(
					poi_id, level, intent, strict_mode
				)
				if not matches_intent and strict_mode:
					continue
				final_score += intent_boost
				intent_match_score = intent_boost
			
			if final_score > 0:
				# Get POI details for return
				det = self.poi_details_by_id.get(poi_id)
				if det:
					info = {
						'name': det.name,
						'category': det.category,
						'type': 'Individual POI' if level == 0 else ('Container/Venue' if level == 1 else 'District'),
						'score_components': {
							'joint_embedding': round(joint_score, 4),
							'spatial': round(dist_score, 4),
							'interest': round(personalization_score, 4),
							'hierarchical': 0.0,  # Calculated separately if needed
							'intent_match': round(intent_match_score, 4) if intent_match_score > 0 else None
						},
						'popularity': det.popularity if level == 0 else None,
						'price': det.price if level == 0 else None,
						'latitude': det.latitude,
						'longitude': det.longitude
					}
					
					# Add hierarchical info
					if level == 1:
						node = self.poi_tree.get(f'level_{level}', {}).get(poi_id, {})
						info['num_pois'] = len(node.get('children', []))
					elif level == 2:
						node = self.poi_tree.get(f'level_{level}', {}).get(poi_id, {})
						info['num_venues'] = len(node.get('children', []))
					
					scored_items.append((poi_id, final_score, info))
					
		scored_items.sort(key=lambda x: x[1], reverse=True)
		return scored_items[:top_k]
			
	def recommend_multi_granularity(self,
							user_id: str,
							levels: List[int] = [0, 1, 2],
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
			raise ValueError(f"Unknown user_id: {user_id}. Please call add_user_profile() first.")
		
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
					current_location=current_location,
					intent=parsed_intent,
					filter_visited=filter_visited,
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
					"name": info.get('name', poi_id),
					"score": float(score) if score else 0.0,
					"type": info.get('type', 'Unknown'),
					"details": info
				}
				level_results.append(item)
			
			results[level_key] = level_results
			
			# Logging
			if prompt_type == 'category_based':
				matches = sum(1 for r in level_results 
							if (r['details'].get('score_components', {}).get('intent_match') or 0) > 0)
				print(f"  [Level {level}] Results: {len(level_results)}/{level_top_k} (matches: {matches})")
			else:
				print(f"  [Level {level}] Results: {len(level_results)}/{level_top_k}")
		
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
			"level_2": "LEVEL 2: DISTRICTS (Neighborhoods)"
		}
		
		print("\n" + "="*80)
		print("🎯 MULTI-GRANULARITY RECOMMENDATIONS")
		if hasattr(self, 'last_prompt_type') and self.last_prompt_type:
			print(f"   Query Type: {self.last_prompt_type.replace('_', ' ').title()}")
		print("="*80)
		
		for level_key in ["level_0", "level_1", "level_2"]:
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
				emoji = {0: "📌", 1: "🏢", 2: "🗺️"}.get(level_num, "•")
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
			"level_2": {"color": "#F18F01", "fill": "#F9D78C", "radius": 20}
		}
		
		level_names = {
			"level_0": "Individual POI",
			"level_1": "Container/Venue", 
			"level_2": "District"
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
						weight=3
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
		"""
		Add a new user to the system for cold start handling.
		Creates embedding vector from interest categories.
		
		Args:
			profile: Dict with 'user_id' and 'interests' (semicolon-separated)
			
		Returns:
			Dict with status, user_id, and idx
		"""
		user_id = str(profile.get("user_id", "")).strip()
		interests_raw = str(profile.get("interests", "")).lower()
		
		if not user_id:
			return {"status": "error", "message": "user_id required"}
		
		# 1. Parse interests
		user_interests = [i.strip() for i in interests_raw.replace(',', ';').split(';') if i.strip()]
		
		# 2. Construct Vector from Interests (Not Random Noise)
		vectors_to_average = []
		for interest in user_interests:
			# Try exact match first
			if interest in self.mean_category_vectors:
				vectors_to_average.append(self.mean_category_vectors[interest])
			else:
				# Try partial match (e.g. "coffee" matches "cafe")
				for cat, vec in self.mean_category_vectors.items():
					if interest in cat or cat in interest:
						vectors_to_average.append(vec)
						break
		
		if vectors_to_average:
			# Average the vectors of what they like
			new_vec = np.mean(vectors_to_average, axis=0, keepdims=True)
		else:
			# True Cold Start (No interests provided): Use Global Average of all users
			new_vec = np.mean(self.U_u, axis=0, keepdims=True)
			
		# Add some slight noise to prevent identical vectors for same-interest users
		noise = np.random.normal(0, 0.005, size=new_vec.shape)
		new_vec = new_vec + noise
		new_vec = new_vec / (np.linalg.norm(new_vec, axis=1, keepdims=True) + 1e-12)
		
		# Add to Matrix
		self.U_u = np.vstack([self.U_u, new_vec])
		new_idx = int(self.U_u.shape[0] - 1)
		self.user_id_to_idx[user_id] = new_idx
		self.idx_to_user_id[new_idx] = user_id
		
		# Initialize empty history
		self.user_history[user_id] = defaultdict(list)
		self.category_affinity[user_id] = Counter()
		
		# Add explicit interests to affinity counter immediately
		for i in user_interests:
			self.category_affinity[user_id][i] += 3 # Start with a strong bias
			
		return {"status": "created", "user_id": user_id, "idx": new_idx}
	
	def compute_interaction_boost(self, user_id: str, poi_id: str, level: int) -> float:
		"""
		Calculates a score (0.0 - 1.0) based on observed behavior.
		Uses the Category Affinity Counter built from visits/ratings.
		"""
		if level != 0: 
			return 0.5 # Interaction logic mostly applies to specific POIs
		
		det = self.poi_details_by_id.get(poi_id)
		if not det: 
			return 0.5
		
		# 1. Get user's category counts
		user_counts = self.category_affinity.get(user_id, Counter())
		if not user_counts: 
			return 0.5
		
		total_interactions = sum(user_counts.values())
		if total_interactions <= 0: 
			return 0.5
		
		# 2. Get specific category score
		cat_score = user_counts.get(det.category, 0)
		
		# Normalize (Frequency / Total Interactions) * Scaling Factor
		affinity_ratio = cat_score / total_interactions
		
		# Cap at 1.0, scale up to make it impactful
		normalized_score = min(1.0, affinity_ratio * 2.0) 
		
		# 3. Handle Penalties (Negative counts from bad ratings)
		if cat_score < 0:
			return 0.0 # Strongly penalize
			
		return normalized_score
	
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
		
		# Removed level_3 from styling
		level_colors = {
			"level_0": {"color": "#2E86AB", "fill": "#A7C6DA", "radius": 6},
			"level_1": {"color": "#A23B72", "fill": "#F18FBB", "radius": 12},
			"level_2": {"color": "#F18F01", "fill": "#F9D78C", "radius": 20}
		}
		
		level_names = {
			"level_0": "Individual POI",
			"level_1": "Container/Venue", 
			"level_2": "District"
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
						weight=3
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

	def _get_similar_visited_pois(self, user_id: str, poi_id: str, top_n: int = 5) -> List[Tuple[str, float]]:
		"""
		Find POIs similar to the target that the user has previously visited.
		Returns list of (poi_name, similarity_score) tuples.
		"""
		if user_id not in self.user_id_to_idx or poi_id not in self.poi_id_to_idx.get(0, {}):
			return []
		
		target_idx = self.poi_id_to_idx[0][poi_id]
		target_vec = self.U_p_levels[0][target_idx]
		
		visited = self.user_history.get(user_id, {}).get(0, [])
		similarities = []
		
		for visited_id in visited:
			if visited_id == poi_id or visited_id not in self.poi_id_to_idx.get(0, {}):
				continue
				
			visited_idx = self.poi_id_to_idx[0][visited_id]
			visited_vec = self.U_p_levels[0][visited_idx]
			
			# Cosine similarity
			dot = np.dot(target_vec, visited_vec)
			norm_target = np.linalg.norm(target_vec)
			norm_visited = np.linalg.norm(visited_vec)
			
			if norm_target > 0 and norm_visited > 0:
				sim = dot / (norm_target * norm_visited)
				if sim > 0.3:  # Similarity threshold
					det = self.poi_details_by_id.get(visited_id)
					name = det.name if det else visited_id
					similarities.append((name, float(sim)))
		
		return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

	def compute_feature_based_score(self, user_idx: int, poi_idx: int, level: int) -> float:
		"""
		Compute feature-based score (wrapper around joint embedding score).
		Scaled to 0-100 range for consistency with explanation framework.
		"""
		joint_score = self.compute_joint_score(user_idx, poi_idx, level)
		return float(joint_score * 100)

	def compute_graph_based_score(self, user_idx: int, poi_idx: int, level: int) -> float:
		"""
		Compute graph-based score using spatial/hierarchical context.
		Uses hierarchical boost as proxy for graph connectivity.
		"""
		poi_id = self.idx_to_poi_id[level][poi_idx]
		hier_score = self.compute_hierarchical_boost(poi_id, user_idx, level)
		return float(hier_score * 100)

	def build_reason_flags(self, 
						  user_id: str, 
						  poi_id: str,
						  level: int = 0) -> Dict[str, bool]:
		"""
		Build interpretable boolean flags explaining WHY this POI is recommended
		"""
		level_key = f'level_{level}'
		
		# Get user profile using correct column name
		user_rows = self.users_df[self.users_df[self.user_id_col] == user_id]
		if user_rows.empty:
			return {}
		user_row = user_rows.iloc[0]
		user_interests = set([i.strip().lower() for i in str(user_row.get('interests', '')).split(';') if i.strip()])
		user_area = user_row.get('area_of_residence', '')
		
		# Get POI data
		if level_key not in self.poi_tree or poi_id not in self.poi_tree[level_key]:
			return {}
			
		poi_data = self.poi_tree[level_key][poi_id]
		poi_info = poi_data.get('data', {}) if isinstance(poi_data.get('data'), dict) else {}
		
		# Initialize flags
		flags = {}
		
		if level == 0:
			# Individual POI flags
			poi_category = str(poi_info.get('category', '')).lower()
			poi_chars = str(poi_info.get('characteristic', '')).lower()
			poi_popularity = float(poi_info.get('popularity', 0) or 0)
			poi_price = str(poi_info.get('price', ''))
			
			# 1. Category match
			interest_category_map = {
				'food': ['restaurant', 'cafe', 'food_court', 'hawker_centre', 'fast_food'],
				'shopping': ['shopping_mall', 'retail', 'store', 'boutique', 'department_store'],
				'movies': ['cinema', 'theatre'],
				'fitness': ['gym', 'sports', 'fitness', 'yoga'],
				'entertainment': ['cinema', 'arcade', 'karaoke', 'bar', 'attraction']
			}
			
			flags['matches_interest'] = any(
				any(cat in poi_category for cat in interest_category_map.get(interest, []))
				for interest in user_interests
			)
			
			# 2. Previously visited
			visited_pois = self.user_history.get(user_id, {}).get(0, [])
			flags['visited_before'] = poi_id in visited_pois
			
			# 3. Nearby (distance check)
			if user_id in self.user_id_to_idx and poi_id in self.poi_id_to_idx.get(0, {}):
				user_idx = self.user_id_to_idx[user_id]
				poi_idx = self.poi_id_to_idx[0][poi_id]
				distance_penalty = self.compute_distance_penalty(user_id, poi_id, level)
				flags['nearby'] = distance_penalty > 0.7
				flags['very_nearby'] = distance_penalty > 0.9
			else:
				flags['nearby'] = False
				flags['very_nearby'] = False
			
			# 4. Popular
			flags['popular'] = poi_popularity >= 4.0
			flags['highly_popular'] = poi_popularity >= 4.5
			
			# 5. Price match
			user_price_sens = str(user_row.get('price_sensitivity', '')).lower()
			flags['matches_budget'] = False
			
			if poi_price and poi_price != '':
				try:
					if '-' in str(poi_price):
						prices = str(poi_price).split('-')
						avg_price = (float(prices[0].strip()) + float(prices[1].strip())) / 2
					else:
						avg_price = float(poi_price)
					
					if user_price_sens == 'low' and avg_price <= 20:
						flags['matches_budget'] = True
					elif user_price_sens == 'medium' and 15 <= avg_price <= 40:
						flags['matches_budget'] = True
					elif user_price_sens == 'high' and avg_price >= 30:
						flags['matches_budget'] = True
				except:
					pass
			
			# 6. Similar to previous visits
			similar_visited = self._get_similar_visited_pois(user_id, poi_id)
			flags['similar_to_past'] = len(similar_visited) > 0 and similar_visited[0][1] > 0.5
			
			# 7. Trending in area (placeholder - could compute from recent interactions)
			flags['trending_in_area'] = False
			
			# 8. Parent venue is popular (hierarchical)
			parent_id = poi_data.get('parent')
			if parent_id and parent_id in self.poi_id_to_idx.get(1, {}):
				parent_idx = self.poi_id_to_idx[1][parent_id]
				user_idx = self.user_id_to_idx[user_id]
				parent_score = self.compute_feature_based_score(user_idx, parent_idx, level=1)
				flags['in_popular_venue'] = parent_score > 50
			else:
				flags['in_popular_venue'] = False
		
		else:
			# Container/District/Region flags
			flags['matches_interest'] = True  # Simplified for higher levels
			flags['visited_before'] = poi_id in self.user_history.get(user_id, {}).get(level, [])
			flags['nearby'] = True  # Higher levels are broader
			flags['popular'] = True  # Placeholder
			flags['has_many_options'] = len(poi_data.get('children', [])) > 10
		
		return flags

	def build_human_explanation(self,
								user_id: str,
								poi_id: str,
								level: int = 0,
								reason_flags: Dict[str, bool] = None,
								score_components: Dict[str, float] = None) -> str:
		"""
		Generate natural language explanation for a recommendation
		"""
		level_key = f'level_{level}'
		
		# Get data with safe access
		user_rows = self.users_df[self.users_df[self.user_id_col] == user_id]
		if user_rows.empty:
			return "Unable to generate explanation: User not found."
		user_row = user_rows.iloc[0]
		user_name = user_row.get('name', 'User')
		
		if level_key not in self.poi_tree or poi_id not in self.poi_tree[level_key]:
			return "Unable to generate explanation: POI not found."
			
		poi_data = self.poi_tree[level_key][poi_id]
		poi_name = poi_data.get('name', 'Unknown')
		
		# Get reason flags
		if reason_flags is None:
			reason_flags = self.build_reason_flags(user_id, poi_id, level)
		
		# Build explanation parts
		parts = []
		
		# Introduction
		if level == 0:
			intro = f"We recommend **{poi_name}** for you because:"
		else:
			intro = f"We suggest exploring **{poi_name}** because:"
		
		# Main reasons (prioritized by importance)
		
		# 1. Visited before (strongest signal)
		if reason_flags.get('visited_before'):
			parts.append("✓ You've visited here before and seem to like it")
		
		# 2. Very nearby (convenience)
		elif reason_flags.get('very_nearby'):
			parts.append("✓ It's very close to your home area")
		elif reason_flags.get('nearby'):
			parts.append("✓ It's within a comfortable distance from you")
		
		# 3. Interest match
		if reason_flags.get('matches_interest'):
			user_interests = user_row.get('interests', '')
			if level == 0:
				poi_info = poi_data.get('data', {}) if isinstance(poi_data.get('data'), dict) else {}
				poi_category = poi_info.get('category', 'place')
				parts.append(f"✓ It matches your interests ({user_interests}) - it's a {poi_category}")
			else:
				parts.append(f"✓ It has options matching your interests ({user_interests})")
		
		# 4. Budget match
		if reason_flags.get('matches_budget'):
			parts.append(f"✓ Prices fit your budget ({user_row.get('price_sensitivity', 'medium')} spending)")
		
		# 5. Popularity
		if reason_flags.get('highly_popular'):
			parts.append("✓ It's highly rated by other users (⭐ 4.5+)")
		elif reason_flags.get('popular'):
			parts.append("✓ It's popular among other users")
		
		# 6. Similar to past
		if reason_flags.get('similar_to_past') and not reason_flags.get('visited_before'):
			similar_pois = self._get_similar_visited_pois(user_id, poi_id)
			if similar_pois:
				similar_name = similar_pois[0][0]
				parts.append(f"✓ It's similar to places you've enjoyed (like {similar_name})")
		
		# 7. In popular venue (hierarchical context)
		if reason_flags.get('in_popular_venue'):
			parent_id = poi_data.get('parent')
			if parent_id and 'level_1' in self.poi_tree and parent_id in self.poi_tree['level_1']:
				parent_name = self.poi_tree['level_1'][parent_id].get('name', 'this area')
				parts.append(f"✓ It's located in {parent_name}, which you might like")
		
		# 8. Many options (for higher levels)
		if reason_flags.get('has_many_options'):
			num_children = len(poi_data.get('children', []))
			parts.append(f"✓ It offers {num_children}+ places to explore")
		
		# Limit to top 3-4 reasons
		parts = parts[:4]
		
		# Construct final explanation
		if not parts:
			explanation = intro + "\n  • It matches your overall preferences and location"
		else:
			explanation = intro + "\n  " + "\n  ".join(parts)
		
		# Add confidence/score context
		if score_components:
			total_score = score_components.get('total_score', 0)
			if total_score > 80:
				explanation += "\n\n🎯 **Strong match** - Highly recommended!"
			elif total_score > 50:
				explanation += "\n\n👍 **Good match** - Worth checking out"
			else:
				explanation += "\n\n💡 **Potential match** - Might be interesting"
		
		return explanation

	def explain_recommendation_enhanced(self,
                                    user_id: str,
                                    poi_id: str,
                                    level: int = 0,
                                    current_location: Optional[Dict] = None) -> Dict[str, any]:
		"""
		Complete enhanced explanation with detailed error reporting
		"""
		# Better error messages for debugging
		if user_id not in self.user_id_to_idx:
			print(f"  [Explain Error] User '{user_id}' not found in user_id_to_idx")
			return {"error": f"User {user_id} not found"}
		
		if level not in self.poi_id_to_idx:
			print(f"  [Explain Error] Level {level} not found")
			return {"error": f"Invalid level {level}"}
		
		if poi_id not in self.poi_id_to_idx[level]:
			print(f"  [Explain Error] POI '{poi_id}' not found at level {level}")
			return {"error": f"POI {poi_id} not found at level {level}"}
		
		user_idx = self.user_id_to_idx[user_id]
		poi_idx = self.poi_id_to_idx[level][poi_id]
		level_key = f'level_{level}'
		
		# 1. Get technical score components (using your existing methods)
		feature_score = self.compute_joint_score(user_idx, poi_idx, level) * 100
		
		# Graph score from hierarchical boost
		hier_boost = self.compute_hierarchical_boost(poi_id, user_idx, level)
		graph_score = hier_boost * 100 if isinstance(hier_boost, (int, float)) else 50.0
		
		# Distance and interest
		if level == 0:
			distance_penalty = self.compute_distance_penalty(user_id, poi_id, level, current_location)
			interest_bonus = self.compute_interest_match(user_id, poi_id, level)
		else:
			distance_penalty = 1.0
			interest_bonus = 0.5
		
		total_score = (0.25 * feature_score + 
					0.15 * graph_score + 
					0.10 * graph_score +  # hierarchical treated as graph here
					0.35 * distance_penalty * 100 +
					0.15 * interest_bonus * 100)
		
		score_components = {
			'total_score': total_score,
			'feature_based': {
				'raw_score': feature_score,
				'contribution': 0.25 * feature_score,
				'weight': 0.25,
				'description': 'Match between your profile and POI attributes'
			},
			'graph_based': {
				'raw_score': graph_score,
				'contribution': 0.15 * graph_score,
				'weight': 0.15,
				'description': 'Spatial context from places you\'ve visited'
			},
			'distance': {
				'raw_score': distance_penalty,
				'contribution': 0.35 * distance_penalty * 100,
				'weight': 0.35,
				'description': 'How convenient it is to reach from your location'
			},
			'interest_match': {
				'raw_score': interest_bonus,
				'contribution': 0.15 * interest_bonus * 100,
				'weight': 0.15,
				'description': 'How well it matches your stated interests'
			}
		}
		
		# Sort components by contribution
		ranked_components = sorted(
			[(k, v) for k, v in score_components.items() if isinstance(v, dict)],
			key=lambda x: x[1].get('contribution', 0),
			reverse=True
		)
		top_factors = [name for name, comp in ranked_components[:3]]
		
		# 2. Get reason flags
		reason_flags = self.build_reason_flags(user_id, poi_id, level)
		
		# 3. Generate human explanation
		human_explanation = self.build_human_explanation(
			user_id, poi_id, level, reason_flags, score_components
		)
		
		# 4. Get contexts
		poi_data = self.poi_tree.get(level_key, {}).get(poi_id, {})
		user_data = self.users_df[self.users_df[self.user_id_col] == user_id]
		user_data = user_data.iloc[0].to_dict() if not user_data.empty else {}
		
		# 5. Similar POIs
		similar_visited = self._get_similar_visited_pois(user_id, poi_id)[:3] if level == 0 else []
		
		return {
			'score_breakdown': score_components,
			'top_contributing_factors': top_factors,
			'reason_flags': reason_flags,
			'active_reasons': [k for k, v in reason_flags.items() if v],
			'human_explanation': human_explanation,
			'user_context': {
				'name': user_data.get('name', 'Unknown'),
				'interests': user_data.get('interests', ''),
				'area': user_data.get('area_of_residence', ''),
				'price_sensitivity': user_data.get('price_sensitivity', 'medium')
			},
			'poi_context': {
				'name': poi_data.get('name', 'Unknown'),
				'type': 'Individual POI' if level == 0 else f'Level {level}',
			},
			'similar_visited_pois': similar_visited,
		}

	def display_explanation(self, explanation: Dict[str, any], detailed: bool = False):
		"""
		Pretty print an explanation
		"""
		print("\n" + "="*70)
		print("RECOMMENDATION EXPLANATION")
		print("="*70)
		
		# Always show human explanation
		print("\n" + explanation.get('human_explanation', 'No explanation available'))
		
		if detailed:
			# Show technical breakdown
			print("\n" + "-"*70)
			print("TECHNICAL BREAKDOWN")
			print("-"*70)
			
			sb = explanation.get('score_breakdown', {})
			print(f"\nTotal Score: {sb.get('total_score', 0):.2f}/100")
			print(f"Top Contributing Factors: {', '.join(explanation.get('top_contributing_factors', []))}")
			
			print("\nScore Components:")
			for name, comp in sb.items():
				if name == 'total_score' or not isinstance(comp, dict):
					continue
				print(f"  • {name}:")
				print(f"      Contribution: {comp.get('contribution', 0):.2f} (weight: {comp.get('weight', 0):.0%})")
				print(f"      Description: {comp.get('description', '')}")
			
			print("\nActive Reasoning Flags:")
			for flag in explanation.get('active_reasons', []):
				print(f"  ✓ {flag}")
			
			similar = explanation.get('similar_visited_pois', [])
			if similar:
				print("\nSimilar Places You've Visited:")
				for poi_name, similarity in similar:
					print(f"  • {poi_name} (similarity: {similarity:.2f})")

	def explain_all_recommendations(self,
                                recommendations: Dict,  
                                user_id: str,
                                detailed: bool = False,
                                top_n_detailed: int = 1) -> None:
		"""
		Display explanations for all recommendations
		Handles both {"level_0": [...]} and {0: [...]} formats
		"""
		level_names = {
			0: "INDIVIDUAL POIs",
			1: "CONTAINERS/VENUES", 
			2: "DISTRICTS"
		}
		
		# Get user name
		user_rows = self.users_df[self.users_df[self.user_id_col] == user_id]
		user_name = user_rows.iloc[0].get('name', user_id) if not user_rows.empty else user_id
		
		print("\n" + "="*70)
		print(f"EXPLANATIONS FOR ALL RECOMMENDATIONS - User: {user_name}")
		print("="*70)
		
		# Normalize keys to integers
		normalized_recs = {}
		for k, v in recommendations.items():
			if isinstance(k, str) and k.startswith('level_'):
				try:
					level_num = int(k.split('_')[1])
					normalized_recs[level_num] = v
				except:
					continue
			elif isinstance(k, int):
				normalized_recs[k] = v
		
		for level in sorted(normalized_recs.keys()):
			recs = normalized_recs[level]
			
			print(f"\n{'='*70}")
			print(f"LEVEL {level}: {level_names.get(level, f'LEVEL {level}')}")
			print(f"{'='*70}\n")
			
			if not recs:
				print("  No recommendations at this level\n")
				continue
			
			for rank, rec in enumerate(recs, 1):
				# Handle both tuple and dict formats
				if isinstance(rec, tuple):
					poi_id, score, poi_info = rec
				elif isinstance(rec, dict):
					poi_id = rec.get('poi_id')
					score = rec.get('score', 0)
					poi_info = rec.get('details', {})
				else:
					continue
				
				poi_name = poi_info.get('name', 'Unknown') if isinstance(poi_info, dict) else 'Unknown'
				
				print(f"\n{'-'*70}")
				print(f"#{rank}. {poi_name} (Score: {float(score):.2f})")
				print(f"{'-'*70}")
				
				explanation = self.explain_recommendation_enhanced(user_id, poi_id, level)
				
				if not explanation or 'error' in explanation:
					error_msg = explanation.get('error', 'Unknown error') if isinstance(explanation, dict) else "Empty result"
					print(f"  Unable to generate explanation: {error_msg}")
					continue
				
				# Print human explanation
				print(explanation.get('human_explanation', ''))
				
				if detailed and rank <= top_n_detailed:
					print(f"\n📊 Technical Breakdown:")
					sb = explanation.get('score_breakdown', {})
					print(f"   Total: {sb.get('total_score', 0):.1f}/100")
					print(f"   Top factors: {', '.join(explanation.get('top_contributing_factors', []))}")
					
					reasons = explanation.get('active_reasons', [])
					if reasons:
						print(f"   Flags: {', '.join(reasons[:3])}")
				
				print()

import pandas as pd
import numpy as np
import json
import pickle
import networkx as nx
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')
import re

class MultiGranularityRecommendationFramework:
    """
    Complete Multi-Granularity POI Recommendation System
    
    Provides recommendations at multiple granularity levels:
    - Level 0: Individual POIs (e.g., "Starbucks @ VivoCity")
    - Level 1: Containers (e.g., "VivoCity Mall")
    - Level 2: Districts (e.g., "HarbourFront District")
    - Level 3: Regions (e.g., "Southern Singapore")
    """
    
    def __init__(self,
                 embeddings_file: str = 'embeddings.pkl',
                 interaction_learning_file: str = 'interaction_learning.pkl',
                 poi_tree_file: str = 'poi_tree.json',
                 users_file: str = 'users.csv',
                 interactions_file: str = 'user_poi_interactions.csv'):
        """Initialize the recommendation framework"""
        print("="*70)
        print("INITIALIZING MULTI-GRANULARITY RECOMMENDATION FRAMEWORK")
        print("="*70)
        
        # Load attribute-based embeddings
        print("\nLoading attribute-based embeddings...")
        with open(embeddings_file, 'rb') as f:
            emb_data = pickle.load(f)
        
        self.user_embeddings = emb_data['user_embeddings']
        self.poi_embeddings = emb_data['poi_embeddings']
        self.user_id_to_idx = emb_data['user_id_to_idx']
        self.X = emb_data['X']
        self.X_A = emb_data['X_A']
        self.X_T = emb_data['X_T']
        
        # Load interaction-based components
        print("Loading interaction-based components...")
        with open(interaction_learning_file, 'rb') as f:
            int_data = pickle.load(f)
        
        self.Theta_u = int_data['Theta_u']
        self.A_l_p = int_data['A_l_p']
        self.G_l = int_data['G_l']
        self.P_l = int_data['P_l']
        self.Q_l = int_data['Q_l']
        self.S_l = int_data['S_l']
        self.U_l_g = int_data['U_l_g']
        
        # Load POI tree
        print("Loading POI tree...")
        with open(poi_tree_file, 'r') as f:
            self.poi_tree = json.load(f)
        
        # Load raw data
        print("Loading user profiles and interactions...")
        self.users_df = pd.read_csv(users_file)
        self.interactions_df = pd.read_csv(interactions_file)
        
        # Build indices
        self._build_indices()
        
        # Initialize hyperparameters
        self.alpha = 0.5   # Weight for feature-based score
        self.beta = 0.3    # Weight for graph-based score
        self.gamma = 0.2   # Weight for hierarchical boost
        
        # Level weights (how much to trust each level)
        self.level_weights = {
            0: 0.6,  # Individual POIs
            1: 0.25, # Containers
            2: 0.10, # Districts
            3: 0.05  # Regions
        }
        
        print("\n" + "="*70)
        print("FRAMEWORK INITIALIZED SUCCESSFULLY")
        print("="*70)
        print(f"Users: {len(self.users_df)}")
        print(f"POIs by level:")
        for level in [0, 1, 2, 3]:
            print(f"  Level {level}: {len(self.poi_tree[f'level_{level}'])} POIs")
        print(f"Interactions: {len(self.interactions_df)}")
    
    def _build_indices(self):
        """Build lookup indices for fast access"""
        # User index
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.users_df['uudi'])}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        
        # POI indices for each level
        self.poi_id_to_idx = {}
        self.idx_to_poi_id = {}
        
        for level in [0, 1, 2, 3]:
            level_key = f'level_{level}'
            poi_ids = self.poi_embeddings[level_key]['poi_ids']
            self.poi_id_to_idx[level] = {pid: idx for idx, pid in enumerate(poi_ids)}
            self.idx_to_poi_id[level] = {idx: pid for pid, idx in self.poi_id_to_idx[level].items()}
        
        # Build user history lookup (at level 0)
        self._build_user_history()
    
    def _build_user_history(self):
        """Build lookup table for user visit history"""
        self.user_history = defaultdict(lambda: defaultdict(list))
        
        visits = self.interactions_df[self.interactions_df['interaction_type'] == 'visit']
        
        for _, row in visits.iterrows():
            user_id = row['user_id']
            poi_id = row['poi_id']
            
            # Store at level 0
            if poi_id not in self.user_history[user_id][0]:
                self.user_history[user_id][0].append(poi_id)
            
            # Propagate to higher levels
            for level in [1, 2, 3]:
                parent_id = self._get_parent_at_level(poi_id, level)
                if parent_id and parent_id not in self.user_history[user_id][level]:
                    self.user_history[user_id][level].append(parent_id)
    
    def _get_parent_at_level(self, poi_id: str, target_level: int) -> Optional[str]:
        """Get parent node of poi_id at target_level"""
        current_level = 0
        current_id = poi_id
        
        while current_level < target_level:
            level_key = f'level_{current_level}'
            if current_id in self.poi_tree[level_key]:
                parent = self.poi_tree[level_key][current_id].get('parent')
                if parent:
                    current_id = parent
                    current_level += 1
                else:
                    break
            else:
                break
        
        return current_id if current_level == target_level else None
    
    # ========================================================================
    # UNDERSTANDING PROMPTS (NLP)
    # ========================================================================

    def extract_categories(self, prompt: str) -> List[str]:
        """Extract POI categories from prompt"""
        prompt_lower = prompt.lower()
        
        category_keywords = {
            'cafe': ['cafe', 'coffee', 'starbucks'],
            'restaurant': ['restaurant', 'food', 'dining', 'eat'],
            'shopping_mall': ['mall', 'shopping', 'shop'],
            'cinema': ['movie', 'cinema', 'film'],
            'gym': ['gym', 'fitness', 'workout'],
            'park': ['park', 'nature', 'outdoor']
        }
        
        categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ['restaurant']  # Default

    def extract_location(self, prompt: str) -> Optional[str]:
        """Extract location mention from prompt"""
        locations = ['orchard', 'jurong', 'tampines', 'woodlands', 'bishan']
        
        prompt_lower = prompt.lower()
        for location in locations:
            if location in prompt_lower:
                return location.title()
        
        return None

    def parse_user_prompt(self, prompt: str, current_location: Optional[Dict]) -> Dict:
        """Parse user prompt into structured intent"""
        return {
            'raw_prompt': prompt,
            'categories': self.extract_categories(prompt),
            'location_mentioned': self.extract_location(prompt),
            'search_location': current_location,
            'keywords': [],
            'max_distance_km': 5.0
        }
    
    # ========================================================================
    # CORE SCORING FUNCTIONS (Multi-level)
    # ========================================================================
    
    def compute_feature_based_score(self, user_idx: int, poi_idx: int, level: int) -> float:
        """Get feature-based affinity score S^l[u, p]"""
        level_key = f'level_{level}'
        return float(self.S_l[level_key][user_idx, poi_idx])
    
    def compute_graph_based_score(self, user_idx: int, poi_idx: int, level: int) -> float:
        """Compute graph-based context score"""
        level_key = f'level_{level}'
        
        U_g_up = self.U_l_g[level_key][user_idx, poi_idx]
        Q_p = self.Q_l[level_key][poi_idx]
        
        min_len = min(len(U_g_up), len(Q_p))
        U_g_up = U_g_up[:min_len]
        Q_p = Q_p[:min_len]
        
        if np.linalg.norm(U_g_up) == 0 or np.linalg.norm(Q_p) == 0:
            return 0.0
        
        score = np.dot(U_g_up, Q_p) / (np.linalg.norm(U_g_up) * np.linalg.norm(Q_p))
        return float(score)
    
    def compute_hierarchical_boost(self, poi_id: str, user_idx: int, level: int) -> float:
        """
        Compute hierarchical boost from parent/child scores
        
        For coarse levels (2, 3): boost from children's popularity
        For fine levels (0, 1): boost from parent's score
        """
        if level >= 2:
            # Coarse level: aggregate from children
            children = self.poi_tree[f'level_{level}'].get(poi_id, {}).get('children', [])
            if not children:
                return 0.0
            
            child_scores = []
            for child_id in children[:10]:  # Limit to top 10 children
                if level == 2:
                    child_level = 1
                elif level == 3:
                    child_level = 2
                else:
                    continue
                
                if child_id in self.poi_id_to_idx[child_level]:
                    child_idx = self.poi_id_to_idx[child_level][child_id]
                    score = self.compute_feature_based_score(user_idx, child_idx, child_level)
                    child_scores.append(score)
            
            return np.mean(child_scores) if child_scores else 0.0
        
        else:
            # Fine level: boost from parent
            poi_data = self.poi_tree[f'level_{level}'].get(poi_id)
            if not poi_data:
                return 0.0
            
            parent_id = poi_data.get('parent')
            if not parent_id:
                return 0.0
            
            parent_level = level + 1
            if parent_id in self.poi_id_to_idx[parent_level]:
                parent_idx = self.poi_id_to_idx[parent_level][parent_id]
                return self.compute_feature_based_score(user_idx, parent_idx, parent_level)
            
            return 0.0
    
    def compute_multi_granularity_score(self, 
                                        user_id: str, 
                                        poi_id: str,
                                        level: int = 0,
                                        use_hierarchical_boost: bool = True,
                                        use_graph_context: bool = True,
                                        use_distance_penalty: bool = True) -> float:
        """
        Updated scoring with distance penalty
        """
        if user_id not in self.user_id_to_idx:
            return 0.0
        
        if poi_id not in self.poi_id_to_idx[level]:
            return 0.0
        
        user_idx = self.user_id_to_idx[user_id]
        poi_idx = self.poi_id_to_idx[level][poi_id]
        
        # 1. Feature-based score
        feature_score = self.compute_feature_based_score(user_idx, poi_idx, level)
        
        # 2. Graph-based score
        graph_score = 0.0
        if use_graph_context:
            graph_score = self.compute_graph_based_score(user_idx, poi_idx, level)
        
        # 3. Hierarchical boost
        hierarchical_score = 0.0
        if use_hierarchical_boost:
            hierarchical_score = self.compute_hierarchical_boost(poi_id, user_idx, level)
        
        # 4. Distance penalty
        distance_penalty = 1.0
        if use_distance_penalty and level == 0:
            distance_penalty = self.compute_distance_penalty(user_id, poi_id, level)
        
        # 5. Interest match bonus
        interest_bonus = self.compute_interest_match(user_id, poi_id, level)
        
        # Combine with updated weights
        # REDUCE feature score weight, ADD distance penalty
        final_score = (0.3 * feature_score +           
                    0.2 * graph_score +              
                    0.1 * hierarchical_score +       
                    0.3 * distance_penalty * 100 +   
                    0.1 * interest_bonus * 100)      
        
        return float(final_score)

    def compute_interest_match(self, user_id: str, poi_id: str, level: int = 0) -> float:
        """
        Match user interests to POI category/characteristics
        
        Args:
            user_id: User ID
            poi_id: POI ID
            level: Granularity level (default: 0)
        
        Returns:
            Interest match score (0.0 - 1.0)
        """
        user_row = self.users_df[self.users_df['uudi'] == user_id].iloc[0]
        user_interests = set([i.strip().lower() for i in user_row['interests'].split(';')])
        
        # Only compute detailed interest match for level 0 (individual POIs)
        if level != 0:
            # For higher levels, return a moderate score (0.5)
            # Or aggregate from children
            return 0.5
        
        level_key = f'level_{level}'
        
        # Check if POI exists at this level
        if poi_id not in self.poi_tree[level_key]:
            return 0.0
        
        poi_data = self.poi_tree[level_key][poi_id]
        poi_category = poi_data['data'].get('category', '').lower()
        poi_chars = poi_data['data'].get('characteristic', '').lower()
        
        # Check matches
        matches = 0
        
        # Direct category match
        if 'food' in user_interests and poi_category in ['restaurant', 'cafe', 'food_court']:
            matches += 1
        
        if 'shopping' in user_interests and poi_category in ['shopping_mall', 'retail', 'store']:
            matches += 1
        
        if 'movies' in user_interests and 'cinema' in poi_category:
            matches += 2  # Strong match
        
        # Characteristic match
        for interest in user_interests:
            if interest in poi_chars:
                matches += 0.5
        
        return min(matches / 3.0, 1.0)  # Normalize to 0-1
    
    def compute_distance_penalty(self, user_id: str, poi_id: str, level: int = 0) -> float:
        """
        Compute distance penalty based on user's home location
        
        Args:
            user_id: User ID
            poi_id: POI ID
            level: Granularity level
        
        Returns:
            Penalty factor (1.0 = nearby, 0.0 = very far)
        """
        # Area to coordinates mapping
        area_coords = {
            'Jurong East': (1.3329, 103.7436),
            'Yishun': (1.4304, 103.8354),
            'Bishan': (1.3526, 103.8352),
            'Tampines': (1.3496, 103.9568),
            'Woodlands': (1.4382, 103.7891),
            'Ang Mo Kio': (1.3691, 103.8454),
            'Bedok': (1.3236, 103.9273),
            'Clementi': (1.3162, 103.7649),
            'Hougang': (1.3612, 103.8864),
            'Punggol': (1.4054, 103.9021),
            'Sengkang': (1.3868, 103.8914),
            'Bukit Batok': (1.3590, 103.7637),
            'Bukit Panjang': (1.3774, 103.7718),
            'Choa Chu Kang': (1.3840, 103.7470),
            'Pasir Ris': (1.3721, 103.9474),
            'Sembawang': (1.4491, 103.8185),
            'Serangoon': (1.3554, 103.8679),
            'Toa Payoh': (1.3343, 103.8564),
        }
        
        user_row = self.users_df[self.users_df['uudi'] == user_id].iloc[0]
        user_area = user_row['area_of_residence']
        
        if user_area not in area_coords:
            return 0.5  # Moderate penalty if unknown
        
        user_lat, user_lon = area_coords[user_area]
        
        # Get POI location based on level
        level_key = f'level_{level}'
        
        if poi_id not in self.poi_tree[level_key]:
            return 0.5
        
        poi_data = self.poi_tree[level_key][poi_id]
        poi_spatial = poi_data.get('spatial')
        
        if not poi_spatial:
            return 0.5  # No spatial data
        
        if isinstance(poi_spatial, str):
            poi_spatial = eval(poi_spatial)
        
        poi_lat, poi_lon = poi_spatial
        
        # Calculate distance
        distance = self._haversine_distance(user_lat, user_lon, poi_lat, poi_lon)
        
        # Transportation-based distance tolerance
        user_transport = user_row['transportation_modes']
        
        if 'car' in user_transport or 'ride-hailing' in user_transport:
            max_comfortable_distance = 15.0  # km
        elif 'MRT' in user_transport or 'bus' in user_transport:
            max_comfortable_distance = 8.0
        else:
            max_comfortable_distance = 3.0  # Walking
        
        # Adjust tolerance for higher levels (broader areas)
        if level == 1:
            max_comfortable_distance *= 1.2
        elif level == 2:
            max_comfortable_distance *= 1.5
        elif level == 3:
            max_comfortable_distance *= 2.0
        
        # Exponential decay penalty
        penalty = np.exp(-distance / max_comfortable_distance)
        
        return float(penalty)
    
    def _diversify_by_category(self, poi_scores: List[Tuple[str, float]], top_k: int) -> List:
        """
        Ensure recommendations span multiple categories
        """
        selected = []
        category_counts = defaultdict(int)
        max_per_category = max(2, top_k // 3)  # Max 2-3 items per category
        
        for poi_id, score in poi_scores:
            poi_data = self.poi_tree['level_0'][poi_id]
            category = poi_data['data'].get('category', 'other')
            
            if category_counts[category] < max_per_category:
                selected.append((poi_id, score))
                category_counts[category] += 1
            
            if len(selected) >= top_k:
                break
        
        return selected
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in km"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    # ========================================================================
    # MULTI-GRANULARITY RECOMMENDATION
    # ========================================================================
    
    def recommend_at_level(self,
                          user_id: str,
                          level: int,
                          top_k: int = 10,
                          filter_visited: bool = True,
                          use_constraints: bool = False,
                          **kwargs) -> List[Tuple[str, float, Dict]]:
        """
        Generate recommendations at specific granularity level
        
        Args:
            user_id: User ID
            level: Granularity level (0=individual, 1=container, 2=district, 3=region)
            top_k: Number of recommendations
            filter_visited: Filter out visited POIs
            use_constraints: Apply user constraints (only for level 0)
        
        Returns:
            List of (poi_id, score, poi_info) tuples
        """
        if user_id not in self.user_id_to_idx:
            return []
        
        level_key = f'level_{level}'
        all_poi_ids = list(self.poi_tree[level_key].keys())
        
        # Get visited POIs at this level
        visited_pois = set(self.user_history.get(user_id, {}).get(level, []))
        
        # Score all POIs
        poi_scores = []
        
        for poi_id in all_poi_ids:
            if filter_visited and poi_id in visited_pois:
                continue
            
            score = self.compute_multi_granularity_score(
                user_id, poi_id, level, **kwargs
            )
            
            poi_scores.append((poi_id, score))
        
        # Sort by score
        poi_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-K
        top_pois = poi_scores[:top_k]
        
        # Add POI information
        recommendations = []
        for poi_id, score in top_pois:
            poi_data = self.poi_tree[level_key][poi_id]
            
            # Build info based on level
            if level == 0:
                poi_info = {
                    'name': poi_data['name'],
                    'category': poi_data['data'].get('category', 'N/A'),
                    'price': poi_data['data'].get('price', 'N/A'),
                    'popularity': poi_data['data'].get('popularity', 'N/A'),
                    'region': poi_data['data'].get('region', 'N/A'),
                    'type': 'Individual POI'
                }
            elif level == 1:
                # Container (e.g., mall, building)
                children = poi_data.get('children', [])
                poi_info = {
                    'name': poi_data['name'],
                    'textual': poi_data.get('textual', '')[:100],
                    'num_pois': len(children),
                    'type': 'Container/Venue'
                }
            elif level == 2:
                # District
                children = poi_data.get('children', [])
                poi_info = {
                    'name': poi_data['name'],
                    'textual': poi_data.get('textual', '')[:100],
                    'num_venues': len(children),
                    'type': 'District'
                }
            else:  # level == 3
                # Region
                children = poi_data.get('children', [])
                poi_info = {
                    'name': poi_data['name'],
                    'textual': poi_data.get('textual', '')[:100],
                    'num_districts': len(children),
                    'type': 'Region'
                }
            
            recommendations.append((poi_id, score, poi_info))
        
        return recommendations
    
    def recommend_multi_granularity(self,
                                   user_id: str,
                                   levels: List[int] = [0, 1, 2, 3],
                                   top_k_per_level: int = 5,
                                   filter_visited: bool = True,
                                   **kwargs) -> Dict[int, List[Tuple[str, float, Dict]]]:
        """
        Generate recommendations at multiple granularity levels
        
        Args:
            user_id: User ID
            levels: List of levels to generate recommendations for
            top_k_per_level: Number of recommendations per level
            filter_visited: Filter visited POIs
        
        Returns:
            Dictionary mapping level -> recommendations
        """
        print(f"\n{'='*70}")
        print(f"MULTI-GRANULARITY RECOMMENDATIONS FOR USER: {user_id}")
        print(f"{'='*70}\n")
        
        results = {}
        
        level_names = {
            0: "INDIVIDUAL POIs",
            1: "CONTAINERS/VENUES",
            2: "DISTRICTS",
            3: "REGIONS"
        }
        
        for level in levels:
            print(f"Generating Level {level} ({level_names[level]}) recommendations...")
            recommendations = self.recommend_at_level(
                user_id=user_id,
                level=level,
                top_k=top_k_per_level,
                filter_visited=filter_visited,
                **kwargs
            )
            results[level] = recommendations
        
        return results
    
    def display_multi_granularity_recommendations(self, 
                                                 recommendations: Dict[int, List],
                                                 show_details: bool = True):
        """
        Pretty print multi-granularity recommendations
        
        Args:
            recommendations: Dict mapping level -> list of recommendations
            show_details: Show detailed information
        """
        level_names = {
            0: "LEVEL 0: INDIVIDUAL POIs (Specific Entities)",
            1: "LEVEL 1: CONTAINERS/VENUES (Malls, Buildings, etc.)",
            2: "LEVEL 2: DISTRICTS (Geographic Clusters)",
            3: "LEVEL 3: REGIONS (Large Areas)"
        }
        
        level_descriptions = {
            0: "Specific places you can visit right now",
            1: "Venues containing multiple places of interest",
            2: "Neighborhoods or districts to explore",
            3: "Broader regions for day trips"
        }
        
        print("\n" + "="*70)
        print("MULTI-GRANULARITY RECOMMENDATIONS")
        print("="*70)
        
        for level in sorted(recommendations.keys()):
            recs = recommendations[level]
            
            print(f"\n{level_names[level]}")
            print(f"({level_descriptions[level]})")
            print("-"*70)
            
            if not recs:
                print("  No recommendations at this level")
                continue
            
            for rank, (poi_id, score, poi_info) in enumerate(recs, 1):
                print(f"\n{rank}. {poi_info['name']}")
                print(f"   Score: {score:.4f}")
                print(f"   Type: {poi_info['type']}")
                
                if show_details:
                    if level == 0:
                        print(f"   Category: {poi_info.get('category', 'N/A')}")
                        print(f"   Price: {poi_info.get('price', 'N/A')}")
                        print(f"   Popularity: {poi_info.get('popularity', 'N/A')}")
                    elif level == 1:
                        print(f"   Contains: {poi_info.get('num_pois', 0)} POIs")
                        if poi_info.get('textual'):
                            print(f"   Description: {poi_info['textual']}")
                    elif level == 2:
                        print(f"   Contains: {poi_info.get('num_venues', 0)} venues")
                        if poi_info.get('textual'):
                            print(f"   Description: {poi_info['textual']}")
                    elif level == 3:
                        print(f"   Contains: {poi_info.get('num_districts', 0)} districts")
                        if poi_info.get('textual'):
                            print(f"   Description: {poi_info['textual']}")
    
    def recommend_adaptive_granularity(self,
                                      user_id: str,
                                      context: str = 'general',
                                      top_k: int = 10) -> List[Tuple[str, float, Dict, int]]:
        """
        Adaptively select granularity level based on context
        
        Args:
            user_id: User ID
            context: Context hint ('specific', 'venue', 'exploration', 'general')
            top_k: Total number of recommendations
        
        Returns:
            List of (poi_id, score, poi_info, level) tuples
        """
        # Context-based level distribution
        context_distributions = {
            'specific': {0: 0.8, 1: 0.15, 2: 0.05, 3: 0.0},
            'venue': {0: 0.3, 1: 0.5, 2: 0.15, 3: 0.05},
            'exploration': {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2},
            'general': {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}
        }
        
        distribution = context_distributions.get(context, context_distributions['general'])
        
        # Calculate number of recommendations per level
        recs_per_level = {
            level: max(1, int(top_k * weight))
            for level, weight in distribution.items() if weight > 0
        }
        
        # Generate recommendations at each level
        all_recommendations = []
        
        for level, k in recs_per_level.items():
            level_recs = self.recommend_at_level(user_id, level, top_k=k)
            for poi_id, score, poi_info in level_recs:
                all_recommendations.append((poi_id, score, poi_info, level))
        
        # Sort by score and return top-K
        all_recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return all_recommendations[:top_k]
    
    # ========================================================================
    # EXPORT & VISUALIZATION
    # ========================================================================
    
    def export_multi_granularity_recommendations(self,
                                                recommendations: Dict[int, List],
                                                user_id: str,
                                                output_prefix: str = 'multi_gran_recs'):
        """
        Export multi-granularity recommendations to CSV files
        
        Args:
            recommendations: Dict of recommendations by level
            user_id: User ID
            output_prefix: Prefix for output files
        """
        user_name = self.users_df[self.users_df['uudi'] == user_id].iloc[0]['name']
        
        for level, recs in recommendations.items():
            rows = []
            for rank, (poi_id, score, poi_info) in enumerate(recs, 1):
                row = {
                    'user_id': user_id,
                    'user_name': user_name,
                    'level': level,
                    'rank': rank,
                    'poi_id': poi_id,
                    'poi_name': poi_info['name'],
                    'score': score,
                    'type': poi_info['type']
                }
                
                # Add level-specific fields
                if level == 0:
                    row.update({
                        'category': poi_info.get('category', ''),
                        'price': poi_info.get('price', ''),
                        'popularity': poi_info.get('popularity', '')
                    })
                elif level == 1:
                    row['num_pois'] = poi_info.get('num_pois', 0)
                elif level == 2:
                    row['num_venues'] = poi_info.get('num_venues', 0)
                elif level == 3:
                    row['num_districts'] = poi_info.get('num_districts', 0)
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            output_file = f'{output_prefix}_level{level}_{user_name}.csv'
            df.to_csv(output_file, index=False)
            print(f"Level {level} recommendations exported to: {output_file}")

    # ========================================================================
    # EXPLAINABILITY 
    # ========================================================================

    def build_reason_flags(self, 
                      user_id: str, 
                      poi_id: str,
                      level: int = 0) -> Dict[str, bool]:
        """
        Build interpretable boolean flags explaining WHY this POI is recommended
        
        Args:
            user_id: User ID
            poi_id: POI ID
            level: Granularity level
        
        Returns:
            Dictionary of reason flags (True/False)
        """
        level_key = f'level_{level}'
        
        # Get user profile
        user_row = self.users_df[self.users_df['uudi'] == user_id].iloc[0]
        user_interests = set([i.strip().lower() for i in user_row['interests'].split(';')])
        user_area = user_row['area_of_residence']
        
        # Get POI data
        poi_data = self.poi_tree[level_key][poi_id]
        
        # Initialize flags
        flags = {}
        
        if level == 0:
            # Individual POI flags
            poi_category = poi_data['data'].get('category', '').lower()
            poi_chars = poi_data['data'].get('characteristic', '').lower()
            poi_popularity = float(poi_data['data'].get('popularity', 0))
            poi_price = poi_data['data'].get('price', '')
            
            # 1. Category match
            interest_category_map = {
                'food': ['restaurant', 'cafe', 'food_court', 'hawker_centre'],
                'shopping': ['shopping_mall', 'retail', 'store', 'boutique'],
                'movies': ['cinema', 'theatre'],
                'fitness': ['gym', 'sports', 'fitness'],
                'entertainment': ['cinema', 'arcade', 'karaoke', 'bar']
            }
            
            flags['matches_interest'] = any(
                any(cat in poi_category for cat in interest_category_map.get(interest, []))
                for interest in user_interests
            )
            
            # 2. Previously visited
            visited_pois = self.user_history.get(user_id, {}).get(0, [])
            flags['visited_before'] = poi_id in visited_pois
            
            # 3. Nearby (distance check)
            distance_penalty = self.compute_distance_penalty(user_id, poi_id)
            flags['nearby'] = distance_penalty > 0.7  # Within comfortable distance
            flags['very_nearby'] = distance_penalty > 0.9  # Very close
            
            # 4. Popular
            flags['popular'] = poi_popularity >= 4.0
            flags['highly_popular'] = poi_popularity >= 4.5
            
            # 5. Price match
            user_price_sens = user_row['price_sensitivity'].lower()
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
            
            # 7. Trending in area
            poi_district = poi_data['data'].get('district', '')
            flags['trending_in_area'] = False  # Placeholder - could compute from recent interactions
            
            # 8. Parent venue is popular (hierarchical)
            parent_id = poi_data.get('parent')
            if parent_id and parent_id in self.poi_id_to_idx[1]:
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
        
        Args:
            user_id: User ID
            poi_id: POI ID
            level: Granularity level
            reason_flags: Pre-computed reason flags (optional)
            score_components: Score breakdown (optional)
        
        Returns:
            Human-readable explanation string
        """
        level_key = f'level_{level}'
        
        # Get data
        user_row = self.users_df[self.users_df['uudi'] == user_id].iloc[0]
        user_name = user_row['name']
        poi_data = self.poi_tree[level_key][poi_id]
        poi_name = poi_data['name']
        
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
            user_interests = user_row['interests']
            if level == 0:
                poi_category = poi_data['data'].get('category', '')
                parts.append(f"✓ It matches your interests ({user_interests}) - it's a {poi_category}")
            else:
                parts.append(f"✓ It has options matching your interests ({user_interests})")
        
        # 4. Budget match
        if reason_flags.get('matches_budget'):
            parts.append(f"✓ Prices fit your budget ({user_row['price_sensitivity']} spending)")
        
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
            if parent_id:
                parent_name = self.poi_tree['level_1'][parent_id]['name']
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
                                    level: int = 0) -> Dict[str, any]:
        """
        Complete enhanced explanation combining:
        - Technical score breakdown
        - Interpretable reason flags
        - Natural language explanation
        
        Args:
            user_id: User ID
            poi_id: POI ID
            level: Granularity level
        
        Returns:
            Comprehensive explanation dictionary
        """
        if user_id not in self.user_id_to_idx or poi_id not in self.poi_id_to_idx[level]:
            return {}
        
        user_idx = self.user_id_to_idx[user_id]
        poi_idx = self.poi_id_to_idx[level][poi_id]
        level_key = f'level_{level}'
        
        # 1. Get technical score components
        feature_score = self.compute_feature_based_score(user_idx, poi_idx, level)
        graph_score = self.compute_graph_based_score(user_idx, poi_idx, level)
        hierarchical_score = self.compute_hierarchical_boost(poi_id, user_idx, level)
        
        if level == 0:
            distance_penalty = self.compute_distance_penalty(user_id, poi_id)
            interest_bonus = self.compute_interest_match(user_id, poi_id)
        else:
            distance_penalty = 1.0
            interest_bonus = 0.0
        
        total_score = (0.25 * feature_score + 
                    0.15 * graph_score + 
                    0.10 * hierarchical_score +
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
            'hierarchical': {
                'raw_score': hierarchical_score,
                'contribution': 0.10 * hierarchical_score,
                'weight': 0.10,
                'description': 'Popularity of the surrounding area'
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
            score_components.items(),
            key=lambda x: x[1]['contribution'] if isinstance(x[1], dict) else 0,
            reverse=True
        )
        top_factors = [name for name, comp in ranked_components[1:4]]  # Skip 'total_score'
        
        # 2. Get reason flags (interpretable)
        reason_flags = self.build_reason_flags(user_id, poi_id, level)
        
        # 3. Generate human explanation
        human_explanation = self.build_human_explanation(
            user_id, poi_id, level, reason_flags, score_components
        )
        
        # 4. Get POI details
        poi_data = self.poi_tree[level_key][poi_id]
        user_data = self.users_df[self.users_df['uudi'] == user_id].iloc[0]
        
        # 5. Get similar visited POIs
        similar_visited = []
        if level == 0:
            similar_visited = self._get_similar_visited_pois(user_id, poi_id)[:3]
        
        # 6. Construct complete explanation
        explanation = {
            # Technical breakdown (for developers/analysts)
            'score_breakdown': score_components,
            'top_contributing_factors': top_factors,
            
            # Interpretable reasons (for UI)
            'reason_flags': reason_flags,
            'active_reasons': [k for k, v in reason_flags.items() if v],
            
            # Natural language (for end users)
            'human_explanation': human_explanation,
            
            # Context
            'user_context': {
                'name': user_data['name'],
                'interests': user_data['interests'],
                'area': user_data['area_of_residence'],
                'price_sensitivity': user_data['price_sensitivity']
            },
            'poi_context': {
                'name': poi_data['name'],
                'type': 'Individual POI' if level == 0 else f'Level {level}',
            },
            
            # Supporting evidence
            'similar_visited_pois': similar_visited,
        }
        
        if level == 0:
            explanation['poi_context'].update({
                'category': poi_data['data'].get('category', 'N/A'),
                'price': poi_data['data'].get('price', 'N/A'),
                'popularity': poi_data['data'].get('popularity', 'N/A'),
                'characteristics': poi_data['data'].get('characteristic', ''),
            })
        
        return explanation
    
    def _get_similar_visited_pois(self, user_id: str, candidate_poi: str) -> List[Tuple[str, float]]:
        """
        Get user's visited POIs that are similar to candidate
        
        Args:
            user_id: User ID
            candidate_poi: Candidate POI ID
        
        Returns:
            List of (poi_id, similarity) tuples
        """
        visited_pois = self.user_history.get(user_id, [])
        
        if not visited_pois or candidate_poi not in self.poi_id_to_idx[0]:
            return []
        
        candidate_idx = self.poi_id_to_idx[0][candidate_poi]
        candidate_emb = self.Q_l['level_0'][candidate_idx]
        
        similarities = []
        
        for visited_poi in visited_pois:
            if visited_poi not in self.poi_id_to_idx[0]:
                continue
            
            visited_idx = self.poi_id_to_idx[0][visited_poi]
            visited_emb = self.Q_l['level_0'][visited_idx]
            
            # Cosine similarity
            min_len = min(len(candidate_emb), len(visited_emb))
            sim = np.dot(candidate_emb[:min_len], visited_emb[:min_len])
            sim /= (np.linalg.norm(candidate_emb[:min_len]) * np.linalg.norm(visited_emb[:min_len]) + 1e-10)
            
            poi_name = self.poi_tree['level_0'][visited_poi]['name']
            similarities.append((poi_name, float(sim)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities


    def display_explanation(self, explanation: Dict[str, any], detailed: bool = False):
        """
        Pretty print an explanation
        
        Args:
            explanation: Explanation dictionary from explain_recommendation_enhanced()
            detailed: Show technical details or just user-friendly explanation
        """
        print("\n" + "="*70)
        print("RECOMMENDATION EXPLANATION")
        print("="*70)
        
        # Always show human explanation
        print("\n" + explanation['human_explanation'])
        
        if detailed:
            # Show technical breakdown
            print("\n" + "-"*70)
            print("TECHNICAL BREAKDOWN")
            print("-"*70)
            
            print(f"\nTotal Score: {explanation['score_breakdown']['total_score']:.2f}/100")
            print(f"Top Contributing Factors: {', '.join(explanation['top_contributing_factors'])}")
            
            print("\nScore Components:")
            for name, comp in explanation['score_breakdown'].items():
                if name == 'total_score' or not isinstance(comp, dict):
                    continue
                print(f"  • {name}:")
                print(f"      Contribution: {comp['contribution']:.2f} (weight: {comp['weight']:.0%})")
                print(f"      Description: {comp['description']}")
            
            print("\nActive Reasoning Flags:")
            for flag in explanation['active_reasons']:
                print(f"  ✓ {flag}")
            
            if explanation.get('similar_visited_pois'):
                print("\nSimilar Places You've Visited:")
                for poi_name, similarity in explanation['similar_visited_pois']:
                    print(f"  • {poi_name} (similarity: {similarity:.2f})")

    def explain_all_recommendations(self,
                                    recommendations: Dict[int, List],
                                    user_id: str,
                                    detailed: bool = False,
                                    top_n_detailed: int = 1) -> None:
        """
        Display explanations for all recommendations across all levels
        
        Args:
            recommendations: Dict mapping level -> list of recommendations
            user_id: User ID
            detailed: Show detailed technical breakdown
            top_n_detailed: Number of recommendations per level to show detailed breakdown for
        """
        level_names = {
            0: "INDIVIDUAL POIs",
            1: "CONTAINERS/VENUES",
            2: "DISTRICTS",
            3: "REGIONS"
        }
        
        user_name = self.users_df[self.users_df['uudi'] == user_id].iloc[0]['name']
        
        print("\n" + "="*70)
        print(f"EXPLANATIONS FOR ALL RECOMMENDATIONS - User: {user_name}")
        print("="*70)
        
        for level in sorted(recommendations.keys()):
            recs = recommendations[level]
            
            print(f"\n{'='*70}")
            print(f"LEVEL {level}: {level_names[level]}")
            print(f"{'='*70}\n")
            
            if not recs:
                print("  No recommendations at this level\n")
                continue
            
            for rank, (poi_id, score, poi_info) in enumerate(recs, 1):
                print(f"\n{'-'*70}")
                print(f"#{rank}. {poi_info['name']} (Score: {score:.2f})")
                print(f"{'-'*70}")
                
                # Get explanation
                explanation = self.explain_recommendation_enhanced(user_id, poi_id, level)
                
                # Always show human explanation
                print(explanation['human_explanation'])
                
                # Show detailed breakdown for top N recommendations
                if detailed and rank <= top_n_detailed:
                    print(f"\n📊 DETAILED SCORE BREAKDOWN:")
                    
                    sb = explanation['score_breakdown']
                    print(f"   Total Score: {sb['total_score']:.2f}/100")
                    print(f"   Top Contributing Factors: {', '.join(explanation['top_contributing_factors'])}")
                    
                    print(f"\n   Components:")
                    for comp_name, comp_data in sb.items():
                        if comp_name == 'total_score' or not isinstance(comp_data, dict):
                            continue
                        print(f"      • {comp_name}: {comp_data['contribution']:.2f} "
                            f"(weight: {comp_data['weight']:.0%})")
                    
                    if explanation['active_reasons']:
                        print(f"\n   Active Flags:")
                        for flag in explanation['active_reasons']:
                            print(f"      ✓ {flag.replace('_', ' ').title()}")
                    
                    if explanation.get('similar_visited_pois'):
                        print(f"\n   Similar Places You've Visited:")
                        for poi_name, similarity in explanation['similar_visited_pois']:
                            print(f"      • {poi_name} (similarity: {similarity:.2f})")
                
                # For non-detailed, just show quick summary
                elif not detailed:
                    print(f"\n   📌 Key factors: {', '.join(explanation['top_contributing_factors'][:3])}")
                
                print()  # Spacing between recommendations
        
        print("="*70)
        print("ALL EXPLANATIONS COMPLETE!")
        print("="*70)

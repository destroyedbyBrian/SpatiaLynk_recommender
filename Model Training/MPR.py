import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiLevelRecommender(keras.Model):
    def __init__(self, levels=[0, 1, 2, 3], embedding_dim=128):
        super().__init__()
        self.levels = levels
        
        # Learnable fusion weights for each level
        self.alpha_weights = {
            level: self.add_weight(
                name=f'alpha_level_{level}',
                shape=(1,),
                initializer='ones',
                trainable=True
            ) for level in levels
        }
        
        self.beta_weights = {
            level: self.add_weight(
                name=f'beta_level_{level}',
                shape=(1,),
                initializer='ones',
                trainable=True
            ) for level in levels
        }
        
        # Optional: Add learnable projection layers per level
        self.projection_layers = {
            level: layers.Dense(embedding_dim, activation='relu', name=f'proj_level_{level}')
            for level in levels
        }
    
    def call(self, inputs, training=False):
        """
        inputs: dict with keys 'S_matrices', 'H_matrices', 'level'
        Returns combined prediction matrix for specified level
        """
        level = inputs['level']
        S_l = inputs['S_matrices'][f'level_{level}']  # Feature-based
        H_l = inputs['H_matrices'][f'level_{level}']  # Historical
        
        # Fusion with learned weights
        alpha = tf.nn.sigmoid(self.alpha_weights[level])  # Constrain to [0,1]
        beta = tf.nn.sigmoid(self.beta_weights[level])
        
        # Combined prediction
        Y_hat_l = alpha * S_l + beta * H_l
        
        return Y_hat_l
    
    def predict_top_k(self, user_id, level, S_matrices, H_matrices, 
                      poi_ids, user_ids, k=10, user_lat=None, user_lon=None):
        """
        Get top-k POI recommendations for a user at specific level
        """
        # Get user index
        user_idx = user_ids.index(user_id)
        
        # Combine matrices
        inputs = {
            'level': level,
            'S_matrices': S_matrices,
            'H_matrices': H_matrices
        }
        Y_hat = self.call(inputs, training=False).numpy()
        
        # Get scores for this user
        user_scores = Y_hat[user_idx, :]
        
        # Optional: Distance-based filtering/boosting
        if user_lat is not None and user_lon is not None:
            distance_weights = self._compute_distance_weights(
                user_lat, user_lon, level, poi_ids[f'level_{level}']
            )
            user_scores = user_scores * distance_weights
        
        # Get top-k POIs
        top_k_indices = np.argsort(user_scores)[-k:][::-1]
        top_k_poi_ids = [poi_ids[f'level_{level}'][idx] for idx in top_k_indices]
        top_k_scores = user_scores[top_k_indices]
        
        return list(zip(top_k_poi_ids, top_k_scores))
    
    def _compute_distance_weights(self, user_lat, user_lon, level, poi_ids_list):
        """
        Compute distance-based weights for POI scores
        Closer POIs get higher weights
        """
        # Load POI tree to get coordinates
        with open('poi_tree_with_uuids.json', 'r') as f:
            import json
            poi_tree = json.load(f)
        
        weights = np.ones(len(poi_ids_list))
        for idx, poi_id in enumerate(poi_ids_list):
            # Find POI in tree (search across all levels)
            poi_data = self._find_poi_in_tree(poi_tree, poi_id)
            if poi_data and 'lat' in poi_data and 'long' in poi_data:
                distance = self._haversine(
                    user_lat, user_lon, 
                    poi_data['lat'], poi_data['long']
                )
                # Decay function: closer = higher weight
                weights[idx] = np.exp(-distance / 5.0)  # 5km decay constant
        
        return weights
    
    def _find_poi_in_tree(self, poi_tree, poi_id):
        """Helper to find POI data in tree"""
        for level_data in poi_tree.values():
            for poi in level_data:
                if poi.get('uuid') == poi_id:
                    return poi.get('data', {})
        return None
    
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Calculate distance in km between two coordinates"""
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
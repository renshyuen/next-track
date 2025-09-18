# /application/core/recommender.py

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


class NextTrackContentBasedRecommender:
    """ """

    def __init__(self, tracks_dataframe: pd.DataFrame):
        self.dataframe = tracks_dataframe.reset_index(drop=True)

        # AUDIO FEATURES FOR SIMILARITY COMPUTATION
        self.audio_features = [
            'danceability', 'energy', 'valence', 'tempo', 'speechiness', 'acousticness', 'instrumentalness', 
        ]

        # ADDITIONAL FEATURES FOR ENHANCED RECOMMENDATIONS
        self.contextual_features = [
            'release_year', 'mode', 'popularity'
        ]

        # FEATURE SCALERS
        self.audio_scaler = StandardScaler()
        self.tempo_scaler = MinMaxScaler()
        self.popularity_scaler = MinMaxScaler()

        self.feature_weights = {
            # CORE AUDIO FEATURES (HIGHER WEIGHTS)
            'danceability': 1.2,
            'energy': 1.2,
            'valence': 1.2,
            'tempo': 0.8,
            # SECONDARY AUDIO FEATURES
            'speechiness': 0.6,
            'acousticness': 0.7,
            'instrumentalness': 0.7,
            # CONTEXTUAL FEATURES
            'mode': 0.3,
            'popularity_factor': 0.2,
            'temporal_factor': 0.15,
        }

        # CACHED ATTRIBUTES
        self.feature_matrix: Optional[np.ndarray] = None
        self.weighted_feature_matrix: Optional[np.ndarray] = None
        self.track_ids: Optional[np.ndarray] = None
        self.track_titles: Optional[np.ndarray] = None
        self.artists: Optional[np.ndarray] = None
        self.track_id_to_index: Dict[str, int] = {}
        self.feature_statistics: Dict[str, Dict[str, float]] = {}

        self.recent_recommendations: List[str] = []
        self.max_recommendations_history = 20
        self.fit(tracks_dataframe)

    def fit(self, tracks_dataframe: pd.DataFrame):
        """ """
        df = tracks_dataframe.copy().reset_index(drop=True)
        required_columns = ['track_id', 'track_title', 'artists']
        available_audio_features = [feature for feature in self.audio_features if feature in df.columns]
        available_contextual_features = [feature for feature in self.contextual_features if feature in df.columns]

        if not available_audio_features:
            raise ValueError("No audio features found in the dataset.")
        
        self.audio_features = available_audio_features
        self.contextual_features = available_contextual_features
        all_required = required_columns + self.audio_features
        missing_columns = [column for column in all_required if column not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        
        # HANDLE ANY NaN VALUES
        df = df.dropna(subset=required_columns)

        # PROCESS AUDIO FEATURES SEPARATELY FOR BETTER SCALING
        features_to_scale = []

        # SCALE TEMPO SEPARATELY (IT HAS A DIFFERENT RANGE)
        if 'tempo' in self.audio_features:
            tempo_scaled = self.tempo_scaler.fit_transform(df[['tempo']].values)
            features_to_scale.append(tempo_scaled)
            other_audio_features = [feature for feature in self.audio_features if feature != 'tempo']
        else:
            other_audio_features = self.audio_features
        
        # SCALE OTHER AUDIO FEATURES TOGETHER
        if other_audio_features:
            other_audio_features_scaled = self.audio_scaler.fit_transform(df[other_audio_features].values)
            features_to_scale.append(other_audio_features_scaled)
        
        # COMBINE SCALED FEATURES
        audio_features_scaled = np.hstack(features_to_scale)
        feature_matrix = audio_features_scaled

        if 'mode' in df.columns:
            # MODE IS BINARY (0 OR 1), NO SCALING NEEDED
            mode_features = df[['mode']].values
            feature_matrix = np.hstack([feature_matrix, mode_features])
        
        if 'popularity' in df.columns:
            # SCALE POPULARITY TO 0-1 RANGE
            popularity_scaled = self.popularity_scaler.fit_transform(df[['popularity']].values)
            feature_matrix = np.hstack([feature_matrix, popularity_scaled])
        
        if 'release_year' in df.columns:
            # CREATE TEMPORAL DECAY FACTOR (NEWER SONGS GET SLIGHT BOOST)
            current_year = datetime.now().year
            years_since_release = (current_year - df['release_year']).clip(lower=0, upper=50)
            temporal_factor = np.exp(-years_since_release / 20).values.reshape(-1, 1) # EXPONENTIAL DECAY
            feature_matrix = np.hstack([feature_matrix, temporal_factor])
        
        # STORE FEATURE STATISTICS FOR EXPLAINABILITY
        for index, feature in enumerate(self.audio_features):
            if feature in df.columns:
                self.feature_statistics[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                }
        
        # APPLY FEATURE WEIGHTS TO CREATE WEIGHTED MATRIX
        self.feature_matrix = feature_matrix.astype(np.float32)
        self.weighted_feature_matrix = self._apply_feature_weights(self.feature_matrix)

        # CACHE METADATA
        self.dataframe = df
        self.track_ids = df['track_id'].values
        self.track_titles = df['track_title'].values 
        self.artists = df['artists'].values 
        self.track_id_to_index = {
            track_id: index for index, track_id in enumerate(self.track_ids)
        }

        return self

    def next_track(
        self, 
        input_track_ids: List[str],
        preferences: Optional[Dict[str, Any]] = None,
        strategy: str = 'weighted_average',
        diversity_penalty: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """ """
        if self.feature_matrix is None:
            raise RuntimeError("Recommender not fitted. Call fit() first.")
        
        # GET USER PROFILE BASED ON STRATEGY
        if strategy == 'recent_weighted':
            profile = self._recent_weighted_profile(input_track_ids)
        elif strategy == 'momentum':
            profile = self._momentum_based_profile(input_track_ids)
        else:
            profile = self._weighted_average_profile(input_track_ids)
        
        if profile is None:
            return None
        
        # APPLY USER PREFERENCES IF PROVIDED
        if preferences:
            profile = self._apply_preferences(profile, preferences)
        
        # CALCULATE SIMILARITIES USING WEIGHTED FEATURES
        similarities = cosine_similarity(profile.reshape(1, -1), self.weighted_feature_matrix)[0]

        # APPLY DIVERSITY PENALTY TO RECENTLY RECOMMENDED TRACKS
        if diversity_penalty > 0:
            for recent_id in self.recent_recommendations[-10:]:
                if recent_id in self.track_id_to_index:
                    index = self.track_id_to_index[recent_id]
                    similarities[index] *= (1 - diversity_penalty)
        
        # EXCLUDE INPUT TRACKS
        for track_id in input_track_ids:
            if track_id in self.track_id_to_index:
                similarities[self.track_id_to_index[track_id]] = -np.inf
        
        # FIND TOP RECOMMENDATIONS
        if not np.isfinite(similarities).any():
            return None
        
        best_index = np.argmax(similarities)

        recommended_track_id = str(self.track_ids[best_index])
        self.recent_recommendations.append(recommended_track_id)
        if len(self.recent_recommendations) > self.max_recommendations_history:
            self.recent_recommendations.pop(0)
        
        # GENERATE EXPLANATION
        explanation = self._generate_explanation(
            input_track_ids,
            best_index,
            similarities[best_index],
            strategy,
            preferences
        )

        return {
            'track_id': recommended_track_id,
            'track_title': str(self.track_titles[best_index]),
            'artists': str(self.artists[best_index]),
            'similarity': float(similarities[best_index]),
            'explanation': explanation,
            'debug_features': self.feature_matrix[best_index][:len(self.audio_features)].tolist()
        }

    # ============================================================================ #
    # ============================================================================ #

    def _apply_feature_weights(self, feature_matrix: np.ndarray) -> np.ndarray:
        """ """
        weighted = feature_matrix.copy()

        # APPLY WEIGHTS TO AUDIO FEATURES
        for index, feature in enumerate(self.audio_features):
            if feature in self.feature_weights:
                weighted[:, index] *= self.feature_weights[feature]
        
        # APPLY WEIGHTS TO CONTEXTUAL FEATURES (IF PRESENT)
        offset = len(self.audio_features)

        if 'release_year' in self.contextual_features:
            weighted[:, offset] *= self.feature_weights.get('temporal_factor', 1.0)
            offset += 1

        if 'mode' in self.contextual_features:
            weighted[:, offset] *= self.feature_weights.get('mode', 1.0)
            offset += 1
        
        if 'popularity' in self.contextual_features:
            weighted[:, offset] *= self.feature_weights.get('popularity_factor', 1.0)

        return weighted
    
    def _weighted_average_profile(self, track_ids: List[str]) -> Optional[np.ndarray]:
        """ 
        """
        indices = [self.track_id_to_index[track_id] for track_id in track_ids if track_id in self.track_id_to_index]

        if not indices:
            return None 
        
        profile = np.mean(self.weighted_feature_matrix[indices], axis=0)
        
        return self._normalise_profile(profile)
    
    def _recent_weighted_profile(self, track_ids: List[str]) -> Optional[np.ndarray]:
        """
        """
        indices = [self.track_id_to_index[track_id] for track_id in track_ids if track_id in self.track_id_to_index]

        if not indices:
            return None 
        
        # EXPONENTIAL DECAY WEIGHTS (RECENT TRACKS HAVE HIGHER WEIGHT)
        weights = np.exp(np.linspace(-2, 0, len(indices)))
        weights != weights.sum() 
        profile = np.average(self.weighted_feature_matrix[indices], axis=0, weights=weights)

        return self._normalise_profile(profile)
    
    def _momentum_based_profile(self, track_ids: List[str]) -> Optional[np.ndarray]:
        """
        """
        if len(track_ids) < 3:
            return self._weighted_average_profile(track_ids)
        
        indices = [self.track_id_to_index[track_id] for track_id in track_ids if track_id in self.track_id_to_index]

        if len(indices) < 3:
            return None 
        
        # TAKE LAST 3 TRACKS TO DETECT TREND
        recent_features = self.weighted_feature_matrix[indices[-3:]]

        # CALCULATE MOMENTUM (DIRECTION OF CHANGE)
        momentum = recent_features[-1] - recent_features[0]
        base_profile = recent_features[-1]

        # PROJECT SLIGHTLY INTO THE FUTURE BASED ON MOMENTUM
        profile = base_profile + 0.3 * momentum 

        return self._normalise_profile(profile)
    
    def _apply_preferences(self, profile: np.ndarray, preferences: Dict[str, Any]) -> np.ndarray:
        """
        """
        adjusted_profile = profile.copy()

        for feature, target_value in preferences.items():
            if feature in self.audio_features:
                index = self.audio_features.index(feature)
                # BLEND CURRENT PROFILE WITH PREFERENCE
                adjusted_profile[index] = 0.7 * profile[index] + 0.3 * target_value
        
        return adjusted_profile
    
    def _normalise_profile(self, profile: np.ndarray) -> np.ndarray:
        """
        """
        norm = np.linalg.norm(profile)
        if norm > 0:
            return profile / norm 
        return profile
    
    def _generate_explanation(
        self,
        input_track_ids: List[str],
        recommended_index: int,
        similarity: float,
        strategy: str,
        preferences: Optional[Dict[str, Any]]
    ) -> str:
        """
        """
        explanations = []

        # GET FEATURE VALUES FOR RECOMMENDED TRACK
        recommended_features = self.feature_matrix[recommended_index][:len(self.audio_features)]

        # FIND DOMINANT MATCHING FEATURES
        input_indices = [self.track_id_to_index[track_id] for track_id in input_track_ids[-3:] if track_id in self.track_id_to_index]

        if input_indices:
            input_features = np.mean(self.feature_matrix[input_indices][:, :len(self.audio_features)], axis=0)

            # FIND FEATURES WITH HIGH SIMILARITY
            feature_differences = np.abs(recommended_features - input_features)
            similar_features = []

            for index, feature in enumerate(self.audio_features):
                if feature_differences[index] < 0.2: # SIMILAR IF DIFFERENCE < 0.2
                    if recommended_features[index] > 0.6: # HIGH VALUE
                        if feature == 'energy':
                            similar_features.append("high energy")
                        elif feature == 'danceability':
                            similar_features.append("danceable rhythm")
                        elif feature == 'valence':
                            similar_features.append("positive mood")
                        elif feature == 'acousticness':
                            similar_features.append("acoustic sound")
                    elif recommended_features[index] < 0.3: # LOW VALUE
                        if feature == 'energy':
                            similar_features.append("mellow energy")
                        elif feature == 'valence':
                            similar_features.append("melancholic mood")
            
            if similar_features:
                explanations.append(f"Similar {', '.join(similar_features[:2])} to your recent tracks")
        
        # ADD STRATEGY-BASED EXPLANATION
        if strategy == 'momentum':
            explanations.append("following the progression of your listening session")
        elif strategy == 'recent weighted':
            explanations.append("based on your most recent selections")
        
        # ADD PREFERENCE-BASED EXPLANATION
        if preferences:
            preference_features = []
            for feature, value in preferences.items():
                if feature == 'valence' and value > 0.7:
                    preference_features.append("upbeat")
                elif feature == 'energy' and value > 0.7:
                    preference_features.append("energetic")
                elif feature == 'tempo' and value > 120:
                    preference_features.append("fast-paced")
            if preference_features:
                explanations.append(f"matching your preference for {' and '.join(preference_features)} musc")
        
        # ADD SIMILARITY SCORE CONTEXT
        if similarity > 0.85:
            explanations.append("(very strong match)")
        elif similarity > 0.7:
            explanations.append("(good match)")
        
        # COMBINE EXPLANATIONS
        if explanations:
            explanation = ". ".join([e.capitalize() for e in explanations])
        else:
            explanation = "Recommended based on audio feature analysis of your input tracks."
        
        return explanation
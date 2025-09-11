# /application/core/recommender.py

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class NextTrackContentBasedRecommender:
    """
    DESCRIPTION:
        A content-based music recommender system that suggests the next track based on audio features.

    ATTRIBUTES:
        attribute1 (type): Description of attribute1.
        attribute2 (type): Description of attribute2.
    """

    def __init__(self, tracks_dataframe: pd.DataFrame):
        """
        DESCRIPTION:
            Creates an instance of the recommender that takes in a dataframe containing track information.

        PARAMETERS:
            param1 (type): Description of param1.
            param2 (type): Description of param2.
        """
        self.dataframe = tracks_dataframe.reset_index(drop=True)
        self.audio_features = ['danceability', 'energy', 'valence', 'tempo']
        self.scaler = StandardScaler() 

        # learned / cached attributes
        self.feature_matrix: Optional[np.ndarray] = None     # [num_tracks, num_features], float32
        self.track_ids: Optional[np.ndarray] = None    # [num_tracks]
        self.track_titles: Optional[np.ndarray] = None      # [num_tracks]
        self.artist_names: Optional[np.ndarray] = None          
        self.track_id_to_index: Dict[str, int] = {}     # map: track_id -> row index in the dataframe

        self.fit(tracks_dataframe)

    # == PUBLIC API 

    @classmethod
    def from_csv(cls, csv_path: str):
        """
        DESCRIPTION:
            Alternative constructor to create an instance from a CSV file.

        PARAMETERS:
            param1 (type): Description of param1.
            param2 (type): Description of param2.
        """
        df = pd.read_csv(csv_path)
        return cls(tracks_dataframe=df)

    def fit(self, tracks_dataframe: pd.DataFrame):
        """
        DESCRIPTION:
            Fit the recommender model to the provided tracks dataframe.
            This method cleans the data, extracts and scales audio features, 
            and caches necessary attributes for faster lookup during recommendation.

        PARAMETERS:
            param1 (type): Description of param1.
        """
        # ensure required columns exist
        required_columns = {'track_id', 'track_name', 'artists', *self.audio_features}
        is_missing = required_columns - set(tracks_dataframe.columns)
        if is_missing:
            raise ValueError(f"Dataset is missing columns: {sorted(is_missing)}")
        
        # clean & align
        df = tracks_dataframe.dropna(subset=['track_id', 'track_name', 'artists', *self.audio_features]).copy()
        df['track_id'] = df['track_id'].astype(str)
        df['track_name'] = df['track_name'].astype(str)
        df['artists'] = df['artists'].astype(str)
        df = df.reset_index(drop=True)

        # extract & scale features
        X = df[self.audio_features].astype(np.float32).values
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        # cache learned attributes
        self.dataframe = df    # keep a copy of cleaned dataframe
        self.feature_matrix = X_scaled
        self.track_ids = df['track_id'].values
        self.track_titles = df['track_name'].values
        self.artist_names = df['artists'].values
        self.track_id_to_index = {track_id: index for index, track_id in enumerate(self.track_ids)}

        return self
    
    def next_track(self, input_track_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        DESCRIPTION:
            Given a list of input track identifiers, recommend the next track that is most similar
            to the average profile of the input tracks, excluding the input tracks themselves.

        PARAMETERS:
            param1 (type): Description of param1.
        """
        if self.feature_matrix is None or self.track_ids is None:
            raise RuntimeError("Recommender is not fitted. Call fit() or from_csv() first.")
        
        profile = self._profile_vector(input_track_ids)
        if profile is None:
            return None
        
        similarities = cosine_similarity(profile.reshape(1, -1), self.feature_matrix)[0]

        # exclude input tracks from recommendations
        recommendation_candidates = np.ones_like(similarities, dtype=bool)
        for track_id in input_track_ids:
            index = self.track_id_to_index.get(track_id)
            if index is not None:
                recommendation_candidates[index] = False
        
        k_internal: int = 50
        valid_similarities = np.where(recommendation_candidates, similarities, -np.inf)
        if not np.isfinite(valid_similarities).any(): # if all candidates are excluded
            return None
        k = min(k_internal, valid_similarities.size)
        top_k_indices = np.argpartition(valid_similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(valid_similarities[top_k_indices])[::-1]]

        index = int(top_k_indices[0])
        return {
            'track_id': str(self.track_ids[index]),
            'track_title': str(self.track_titles[index]),
            'artist_name': str(self.artist_names[index]),
            'similarity': float(valid_similarities[index]),
            'z_score_features': self.feature_matrix[index].tolist(),    # for debugging
        }
        
    
    # == INTERNAL HELPERS

    def _profile_vector(self, input_track_ids: List[str]) -> Optional[np.ndarray]:
        """
        DESCRIPTION:
            Compute the average profile vector from the audio features of the input tracks.

        PARAMETERS:
            param1 (type): Description of param1.
        """
        indices = [self.track_id_to_index[track_id] for track_id in input_track_ids if track_id in self.track_id_to_index]
        if not indices:
            return None
        
        profile = np.mean(self.feature_matrix[indices], axis=0)     # [num_features]

        # L2 normalise so that cosine similarity is equivalent to dot product after weighting
        normalised = np.linalg.norm(profile)
        if normalised > 0:
            profile = profile / normalised
        
        return profile.astype(np.float32)
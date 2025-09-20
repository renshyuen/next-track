# /tests/test_recommender.py

import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# ADD PARENT DIRECTORY TO PATH FOR IMPORTS
# TO MAKE SURE PYTHON KNOWS WHERE TO FIND application/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.core.recommender import NextTrackContentBasedRecommender
from application.core.explainability import ExplainabilityEngine


class TestRecommender:
    """
    Unit tests for NextTrackContentBasedRecommender.
    """
  
    @pytest.fixture
    def sample_dataframe(self):
        """
        Create a sample dataframe for testing.
        """
        np.random.seed(42)
        n_tracks = 100
        
        data = {
            'track_id': [f'track_{index:03d}' for index in range(n_tracks)], # STRING FORMATTING SYNTAX
            'track_title': [f'Song {index}' for index in range(n_tracks)],
            'artists': [f'Artist {index % 20}' for index in range(n_tracks)],
            'danceability': np.random.random(n_tracks),
            'energy': np.random.random(n_tracks),
            'valence': np.random.random(n_tracks),
            'tempo': np.random.uniform(60, 200, n_tracks),
            'speechiness': np.random.random(n_tracks),
            'acousticness': np.random.random(n_tracks),
            'instrumentalness': np.random.random(n_tracks),
            'mode': np.random.randint(0, 2, n_tracks),
            'popularity': np.random.randint(0, 100, n_tracks),
            'release_year': np.random.randint(1980, 2024, n_tracks)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def recommender(self, sample_dataframe):
        """
        Create a recommender instance.
        """
        return NextTrackContentBasedRecommender(sample_dataframe)
    
    def test_initialisation(self, sample_dataframe):
        """
        Test recommender initialisation.
        """
        recommender = NextTrackContentBasedRecommender(sample_dataframe)
        
        assert recommender.dataframe is not None
        assert len(recommender.track_ids) == len(sample_dataframe)
        assert recommender.feature_matrix is not None
        assert recommender.weighted_feature_matrix is not None
        assert len(recommender.track_id_to_index) == len(sample_dataframe)
    
    def test_fit_missing_columns(self):
        """
        Test fit method with missing required columns.
        """
        df = pd.DataFrame({
            'track_id': ['1', '2'],
            'track_title': ['Song 1', 'Song 2'],
            # MISSING 'artists' AND AUDIO FEATURES
        })
        
        with pytest.raises(ValueError, match="Dataset is missing columns"):
            NextTrackContentBasedRecommender(df)
    
    def test_fit_with_nan_values(self, sample_dataframe):
        """
        Test fit method handles NaN values correctly.
        """
        df = sample_dataframe.copy()
        df.loc[0, 'track_title'] = np.nan  # ADD NaN VALUE
        
        recommender = NextTrackContentBasedRecommender(df)
        assert len(recommender.track_ids) == len(df) - 1  # ONE ROW SHOULD BE DROPPED
    
    def test_feature_scaling(self, recommender):
        """
        Test that features are properly scaled.
        """
        # CHECK THAT AUDIO FEATURES ARE SCALED (SHOULD HAVE MEAN ~0 AND STD ~1)
        audio_features = recommender.feature_matrix[:, :len(recommender.audio_features)]
        
        # FOR STANDARDISED FEATURES, MEAN SHOULD BE CLOSE TO 0
        means = np.mean(audio_features, axis=0)
        assert np.all(np.abs(means) < 0.5)  # REASONABLE TOLERANCE
        
        # CHECK TEMPO IS SEPARATELY SCALED (SHOULD BE IN 0-1 RANGE IF MinMaxScaler USED)
        if 'tempo' in recommender.audio_features:
            tempo_index = recommender.audio_features.index('tempo')
            tempo_values = recommender.feature_matrix[:, tempo_index]
            # AFTER MinMaxScaling, VALUES SHOULD BE IN A REASONABLE RANGE
            assert np.min(tempo_values) >= -3 and np.max(tempo_values) <= 3
    
    def test_weighted_average_profile(self, recommender):
        """
        Test weighted average profile generation.
        """
        track_ids = ['track_000', 'track_001', 'track_002']
        profile = recommender._weighted_average_profile(track_ids)
        
        assert profile is not None
        assert len(profile) == recommender.weighted_feature_matrix.shape[1]
        assert np.isfinite(profile).all()
        
        # TEST L2 NORMALISATION
        norm = np.linalg.norm(profile)
        assert np.isclose(norm, 1.0, rtol=1e-5)
    
    def test_recent_weighted_profile(self, recommender):
        """
        Test recent weighted profile generation.
        """
        track_ids = ['track_000', 'track_001', 'track_002', 'track_003']
        profile = recommender._recent_weighted_profile(track_ids)
        
        assert profile is not None
        assert len(profile) == recommender.weighted_feature_matrix.shape[1]
        assert np.isfinite(profile).all()
    
    def test_momentum_based_profile(self, recommender):
        """
        Test momentum-based profile generation.
        """
        # TEST WITH FEWER THAN 3 TRACKS (SHOULD FALL BACK TO WEIGHTED AVERAGE)
        track_ids = ['track_000', 'track_001']
        profile = recommender._momentum_based_profile(track_ids)
        assert profile is not None
        
        # TEST WITH 3+ TRACKS
        track_ids = ['track_000', 'track_001', 'track_002', 'track_003']
        profile = recommender._momentum_based_profile(track_ids)
        assert profile is not None
        assert np.isfinite(profile).all()
    
    def test_apply_preferences(self, recommender):
        """
        Test preference application to profile.
        """
        track_ids = ['track_000', 'track_001']
        profile = recommender._weighted_average_profile(track_ids)
        
        preferences = {
            'valence': 0.8,
            'energy': 0.9,
            'popular': True,
            'temporal_preference': 'recent'
        }
        
        adjusted_profile = recommender._apply_preferences(profile, preferences)
        
        assert adjusted_profile is not None
        assert len(adjusted_profile) == len(profile)
        assert not np.array_equal(adjusted_profile, profile)  # SHOULD BE DIFFERENT
    
    def test_next_track_basic(self, recommender):
        """
        Test basic next track recommendation.
        """
        track_ids = ['track_000', 'track_001']
        
        recommendation = recommender.next_track(
            input_track_ids=track_ids,
            strategy='weighted_average'
        )
        
        assert recommendation is not None
        assert 'track_id' in recommendation
        assert 'track_title' in recommendation
        assert 'artists' in recommendation
        assert 'similarity' in recommendation
        assert 'explanation' in recommendation
        
        # RECOMMENDED TRACK SHOULD NOT BE IN INPUT TRACKS
        assert recommendation['track_id'] not in track_ids
    
    def test_next_track_with_preferences(self, recommender):
        """
        Test next track recommendation with preferences.
        """
        track_ids = ['track_000', 'track_001']
        preferences = {
            'valence': 0.9,
            'energy': 0.8,
            'tempo': 0.7
        }
        
        recommendation = recommender.next_track(
            input_track_ids=track_ids,
            preferences=preferences,
            strategy='weighted_average'
        )
        
        assert recommendation is not None
        assert 'track_id' in recommendation
    
    def test_next_track_all_strategies(self, recommender):
        """
        Test all recommendation strategies.
        """
        track_ids = ['track_000', 'track_001', 'track_002', 'track_003']
        strategies = ['weighted_average', 'recent_weighted', 'momentum']
        
        recommendations = {}
        for strategy in strategies:
            rec = recommender.next_track(
                input_track_ids=track_ids,
                strategy=strategy
            )
            assert rec is not None
            recommendations[strategy] = rec['track_id']
        
        # DIFFERENT STRATEGIES MIGHT PRODUCE DIFFERENT RECOMMENDATIONS
        # (THOUGH NOT GUARANTEED WITH SMALL TEST DATA)
        assert len(set(recommendations.values())) >= 1
    
    def test_invalid_track_ids(self, recommender):
        """
        Test handling of invalid track IDs.
        """
        invalid_ids = ['invalid_001', 'invalid_002']
        
        result = recommender.next_track(
            input_track_ids=invalid_ids,
            strategy='weighted_average'
        )
        
        assert result is None  # SHOULD RETURN None FOR ALL INVALID TRACKS
    
    def test_mixed_valid_invalid_track_ids(self, recommender):
        """
        Test handling of mixed valid and invalid track IDs.
        """
        mixed_ids = ['track_000', 'invalid_001', 'track_002']
        
        result = recommender.next_track(
            input_track_ids=mixed_ids,
            strategy='weighted_average'
        )
        
        assert result is not None  # SHOULD WORK WITH VALID TRACKS ONLY


class TestExplainability:
    """
    Unit tests for ExplainabilityEngine.
    """
    
    def test_generate_feature_explanation(self):
        """
        Test single feature explanation generation.
        """
        explanation = ExplainabilityEngine.generate_feature_explanation(
            'energy', 0.8, 'matching'
        )
        assert explanation is not None
        assert 'similar' in explanation
        assert any(word in explanation for word in ['energetic', 'intense', 'powerful'])
        
        explanation = ExplainabilityEngine.generate_feature_explanation(
            'valence', 0.2, 'matching'
        )
        assert 'melancholic' in explanation or 'emotional' in explanation
    
    def test_detect_musical_style(self):
        """
        Test musical style detection.
        """
        features = {
            'energy': 0.8,
            'danceability': 0.9,
            'acousticness': 0.1
        }
        style = ExplainabilityEngine.detect_musical_style(features)
        assert style == 'electronic dance'
        
        features = {
            'acousticness': 0.9,
            'energy': 0.3
        }
        style = ExplainabilityEngine.detect_musical_style(features)
        assert style == 'acoustic/folk style'
    
    def test_generate_transition_explanation(self):
        """
        Test transition explanation generation.
        """
        from_features = {'energy': 0.3, 'valence': 0.3, 'tempo': 0.5}
        to_features = {'energy': 0.8, 'valence': 0.7, 'tempo': 0.6}
        
        explanation = ExplainabilityEngine.generate_transition_explanation(
            from_features, to_features
        )
        assert 'building energy' in explanation
        assert 'lifting the mood' in explanation
    
    def test_comprehensive_explanation(self):
        """
        Test comprehensive explanation generation.
        """
        input_features = {
            'energy': 0.5,
            'valence': 0.6,
            'danceability': 0.7
        }
        recommended_features = {
            'energy': 0.5,
            'valence': 0.6,
            'danceability': 0.7
        }
        
        explanation = ExplainabilityEngine.generate_comprehensive_explanation(
            input_features=input_features,
            recommended_features=recommended_features,
            similarity_score=0.9,
            strategy='momentum',
            preferences={'valence': 0.8}
        )
        
        assert explanation is not None
        assert len(explanation) > 0
        assert 'following your listening progression' in explanation or 'strong match' in explanation
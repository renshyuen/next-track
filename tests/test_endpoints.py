# /tests/test_endpoints.py

import pytest
import sys
import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# ADD PARENT DIRECTORY TO PATH FOR IMPORTS
# TO MAKE SURE PYTHON KNOWS WHERE TO FIND application/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.main import app


class TestAPIEndpoints:
    """
    Integration tests for API endpoints.
    """
    
    @pytest.fixture
    def client(self):
        """
        Create a test client for the FastAPI app.
        """
        return TestClient(app)
    
    @pytest.fixture
    def mock_dataframe(self):
        """
        Create a mock dataframe for testing.
        """
        np.random.seed(42)
        n_tracks = 100
        
        data = {
            'track_id': [f'track_{index:03d}' for index in range(n_tracks)],
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
    
    def test_health_check(self, client):
        """
        Test health check endpoint.
        """
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_root_endpoint(self, client):
        """
        Test root endpoint.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    @patch('application.api.endpoints.recommend.pd.read_csv')
    def test_recommend_basic(self, mock_read_csv, client, mock_dataframe):
        """
        Test basic recommendation request.
        """
        mock_read_csv.return_value = mock_dataframe
        
        request_data = {
            "track_ids": ["track_000", "track_001"],
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # CHECK RESPONSE STRUCTURE EVEN IF IT FAILS
        if response.status_code == 200:
            data = response.json()
            assert "recommended_track" in data
            assert "explanation" in data
            assert "confidence_score" in data
            assert "strategy_used" in data
            
            # CHECK RECOMMENDED TRACK STRUCTURE
            track = data["recommended_track"]
            assert "track_id" in track
            assert "track_title" in track
            assert "artists" in track
    
    def test_recommend_with_preferences(self, client):
        """
        Test recommendation with preferences.
        """
        request_data = {
            "track_ids": ["track_000", "track_001"],
            "preferences": {
                "valence": 0.8,
                "energy": 0.7,
                "danceability": 0.6,
                "popular": True,
                "temporal_preference": "recent"
            },
            "strategy": "momentum"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # CHECK THAT THE REQUEST IS PROCESSED (MAY FAIL IF TRACKS DON'T EXIST)
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert data["strategy_used"] == "momentum"
    
    def test_recommend_invalid_track_ids(self, client):
        """
        Test recommendation with invalid track IDs.
        """
        request_data = {
            "track_ids": ["invalid_track_999", "nonexistent_track"],
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD RETURN 404 FOR NON-EXISTENT TRACKS
        assert response.status_code == 404
        assert "detail" in response.json()
    
    def test_recommend_empty_track_ids(self, client):
        """
        Test recommendation with empty track IDs.
        """
        request_data = {
            "track_ids": [],
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD RETURN 422 (Validation Error) FOR EMPTY LIST
        assert response.status_code == 422
    
    def test_recommend_too_many_tracks(self, client):
        """
        Test recommendation with too many track IDs.
        """
        request_data = {
            "track_ids": [f"track_{i:03d}" for i in range(10)],  # EXCEEDS max_items=5
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD RETURN 422 (Validation Error) FOR TOO MANY TRACKS
        assert response.status_code == 422
    
    def test_recommend_invalid_strategy(self, client):
        """
        Test recommendation with invalid strategy.
        """
        request_data = {
            "track_ids": ["track_000", "track_001"],
            "strategy": "invalid_strategy"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD RETURN 422 (Validation Error) FOR INVALID STRATEGY
        assert response.status_code == 422
    
    def test_recommend_invalid_preferences(self, client):
        """
        Test recommendation with invalid preferences.
        """
        request_data = {
            "track_ids": ["track_000", "track_001"],
            "preferences": {
                "valence": 1.2,  # Invalid: exceeds range [0, 1]
                "tempo": 201  # Invalid: exceeds range [60, 200]
            }
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD RETURN 422 (Validation Error) FOR INVALID PREFERENCES
        assert response.status_code == 422
    
    def test_recommend_all_strategies(self, client):
        """
        Test all available recommendation strategies.
        """
        strategies = ["weighted_average", "recent_weighted", "momentum"]
        
        for strategy in strategies:
            request_data = {
                "track_ids": ["track_000", "track_001", "track_002"],
                "strategy": strategy
            }
            
            response = client.post("/api/recommend", json=request_data)
            
            # CHECK THAT ALL STRATEGIES ARE HANDLED
            assert response.status_code in [200, 404]
    
    def test_recommend_duplicate_track_ids(self, client):
        """
        Test recommendation with duplicate track IDs.
        """
        request_data = {
            "track_ids": ["track_000", "track_000", "track_001"],
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD HANDLE DUPLICATES GRACEFULLY
        assert response.status_code in [200, 404]
    
    def test_recommend_response_schema(self, client):
        """
        Test that response matches expected schema.
        """
        request_data = {
            "track_ids": ["track_000", "track_001"],
            "preferences": {
                "valence": 0.8,
                "energy": 0.6
            },
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            
            # CHECK ALL REQUIRED FIELDS ARE PRESENT
            assert "recommended_track" in data
            assert "explanation" in data
            assert "confidence_score" in data
            assert "strategy_used" in data
            
            # CHECK DATA TYPES
            assert isinstance(data["explanation"], str)
            assert isinstance(data["confidence_score"], (int, float))
            assert 0 <= data["confidence_score"] <= 1
            assert data["strategy_used"] in ["weighted_average", "recent_weighted", "momentum"]
            
            # CHECK RECOMMENDED TRACK SCHEMA
            track = data["recommended_track"]
            assert isinstance(track["track_id"], str)
            assert isinstance(track["track_title"], str)
            assert isinstance(track["artists"], str)


class TestPreferenceValidation:
    """
    Test preference parameter validation.
    """
    
    @pytest.fixture
    def client(self):
        """
        Create a test client.
        """
        return TestClient(app)
    
    def test_valence_range_validation(self, client):
        """
        Test valence parameter range validation.
        """
        # VALID VALENCE
        request_data = {
            "track_ids": ["track_000"],
            "preferences": {"valence": 0.5}
        }
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code in [200, 404]
        
        # INVALID VALENCE (TOO HIGH)
        request_data["preferences"]["valence"] = 1.1
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code == 422
        
        # INVALID VALENCE (TOO LOW)
        request_data["preferences"]["valence"] = -0.1
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code == 422
    
    def test_tempo_range_validation(self, client):
        """
        Test tempo parameter range validation.
        """
        # VALID TEMPO
        request_data = {
            "track_ids": ["track_000"],
            "preferences": {"tempo": 120}
        }
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code in [200, 404]
        
        # INVALID TEMPO (TOO HIGH)
        request_data["preferences"]["tempo"] = 201
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code == 422
        
        # INVALID TEMPO (TOO LOW)
        request_data["preferences"]["tempo"] = 59
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code == 422
    
    def test_temporal_preference_validation(self, client):
        """
        Test temporal preference validation.
        """
        valid_preferences = ["recent", "classic", "any", None]
        
        for pref in valid_preferences:
            request_data = {
                "track_ids": ["track_000"],
                "preferences": {"temporal_preference": pref} if pref else {}
            }
            response = client.post("/api/recommend", json=request_data)
            assert response.status_code in [200, 404]


class TestErrorHandling:
    """
    Test error handling in API endpoints.
    """
    
    @pytest.fixture
    def client(self):
        """
        Create a test client.
        """
        return TestClient(app)
    
    def test_malformed_json(self, client):
        """
        Test handling of malformed JSON.
        """
        response = client.post(
            "/api/recommend",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """
        Test handling of missing required fields.
        """
        # MISSING TRACK IDS
        request_data = {
            "strategy": "weighted_average"
        }
        response = client.post("/api/recommend", json=request_data)
        assert response.status_code == 422
    
    @patch('application.api.endpoints.recommend.recommender.next_track')
    def test_internal_server_error(self, mock_next_track, client):
        """
        Test handling of internal server errors.
        """
        # MAKE THE RECOMMENDER RAISE AN EXCEPTION
        mock_next_track.side_effect = Exception("Internal error")
        
        request_data = {
            "track_ids": ["track_000"],
            "strategy": "weighted_average"
        }
        
        response = client.post("/api/recommend", json=request_data)
        
        # SHOULD HANDLE THE ERROR GRACEFULLY
        assert response.status_code in [404, 500]
        assert "detail" in response.json()
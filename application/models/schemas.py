# /application/models/schemas.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any 
from enum import Enum


class TrackInfo(BaseModel):
    """
    Information about a track.
    """
    track_id: str
    track_title: str
    artists: str


class RecommendationStrategy(str, Enum):
    """
    Available recommendation strategies.
    """
    WEIGHTED_AVERAGE = "weighted_average"
    RECENT_WEIGHTED = "recent_weighted" # HIGHER WEIGHT ON RECENT TRACKS
    MOMENTUM = "momentum"


class PreferenceParameters(BaseModel):
    """
    Optional preference parameters for fine-tuning recommendations.
    """
    # AUDIO FEATURES PREFERENCES (0 - 1 RANGE)
    valence: Optional[float] = Field(None, ge=0, le=1, description="Musical positivity (0=sad, 1=happy)")
    energy: Optional[float] = Field(None, ge=0, le=1, description="Energy level (0=low, 1=high)")
    danceability: Optional[float] = Field(None, ge=0, le=1, description="How suitable for dancing")
    acousticness: Optional[float] = Field(None, ge=0, le=1, description="Acoustic vs electronic (0=electronic, 1=acoustic)")
    instrumentalness: Optional[float] = Field(None, ge=0, le=1, description="Vocal vs instrumental (0=vocal, 1=instrumental)")

    # TEMPO PREFERENCE (BPM)
    tempo: Optional[float] = Field(None, ge=60, le=200, description="Preferred tempo in BPM")

    # ADDITIONAL PREFERENCES
    popular: Optional[bool] = Field(False, description="Prefer more popular tracks")
    temporal_preference: Optional[str] = Field(None, description="Preference for track era: 'recent', 'classic', 'any'")

    @validator('valence', 'energy', 'danceability', 'acousticness', 'instrumentalness')
    def validate_audio_feature(cls, v):
        if v is not None and not (0 <= v <= 1):
            raise ValueError("Audio feature preferences must be between 0 and 1.")
        return v
    
    @validator('tempo')
    def validate_tempo(cls, v):
        if v is not None and not (60 <= v <= 200):
            raise ValueError("Tempo preference must be between 60 and 200 BPM.")
        return v


class TrackRequest(BaseModel):
    """
    Request model for track recommendation.
    """
    track_ids: List[str] = Field(..., min_items=1, max_items=5, description="List of track IDs from listening history")
    preferences: Optional[PreferenceParameters] = Field(None, description="Optional preference parameters")
    strategy: Optional[RecommendationStrategy] = Field(RecommendationStrategy.WEIGHTED_AVERAGE, description="Recommendation strategy to use")
    
    class Config:
        schema_extra = {
            "example": {
                "track_ids": ["5rgu12WBIHQtvej2MdHSH0", "0NuWgxEp51CutD2pJoF4OM"],
                "preferences": {
                    "valence": 0.8,
                    "energy": 0.7,
                    "popular": True
                },
                "strategy": "recent_weighted"
            }
        }


class RecommendationResponse(BaseModel):
    """
    Response model for track recommendation.
    """
    recommended_track: TrackInfo
    explanation: str
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence in the recommendation (0-1 range)")
    strategy_used: Optional[RecommendationStrategy] = Field(None, description="Strategy used for recommendation")

    class Config:
        schema_extra = {
            "example": {
                "recommended_track": {
                    "track_id": "3n3Ppam7vgaVa1iaRUc9Lp",
                    "track_title": "Mr. Brightside",
                    "artists": "The Killers"
                },
                "explanation": "Similar high energy and positive mood to your recent tracks. Following the progression of your listening session (very strong match).",
                "confidence_score": 0.87,
                "strategy_used": "momentum"
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response model.
    """
    detail: str
    error_type: Optional[str] = None
    suggestions: Optional[List[str]] = None
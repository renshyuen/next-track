# /application/models/schemas.py

from pydantic import BaseModel
from typing import List


class TrackInfo(BaseModel):
    track_id: str
    track_title: str
    artist_name: str


class TrackRequest(BaseModel):
    track_ids: List[str]
    

class RecommendationResponse(BaseModel):
    recommended_track: TrackInfo
    explanation: str

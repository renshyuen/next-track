from pydantic import BaseModel
from typing import List


# ====== Input Schemas ======

class TrackInfo(BaseModel):
    track_id: str
    track_title: str
    artist_name: str


class TrackRequest(BaseModel):
    track_ids: List[str]        # track IDs
    language: str



# ====== Output Schemas ======

class RecommendationResponse(BaseModel):
    recommended_track: TrackInfo
    explanation: str
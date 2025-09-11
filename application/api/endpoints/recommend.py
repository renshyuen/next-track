# /application/api/endpoints/recommend.py

import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from application.models.schemas import RecommendationResponse, TrackRequest, TrackInfo
from application.core.recommender import NextTrackContentBasedRecommender


router = APIRouter()    # creates an API router object to attach endpoints

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'spotify_tracks_reduced.csv')
# Add debug print to verify path (you can remove this after confirming)
print(f"Looking for data file at: {DATA_CSV_PATH}")
# Verify the file exists
if not os.path.exists(DATA_CSV_PATH):
    raise FileNotFoundError(f"Data file not found at: {DATA_CSV_PATH}")

# == RECOMMENDER LIFECYCLE
try:
    recommender = NextTrackContentBasedRecommender.from_csv(DATA_CSV_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to initialise recommender: {e}")    # fail fast on startup if model loading fails, rather than per request


def _validate_input_track_ids(track_ids: List[str]) -> List[str]:
    """
    DESCRIPTION:
        This helper method validates and processes the input track identifiers from the API request.
    """
    if not track_ids:
        raise HTTPException(status_code=400, detail="Provide at least one input track identifier.")

    # remove duplicates while preserving order
    # in Python, None evaluates to Fals in boolean context
    seen = set()
    input_tracks_ids = [track_id for track_id in track_ids if not (track_id in seen or seen.add(track_id))]

    validated_input_track_ids: List[str] = []

    # validate each input track_id exists in the dataset
    # prefer using recommender's internal mapping if available
    # otherwise, fallback to searching the dataframe
    for track_id in input_tracks_ids:
        if track_id in recommender.track_id_to_index:
            validated_input_track_ids.append(track_id)
        elif hasattr(recommender, 'dataframe'):
            dataframe = recommender.dataframe
            if 'track_id' in dataframe.columns and (dataframe['track_id'] == track_id).any():
                validated_input_track_ids.append(track_id)

    if not validated_input_track_ids:
        raise HTTPException(status_code=400, detail="The provided track ID(s) do not exist in the dataset.")
    
    return validated_input_track_ids


def _to_track_info(track_id: str) -> TrackInfo:
    """
    DESCRIPTION:
        This helper method converts a track identifier to a TrackInfo object using recommender's metadata.
        Maps:
        - dataframe['track_name'] -> track_title
        - dataframe['artists'] -> artist_name
    """
    track_title: Optional[str] = None
    artist_name: Optional[str] = None

    if hasattr(recommender, 'dataframe'):
        dataframe = recommender.dataframe
        row = dataframe.loc[dataframe['track_id'] == track_id]
        if not row.empty:
            track_title = row['track_name'] if 'track_name' in row else None
            artist_name = row['artists'] if 'artists' in row else None
    
    # ensure required fields (schema requires all three)
    return TrackInfo(
        track_id=str(track_id),
        track_title=str(track_title) if track_title else "",
        artist_name=str(artist_name) if artist_name else "",
    )


def _default_explanation() -> str:
    return "Recommended based on similar audio features to your input track(s)."
    

@router.post('/recommend', response_model=RecommendationResponse)
async def recommend_next_track(request: TrackRequest):
    """
    DESCRIPTION:
        This endpoint generates track recommendations based on input track identifiers.
        It uses a content-based filtering approach to find similar tracks.
    """
    try:
        # 1. Validate and resolve input track identifiers
        input_track_ids = _validate_input_track_ids(request.track_ids)

        # 2. Ask recommender for the best matching track
        recommendation = recommender.next_track(input_track_ids)
        if recommendation is None:
            raise HTTPException(status_code=404, detail="No suitable recommendation found.")
        
        # 3. Map to TrackInfo
        recommended_track = TrackInfo(
            track_id=recommendation['track_id'],
            track_title=recommendation['track_title'],
            artist_name=recommendation['artist_name'],
        )

        explanation = _default_explanation()
        return RecommendationResponse(
            recommended_track=recommended_track,
            explanation=explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
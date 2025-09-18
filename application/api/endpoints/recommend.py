# /application/api/endpoints/recommend.py

import os
import logging
import pandas as pd 
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from typing import List, Optional, Dict, Any 
from application.core.recommender import NextTrackContentBasedRecommender
from application.models.schemas import RecommendationResponse, TrackRequest, TrackInfo, PreferenceParameters, RecommendationStrategy, ErrorResponse


router = APIRouter() 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_path = Path(os.getenv('DATASET_PATH', 'application/data/cleaned_spotify_tracks.csv'))

if not dataset_path.is_file():
    raise RuntimeError(f"Dataset file not found at {dataset_path}")

try:
    logger.info(f"Loading dataset from {dataset_path}") 
    dataframe = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(dataframe)} tracks")
    recommender = NextTrackContentBasedRecommender(dataframe)
    logger.info("Recommender initialised successfully")
    
except Exception as e:
    # FAIL FAST ON STARTUP IF MODEL LOADING FAILS, RATHER THAN PER REQUEST
    logger.error(f"Failed to initialise recommender: {e}")
    raise RuntimeError(f"Failed to initialise recommender: {e}")    

def _validate_and_process_user_input(request: TrackRequest) -> tuple:
    """ """
    if not request.track_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one track identifier."
        )
    
    # REMOVE DUPLICATES WHILE PRESERVING ORDER
    seen = set()
    unique_track_ids = [track_id for track_id in request.track_ids if not (track_id in seen or seen.add(track_id))]

    # VALIDATE TRACK IDS EXIST IN DATASET
    validated_track_ids = []
    invalid_track_ids = []

    for track_id in unique_track_ids:
        if track_id in recommender.track_id_to_index:
            validated_track_ids.append(track_id)
        else:
            invalid_track_ids.append(track_id)
    
    if not validated_track_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"None of the provided track IDs exist in the dataset. Invalid IDs: {invalid_track_ids}"
        )
    
    # LOG WARNING FOR INVALID IDS BUT CONTINUE WITH VALID ONES
    if invalid_track_ids:
        logger.warning(f"Some track IDs not found: {invalid_ids}")
    
    # PROCESS PREFERENCES IF PROVIDED
    processed_preferences = None 
    if request.preferences:
        processed_preferences = {}

        # MAP PREFERENCE PARAMETERS TO FEATURE NAMES
        if request.preferences.valence is not None:
            processed_preferences['valence'] = request.preferences.valence
        if request.preferences.energy is not None:
            processed_preferences['energy'] = request.preferences.energy
        if request.preferences.danceability is not None:
            processed_preferences['danceability'] = request.preferences.danceability
        if request.preferences.acousticness is not None:
            processed_preferences['acousticness'] = request.preferences.acousticness
        if request.preferences.instrumentalness is not None:
            processed_preferences['instrumentalness'] = request.preferences.instrumentalness
        if request.preferences.tempo is not None:
            # NORMALISE TEMPO TO 0-1 RANGE (60-200 BPM RANGE)
            processed_preferences['tempo'] = (request.preferences.tempo - 60) / 140
        
        # HANDLE ADDITIONAL PREFERENCES
        if request.preferences.popular:
            processed_preferences['popular'] = True
        if request.preferences.temporal_preference:
            processed_preferences['temporal_preference'] = request.preferences.temporal_preference
    
    return validated_track_ids, processed_preferences

def _calculate_confidence_score(similarity: float, strategy: str, num_input_tracks: int) -> float:
    """
    """
    base_confidence_score = similarity

    # ADJUST BASED ON NUMBER OF INPUT TRACKS (MORE TRACKS = HIGHER CONFIDENCE)
    track_factor = min(1.0, num_input_tracks / 5) # MAX BOOST AT 5 TRACKS

    # STRATEGY CONFIDENCE MULTIPLIERS
    strategy_multipliers = {
        'weighted_average': 1.0,
        'recent_weighted': 0.95,
        'momentum': 0.9 if num_input_tracks >= 3 else 0.7
    }
    strategy_mult = strategy_multipliers.get(strategy, 1.0)

    # CALCULATE FINAL CONFIDENCE
    confidence_score = base_confidence_score * (0.7 + 0.3 * track_factor) * strategy_mult

    return min(1.0, max(0.0, confidence_score))

@router.post('/recommend', response_model=RecommendationResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid request"},
    404: {"model": ErrorResponse, "description": "No suitable recommendation found"},
    500: {"model": ErrorResponse, "description": "Internal server error"}
}, 
summary="Get next track recommendation", 
description="Returns a recommended track based on input tracks and optional preferences"
)
async def recommend_next_track(request: TrackRequest):
    """ """
    try:
        # VALIDATE AND PROCESS USER INPUT
        validated_track_ids, processed_preferences = _validate_and_process_user_input(request)

        # LOG REQUEST DETAILS
        logger.info(f"Processing recommendation request: {len(validated_track_ids)} tracks, strategy: {request.strategy}, preferences: {bool(processed_preferences)}")

        # DETERMINE STRATEGY
        strategy = request.strategy or RecommendationStrategy.WEIGHTED_AVERAGE

        # GET DIVERSITY PENALTY
        diversity_penalty = 0.0

        # GET RECOMMENDATION FROM THE MODEL
        recommendation = recommender.next_track(
            input_track_ids=validated_track_ids,
            preferences=processed_preferences,
            strategy=strategy.value,
            diversity_penalty=diversity_penalty
        )

        if recommendation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No suitable recommendation found. Try different input tracks or adjust preferences."
            )
        
        # CREATE A RESPONSE
        recommended_track = TrackInfo(
            track_id=recommendation['track_id'],
            track_title=recommendation['track_title'],
            artists=recommendation['artists']
        )

        # CALCULATE CONFIDENCE SCORE
        confidence_score = _calculate_confidence_score(
            recommendation['similarity'],
            strategy.value,
            len(validated_track_ids)
        )

        response = RecommendationResponse(
            recommended_track=recommended_track,
            explanation=recommendation['explanation'],
            confidence_score=confidence_score,
            strategy_used=strategy.value
        )

        logger.info(f"Recommendation successful: {recommendation['track_id']}, confidence: {confidence_score:.2f}")

        return response

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error in recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating recommendation"
        )
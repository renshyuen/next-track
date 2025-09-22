# tests/diagnostic_test.py

"""
Diagnostic script to debug why evaluation metrics are showing 0.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from application.core.recommender import NextTrackContentBasedRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_evaluation():
    """
    Run diagnostic tests to identify why metrics are 0.
    """
    
    logger.info("   Loading dataset...")
    df = pd.read_csv('application/data/cleaned_spotify_tracks.csv')
    logger.info(f"  Loaded {len(df)} tracks")
    
    # Initialize recommender
    logger.info("   Initializing recommender...")
    recommender = NextTrackContentBasedRecommender(df)
    
    # TEST 1: Check if recommender works at all
    logger.info("\n\n=== TEST 1: Basic Recommendation Test ===")
    sample_tracks = df.sample(n=5)['track_id'].tolist()
    logger.info(f"  Input tracks: {sample_tracks[:2]}...")
    
    rec = recommender.next_track(sample_tracks, strategy='weighted_average')
    if rec:
        logger.info(f"  ✓ Got recommendation: {rec['track_id']}")
    else:
        logger.error("  ✗ No recommendation returned!")
        return
    
    # TEST 2: Check if we can find any matches
    logger.info("\n\n=== TEST 2: Similarity-Based Matching ===")
    
    # Find tracks with similar features
    if 'energy' in df.columns and 'valence' in df.columns:
        # Pick tracks with specific energy/valence
        high_energy = df[df['energy'] > 0.8]
        if len(high_energy) > 10:
            similar_tracks = high_energy.sample(n=10)['track_id'].tolist()
            input_tracks = similar_tracks[:5]
            potential_matches = similar_tracks[5:]
            
            logger.info(f"  Testing with high-energy tracks...")
            logger.info(f"  Input: {len(input_tracks)} tracks")
            logger.info(f"  Potential matches: {len(potential_matches)} tracks")
            
            # Get recommendations
            recommendations = []
            for i in range(5):
                rec = recommender.next_track(input_tracks, strategy='weighted_average')
                if rec and rec['track_id'] not in recommendations:
                    recommendations.append(rec['track_id'])
            
            # Check for matches
            matches = [r for r in recommendations if r in potential_matches]
            logger.info(f"  Recommendations: {len(recommendations)}")
            logger.info(f"  Matches found: {len(matches)}")
            
            if matches:
                logger.info(f"  ✓ Found matches! Precision would be: {len(matches)/len(recommendations):.3f}")
            else:
                logger.info("   ✗ No matches (expected with random ground truth)")
    
    # TEST 3: Artist-based similarity test
    logger.info("\n\n=== TEST 3: Artist-Based Test ===")
    if 'artists' in df.columns:
        # Find an artist with multiple tracks
        artist_counts = df['artists'].value_counts()
        popular_artists = artist_counts[artist_counts >= 20].index[:5]
        
        for artist in popular_artists:
            artist_tracks = df[df['artists'] == artist]['track_id'].tolist()
            
            if len(artist_tracks) >= 10:
                logger.info(f"  Testing with artist: {artist}")
                logger.info(f"  Artist has {len(artist_tracks)} tracks")
                
                # Use half for input, half for ground truth
                input_tracks = artist_tracks[:5]
                ground_truth = set(artist_tracks[5:10])
                
                # Get recommendations
                recommendations = []
                seen = set(input_tracks)
                for i in range(5):
                    rec = recommender.next_track(input_tracks, strategy='weighted_average')
                    if rec and rec['track_id'] not in seen:
                        recommendations.append(rec['track_id'])
                        seen.add(rec['track_id'])
                
                # Check matches
                matches = [r for r in recommendations if r in ground_truth]
                
                if matches:
                    precision = len(matches) / len(recommendations) if recommendations else 0
                    logger.info(f"  ✓ Found {len(matches)} matches! Precision: {precision:.3f}")
                    break
                else:
                    logger.info(f"  No matches for {artist}")
    
    # TEST 4: Check the actual evaluation setup
    logger.info("\n\n=== TEST 4: Evaluation Setup Check ===")
    from tests.evaluation_metrics import RecommendationEvaluator
    
    evaluator = RecommendationEvaluator(df, test_size=0.2)
    logger.info(f"  Created {len(evaluator.playlists)} playlists")
    logger.info(f"  Test playlists: {len(evaluator.test_playlists)}")
    
    if evaluator.test_playlists:
        # Check a sample test playlist
        test_playlist = evaluator.test_playlists[0]
        logger.info(f"  Sample test playlist:")
        logger.info(f"  Input tracks: {len(test_playlist['input_tracks'])}")
        logger.info(f"  Ground truth: {len(test_playlist['ground_truth'])}")
        
        # Try to get recommendations for this playlist
        input_tracks = test_playlist['input_tracks']
        ground_truth = set(test_playlist['ground_truth'])
        
        recommendations = []
        seen = set(input_tracks)
        
        for _ in range(5):
            rec = recommender.next_track(input_tracks, strategy='weighted_average')
            if rec and rec['track_id'] not in seen:
                recommendations.append(rec['track_id'])
                seen.add(rec['track_id'])
        
        logger.info(f"  Got {len(recommendations)} recommendations")
        
        # Check for any matches
        matches = [r for r in recommendations if r in ground_truth]
        if matches:
            logger.info(f"  ✓ Found {len(matches)} matches!")
        else:
            logger.info(f"  ✗ No matches with ground truth")
            
            # Check if ground truth tracks even exist in the dataset
            existing_ground_truth = [gt for gt in ground_truth if gt in df['track_id'].values]
            logger.info(f"  Ground truth tracks in dataset: {len(existing_ground_truth)}/{len(ground_truth)}")
    
    # TEST 5: Feature-based playlist creation
    logger.info("\n\n=== TEST 5: Smart Playlist Creation ===")
    
    # Create a playlist of truly similar tracks
    if 'energy' in df.columns and 'valence' in df.columns and 'danceability' in df.columns:
        # Find cluster of very similar tracks
        target_energy = 0.7
        target_valence = 0.7
        target_dance = 0.7
        tolerance = 0.1
        
        similar = df[
            (df['energy'].between(target_energy - tolerance, target_energy + tolerance)) &
            (df['valence'].between(target_valence - tolerance, target_valence + tolerance)) &
            (df['danceability'].between(target_dance - tolerance, target_dance + tolerance))
        ]
        
        if len(similar) >= 10:
            logger.info(f"  Found {len(similar)} tracks with similar features")
            
            similar_ids = similar['track_id'].tolist()
            input_tracks = similar_ids[:5]
            ground_truth = set(similar_ids[5:10])
            
            recommendations = []
            seen = set(input_tracks)
            
            for _ in range(5):
                rec = recommender.next_track(input_tracks, strategy='weighted_average')
                if rec and rec['track_id'] not in seen:
                    recommendations.append(rec['track_id'])
                    seen.add(rec['track_id'])
            
            matches = [r for r in recommendations if r in ground_truth]
            
            if matches:
                logger.info(f"  ✓ Smart playlist got {len(matches)} matches!")
                precision = len(matches) / len(recommendations)
                logger.info(f"  Precision: {precision:.3f}")
            else:
                logger.info("   ✗ Even similar tracks didn't match")
                
                # Debug: Check feature similarity of recommendations
                if recommendations:
                    rec_track = df[df['track_id'] == recommendations[0]].iloc[0]
                    input_track = df[df['track_id'] == input_tracks[0]].iloc[0]
                    
                    logger.info("\n  Feature comparison (first rec vs first input):")
                    for feature in ['energy', 'valence', 'danceability']:
                        if feature in df.columns:
                            rec_val = rec_track[feature]
                            inp_val = input_track[feature]
                            diff = abs(rec_val - inp_val)
                            logger.info(f"    {feature}: {rec_val:.3f} vs {inp_val:.3f} (diff: {diff:.3f})")


if __name__ == "__main__":
    diagnose_evaluation()
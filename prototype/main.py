from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

df = pd.read_csv('spotify_tracks_reduced.csv')

AUDIO_FEATURES = ['danceability', 'energy', 'valence']
filtered_df = df.dropna(subset=AUDIO_FEATURES)
track_features = filtered_df[AUDIO_FEATURES].values

class TrackRequest(BaseModel):
    track_ids: list

@app.post('/recommend')
def recommend_tracks(request_data: TrackRequest):

    track_ids = [str(track_id) for track_id in request_data.track_ids]
    df_input = df[df['track_id'].isin(track_ids)]

    if not track_ids:
        raise HTTPException(status_code=400, detail='no track_ids provided')
    if df_input.empty:
        raise HTTPException(status_code=400, detail='track_ids provided not found')
    
    # random (not real logic yet)
    # recommendation = df.sample(1).iloc[0]

    session_vector = df_input[AUDIO_FEATURES].mean().values.reshape(1, -1)
    similarity_scores = cosine_similarity(session_vector, track_features).flatten()
    recommendation_df = filtered_df.copy()
    recommendation_df['similarity'] = similarity_scores
    recommendation_df = recommendation_df[~recommendation_df['track_id'].isin(track_ids)]

    if recommendation_df.empty:
        raise HTTPException(status_code=404, detail='no recommendation found based on the provided track IDs')
    
    recommendation = recommendation_df.sort_values(by='similarity', ascending=False).iloc[0]
    explanation = f"recommended because it has similar danceability ({recommendation['danceability']:.2f}), energy ({recommendation['energy']:.2f}), and valence ({recommendation['valence']:.2f}) to your provided tracks"

    return {
        'recommended_track': {
            'track_id': recommendation['track_id'],
            'artist': recommendation['artists'],
            'title': recommendation['track_name'],
            'album': recommendation['album_name'],
        },
        'explanation': explanation,
    }



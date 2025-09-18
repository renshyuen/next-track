# /application/utilities/scripts.py

import pandas as pd 
import ast 

# THIS IS THE CODE FOR DATA CLEANING / PREPARATION IN GOOGLE COLAB
def clean_dataset(path: str) -> pd.DataFrame:
    """
    DESCRIPTION:
    PARAMETERS:
    """
    if not path:
        raise ValueError("A valid file path must be provided.")
    
    raw_dataframe = pd.read_csv(path)

    # BASIC CLEANUP
    original_dataframe = raw_dataframe.dropna(subset=['name'])

    # SELECT & RENAME TO TARGET SCHEMA  
    filtered_dataframe = original_dataframe[['id', 'name', 'artists']].rename(columns={
        'id': 'track_id',
        'name': 'track_title',
    }).copy()

    # ARTISTS: STRINGIFIED LIST -> "A, B"
    filtered_dataframe['artists'] = original_dataframe['artists'].apply(lambda x: ast.literal_eval(x))
    filtered_dataframe['artists'] = filtered_dataframe['artists'].apply(lambda x: ", ".join(x))


    filtered_dataframe['release_year'] = pd.to_datetime(original_dataframe['release_date'], format='%m/%d/%Y', errors='coerce')

    mask_indices = filtered_dataframe['release_year'].isna()
    filtered_dataframe.loc[mask_indices, 'release_year'] = pd.to_datetime(original_dataframe.loc[mask_indices, 'release_date'], format='%Y', errors='coerce')
    mask_indices = filtered_dataframe['release_year'].isna()
    filtered_dataframe.loc[mask_indices, 'release_year'] = pd.to_datetime(original_dataframe.loc[mask_indices, 'release_date'], format='%Y-%m', errors='coerce')
    filtered_dataframe['release_year'] = filtered_dataframe['release_year'].dt.year.astype('Int64')

    # CORE AUDIO FEATURES (REQUIRED)
    filtered_dataframe['danceability'] = original_dataframe['danceability']
    filtered_dataframe['energy'] = original_dataframe['energy']
    filtered_dataframe['valence'] = original_dataframe['valence']
    filtered_dataframe['tempo'] = original_dataframe['tempo']

    # INTERPRETABLE AUDIO FEATURES THAT COULD HELP ENRICH EXPLANATIONS
    filtered_dataframe['mode'] = original_dataframe['mode']
    filtered_dataframe['speechiness'] = original_dataframe['speechiness']
    filtered_dataframe['acousticness'] = original_dataframe['acousticness']
    filtered_dataframe['instrumentalness'] = original_dataframe['instrumentalness']

    # RESET INDEX
    filtered_dataframe = filtered_dataframe.reset_index(drop=True)

    return filtered_dataframe
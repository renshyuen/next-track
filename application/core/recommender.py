import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity



class ContentBasedRecommender:
    """
    A short summary of the ContentBasedRecommender class.
    A more detailed description of the ContentBasedRecommender class, 
    including its purpose, how it works, and any important details that users should know.
    Attributes:
        attribute1 (type): Description of attribute1.
        attribute2 (type): Description of attribute2.
    Methods:
        method1(param1, param2): Description of method1.
        method2(param1): Description of method2.
    """
    def __init__(self, tracks_df: pd.DataFrame):
        self.tracks_df = tracks_df
        
    
    def _prepare_features(self):
        """
        Prepares the features for the content-based recommended.
        This method extracts relevant features from the tracks
        DataFrame and prepares them for similarity calculations.
        """

    def recommend(self):
        
# /application/core/explainability.py

from typing import List, Dict, Optional, Any 
import numpy as np 


class ExplainabilityEngine:
    """
    Explainability engine for generating human-readable explanations for recommended tracks.
    """

    FEATURE_INTERPRETATIONS = {
        'energy': {
            'high': (0.7, 1.0, ['energetic', 'intense', 'powerful']),
            'medium': (0.3, 0.7, ['moderate energy', 'balanced']),
            'low': (0.0, 0.3, ['mellow', 'relaxed', 'calm'])
        },
        'valence': {
            'high': (0.7, 1.0, ['upbeat', 'positive', 'happy']),
            'medium': (0.3, 0.7, ['neutral mood', 'balanced emotion']),
            'low': (0.0, 0.3, ['melancholic', 'emotional', 'introspective'])
        },
        'danceability': {
            'high': (0.7, 1.0, ['highly danceable', 'groove-driven', 'rhythmic']),
            'medium': (0.4, 0.7, ['moderately danceable', 'steady beat']),
            'low': (0.0, 0.4, ['less danceable', 'free-form', 'ambient'])
        },
        'acousticness': {
            'high': (0.7, 1.0, ['acoustic', 'organic', 'unplugged']),
            'medium': (0.3, 0.7, ['mixed instrumentation', 'hybrid sound']),
            'low': (0.0, 0.3, ['electronic', 'synthesized', 'produced'])
        },
        'instrumentalness': {
            'high': (0.7, 1.0, ['instrumental', 'no vocals', 'purely musical']),
            'medium': (0.3, 0.7, ['some vocals', 'mixed vocal/instrumental']),
            'low': (0.0, 0.3, ['vocal-focused', 'lyrical', 'sung'])
        },
        'speechiness': {
            'high': (0.3, 1.0, ['spoken word', 'rap', 'talk-heavy']),
            'medium': (0.1, 0.3, ['some spoken elements', 'vocal variety']),
            'low': (0.0, 0.1, ['melodic', 'sung throughout', 'no speech'])
        },
        'tempo': {
            'high': (140, 200, ['fast-paced', 'high tempo', 'quick']),
            'medium': (100, 140, ['moderate tempo', 'mid-paced']),
            'low': (60, 100, ['slow', 'relaxed tempo', 'laid-back'])
        },
    }

    # MUSICAL GENRE / STYLE ASSOCIATIONS BASED ON FEATURE COMBINATIONS
    STYLE_PATTERNS = {
        'electronic_dance': {
            'features': {
                'energy': (0.7, 1.0), 
                'danceability': (0.7, 1.0), 
                'acousticness': (0.0, 0.3)
            },
            'description': 'electronic dance'
        },
        'acoustic_folk': {
            'features': {
                'acousticness': (0.7, 1.0), 
                'energy': (0.2, 0.6)
            },
            'description': 'acoustic/folk style'
        },
        'ambient_chill': {
            'features': {
                'energy': (0.0, 0.4), 
                'instrumentalness': (0.5, 1.0)
            },
            'description': 'ambient/chill'
        },
        'upbeat_pop': {
            'features': {
                'valence': (0.6, 1.0), 
                'energy': (0.5, 0.9), 
                'danceability': (0.6, 1.0)
            },
            'description': 'upbeat pop'
        },
        'melancholic': {
            'features': {
                'valence': (0.0, 0.4), 
                'energy': (0.2, 0.6)
            },
            'description': 'melancholic/emotional'
        }
    }  

    @classmethod
    def generate_feature_explanation(
        cls,
        feature_name: str,
        feature_value: float,
        context: str = 'matching',
    ) -> Optional[str]:
        """ 
        Generate explanation for a single feature value.

        Parameters:
            feature_name [string]: Name of the audio feature
            feature_value [float]: Normalised feature value
            context [str]: Context for explanation ('matching', 'contrasting', 'neutral')
        
        Returns:
            Human-readable explanation or None
        """
        if feature_name not in cls.FEATURE_INTERPRETATIONS:
            return None 
        
        interpretations = cls.FEATURE_INTERPRETATIONS[feature_name]

        for level, (min_value, max_value, descriptions) in interpretations.items():
            if min_value <= feature_value <= max_value:
                description = np.random.choice(descriptions) # VARY THE LANGUAGE

                if context == 'matching':
                    return f"similar {description}"
                elif context == 'contrasting':
                    return f"contrasting {description}"
                else:
                    return description 
        
        return None 
    
    @classmethod
    def detect_musical_style(cls, features: Dict[str, float]) -> Optional[str]:
        """
        Detect musical style based on feature combinations.

        Parameters:
            features [dict]: Dictionary of feature names to feature values
        
        Returns:
            Style description or None
        """
        best_match = None 
        best_score = 0

        for style_name, style_info in cls.STYLE_PATTERNS.items():
            score = 0
            required_features = len(style_info['features'])

            for feature, (min_value, max_value) in style_info['features'].items():
                if feature in features and min_value <= features[feature] <= max_value:
                    score += 1
            
            if score > best_score and score >= required_features * 0.7:
                best_score = score 
                best_match = style_info['description']
        
        return best_match 
    
    @classmethod 
    def generate_transition_explanation(
        cls,
        from_features: Dict[str, float],
        to_features: Dict[str, float],
        threshold: float = 0.3
    ) -> str:
        """
        Generate explanation for the transition between tracks.

        Parameters:
            from_features [dict]: Features of input tracks
            to_features [dict]: Features of recommended track
            threshold: Threshold for significant change
        
        Returns:
            Explanation of the transition
        """
        transitions = []

        for feature in ['energy', 'valence', 'tempo']:
            if feature in from_features and feature in to_features:
                difference = to_features[feature] - from_features[feature]

                if abs(difference) > threshold:
                    if difference > 0:
                        if feature == 'energy':
                            transitions.append("building energy")
                        elif feature == 'valence':
                            transitions.append("lifting the mood")
                        elif feature == 'tempo':
                            transitions.append("picking up the pace")
                    else:
                        if feature == 'energy':
                            transitions.append("winding down")
                        elif feature == 'valence':
                            transitions.append("deepening the emotion")
                        elif feature == 'tempo':
                            transitions.append("slowing the tempo")
        if transitions:
            return f"Smoothly {' and '.join(transitions)}"
        else:
            return "Maintaining the musical flow"
    
    @classmethod
    def generate_comprehensive_explanation(
        cls,
        input_features: Dict[str, float],
        recommended_features: Dict[str, float],
        similarity_score: float,
        strategy: str,
        preferences: Optional[Dict[str, Any]] = None,
        artist_similarity: Optional[float] = None
    ) -> str:
        """
        Generate a comprehensive explanation combining multiple factors.
        
        Parameters:
            input_features [dict]: Average features of input tracks
            recommended_features [dict]: Features of recommended track
            similarity_score [float]: Overall similarity score
            strategy [str]: Recommendation strategy used
            preferences [dict]: User preferences applied
            artist_similarity [float]: Optional artist similarity score
        
        Returns:
            Comprehensive explanation string
        """
        explanations = []

        # 1. STYLE DETECTION
        recommendation_style = cls.detect_musical_style(recommended_features)
        if recommendation_style:
            explanations.append(f"This {recommendation_style} track")
        else:
            explanations.append("This track")
        
        # 2. KEY MATCHING FEATURES
        matching_features = []
        for feature in ['energy', 'valence', 'danceability']:
            if feature in input_features and feature in recommended_features:
                difference = abs(recommended_features[feature] - input_features[feature])
                if difference < 0.2: # SIMILAR VALUES
                    explanation = cls.generate_feature_explanation(
                        feature,
                        recommended_features[feature], 
                        'matching'
                    )
                    if explanation:
                        matching_features.append(explanation)
        
        if matching_features:
            explanations.append(f"features {', '.join(matching_features[:2])}")
        
        # 3. TRANSITION EXPLANATION
        transition = cls.generate_transition_explanation(
            input_features,
            recommended_features
        )

        if transition != "Maintaining the musical flow":
            explanations.append(f"while {transition.lower()}")
        
        # 4. STRATEGY-SPECIFIC EXPLANATION
        strategy_explanations = {
            'momentum': "following your listening progression",
            'recent_weighted': "based on your recent selections",
            'weighted_average': "balancing your entire listening history"
        }

        if strategy in strategy_explanations:
            explanations.append(strategy_explanations[strategy])
        
        # 5. PREFERENCE INFLUENCE
        if preferences:
            preference_mentions = []
            if 'valence' in preferences and preferences['valence'] > 0.7:
                preference_mentions.append("upbeat preference")
            if 'energy' in preferences and preferences['energy'] > 0.7:
                preference_mentions.append("high-energy request")
            if preference_mentions:
                explanations.append(f"honoring your {' and '.join(preference_mentions)}")
        
        # 6. CONFIDENCE INDICATOR
        if similarity_score > 0.85:
            explanations.append("(strong match)")
        elif similarity_score > 0.6:
            explanations.append("(good match)")
        
        # COMBINE ALL EXPLANATIONS
        if len(explanations) > 1:
            # CAPITALIZE FIRST WORD AND JOIN WITH APPROPRIATE PUNCTUATION
            result = explanations[0]
            for index, explain in enumerate(explanations[1:], 1):
                if explain.startswith("("):
                    result += f" {explain}"
                elif index == len(explanations) - 1:
                    result += f", {explain}"
                else:
                    result += f", {explain}"
            return result + "."
        
        return "Recommended based on audio feature analysis."
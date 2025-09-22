# /tests/evaluation_metrics.py

# /tests/evaluation_metrics.py

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from sklearn.model_selection import train_test_split
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """
    Evaluator for Next-Track's recommender system using Precision@k, Recall@k, and other metrics.
    """
    
    def __init__(self, dataframe: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Initialise evaluator with dataset.
        
        Parameters:
            dataframe [pd.DataFrame]: Full dataset
            test_size [float]: Proportion of data to use for testing
            random_state [int]: Random seed for reproducibility
        """
        self.dataframe = dataframe
        self.test_size = test_size
        self.random_state = random_state
        
        # CREATE SYNTHETIC PLAYLISTS FOR EVALUATION
        self.playlists = self._create_synthetic_playlists()
        
        # SPLIT PLAYLISTS INTO TRAINING/TESTINGS
        self.train_playlists, self.test_playlists = self._split_playlists()
        
        # STORE EVALUATION RESULTS
        self.results = defaultdict(list)
    
    def _create_synthetic_playlists(self, max_playlists: int = 500) -> List[List[str]]:
        """
        Create synthetic playlists by grouping tracks with similar features, simulating real user listening sessions.

        This creates test playlists using three strategies:
            - Mood-based: Groups songs by energy and valence
            - Artist-based: Groups songs by the same artist
            - Activity-based: Groups songs by tempo and danceability
        
        Parameters:
            max_playlists [int]: Maximum number of playlists to create for reasonable runtime
        """
        playlists = []

        df = self.dataframe
        
        # STRATEGY 1: GROUP BY SIMILAR ENERGY AND VALENCE (MOOD-BASED PLAYLISTS)
        if 'energy' in self.dataframe.columns and 'valence' in self.dataframe.columns:
            try:
                # CREATE ENERGY-VALENCE BINS
                energy_bins = pd.qcut(self.dataframe['energy'], q=3, labels=['low', 'medium', 'high'], duplicates='drop')
                valence_bins = pd.qcut(self.dataframe['valence'], q=3, labels=['sad', 'neutral', 'happy'], duplicates='drop')
            
                # GROUP TRACKS
                for energy_level in energy_bins.unique():
                    for valence_level in valence_bins.unique():
                        if len(playlists) >= max_playlists // 3:
                            break
                        mask = (energy_bins == energy_level) & (valence_bins == valence_level)
                        cluster_tracks = df[mask]['track_id'].tolist()
                        
                        # CREATE 1-2 PLAYLISTS PER CLUSTER
                        if len(cluster_tracks) >= 10:
                            # RANDOM SAMPLE TRACKS FOR PLAYLIST
                            n_playlists_from_cluster = min(2, len(cluster_tracks) // 20)
                            for _ in range(max(1, n_playlists_from_cluster)):
                                if len(playlists) >= max_playlists:
                                    break
                                playlist_size = min(20, len(cluster_tracks))
                                playlist = np.random.choice(
                                    cluster_tracks, 
                                    size=playlist_size, 
                                    replace=False
                                ).tolist()
                                playlists.append(playlist)
            except Exception as e:
                logger.warning(f"Error creating energy/valence playlists: {e}")
        
        # STRATEGY 2: RANDOM SOME RANDOM PLAYLISTS FOR DIVERSITY
        if len(playlists) < max_playlists:
            n_random = min(20, max_playlists - len(playlists))
            for _ in range(n_random):
                playlist_size = np.random.randint(10, 21)
                random_tracks = df.sample(n=min(playlist_size, len(df)))['track_id'].tolist()
                if len(random_tracks) >= 5:
                    playlists.append(random_tracks)
        
        # STRATEGY 3: CREATE SOME ARTIST-BASED PLAYLISTS IF POSSIBLE
        if 'artists' in df.columns and len(playlists) < max_playlists:
            # GET TOP ARTISTS BY TRACK COUNT
            artist_counts = df['artists'].value_counts()
            top_artists = artist_counts[artist_counts >= 5].head(20).index
            
            for artist in top_artists:
                if len(playlists) >= max_playlists:
                    break
                artist_tracks = df[df['artists'] == artist]['track_id'].tolist()
                if len(artist_tracks) >= 5:
                    playlist = artist_tracks[:min(15, len(artist_tracks))]
                    playlists.append(playlist)
        
        # ENSURE WE HAVE AT LEAST SOME PLAYLISTS
        if len(playlists) == 0:
            logger.warning("No playlists created from clustering, using random sampling")
            for _ in range(min(50, max_playlists)):
                playlist_size = np.random.randint(8, 16)
                random_tracks = df.sample(n=min(playlist_size, len(df)))['track_id'].tolist()
                playlists.append(random_tracks)
        
        logger.info(f"Created {len(playlists)} synthetic playlists (limited from larger dataset)")
        return playlists
    
    def _split_playlists(self) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Split playlists into train and test sets.
        For each playlist, use first n-1 tracks for input and last tracks for validation.
        """
        train_playlists = []
        test_playlists = []
        
        for playlist in self.playlists:
            if len(playlist) >= 5:
                # USE 70% OF TRACKS AS INPUT, 30% AS GROUND TRUTH
                split_point = max(3, int(len(playlist) * 0.7))
                
                train_playlists.append({
                    'input_tracks': playlist[:split_point],
                    'full_playlist': playlist
                })
                
                test_playlists.append({
                    'input_tracks': playlist[:split_point],
                    'ground_truth': playlist[split_point:],
                    'full_playlist': playlist
                })
        
        # SPLIT INTO TRAIN/TEST SETS
        split_index = int(len(test_playlists) * (1 - self.test_size))
        
        return train_playlists[:split_index], test_playlists[split_index:]
    
    def precision_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate Precision@k, measures accuracy of Next-Track's recommendations.
        
        Precision@k = (# of recommended items @k that are relevant) / k
        
        Parameters:
            recommended [list]: List of recommended track IDs
            ground_truth [list]: List of ground truth track IDs
            k [int]: Number of recommendations to consider
        
        Returns:
            Precision@k score
        """
        if k <= 0:
            return 0.0
        
        recommended_k = recommended[:k]
        ground_truth_set = set(ground_truth)
        
        relevant_and_recommended = sum(1 for track in recommended_k if track in ground_truth_set)
        
        return relevant_and_recommended / k
    
    def recall_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate Recall@k, measures completeness of recommendations.
        
        Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
        
        Parameters:
            recommended [list]: List of recommended track IDs
            ground_truth [list]: List of ground truth track IDs
            k [int]: Number of recommendations to consider
        
        Returns:
            Recall@k score
        """
        if len(ground_truth) == 0:
            return 0.0
        
        recommended_k = set(recommended[:k])
        ground_truth_set = set(ground_truth)
        
        relevant_and_recommended = len(recommended_k.intersection(ground_truth_set))
        
        return relevant_and_recommended / len(ground_truth_set)
    
    def f1_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate F1@k score (harmonic mean of precision and recall).
        
        Parameters:
            recommended [list]: List of recommended track IDs
            ground_truth [list]: List of ground truth track IDs
            k [int]: Number of recommendations to consider
        
        Returns:
            F1@k score
        """
        precision = self.precision_at_k(recommended, ground_truth, k)
        recall = self.recall_at_k(recommended, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended: List[str], ground_truth: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain @k.
        
        Parameters:
            recommended [list]: List of recommended track IDs
            ground_truth [list]: List of ground truth track IDs
            k [int]: Number of recommendations to consider
        
        Returns:
            NDCG@k score
        """
        recommended_k = recommended[:k]
        ground_truth_set = set(ground_truth)
        
        # CALCULATE DCG
        dcg = 0.0
        for index, track in enumerate(recommended_k):
            if track in ground_truth_set:
                # RELEVANCE IS BINARY (1 IF IN GROUND TRUTH, 0 OTHERWISE)
                dcg += 1.0 / np.log2(index + 2)  # index+2 BECAUSE POSITIONS START AT 1
        
        # CALCULATE IDEAL DCG (BEST POSSIBLE RANKING)
        idcg = sum(1.0 / np.log2(index + 2) for index in range(min(k, len(ground_truth))))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mean_reciprocal_rank(self, recommended: List[str], ground_truth: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / (rank of first relevant item)
        
        Parameters:
            recommended [list]: List of recommended track IDs
            ground_truth [list]: List of ground truth track IDs
        
        Returns:
            MRR score
        """
        ground_truth_set = set(ground_truth)
        
        for index, track in enumerate(recommended):
            if track in ground_truth_set:
                return 1.0 / (index + 1)
        
        return 0.0
    
    def evaluate_recommender(
        self, 
        recommender, 
        k_values: List[int] = [1, 3, 5, 10], 
        strategies: List[str] = ['weighted_average', 'recent_weighted', 'momentum'],
        max_test_playlists: int = 100
    ) -> Dict:
        """
        Evaluate recommender system using multiple metrics.
        
        Parameters:
            recommender: The recommender system to evaluate
            k_values [list]: List of k values to evaluate
            strategies [list]: List of strategies to test
        
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'precision': defaultdict(lambda: defaultdict(list)),
            'recall': defaultdict(lambda: defaultdict(list)),
            'f1': defaultdict(lambda: defaultdict(list)),
            'ndcg': defaultdict(lambda: defaultdict(list)),
            'mrr': defaultdict(list)
        }

        # LIMIT TEST PLAYLISTS FOR REASONABLE RUNTIME
        test_subset = self.test_playlists[:max_test_playlists]
        logger.info(f"Evaluating on {len(test_subset)} test playlists")
        
        for strategy in strategies:
            logger.info(f"Evaluating strategy: {strategy}")

            # PROGRESS TRACKING
            evaluated = 0
            
            for test_playlist in test_subset:
                input_tracks = test_playlist['input_tracks']
                ground_truth = test_playlist['ground_truth']

                # SKIP IF NOT ENOUGH GROUND TRUTH
                if len(ground_truth) < 2:
                    continue
                
                # GET RECOMMENDATIONS EFFICIENTLY
                recommendations = []
                seen_tracks = set(input_tracks)

                # GET BATCH RECOMMENDATIONS (FASTER THAN ONE AT A TIME)
                try:
                    for _ in range(max(k_values)):
                        recommendation = recommender.next_track(
                                input_track_ids=input_tracks,
                                strategy=strategy
                            )
                        if recommendation and recommendation['track_id'] not in seen_tracks:
                            recommendations.append(recommendation['track_id'])
                            seen_tracks.add(recommendation['track_id'])
                            # UPDATE INPUT FOR NEXT ITERATION (OPTIONAL FOR MOMENTUM)
                            if strategy == 'momentum' and len(recommendations) % 3 == 0:
                                input_tracks = input_tracks[-2:] + [recommendation['track_id']]
                        else:
                            break # NO MORE UNIQUE RECOMMENDATIONS
                except Exception as e:
                    logger.debug(f"Error getting recommendation: {e}")
                    continue
                
                if not recommendations:
                    continue
                
                # CALCULATE METRICS FOR DIFFERENT K VALUES
                for k in k_values:
                    results['precision'][strategy][k].append(
                        self.precision_at_k(recommendations, ground_truth, k)
                    )
                    results['recall'][strategy][k].append(
                        self.recall_at_k(recommendations, ground_truth, k)
                    )
                    results['f1'][strategy][k].append(
                        self.f1_at_k(recommendations, ground_truth, k)
                    )
                    results['ndcg'][strategy][k].append(
                        self.ndcg_at_k(recommendations, ground_truth, k)
                    )
                
                # CALCULATE MRR (DOESN'T DEPEND ON K)
                results['mrr'][strategy].append(
                    self.mean_reciprocal_rank(recommendations, ground_truth)
                )

                evaluated += 1
                if evaluated % 20 == 0:
                    logger.debug(f"  Evaluated {evaluated}/{len(test_subset)} playlists")
        
        # CALCULATE AVERAGE SCORES
        summary = {}
        for strategy in strategies:
            summary[strategy] = {}
            
            for k in k_values:
                if results['precision'][strategy][k]:
                    summary[strategy][f'precision@{k}'] = np.mean(results['precision'][strategy][k])
                    summary[strategy][f'recall@{k}'] = np.mean(results['recall'][strategy][k])
                    summary[strategy][f'f1@{k}'] = np.mean(results['f1'][strategy][k])
                    summary[strategy][f'ndcg@{k}'] = np.mean(results['ndcg'][strategy][k])
                else:
                    summary[strategy][f'precision@{k}'] = 0.0
                    summary[strategy][f'recall@{k}'] = 0.0
                    summary[strategy][f'f1@{k}'] = 0.0
                    summary[strategy][f'ndcg@{k}'] = 0.0
            
            summary[strategy]['mrr'] = np.mean(results['mrr'][strategy]) if results['mrr'][strategy] else 0.0
        
        return summary
    
    def cross_validate(self, recommender, n_folds: int = 5, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Perform cross-validation on the recommender system.
        This splits data into n_folds, tests recommender performance across different data splits.
        
        Parameters:
            recommender: The recommender system to evaluate
            n_folds [int]: Number of cross-validation folds
            k_values [list]: List of k values to evaluate
        
        Returns:
            Dictionary containing cross-validation results
        """
        fold_results = []
        fold_size = len(self.playlists) // n_folds
        
        for fold in range(n_folds):
            logger.info(f"Cross-validation fold {fold + 1}/{n_folds}")
            
            # CREATE FOLD-SPECIFIC TRAIN/TEST SPLIT
            start_index = fold * fold_size
            end_index = start_index + fold_size if fold < n_folds - 1 else len(self.playlists)
            
            test_fold = self.playlists[start_index:end_index]
            train_fold = self.playlists[:start_index] + self.playlists[end_index:]
            
            # TEMPORARILY UPDATE TEST PLAYLISTS
            original_test = self.test_playlists
            self.test_playlists = [
                {
                    'input_tracks': playlist[:int(len(playlist) * 0.7)],
                    'ground_truth': playlist[int(len(playlist) * 0.7):],
                    'full_playlist': playlist
                }
                for playlist in test_fold if len(playlist) >= 5
            ]
            
            # EVALUATE ON THIS FOLD
            fold_result = self.evaluate_recommender(recommender, k_values)
            fold_results.append(fold_result)
            
            # RESTORE ORIGINAL TEST PLAYLISTS
            self.test_playlists = original_test
        
        # AVERAGE RESULTS ACROSS FOLDS
        cv_summary = defaultdict(lambda: defaultdict(list))
        for fold_result in fold_results:
            for strategy, metrics in fold_result.items():
                for metric, value in metrics.items():
                    cv_summary[strategy][metric].append(value)
        
        # CALCULATE MEAN AND STD
        final_summary = {}
        for strategy in cv_summary:
            final_summary[strategy] = {}
            for metric in cv_summary[strategy]:
                values = cv_summary[strategy][metric]
                final_summary[strategy][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return final_summary
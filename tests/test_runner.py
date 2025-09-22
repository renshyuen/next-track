# /tests/test_runner.py

"""
Complete test suite runner for Next-Track API.
Runs unit tests, integration tests, and performance evaluation with metrics.
"""

import pytest
import sys
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# ADD PARENT DIRECTORY TO PATH FOR IMPORTS
# TO MAKE SURE PYTHON KNOWS WHERE TO FIND application/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.evaluation_metrics import RecommendationEvaluator
from application.core.recommender import NextTrackContentBasedRecommender


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """
    Orchestrates all testing activities.
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialise test runner.
        
        Parameters:
            dataset_path [str]: Path to the dataset CSV file
        """
        self.dataset_path = dataset_path or 'application/data/cleaned_spotify_tracks.csv'
        self.results_dir = Path('tests') / 'test_results'
        self.results_dir.mkdir(exist_ok=True)
        
        # CREATE TIMESTAMPED RESULTS SUBDIRECTORY
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_results_dir = self.results_dir / timestamp
        self.current_results_dir.mkdir(exist_ok=True)
        
        self.test_results = {
            'timestamp': timestamp,
            'unit_tests': {},
            'integration_tests': {},
            'evaluation_metrics': {},
            'performance_benchmarks': {}
        }
    
    def run_unit_tests(self) -> Dict:
        """
        Run unit tests using pytest.
        """
        logger.info("Running unit tests...")
        
        # RUN pytest FOR UNIT TESTS
        exit_code = pytest.main([
            'tests/test_recommender.py',
            '-v',
            '--tb=short',
            f'--junit-xml={self.current_results_dir}/unit_tests.xml',
            f'--html={self.current_results_dir}/unit_tests.html',
            '--self-contained-html'
        ])
        
        self.test_results['unit_tests'] = {
            'passed': exit_code == 0,
            'exit_code': exit_code,
            'report_path': str(self.current_results_dir / 'unit_tests.html')
        }
        
        logger.info(f"Unit tests {'PASSED' if exit_code == 0 else 'FAILED'}")
        return self.test_results['unit_tests']
    
    def run_integration_tests(self) -> Dict:
        """
        Run integration tests using pytest.
        """
        logger.info("Running integration tests...")
        
        # RUN pytest FOR INTEGRATION TESTS
        exit_code = pytest.main([
            'tests/test_endpoints.py',
            '-v',
            '--tb=short',
            f'--junit-xml={self.current_results_dir}/integration_tests.xml',
            f'--html={self.current_results_dir}/integration_tests.html',
            '--self-contained-html'
        ])
        
        self.test_results['integration_tests'] = {
            'passed': exit_code == 0,
            'exit_code': exit_code,
            'report_path': str(self.current_results_dir / 'integration_tests.html')
        }
        
        logger.info(f"Integration tests {'PASSED' if exit_code == 0 else 'FAILED'}")
        return self.test_results['integration_tests']
    
    def run_evaluation_metrics(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Run recommendation quality evaluation with Precision@k and Recall@k.
        
        Parameters:
            k_values [list]: List of k values for evaluation metrics
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Running evaluation metrics...")
        
        try:
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded {len(df)} tracks for evaluation")
            
            evaluator = RecommendationEvaluator(df, test_size=0.2)
            recommender = NextTrackContentBasedRecommender(df)
            
            # RUN EVALUATION
            strategies = ['weighted_average', 'recent_weighted', 'momentum']
            results = evaluator.evaluate_recommender(
                recommender, 
                k_values=k_values,
                strategies=strategies
            )
            
            # STORE RESULTS
            self.test_results['evaluation_metrics'] = results
            
            # SAVE DETAILED RESULTS TO FILE
            results_file = self.current_results_dir / 'evaluation_metrics.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            # PRINT SUMMARY
            self._print_evaluation_summary(results)
            
            logger.info("Evaluation metrics completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in evaluation metrics: {e}")
            self.test_results['evaluation_metrics'] = {
                'error': str(e),
                'passed': False
            }
            return self.test_results['evaluation_metrics']
    
    def run_cross_validation(self, n_folds: int = 5) -> Dict:
        """
        Run cross-validation evaluation.
        
        Parameters:
            n_folds [int]: Number of cross-validation folds
        
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Running {n_folds}-fold cross-validation...")
        
        try:
            df = pd.read_csv(self.dataset_path)
        
            evaluator = RecommendationEvaluator(df)
            recommender = NextTrackContentBasedRecommender(df)
            
            # RUN CROSS-VALIDATION
            cv_results = evaluator.cross_validate(
                recommender,
                n_folds=n_folds,
                k_values=[1, 3, 5, 10]
            )
            
            # STORE RESULTS
            self.test_results['cross_validation'] = cv_results
            
            # SAVE TO FILE
            cv_file = self.current_results_dir / 'cross_validation.json'
            with open(cv_file, 'w') as f:
                json.dump(cv_results, f, indent=4, default=str)
            
            logger.info("Cross-validation completed")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}
    
    def run_performance_benchmarks(self) -> Dict:
        """
        Run performance benchmarks for the recommender.
        """
        logger.info("Running performance benchmarks...")
        
        import time
        
        try:
            df = pd.read_csv(self.dataset_path)
            recommender = NextTrackContentBasedRecommender(df)
            
            benchmarks = {}
            
            # BENCHMARK: INITIALISATION TIME
            start_time = time.time()
            _ = NextTrackContentBasedRecommender(df)
            benchmarks['initialisation_time'] = time.time() - start_time
            
            # BENCHMARK: SINGLE RECOMMENDATION TIME
            track_ids = df['track_id'].sample(3).tolist()
            
            times = []
            for _ in range(100):  # RUN 100 TIMES FOR AVERAGE
                start_time = time.time()
                _ = recommender.next_track(track_ids, strategy='weighted_average')
                times.append(time.time() - start_time)
            
            benchmarks['avg_recommendation_time'] = np.mean(times)
            benchmarks['std_recommendation_time'] = np.std(times)
            benchmarks['min_recommendation_time'] = np.min(times)
            benchmarks['max_recommendation_time'] = np.max(times)
            
            # BENCHMARK: DIFFERENT STRATEGIES
            for strategy in ['weighted_average', 'recent_weighted', 'momentum']:
                times = []
                for _ in range(50):
                    start_time = time.time()
                    _ = recommender.next_track(track_ids, strategy=strategy)
                    times.append(time.time() - start_time)
                benchmarks[f'{strategy}_avg_time'] = np.mean(times)
            
            # MEMORY USAGE (APPROXIMATE)
            import psutil
            process = psutil.Process(os.getpid())
            benchmarks['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            
            self.test_results['performance_benchmarks'] = benchmarks
            
            # SAVE BENCHMARKS
            benchmark_file = self.current_results_dir / 'performance_benchmarks.json'
            with open(benchmark_file, 'w') as f:
                json.dump(benchmarks, f, indent=2)
            
            logger.info("Performance benchmarks completed")
            self._print_performance_summary(benchmarks)
            
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error in performance benchmarks: {e}")
            return {'error': str(e)}
    
    def _print_evaluation_summary(self, results: Dict):
        """
        Print a summary of evaluation metrics.
        """
        print("\n" + "="*60)
        print("EVALUATION METRICS SUMMARY")
        print("="*60)
        
        for strategy, metrics in results.items():
            print(f"\nStrategy: {strategy}")
            print("-" * 40)
            
            # PRINT METRICS IN A FORMATTED TABLE
            metric_names = ['precision', 'recall', 'f1', 'ndcg', 'mrr']
            k_values = [1, 3, 5, 10]
            
            # PRINT HEADER
            print(f"{'Metric':<15}", end="")
            for k in k_values:
                print(f"@{k:<3}", end="  ")
            print()
            
            # PRINT EACH METRIC
            for metric_name in metric_names[:-1]:  # All except MRR
                print(f"{metric_name.upper():<15}", end="")
                for k in k_values:
                    key = f"{metric_name}@{k}"
                    if key in metrics:
                        print(f"{metrics[key]:.3f}", end="  ")
                    else:
                        print("N/A  ", end="  ")
                print()
            
            # PRINT MRR (DOESN'T HAVE @k)
            if 'mrr' in metrics:
                print(f"{'MRR':<15}{metrics['mrr']:.3f}")
        
        print("="*60 + "\n")
    
    def _print_performance_summary(self, benchmarks: Dict):
        """
        Print a summary of performance benchmarks.
        """
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARKS")
        print("="*60)
        
        print(f"Initialisation Time: {benchmarks.get('initialisation_time', 'N/A'):.3f} seconds")
        print(f"Avg Recommendation Time: {benchmarks.get('avg_recommendation_time', 'N/A'):.4f} seconds")
        print(f"Std Recommendation Time: {benchmarks.get('std_recommendation_time', 'N/A'):.4f} seconds")
        print(f"Memory Usage: {benchmarks.get('memory_usage_mb', 'N/A'):.2f} MB")
        
        print("\nStrategy-specific times:")
        for strategy in ['weighted_average', 'recent_weighted', 'momentum']:
            key = f'{strategy}_avg_time'
            if key in benchmarks:
                print(f"  {strategy}: {benchmarks[key]:.4f} seconds")
        
        print("="*60 + "\n")
    
    def generate_report(self):
        """
        Generate a comprehensive test report.
        """
        report_file = self.current_results_dir / 'test_report.json'
        
        # ADD A SUMMARY
        self.test_results['summary'] = {
            'total_tests_passed': all([
                self.test_results.get('unit_tests', {}).get('passed', False),
                self.test_results.get('integration_tests', {}).get('passed', False)
            ]),
            'dataset_path': self.dataset_path,
            'results_directory': str(self.current_results_dir)
        }
        
        # SAVE THE REPORT
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=4, default=str)
        
        # GENERATE A MARKDWON REPORT
        self._generate_markdown_report()
        
        logger.info(f"Test report saved to {report_file}")
        return self.test_results
    
    def _generate_markdown_report(self):
        """
        Generate a markdown version of the test report.
        """
        report_md = self.current_results_dir / 'report.md'
        
        with open(report_md, 'w') as f:
            f.write("# Next-Track API Test Report\n\n")
            f.write(f"**Timestamp:** {self.test_results['timestamp']}\n\n")
            
            # UNIT TESTS
            f.write("## Unit Tests\n")
            unit_results = self.test_results.get('unit_tests', {})
            status = "PASSED" if unit_results.get('passed') else "FAILED"
            f.write(f"**Status:** {status}\n\n")
            
            # INTEGRATION TESTS
            f.write("## Integration Tests\n")
            integration_results = self.test_results.get('integration_tests', {})
            status = "PASSED" if integration_results.get('passed') else "FAILED"
            f.write(f"**Status:** {status}\n\n")
            
            # EVALUATION METRICS
            f.write("## Evaluation Metrics\n\n")
            eval_results = self.test_results.get('evaluation_metrics', {})
            
            if eval_results and 'error' not in eval_results:
                for strategy, metrics in eval_results.items():
                    f.write(f"### Strategy: {strategy}\n\n")
                    f.write("| Metric | @1 | @3 | @5 | @10 |\n")
                    f.write("|--------|-----|-----|-----|-----|\n")
                    
                    for metric in ['precision', 'recall', 'f1', 'ndcg']:
                        row = f"| {metric.upper()} |"
                        for k in [1, 3, 5, 10]:
                            key = f"{metric}@{k}"
                            value = metrics.get(key, 'N/A')
                            if isinstance(value, float):
                                row += f" {value:.3f} |"
                            else:
                                row += f" {value} |"
                        f.write(row + "\n")
                    
                    if 'mrr' in metrics:
                        f.write(f"\n**MRR:** {metrics['mrr']:.3f}\n\n")
            
            # PERFORMANCE BENCHMARKS
            f.write("## Performance Benchmarks\n\n")
            perf_results = self.test_results.get('performance_benchmarks', {})
            
            if perf_results and 'error' not in perf_results:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Initialisation Time | {perf_results.get('initialisation_time', 'N/A'):.3f}s |\n")
                f.write(f"| Avg Recommendation Time | {perf_results.get('avg_recommendation_time', 'N/A'):.4f}s |\n")
                f.write(f"| Memory Usage | {perf_results.get('memory_usage_mb', 'N/A'):.2f} MB |\n")
    
    def run_all_tests(self):
        """
        Run all tests and generate comprehensive report.
        """
        logger.info("Starting comprehensive test suite...")
        
        # RUN ALL TEST CATEGORIES
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_evaluation_metrics()
        self.run_performance_benchmarks()
        
        # OPTIONAL: RUN CROSS-VALIDATION (THIS TAKES LONGER)
        # self.run_cross_validation(n_folds=5)
        
        self.generate_report()
        
        logger.info(f"All tests completed. Results saved to {self.current_results_dir}")
        
        return self.test_results


if __name__ == "__main__":
    # PARSE COMMAND LINE ARGUMENTS
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Next-Track API test suite")
    parser.add_argument('--dataset', type=str, help='Path to dataset CSV')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--evaluation', action='store_true', help='Run only evaluation metrics')
    parser.add_argument('--performance', action='store_true', help='Run only performance benchmarks')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 3],
                       help='K values for precision/recall metrics')
    
    args = parser.parse_args()
    
    # INITIALISE TEST RUNNER
    runner = TestRunner(dataset_path=args.dataset)
    
    # RUN REQUESTED TESTS
    if args.unit:
        runner.run_unit_tests()
    elif args.integration:
        runner.run_integration_tests()
    elif args.evaluation:
        runner.run_evaluation_metrics(k_values=args.k_values)
    elif args.performance:
        runner.run_performance_benchmarks()
    elif args.cv:
        runner.run_cross_validation()
    else:
        # RUN ALL TESTS BY DEFAULT
        runner.run_all_tests()
    
    runner.generate_report()
    
    print(f"\n Test suite completed. Results saved to: {runner.current_results_dir}")
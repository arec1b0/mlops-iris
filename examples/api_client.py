#!/usr/bin/env python3
"""Example API client for the MLOps Iris Classification API.

This script demonstrates how to interact with the Iris classification API
programmatically, including making predictions and handling responses.
"""

import json
import time
from typing import List, Dict, Any
import requests
from pathlib import Path


class IrisAPIClient:
    """Client for interacting with the Iris Classification API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and available endpoints.

        Returns:
            Dictionary containing API metadata
        """
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

    def check_health(self) -> Dict[str, Any]:
        """Check API health status.

        Returns:
            Dictionary containing health information
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def predict_single(self, features: List[float]) -> Dict[str, Any]:
        """Make a single prediction.

        Args:
            features: List of 4 features [sepal_length, sepal_width, petal_length, petal_width]

        Returns:
            Prediction response with class and confidence
        """
        payload = {"features": features}
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def predict_batch(self, samples: List[List[float]]) -> List[Dict[str, Any]]:
        """Make batch predictions.

        Args:
            samples: List of feature lists for batch prediction

        Returns:
            List of prediction responses
        """
        results = []
        for sample in samples:
            try:
                result = self.predict_single(sample)
                results.append(result)
            except requests.RequestException as e:
                print(f"Error predicting sample {sample}: {e}")
                results.append({"error": str(e)})
        return results

    def benchmark_predictions(self, samples: List[List[float]], num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark prediction performance.

        Args:
            samples: List of feature samples to test
            num_runs: Number of prediction runs for timing

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(num_runs):
            for sample in samples:
                start_time = time.time()
                try:
                    self.predict_single(sample)
                    end_time = time.time()
                    times.append(end_time - start_time)
                except requests.RequestException:
                    continue

        if not times:
            return {"error": "No successful predictions"}

        return {
            "total_predictions": len(times),
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "samples_per_second": len(times) / sum(times) if sum(times) > 0 else 0
        }


def main():
    """Demonstrate API usage with example data."""
    client = IrisAPIClient()

    print("🌸 MLOps Iris API Client Demo")
    print("=" * 40)

    try:
        # 1. Get API information
        print("\n1. Getting API information...")
        api_info = client.get_api_info()
        print(f"API Title: {api_info.get('message', 'Unknown')}")
        print(f"Version: {api_info.get('version', 'Unknown')}")

        # 2. Check health
        print("\n2. Checking API health...")
        health = client.check_health()
        print(f"Status: {health.get('status', 'Unknown')}")
        print(f"Model loaded: {health.get('model_loaded', False)}")
        if health.get('target_classes'):
            print(f"Target classes: {health['target_classes']}")

        # 3. Example predictions
        print("\n3. Making predictions...")

        # Sample iris measurements
        test_samples = [
            [5.1, 3.5, 1.4, 0.2],  # Should be setosa
            [6.3, 3.3, 6.0, 2.5],  # Should be virginica
            [5.7, 2.8, 4.5, 1.3],  # Should be versicolor
        ]

        for i, sample in enumerate(test_samples, 1):
            print(f"\nSample {i}: {sample}")
            result = client.predict_single(sample)
            print(f"Prediction: {result['class_name']} (class {result['prediction']})")
            print(f"Confidence: {result['confidence']}")

        # 4. Batch predictions
        print("\n4. Batch predictions...")
        batch_results = client.predict_batch(test_samples)
        for i, result in enumerate(batch_results, 1):
            if 'error' not in result:
                print(f"Sample {i}: {result['class_name']}")
            else:
                print(f"Sample {i}: Error - {result['error']}")

        # 5. Performance benchmark
        print("\n5. Performance benchmark...")
        benchmark = client.benchmark_predictions(test_samples, num_runs=50)
        if 'error' not in benchmark:
            print(f"Average prediction time: {benchmark['average_time']".4f"}s")
            print(f"Predictions per second: {benchmark['samples_per_second']".2f"}")
            print(f"Min/Max time: {benchmark['min_time']".4f"}s / {benchmark['max_time']".4f"}s")
        else:
            print(f"Benchmark error: {benchmark['error']}")

    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
        print("Make sure the API server is running on http://localhost:8000")
        print("Start it with: python run_api.py")
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()

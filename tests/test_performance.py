"""Performance and load testing for the MLOps Iris system."""

import pytest
import time
import statistics
import threading
import concurrent.futures
from unittest.mock import patch
import psutil
import os

from src.data import load_iris_data, split_data
from src.model import IrisModel


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for the ML model."""

    @pytest.fixture
    def trained_model(self, tmp_path):
        """Create trained model for performance testing."""
        X, y, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )

        model = IrisModel(tmp_path / "perf_model.pkl")
        model.train(X_train, y_train)

        return model, X_test

    def test_single_prediction_performance(self, trained_model):
        """Test single prediction response time."""
        model, X_test = trained_model

        # Warm up
        model.predict(X_test[0])

        # Measure performance
        times = []
        for _ in range(100):
            start = time.perf_counter()
            model.predict(X_test[0])
            end = time.perf_counter()
            times.append(end - start)

        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile

        # Performance assertions
        assert avg_time < 0.001, ".4f"  # Average < 1ms
        assert p95_time < 0.005, ".4f"  # P95 < 5ms

        print(".6f")
        print(".6f")

    def test_batch_prediction_performance(self, trained_model):
        """Test batch prediction performance."""
        model, X_test = trained_model

        batch_sizes = [1, 10, 50, 100]
        results = {}

        for batch_size in batch_sizes:
            batch_data = X_test[:batch_size]

            # Warm up
            for features in batch_data[:5]:  # Warm up with first 5
                model.predict(features)

            # Measure batch performance
            start = time.perf_counter()
            predictions = [model.predict(features) for features in batch_data]
            end = time.perf_counter()

            total_time = end - start
            avg_time_per_prediction = total_time / batch_size

            results[batch_size] = {
                "total_time": total_time,
                "avg_time_per_pred": avg_time_per_prediction,
                "predictions_per_sec": batch_size / total_time,
            }

            print(
                f"Batch size {batch_size}: {avg_time_per_prediction:.6f}s per prediction, "
                f"{results[batch_size]['predictions_per_sec']:.1f} pred/sec"
            )

        # Performance should scale reasonably
        assert (
            results[100]["avg_time_per_pred"] < 0.01
        )  # < 10ms per prediction even in batch

    def test_memory_usage(self, trained_model):
        """Test model memory usage."""
        model, _ = trained_model

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load model into memory (if not already loaded)
        _ = model.model

        # Get memory after model loading
        loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = loaded_memory - initial_memory

        print(".2f")
        print(".2f")

        # Model should not consume excessive memory
        assert memory_increase < 50  # Less than 50MB increase

    def test_model_inference_scalability(self, trained_model):
        """Test model performance with different input scales."""
        model, X_test = trained_model

        # Test with different numbers of concurrent predictions
        concurrency_levels = [1, 5, 10, 20]

        for concurrency in concurrency_levels:
            start_time = time.time()

            def predict_worker():
                for _ in range(10):  # Each worker makes 10 predictions
                    model.predict(X_test[0])

            threads = []
            for _ in range(concurrency):
                thread = threading.Thread(target=predict_worker)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            total_time = time.time() - start_time
            total_predictions = concurrency * 10
            predictions_per_second = total_predictions / total_time

            print(f"Concurrency {concurrency}: {predictions_per_second:.1f} pred/sec")

            # Should maintain reasonable performance
            assert predictions_per_second > 10  # At least 10 predictions per second


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for the API."""

    @pytest.fixture
    def api_client(self):
        """Create FastAPI test client."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        return TestClient(app)

    def test_api_response_time(self, api_client):
        """Test API response time with mocked model."""
        features = [5.1, 3.5, 1.4, 0.2]

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")

            # Warm up
            api_client.post("/predict", json={"features": features})

            # Measure response times
            times = []
            for _ in range(50):
                start = time.perf_counter()
                response = api_client.post("/predict", json={"features": features})
                end = time.perf_counter()

                assert response.status_code == 200
                times.append(end - start)

            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]

            print(".4f")
            print(".4f")

            # API should respond quickly
            assert avg_time < 0.1  # Average < 100ms
            assert p95_time < 0.5  # P95 < 500ms

    def test_api_concurrent_load(self, api_client):
        """Test API performance under concurrent load."""
        features = [5.1, 3.5, 1.4, 0.2]

        def make_request(request_id):
            with patch("src.api.main.get_model") as mock_get_model:
                mock_model = mock_get_model.return_value
                mock_model.predict.return_value = (
                    request_id % 3,
                    "setosa",
                )  # Vary response

                start = time.perf_counter()
                response = api_client.post("/predict", json={"features": features})
                end = time.perf_counter()

                return response.status_code, end - start

        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10]

        for concurrency in concurrency_levels:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                futures = [
                    executor.submit(make_request, i) for i in range(concurrency * 2)
                ]

                results = []
                for future in concurrent.futures.as_completed(futures):
                    status, response_time = future.result()
                    results.append((status, response_time))

                # All requests should succeed
                assert all(status == 200 for status, _ in results)

                avg_response_time = statistics.mean([rt for _, rt in results])
                max_response_time = max([rt for _, rt in results])

                print(
                    f"Concurrency {concurrency}: avg={avg_response_time:.4f}s, "
                    f"max={max_response_time:.4f}s"
                )

                # Performance should degrade gracefully
                assert avg_response_time < 1.0  # Still under 1 second average
                assert max_response_time < 5.0  # Max response under 5 seconds

    def test_api_memory_leak_detection(self, api_client):
        """Test for potential memory leaks in API."""
        features = [5.1, 3.5, 1.4, 0.2]

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")

            # Make many requests
            for i in range(100):
                response = api_client.post("/predict", json={"features": features})
                assert response.status_code == 200

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(".2f")

        # Memory increase should be minimal (allowing for some GC variations)
        assert memory_increase < 10  # Less than 10MB increase after 100 requests


@pytest.mark.performance
@pytest.mark.slow
class TestSystemLoad:
    """System-wide load and stress tests."""

    def test_sustained_load(self, api_client):
        """Test sustained load over time."""
        features = [5.1, 3.5, 1.4, 0.2]
        duration = 10  # 10 seconds
        request_count = 0

        start_time = time.time()

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")

            while time.time() - start_time < duration:
                response = api_client.post("/predict", json={"features": features})
                assert response.status_code == 200
                request_count += 1

        requests_per_second = request_count / duration

        print(f"Sustained load: {requests_per_second:.1f} req/sec over {duration}s")

        # Should maintain reasonable throughput
        assert requests_per_second > 5  # At least 5 requests per second sustained

    def test_large_payload_handling(self, api_client):
        """Test handling of large numbers of features."""
        # Test with maximum reasonable batch size
        batch_sizes = [10, 50, 100]

        for batch_size in batch_sizes:
            # Create batch request (simulate multiple features)
            batch_features = [[5.1, 3.5, 1.4, 0.2]] * batch_size

            with patch("src.api.main.get_model") as mock_get_model:
                mock_model = mock_get_model.return_value
                mock_model.predict.return_value = (0, "setosa")

                start = time.perf_counter()

                # Make multiple requests to simulate batch processing
                for features in batch_features:
                    response = api_client.post("/predict", json={"features": features})
                    assert response.status_code == 200

                end = time.perf_counter()
                total_time = end - start

                print(
                    f"Batch size {batch_size}: {total_time:.4f}s total, "
                    f"{total_time / batch_size:.6f}s per request"
                )

                # Should handle batch processing reasonably
                assert total_time / batch_size < 0.1  # Less than 100ms per request


# Performance benchmarking utilities
def benchmark_function(func, iterations=100, warmup=10):
    """Benchmark a function's performance."""
    # Warm up
    for _ in range(warmup):
        func()

    # Measure performance
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)

        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "p95": statistics.quantiles(times, n=20)[18]
            if len(times) >= 20
            else max(times),
            "p99": statistics.quantiles(times, n=20)[19]
            if len(times) >= 20
            else max(times),
        }


@pytest.mark.performance
def test_model_benchmark(trained_model):
    """Comprehensive model performance benchmark."""
    model, X_test, _ = trained_model

    def single_prediction():
        return model.predict(X_test[0])

    # Benchmark single predictions
    results = benchmark_function(single_prediction, iterations=1000, warmup=100)

    print("\nModel Performance Benchmark:")
    print(f"Mean: {results['mean']:.6f}s")
    print(f"Median: {results['median']:.6f}s")
    print(f"P95: {results['p95']:.6f}s")
    print(f"P99: {results['p99']:.6f}s")
    print(f"Std Dev: {results['std']:.6f}s")

    # Performance assertions
    assert results["mean"] < 0.001  # Average < 1ms
    assert results["p95"] < 0.005  # P95 < 5ms
    assert results["p99"] < 0.01  # P99 < 10ms

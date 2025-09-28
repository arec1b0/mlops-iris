#!/usr/bin/env python3
"""
Health check script for MLOps Iris API
Usage: python scripts/health_check.py [host] [port]
"""

import sys
import requests
import time
from typing import Optional


def check_api_health(host: str = "localhost", port: int = 8000, timeout: int = 30) -> bool:
    """Check if the API is healthy."""
    url = f"http://{host}:{port}/health"

    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        response_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy"            print(".3f"            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Classes: {len(data.get('target_classes', []))}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to API: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_api_predictions(host: str = "localhost", port: int = 8000) -> bool:
    """Test API prediction functionality."""
    url = f"http://{host}:{port}/predict"
    test_data = {"features": [5.1, 3.5, 1.4, 0.2]}

    try:
        response = requests.post(url, json=test_data, timeout=10)

        if response.status_code == 200:
            data = response.json()
            required_fields = ["prediction", "class_name", "confidence"]

            if all(field in data for field in required_fields):
                print("✅ Prediction API working"                print(f"   Prediction: {data['prediction']}")
                print(f"   Class: {data['class_name']}")
                return True
            else:
                print("❌ Prediction response missing required fields"                return False
        else:
            print(f"❌ Prediction API returned status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False


def main():
    """Main health check function."""
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    print(f"🔍 Checking MLOps Iris API at {host}:{port}")
    print("=" * 50)

    # Check health endpoint
    health_ok = check_api_health(host, port)
    print()

    # Check prediction functionality
    prediction_ok = check_api_predictions(host, port)
    print()

    # Overall result
    if health_ok and prediction_ok:
        print("🎉 All health checks passed!")
        sys.exit(0)
    else:
        print("💥 Some health checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

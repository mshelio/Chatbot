import requests
import json
import time

# Base URL of your running API
BASE_URL = "http://localhost:5000"


def test_endpoint(endpoint, payload=None):
    """Generic function to test API endpoints"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if payload:
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url)

        print(f"\nEndpoint: {endpoint}")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except Exception as e:
        print(f"Error testing {endpoint}: {str(e)}")
        return None


def main():
    # Test health check endpoint
    test_endpoint("/health")

    # Test chat endpoint with various financial queries
    test_cases = [
        {"message": "Hello!", "description": "Greeting"},
        {"message": "How do I pay off my credit card debt?", "description": "Debt advice"},
        {"message": "What's a good emergency fund amount?", "description": "Emergency fund"},
        {"message": "How should I invest for retirement?", "description": "Retirement investing"},
        {"message": "What's the best way to protect money in Lebanese banks?", "description": "Lebanon banking"},
        {"message": "Explain the debt snowball method", "description": "Debt snowball"},
        {"message": "What are the 7 baby steps?", "description": "Baby steps overview"},
        {"message": "I lost my job, what should I do?", "description": "Crisis handling"},
        {"message": "How can I make extra money?", "description": "Side hustles"},
        {"message": "Tell me a Dave Ramsey quote", "description": "Inspirational quote"}
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n{'=' * 50}")
        print(f"TEST CASE #{i + 1}: {test_case['description']}")
        print(f"{'=' * 50}")

        payload = {
            "message": test_case["message"],
            "user_id": f"test_user_{i + 1}"
        }

        response = test_endpoint("/chat", payload)

        # Add delay between tests to avoid overwhelming the server
        time.sleep(0.5)


if __name__ == "__main__":
    main()
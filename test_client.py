"""
Test client for the image processing service.
Usage: python test_client.py
"""
import asyncio
import httpx
import json
from datetime import datetime


async def test_health_check():
    """Test the health check endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get("http://192.168.10.74:8008/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200


async def test_process_images():
    """Test the main image processing endpoint."""

    # Sample request payload (remains the same)
    request_data = {
        "id": 245,
        "conversation_id": 196,
        "message_id": 1420,
        "template_id": 1,
        "template_color_id": 1,
        "parameters": None,
        "output": {
            "front": "https://acquires.in/cdn/shop/files/smooth-white-cotton-t-shirt-with-beautiful-3d-design-879335.jpg?v=1723878953",
            "back": "https://vectorbazarbd.com/cdn/shop/files/VectorB-104..jpg?v=1736258208"
        }
    }

    print("Sending process request...")
    print(f"Request: {json.dumps(request_data, indent=2)}\n")

    start_time = datetime.now()

    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
        try:
            response = await client.post(
                "http://192.168.10.74:8008/process",
                json=request_data
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"Process Status: {response.status_code}")
            print(f"Client-side elapsed time: {elapsed:.2f}s")

            if response.status_code == 200:
                result = response.json()
                print(f"\nResponse:")
                print(json.dumps(result, indent=2))

                # --- MODIFIED: Check for new response fields ---
                print(f"\nOutputs generated:")
                print(f"  - Front Output: {result['front_output']}")
                print(f"  - Back Output: {result['back_output']}")
                print(f"\nServer processing time: {result['processing_time_seconds']:.2f}s")
            else:
                print(f"Error: {response.text}")

            return response.status_code == 200

        except httpx.TimeoutException:
            print("Request timed out!")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Image Processing Service Test Suite")
    print("=" * 60)
    print()

    # Test 1: Health check
    print("Test 1: Health Check")
    print("-" * 60)
    health_ok = await test_health_check()

    if not health_ok:
        print("⚠️  Service is not healthy. Please check if the service is running.")
        return

    print("✅ Health check passed\n")

    # Test 2: Single request
    print("Test 2: Single Image Processing Request")
    print("-" * 60)
    process_ok = await test_process_images()

    if process_ok:
        print("\n✅ Single request test passed\n")
    else:
        print("\n❌ Single request test failed\n")

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
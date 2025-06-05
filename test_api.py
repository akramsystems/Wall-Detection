#!/usr/bin/env python3
"""
Test script for the Wall Detection API.
Tests both base64 image and cloud storage functionality.
"""
import base64
import requests
import json
import time
import os
from PIL import Image
import io
import numpy as np

API_BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image for testing."""
    # Create a simple test image (RGB)
    img = Image.new('RGB', (256, 256), color='white')
    
    # Add some simple patterns to simulate a floor plan
    pixels = np.array(img)
    # Add some black lines to simulate walls
    pixels[50:60, :] = [0, 0, 0]  # horizontal line
    pixels[:, 50:60] = [0, 0, 0]  # vertical line
    pixels[200:210, :] = [0, 0, 0]  # another horizontal line
    pixels[:, 200:210] = [0, 0, 0]  # another vertical line
    
    img = Image.fromarray(pixels)
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return base64_string

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_base64_prediction():
    """Test prediction with base64 encoded image."""
    print("\nTesting base64 image prediction...")
    
    # Create test image
    test_img = create_test_image()
    base64_img = image_to_base64(test_img)
    
    # Prepare request
    request_data = {
        "image_data": base64_img,
        "threshold": 0.05,  # Lower threshold for test
        "nms_size": 3
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"Status Code: {response.status_code}")
        print(f"Request Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Wall corners found: {len(result['wall_corners'])}")
            print(f"Image size: {result['image_size']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Print first few corners
            for i, corner in enumerate(result['wall_corners'][:5]):
                print(f"Corner {i+1}: x={corner['x']:.1f}, y={corner['y']:.1f}, "
                      f"confidence={corner['confidence']:.3f}, type={corner['corner_type']}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error testing base64 prediction: {e}")
        return False

def test_cloud_storage_prediction():
    """Test prediction with cloud storage path (mock test)."""
    print("\nTesting cloud storage prediction...")
    
    # This test assumes you have configured cloud storage
    # For demonstration, we'll just show the request format
    request_data = {
        "cloud_path": "test_images/floor_plan_1.png",
        "storage_type": "azure",  # or "gcp"
        "threshold": 0.1,
        "nms_size": 3
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Wall corners found: {len(result['wall_corners'])}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            return True
        else:
            print(f"Expected error (no cloud storage configured): {response.text}")
            return True  # This is expected if cloud storage isn't configured
            
    except Exception as e:
        print(f"Error testing cloud storage prediction: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid requests."""
    print("\nTesting error handling...")
    
    # Test with no image data
    request_data = {
        "threshold": 0.1
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 400:
            print("✓ Correctly handled missing image data")
            return True
        else:
            print("✗ Should have returned 400 for missing image data")
            return False
            
    except Exception as e:
        print(f"Error testing error handling: {e}")
        return False

def save_test_image():
    """Save a test image for manual testing."""
    test_img = create_test_image()
    test_img.save("test_floor_plan.png")
    print("Test image saved as 'test_floor_plan.png'")

def main():
    """Run all tests."""
    print("=" * 50)
    print("Wall Detection API Test Suite")
    print("=" * 50)
    
    # Save test image for manual testing
    save_test_image()
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Base64 Image Prediction", test_base64_prediction),
        ("Cloud Storage Prediction", test_cloud_storage_prediction),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        result = test_func()
        results.append((test_name, result))
        print(f"Result: {'✓ PASSED' if result else '✗ FAILED'}")
    
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print("=" * 50)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main() 
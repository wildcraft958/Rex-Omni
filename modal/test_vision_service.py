#!/usr/bin/env python3
"""Test Rex Vision Service (SAM)"""

import base64
import json
import requests

# Read and encode image
with open("tutorials/detection_example/test_images/cafe.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

print("Testing Rex Vision Service (SAM)...")
print(f"Image size: {len(image_b64)} chars")

# Test SAM endpoint
url = "https://animeshraj958--rex-vision-service-api-sam.modal.run"
payload = {
    "image": image_b64,
    "categories": ["person", "laptop"]
}

try:
    response = requests.post(url, json=payload, timeout=120)
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data.get('success')}")
        print(f"Number of SAM results: {len(data.get('sam_results', []))}")
        print(f"\nFirst SAM result:")
        if data.get('sam_results'):
            result = data['sam_results'][0]
            print(f"  Category: {result['category']}")
            print(f"  Box: {result['box']}")
            print(f"  Score: {result['score']}")
            print(f"  Polygons: {len(result['polygons'])} polygon(s)")
    else:
        print(f"Response:\n{response.text}")
except requests.exceptions.Timeout:
    print("Request timed out (model still loading...)")
except Exception as e:
    print(f"Error: {e}")
    if 'response' in locals():
        print(f"Response text: {response.text}")

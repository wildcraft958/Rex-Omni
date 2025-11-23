#!/usr/bin/env python3
"""Test Rex LLM Service"""

import base64
import json
import requests

# Read and encode image
with open("tutorials/detection_example/test_images/cafe.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

print("Testing Rex LLM Service...")
print(f"Image size: {len(image_b64)} chars")

# Test endpoint
url = "https://animeshraj958--rex-llm-service-rex-inference.modal.run"
payload = {
    "image": image_b64,
    "task": "detection",
    "categories": ["person", "cup", "laptop"]
}

try:
    response = requests.post(url, json=payload, timeout=120)
    print(f"\nStatus: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
except requests.exceptions.Timeout:
    print("Request timed out (model still loading...)")
except Exception as e:
    print(f"Error: {e}")
    print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")

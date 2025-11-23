import requests
import base64
import json
import argparse
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_inference(base_url, image_path):
    # Construct URL: base-api-inference.modal.run
    url = f"{base_url}-api-inference.modal.run"
    print(f"\nTesting Inference Endpoint: {url}")
    
    payload = {
        "image": encode_image(image_path),
        "task": "detection",
        "categories": ["person", "cup", "laptop"]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Success!")
        print(json.dumps(result, indent=2)[:500] + "...") # Print first 500 chars
    except Exception as e:
        print(f"Failed: {e}")
        if 'response' in locals():
            print(response.text)

def test_sam(base_url, image_path):
    url = f"{base_url}-api-sam.modal.run"
    print(f"\nTesting SAM Endpoint: {url}")
    
    payload = {
        "image": encode_image(image_path),
        "categories": ["person"]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Success!")
        # Print summary of SAM results
        if result.get("success"):
            print(f"Found {len(result.get('sam_results', []))} segmented objects.")
        else:
            print(result)
    except Exception as e:
        print(f"Failed: {e}")

def test_grounding(base_url, image_path):
    url = f"{base_url}-api-grounding.modal.run"
    print(f"\nTesting Grounding Endpoint: {url}")
    
    payload = {
        "image": encode_image(image_path),
        "caption": "A person is sitting at a table with a laptop."
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Success!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Failed: {e}")

def test_health(base_url):
    url = f"{base_url}-health.modal.run"
    print(f"\nTesting Health Endpoint: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Rex-Omni Modal Deployment")
    parser.add_argument("--url", required=True, help="Base URL of the deployed Modal app")
    parser.add_argument("--image", default="tutorials/detection_example/test_images/cafe.jpg", help="Path to test image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        exit(1)
        
    test_health(args.url)
    test_inference(args.url, args.image)
    test_sam(args.url, args.image)
    test_grounding(args.url, args.image)

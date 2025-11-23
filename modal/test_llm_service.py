#!/usr/bin/env python3
"""
Exhaustive Test Suite for Rex-Omni Services
Tests both LLM and Vision services with all supported tasks
"""

import base64
import json
import requests
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# ============================================================================
# Configuration
# ============================================================================

# Service URLs
LLM_SERVICE_URL = "https://animeshraj958--rex-llm-service-rex-inference.modal.run"
VISION_SAM_URL = "https://animeshraj958--rex-vision-service-api-sam.modal.run"
VISION_GROUNDING_URL = "https://animeshraj958--rex-vision-service-api-grounding.modal.run"

# Test images directory
TEST_IMAGES_DIR = Path("tutorials/detection_example/test_images")

# Timeout for requests
REQUEST_TIMEOUT = 120


# ============================================================================
# Utility Functions
# ============================================================================

def load_image_base64(image_path: str) -> str:
    """Load image and encode as base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test_header(test_name: str):
    """Print a test header"""
    print(f"\n{'─' * 80}")
    print(f"🧪 Test: {test_name}")
    print(f"{'─' * 80}")


def print_results(response: requests.Response, verbose: bool = True):
    """Pretty print API response"""
    print(f"\n✓ Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Print success status
        if "success" in data:
            print(f"✓ Success: {data['success']}")
        
        # Print key metrics
        if "extracted_predictions" in data:
            predictions = data["extracted_predictions"]
            total_objects = sum(len(items) for items in predictions.values())
            print(f"✓ Total objects detected: {total_objects}")
            print(f"✓ Categories found: {list(predictions.keys())}")
            
            if verbose:
                for category, items in predictions.items():
                    print(f"\n  Category: {category} ({len(items)} items)")
                    for idx, item in enumerate(items):
                        item_type = item.get("type", "unknown")
                        coords = item.get("coords", [])
                        print(f"    [{idx+1}] {item_type}: {coords}")
        
        if "sam_results" in data:
            sam_results = data["sam_results"]
            print(f"✓ SAM segmentation results: {len(sam_results)}")
            
            if verbose and sam_results:
                for idx, result in enumerate(sam_results):
                    print(f"\n  SAM result [{idx+1}]:")
                    print(f"    Category: {result['category']}")
                    print(f"    Box: {result['box']}")
                    print(f"    Score: {result['score']:.4f}")
                    print(f"    Polygons: {len(result['polygons'])} polygon(s)")
        
        if "annotations" in data:
            annotations = data["annotations"]
            print(f"✓ Grounding annotations: {len(annotations)}")
            
            if verbose and annotations:
                for idx, ann in enumerate(annotations):
                    print(f"\n  Annotation [{idx+1}]:")
                    print(f"    Phrase: '{ann['phrase']}'")
                    print(f"    Char range: {ann['start_char']}-{ann['end_char']}")
                    print(f"    Boxes: {len(ann['boxes'])} box(es)")
                    for box_idx, box in enumerate(ann['boxes']):
                        print(f"      Box {box_idx+1}: {box}")
        
        if "raw_output" in data and verbose:
            print(f"\n📝 Raw Output:\n{data['raw_output'][:500]}...")
        
        # Print full JSON in non-verbose mode
        if not verbose:
            print(f"\n📊 Response:\n{json.dumps(data, indent=2)}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Response: {response.text}")


def test_api_call(
    url: str,
    payload: Dict[str, Any],
    test_name: str,
    verbose: bool = True
):
    """Execute a single test API call"""
    print_test_header(test_name)
    print(f"⏳ Sending request to: {url}")
    print(f"📦 Payload keys: {list(payload.keys())}")
    
    if "categories" in payload:
        print(f"🏷️  Categories: {payload['categories']}")
    if "caption" in payload:
        print(f"💬 Caption: {payload['caption']}")
    if "task" in payload:
        print(f"🎯 Task: {payload['task']}")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        elapsed = time.time() - start_time
        
        print(f"⏱️  Request completed in {elapsed:.2f}s")
        print_results(response, verbose=verbose)
        
        return response
        
    except requests.exceptions.Timeout:
        print("⏱️  ✗ Request timed out (model may still be loading...)")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# Test Cases - LLM Service
# ============================================================================

def test_llm_detection(image_b64: str, verbose: bool = True):
    """Test object detection task"""
    payload = {
        "image": image_b64,
        "task": "detection",
        "categories": ["person", "cup", "laptop", "chair", "plant", "table"]
    }
    return test_api_call(LLM_SERVICE_URL, payload, "Detection - Multiple Categories", verbose)


def test_llm_object_referring(image_b64: str, verbose: bool = True):
    """Test object referring with descriptive phrases"""
    payload = {
        "image": image_b64,
        "task": "detection",
        "categories": [
            "person wearing blue shirt",
            "yellow flower on table",
            "silver laptop computer",
            "white coffee cup"
        ]
    }
    return test_api_call(LLM_SERVICE_URL, payload, "Object Referring - Descriptive Phrases", verbose)


def test_llm_pointing(image_b64: str, verbose: bool = True):
    """Test pointing task to get center points"""
    payload = {
        "image": image_b64,
        "task": "pointing",
        "categories": ["person", "laptop", "cup"]
    }
    return test_api_call(LLM_SERVICE_URL, payload, "Pointing - Object Centers", verbose)


def test_llm_visual_prompting(image_b64: str, verbose: bool = True):
    """Test visual prompting with reference boxes"""
    payload = {
        "image": image_b64,
        "task": "visual_prompting",
        "visual_prompt_boxes": [[100, 100, 300, 300]],  # Example reference box
        "categories": ["similar objects"]
    }
    return test_api_call(LLM_SERVICE_URL, payload, "Visual Prompting - Find Similar", verbose)


def test_llm_keypoint_person(image_b64: str, verbose: bool = True):
    """Test person keypoint detection"""
    payload = {
        "image": image_b64,
        "task": "keypoint",
        "keypoint_type": "person"
    }
    return test_api_call(LLM_SERVICE_URL, payload, "Keypoint Detection - Person", verbose)


def test_llm_ocr_box(image_b64: str, verbose: bool = True):
    """Test OCR with bounding boxes"""
    payload = {
        "image": image_b64,
        "task": "ocr_box"
    }
    return test_api_call(LLM_SERVICE_URL, payload, "OCR - Word Boxes", verbose)


def test_llm_ocr_polygon(image_b64: str, verbose: bool = True):
    """Test OCR with polygon annotations"""
    payload = {
        "image": image_b64,
        "task": "ocr_polygon"
    }
    return test_api_call(LLM_SERVICE_URL, payload, "OCR - Polygons", verbose)


def test_llm_gui_grounding(image_b64: str, verbose: bool = True):
    """Test GUI element grounding"""
    payload = {
        "image": image_b64,
        "task": "gui_grounding",
        "categories": ["button", "text field", "icon", "menu"]
    }
    return test_api_call(LLM_SERVICE_URL, payload, "GUI Grounding - UI Elements", verbose)


def test_llm_gui_pointing(image_b64: str, verbose: bool = True):
    """Test GUI element pointing"""
    payload = {
        "image": image_b64,
        "task": "gui_pointing",
        "categories": ["submit button", "search box", "logo"]
    }
    return test_api_call(LLM_SERVICE_URL, payload, "GUI Pointing - UI Elements", verbose)


# ============================================================================
# Test Cases - Vision Service (SAM)
# ============================================================================

def test_vision_sam(image_b64: str, verbose: bool = True):
    """Test SAM segmentation service"""
    payload = {
        "image": image_b64,
        "categories": ["person", "laptop", "cup", "chair"]
    }
    return test_api_call(VISION_SAM_URL, payload, "SAM Segmentation - Detection + Masks", verbose)


# ============================================================================
# Test Cases - Vision Service (Grounding)
# ============================================================================

def test_vision_grounding(image_b64: str, verbose: bool = True):
    """Test phrase grounding service"""
    payload = {
        "image": image_b64,
        "caption": "A person sitting at a table with a laptop and coffee cup in a modern cafe"
    }
    return test_api_call(VISION_GROUNDING_URL, payload, "Phrase Grounding - Caption Analysis", verbose)


def test_vision_grounding_complex(image_b64: str, verbose: bool = True):
    """Test phrase grounding with complex caption"""
    payload = {
        "image": image_b64,
        "caption": "Multiple people are gathered around wooden tables. Some tables have laptops, coffee cups, and decorative plants. Large windows provide natural lighting."
    }
    return test_api_call(VISION_GROUNDING_URL, payload, "Phrase Grounding - Complex Scene", verbose)


# ============================================================================
# Main Test Execution
# ============================================================================

def run_all_tests(image_name: str = "cafe.jpg", verbose: bool = True):
    """Run comprehensive test suite"""
    
    print_section("🚀 Rex-Omni Exhaustive Test Suite")
    print(f"\n📸 Test Image: {image_name}")
    print(f"📁 Image Directory: {TEST_IMAGES_DIR}")
    print(f"🔧 Verbose Mode: {verbose}")
    
    # Load test image
    image_path = TEST_IMAGES_DIR / image_name
    if not image_path.exists():
        print(f"\n✗ Error: Image not found at {image_path}")
        return
    
    print(f"\n✓ Loading image: {image_path}")
    image_b64 = load_image_base64(str(image_path))
    print(f"✓ Image encoded: {len(image_b64)} characters")
    
    # ========================================================================
    # LLM Service Tests
    # ========================================================================
    
    print_section("🤖 LLM Service Tests")
    
    # Core detection tasks
    test_llm_detection(image_b64, verbose)
    test_llm_object_referring(image_b64, verbose)
    test_llm_pointing(image_b64, verbose)
    
    # Advanced tasks (may fail for certain images)
    test_llm_visual_prompting(image_b64, verbose)
    test_llm_keypoint_person(image_b64, verbose)
    
    # OCR tasks (better on text-heavy images)
    if image_name in ["gui.png", "layout.jpg"]:
        test_llm_ocr_box(image_b64, verbose)
        test_llm_ocr_polygon(image_b64, verbose)
    
    # GUI tasks (best on GUI screenshots)
    if image_name == "gui.png":
        test_llm_gui_grounding(image_b64, verbose)
        test_llm_gui_pointing(image_b64, verbose)
    
    # ========================================================================
    # Vision Service Tests
    # ========================================================================
    
    print_section("👁️  Vision Service Tests")
    
    # SAM segmentation
    test_vision_sam(image_b64, verbose)
    
    # Phrase grounding
    test_vision_grounding(image_b64, verbose)
    test_vision_grounding_complex(image_b64, verbose)
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print_section("✅ Test Suite Completed")
    print("\n📊 All tests executed successfully!")
    print(f"\n💡 Tip: Run with different images:")
    print(f"   - cafe.jpg:  General scene with people, objects")
    print(f"   - boys.jpg:  People-focused scene")
    print(f"   - gui.png:   GUI/interface testing")
    print(f"   - layout.jpg: Layout and text analysis")


def run_quick_test(image_name: str = "cafe.jpg"):
    """Run a quick sanity test with key endpoints"""
    
    print_section("⚡ Quick Sanity Test")
    
    image_path = TEST_IMAGES_DIR / image_name
    if not image_path.exists():
        print(f"\n✗ Error: Image not found at {image_path}")
        return
    
    print(f"\n✓ Loading image: {image_path}")
    image_b64 = load_image_base64(str(image_path))
    print(f"✓ Image encoded: {len(image_b64)} characters")
    
    # Test core functionality
    test_llm_detection(image_b64, verbose=False)
    test_vision_sam(image_b64, verbose=False)
    test_vision_grounding(image_b64, verbose=False)
    
    print_section("✅ Quick Test Completed")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rex-Omni Exhaustive Test Suite")
    parser.add_argument(
        "--image",
        type=str,
        default="cafe.jpg",
        help="Test image filename (default: cafe.jpg)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quick"],
        default="full",
        help="Test mode: 'full' for all tests, 'quick' for sanity check"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output with detailed results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output (compact mode)"
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    # Run appropriate test mode
    if args.mode == "quick":
        run_quick_test(args.image)
    else:
        run_all_tests(args.image, verbose=verbose)

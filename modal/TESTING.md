# Rex-Omni API Testing - Quick Reference

Complete suite of testing tools for Rex-Omni API services.

---

## 📋 Quick Links

- **[cURL Commands Documentation](CURL_COMMANDS.md)** - Complete reference with all endpoints
- **[Python Test Suite](#python-test-suite)** - Exhaustive automated testing
- **[Shell Scripts](#shell-scripts)** - Quick command-line testing

---

## 🚀 Quick Start

### Option 1: cURL (Fastest)

```bash
# Make script executable
chmod +x curl_examples.sh

# Run quick test
./curl_examples.sh quick

# Run specific endpoint
./curl_examples.sh detection
./curl_examples.sh sam
./curl_examples.sh grounding

# Run all examples
./curl_examples.sh all
```

### Option 2: Python Test Suite (Most Comprehensive)

```bash
# Quick sanity test
python3 test_llm_service.py --mode quick

# Full test suite (verbose)
python3 test_llm_service.py

# Test specific image
python3 test_llm_service.py --image boys.jpg --mode full

# Quiet mode (less output)
python3 test_llm_service.py --quiet
```

### Option 3: Shell Script Wrapper

```bash
# Make executable
chmod +x test_llm_service.sh

# Run different test modes
./test_llm_service.sh quick      # Quick sanity check
./test_llm_service.sh cafe       # Full test on cafe.jpg
./test_llm_service.sh boys       # Full test on boys.jpg  
./test_llm_service.sh gui        # Full test on gui.png
./test_llm_service.sh all        # Test all images
```

---

## 📁 Files Overview

### Documentation
- **`CURL_COMMANDS.md`** - Complete cURL reference for all API endpoints

### Testing Scripts
- **`test_llm_service.py`** - Exhaustive Python test suite for all services
- **`test_llm_service.sh`** - Shell wrapper for Python tests
- **`curl_examples.sh`** - Standalone cURL examples
- **`test_vision_service.py`** - Legacy vision service test

### Support Files
- **`README.md`** - Main project documentation
- **`TESTING.md`** - This file

---

## 🎯 API Endpoints

### LLM Service
**Base URL:** `https://animeshraj958--rex-llm-service-rex-inference.modal.run`

**Supported Tasks:**
- ✅ `detection` - Object detection with bounding boxes
- ✅ `pointing` - Object center point detection
- ✅ `visual_prompting` - Find similar objects
- ✅ `keypoint` - Keypoint detection (person/hand/animal)
- ✅ `ocr_box` - OCR with word-level boxes
- ✅ `ocr_polygon` - OCR with polygon annotations
- ✅ `gui_grounding` - UI element detection
- ✅ `gui_pointing` - UI element center points

### Vision Service
**SAM Endpoint:** `https://animeshraj958--rex-vision-service-api-sam.modal.run`
- Combines Rex-Omni detection + SAM segmentation
- Returns bounding boxes + segmentation polygons

**Grounding Endpoint:** `https://animeshraj958--rex-vision-service-api-grounding.modal.run`
- Phrase grounding from natural language captions
- Extracts noun phrases using SpaCy
- Returns phrase-to-region mappings

---

## 🧪 Test Examples

### cURL Example: Object Detection

```bash
# Encode image
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

# Send request
curl -X POST "https://animeshraj958--rex-llm-service-rex-inference.modal.run" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\", \"cup\", \"laptop\"]
  }" | jq '.extracted_predictions'
```

### Python Example: SAM Segmentation

```python
import base64
import requests

# Load and encode image
with open("tutorials/detection_example/test_images/cafe.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Call SAM endpoint
response = requests.post(
    "https://animeshraj958--rex-vision-service-api-sam.modal.run",
    json={
        "image": image_b64,
        "categories": ["person", "laptop", "cup"]
    },
    timeout=120
)

# Get results
data = response.json()
sam_results = data["sam_results"]  # List of segmentation results
```

---

## 📊 Response Formats

### Detection Response
```json
{
  "success": true,
  "extracted_predictions": {
    "person": [
      {"type": "box", "coords": [100, 150, 250, 450]}
    ],
    "cup": [
      {"type": "box", "coords": [300, 200, 350, 280]}
    ]
  },
  "raw_output": "..."
}
```

### SAM Response
```json
{
  "success": true,
  "predictions": { ... },
  "sam_results": [
    {
      "category": "person",
      "box": [100, 150, 250, 450],
      "score": 0.9876,
      "polygons": [[x1,y1,x2,y2,...]]
    }
  ]
}
```

### Grounding Response
```json
{
  "success": true,
  "image_size": [800, 600],
  "caption": "A person with a laptop",
  "annotations": [
    {
      "phrase": "person",
      "start_char": 2,
      "end_char": 8,
      "boxes": [[100, 150, 250, 450]]
    }
  ]
}
```

---

## 🛠️ Python Test Suite Features

The `test_llm_service.py` script provides:

### ✨ Comprehensive Testing
- All LLM service tasks (9 different tasks)
- Vision service endpoints (SAM + Grounding)
- Multiple test images
- Verbose and quiet modes

### 📋 CLI Options
```bash
# Usage
python3 test_llm_service.py [OPTIONS]

# Options
--image IMAGE          Test image (cafe.jpg, boys.jpg, gui.png, layout.jpg)
--mode MODE           Test mode: 'full' or 'quick'
--verbose             Detailed output with predictions
--quiet               Compact output mode

# Examples
python3 test_llm_service.py --image cafe.jpg --mode full
python3 test_llm_service.py --image gui.png --mode quick --quiet
```

### 📦 Test Coverage

**LLM Service Tests:**
1. Object Detection - Multiple categories
2. Object Referring - Descriptive phrases
3. Pointing - Object center points
4. Visual Prompting - Find similar objects
5. Keypoint Detection - Person/hand/animal
6. OCR Word Boxes - Text extraction
7. OCR Polygons - Rotated text
8. GUI Grounding - UI elements
9. GUI Pointing - UI element centers

**Vision Service Tests:**
1. SAM Segmentation - Detection + masks
2. Phrase Grounding - Simple captions
3. Complex Grounding - Multi-phrase scenes

---

## 🔧 Advanced Usage

### Custom Categories
```bash
# With cURL
curl -X POST "..." -d "{
  \"task\": \"detection\",
  \"categories\": [\"red car\", \"person wearing hat\", \"dog\"]
}"

# With Python
test_api_call(
    url=LLM_SERVICE_URL,
    payload={
        "image": image_b64,
        "task": "detection",
        "categories": ["custom", "object", "names"]
    }
)
```

### Visual Prompting
```bash
# Provide reference boxes to find similar objects
{
  "task": "visual_prompting",
  "visual_prompt_boxes": [[x1, y1, x2, y2]],
  "categories": ["similar objects"]
}
```

### Keypoint Types
```bash
# Person keypoints (COCO 17-point format)
{"task": "keypoint", "keypoint_type": "person"}

# Hand keypoints
{"task": "keypoint", "keypoint_type": "hand"}

# Animal keypoints  
{"task": "keypoint", "keypoint_type": "animal"}
```

---

## 📸 Test Images

Available in `tutorials/detection_example/test_images/`:

- **`cafe.jpg`** - General scene with people, furniture, objects
- **`boys.jpg`** - People-focused scene
- **`gui.png`** - GUI screenshot for UI testing
- **`layout.jpg`** - Document layout for OCR testing

---

## ⚡ Performance Tips

1. **First Request is Slow**: Models need to load (cold start)
   - Subsequent requests are faster
   - Expect 30-60s for first request

2. **Batch Testing**: Use parallel requests for multiple images
   ```bash
   ./curl_examples.sh detection &
   ./curl_examples.sh pointing &
   wait
   ```

3. **Image Size**: Smaller images = faster inference
   ```bash
   # Resize large images
   convert input.jpg -resize 1024x1024 output.jpg
   ```

4. **Timeout Settings**: Increase for large images
   ```bash
   curl --max-time 180 ...  # 3 minutes
   ```

---

## 🐛 Troubleshooting

### Request Timeout
```bash
# Increase timeout
curl --max-time 180 ...

# In Python
requests.post(..., timeout=180)
```

### Invalid Base64
```bash
# Use -w 0 to avoid line breaks
base64 -w 0 image.jpg  # ✅ Correct
base64 image.jpg       # ❌ May have newlines
```

### Empty Response
- Check if service URL is correct
- Verify image encoding
- Check categories list is not empty

### Service Unavailable
- First request may take time (cold start)
- Retry after 30-60 seconds

---

## 📚 Additional Resources

- **Rex-Omni Paper**: https://arxiv.org/abs/2510.12798
- **GitHub Repository**: https://github.com/IDEA-Research/Rex-Omni
- **HuggingFace Demo**: https://huggingface.co/spaces/Mountchicken/Rex-Omni
- **Model Weights**: https://huggingface.co/IDEA-Research/Rex-Omni

---

## 💡 Pro Tips

1. **Use jq for JSON parsing** (install with `apt install jq` or `brew install jq`)
2. **Save responses** for offline analysis: `curl ... -o response.json`
3. **Test incrementally**: Start with quick tests, then full suite
4. **Check logs**: Services print debug info for troubleshooting
5. **Monitor timing**: Use `time` command or `-w` flag to measure latency

---

## ✅ Quick Checklist

Before sharing your API:

- [ ] Test all endpoints with `curl_examples.sh all`
- [ ] Run Python test suite: `python3 test_llm_service.py`
- [ ] Verify all response formats
- [ ] Test with different images
- [ ] Document any custom categories
- [ ] Check timeout settings
- [ ] Validate error handling

---

**Last Updated**: 2025-11-23  
**Maintained By**: Rex-Omni Team

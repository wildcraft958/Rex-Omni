# Rex-Omni API - cURL Commands Reference

Complete reference of cURL commands for testing all Rex-Omni API endpoints.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [LLM Service Endpoints](#llm-service-endpoints)
  - [Object Detection](#1-object-detection)
  - [Object Referring](#2-object-referring)
  - [Pointing](#3-pointing)
  - [Visual Prompting](#4-visual-prompting)
  - [Person Keypoint Detection](#5-person-keypoint-detection)
  - [OCR - Word Boxes](#6-ocr---word-boxes)
  - [OCR - Polygons](#7-ocr---polygons)
  - [GUI Grounding](#8-gui-grounding)
  - [GUI Pointing](#9-gui-pointing)
- [Vision Service Endpoints](#vision-service-endpoints)
  - [SAM Segmentation](#1-sam-segmentation)
  - [Phrase Grounding](#2-phrase-grounding)
  - [Complex Caption Grounding](#3-complex-caption-grounding)
- [Advanced Usage](#advanced-usage)
- [Batch Testing](#batch-testing)

---

## Prerequisites

### 1. Encode Image to Base64

```bash
# Method 1: Using base64 command
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

# Method 2: Using openssl
IMAGE_B64=$(openssl base64 -A -in tutorials/detection_example/test_images/cafe.jpg)

# Method 3: Using Python (for cross-platform compatibility)
IMAGE_B64=$(python3 -c "import base64; print(base64.b64encode(open('tutorials/detection_example/test_images/cafe.jpg', 'rb').read()).decode())")
```

### 2. Set Service URLs

```bash
LLM_URL="https://animeshraj958--rex-llm-service-rex-inference.modal.run"
SAM_URL="https://animeshraj958--rex-vision-service-api-sam.modal.run"
GROUNDING_URL="https://animeshraj958--rex-vision-service-api-grounding.modal.run"
```

---

## Quick Start

### Test Connection

```bash
# Encode image
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

# Basic detection test
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\", \"cup\", \"laptop\"]
  }" | jq '.'
```

---

## LLM Service Endpoints

Base URL: `https://animeshraj958--rex-llm-service-rex-inference.modal.run`

### 1. Object Detection

Detect multiple object categories in an image.

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\", \"cup\", \"laptop\", \"chair\", \"table\", \"plant\"]
  }" | jq '.extracted_predictions'
```

**Response:**
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

### 2. Object Referring

Detect objects using descriptive phrases.

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [
      \"person wearing blue shirt\",
      \"yellow flower on table\",
      \"silver laptop computer\",
      \"white coffee cup\"
    ]
  }" | jq '.extracted_predictions'
```

### 3. Pointing

Get center points of objects instead of bounding boxes.

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"pointing\",
    \"categories\": [\"person\", \"laptop\", \"cup\"]
  }" | jq '.extracted_predictions'
```

**Response:**
```json
{
  "person": [
    {"type": "point", "coords": [175, 300]}
  ],
  "laptop": [
    {"type": "point", "coords": [420, 250]}
  ]
}
```

### 4. Visual Prompting

Find similar objects based on reference boxes.

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"visual_prompting\",
    \"visual_prompt_boxes\": [[100, 100, 300, 300]],
    \"categories\": [\"similar objects\"]
  }" | jq '.extracted_predictions'
```

### 5. Person Keypoint Detection

Detect human body keypoints (17 keypoints in COCO format).

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"keypoint\",
    \"keypoint_type\": \"person\"
  }" | jq '.extracted_predictions'
```

**Other keypoint types:**
- `"person"` - Human body keypoints
- `"hand"` - Hand keypoints
- `"animal"` - Animal keypoints

### 6. OCR - Word Boxes

Extract text with word-level bounding boxes.

```bash
# Better results with text-heavy images
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/layout.jpg)

curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"ocr_box\"
  }" | jq '.extracted_predictions'
```

### 7. OCR - Polygons

Extract text with polygon annotations (for rotated/curved text).

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"ocr_polygon\"
  }" | jq '.extracted_predictions'
```

### 8. GUI Grounding

Detect and locate UI elements in screenshots.

```bash
# Best results with GUI screenshots
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/gui.png)

curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"gui_grounding\",
    \"categories\": [\"button\", \"text field\", \"icon\", \"menu\", \"checkbox\"]
  }" | jq '.extracted_predictions'
```

### 9. GUI Pointing

Get center points of UI elements.

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"gui_pointing\",
    \"categories\": [\"submit button\", \"search box\", \"logo\", \"menu icon\"]
  }" | jq '.extracted_predictions'
```

---

## Vision Service Endpoints

### 1. SAM Segmentation

Combine Rex-Omni detection with SAM for pixel-perfect segmentation.

**Endpoint:** `https://animeshraj958--rex-vision-service-api-sam.modal.run`

```bash
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

curl -X POST "${SAM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"categories\": [\"person\", \"laptop\", \"cup\", \"chair\"]
  }" | jq '.'
```

**Response:**
```json
{
  "success": true,
  "predictions": {
    "person": [{"type": "box", "coords": [...]}]
  },
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

### 2. Phrase Grounding

Extract noun phrases from captions and ground them to image regions.

**Endpoint:** `https://animeshraj958--rex-vision-service-api-grounding.modal.run`

```bash
curl -X POST "${GROUNDING_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"caption\": \"A person sitting at a table with a laptop and coffee cup in a modern cafe\"
  }" | jq '.'
```

**Response:**
```json
{
  "success": true,
  "image_size": [800, 600],
  "caption": "A person sitting at a table...",
  "annotations": [
    {
      "phrase": "person",
      "start_char": 2,
      "end_char": 8,
      "boxes": [[100, 150, 250, 450]]
    },
    {
      "phrase": "laptop",
      "start_char": 35,
      "end_char": 41,
      "boxes": [[400, 200, 550, 320]]
    }
  ]
}
```

### 3. Complex Caption Grounding

Test with more complex scene descriptions.

```bash
curl -X POST "${GROUNDING_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"caption\": \"Multiple people are gathered around wooden tables. Some tables have laptops, coffee cups, and decorative plants. Large windows provide natural lighting.\"
  }" | jq '.annotations | length'
```

---

## Advanced Usage

### Save Response to File

```bash
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\", \"cup\", \"laptop\"]
  }" -o response.json

# Pretty print
jq '.' response.json
```

### Extract Specific Fields

```bash
# Get only bounding boxes
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\"]
  }" | jq '.extracted_predictions.person[].coords'

# Count detected objects
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\", \"cup\"]
  }" | jq '[.extracted_predictions | to_entries | .[].value | length] | add'
```

### Timing Requests

```bash
# Measure response time
time curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\"]
  }" -s -o /dev/null

# With verbose timing
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\"]
  }" -w "\nTotal time: %{time_total}s\n" | jq '.success'
```

### Handle Errors

```bash
# Check HTTP status
HTTP_CODE=$(curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\"]
  }" -w "%{http_code}" -s -o response.json)

if [ "$HTTP_CODE" -eq 200 ]; then
  echo "Success!"
  jq '.' response.json
else
  echo "Error: HTTP $HTTP_CODE"
  cat response.json
fi
```

### Test with Multiple Images

```bash
# Loop through all test images
for IMAGE in tutorials/detection_example/test_images/*.jpg; do
  echo "Testing: $IMAGE"
  IMAGE_B64=$(base64 -w 0 "$IMAGE")
  curl -X POST "${LLM_URL}" \
    -H "Content-Type: application/json" \
    -d "{
      \"image\": \"${IMAGE_B64}\",
      \"task\": \"detection\",
      \"categories\": [\"person\", \"object\"]
    }" | jq '.extracted_predictions | keys'
  echo "---"
done
```

---

## Batch Testing

### Quick Test Script

```bash
#!/bin/bash
# quick_test.sh

LLM_URL="https://animeshraj958--rex-llm-service-rex-inference.modal.run"
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

echo "Testing LLM Service..."
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${IMAGE_B64}\",
    \"task\": \"detection\",
    \"categories\": [\"person\", \"cup\"]
  }" | jq '.success, .extracted_predictions | keys'

echo -e "\nDone!"
```

### Parallel Requests

```bash
# Test multiple tasks simultaneously
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

# Detection
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_B64}\", \"task\": \"detection\", \"categories\": [\"person\"]}" \
  -o detection.json &

# Pointing
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_B64}\", \"task\": \"pointing\", \"categories\": [\"person\"]}" \
  -o pointing.json &

# SAM
curl -X POST "${SAM_URL}" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_B64}\", \"categories\": [\"person\"]}" \
  -o sam.json &

# Wait for all
wait
echo "All requests completed!"
```

### Benchmark Performance

```bash
#!/bin/bash
# benchmark.sh

IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)
ITERATIONS=5

echo "Running $ITERATIONS iterations..."
total_time=0

for i in $(seq 1 $ITERATIONS); do
  start=$(date +%s.%N)
  
  curl -X POST "${LLM_URL}" \
    -H "Content-Type: application/json" \
    -d "{
      \"image\": \"${IMAGE_B64}\",
      \"task\": \"detection\",
      \"categories\": [\"person\", \"cup\"]
    }" -s -o /dev/null
  
  end=$(date +%s.%N)
  duration=$(echo "$end - $start" | bc)
  total_time=$(echo "$total_time + $duration" | bc)
  
  echo "Request $i: ${duration}s"
done

avg_time=$(echo "scale=2; $total_time / $ITERATIONS" | bc)
echo "Average time: ${avg_time}s"
```

---

## Common Payloads

### Detection Tasks

```bash
# Object detection
{
  "image": "${IMAGE_B64}",
  "task": "detection",
  "categories": ["person", "car", "dog"]
}

# Pointing
{
  "image": "${IMAGE_B64}",
  "task": "pointing",
  "categories": ["person"]
}

# Visual prompting
{
  "image": "${IMAGE_B64}",
  "task": "visual_prompting",
  "visual_prompt_boxes": [[x1,y1,x2,y2]],
  "categories": ["similar objects"]
}
```

### Perception Tasks

```bash
# Keypoint detection
{
  "image": "${IMAGE_B64}",
  "task": "keypoint",
  "keypoint_type": "person"  # or "hand", "animal"
}

# OCR
{
  "image": "${IMAGE_B64}",
  "task": "ocr_box"  # or "ocr_polygon"
}

# GUI
{
  "image": "${IMAGE_B64}",
  "task": "gui_grounding",  # or "gui_pointing"
  "categories": ["button", "text field"]
}
```

### Vision Service

```bash
# SAM segmentation
{
  "image": "${IMAGE_B64}",
  "categories": ["person", "object"]
}

# Phrase grounding
{
  "image": "${IMAGE_B64}",
  "caption": "Description of the scene"
}
```

---

## Tips & Best Practices

1. **Always use `-w 0` with base64** to avoid line breaks:
   ```bash
   base64 -w 0 image.jpg  # Good
   base64 image.jpg       # Bad (includes newlines)
   ```

2. **Use jq for JSON parsing**:
   ```bash
   curl ... | jq '.extracted_predictions'
   ```

3. **Set timeout for long requests**:
   ```bash
   curl --max-time 120 ...
   ```

4. **Test connectivity first**:
   ```bash
   curl -I "${LLM_URL}"
   ```

5. **Save large responses**:
   ```bash
   curl ... -o response.json
   ```

6. **Check for cold starts**: First request may be slow as models load.

---

## Troubleshooting

### Request Timeout
```bash
# Increase timeout
curl --max-time 180 ...
```

### Image Too Large
```bash
# Resize before encoding
convert input.jpg -resize 1024x1024 output.jpg
IMAGE_B64=$(base64 -w 0 output.jpg)
```

### Invalid JSON
```bash
# Validate JSON payload
echo '{"image":"...","task":"detection"}' | jq '.'
```

### Check Service Status
```bash
# Test if service is up
curl -X POST "${LLM_URL}" \
  -H "Content-Type: application/json" \
  -d '{"image":"","task":"detection","categories":[]}' \
  -w "HTTP: %{http_code}\n"
```

---

## Additional Resources

- **Python Test Suite**: `test_llm_service.py`
- **Shell Script Runner**: `test_llm_service.sh`
- **Service Documentation**: `rex_llm_app.py`, `rex_vision_app.py`
- **Rex-Omni Repository**: https://github.com/IDEA-Research/Rex-Omni
- **Paper**: https://arxiv.org/abs/2510.12798

---

**Last Updated**: 2025-11-23

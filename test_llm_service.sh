#!/bin/bash
# Test script for Rex LLM Service

echo "Encoding image..."
IMAGE_B64=$(base64 -w 0 tutorials/detection_example/test_images/cafe.jpg)

echo "Testing Rex LLM Service..."
curl -X POST https://animeshraj958--rex-llm-service-rex-inference.modal.run \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_B64\", \"task\": \"detection\", \"categories\": [\"person\", \"cup\", \"laptop\"]}"

echo ""
echo "Done!"

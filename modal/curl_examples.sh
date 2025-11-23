#!/bin/bash
# Quick cURL Examples for Rex-Omni API Testing
# Ready-to-run curl commands for all endpoints

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Rex-Omni API - Quick cURL Examples                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Service URLs
LLM_URL="https://animeshraj958--rex-llm-service-rex-inference.modal.run"
SAM_URL="https://animeshraj958--rex-vision-service-api-sam.modal.run"
GROUNDING_URL="https://animeshraj958--rex-vision-service-api-grounding.modal.run"

# Default image
DEFAULT_IMAGE="tutorials/detection_example/test_images/cafe.jpg"

# Encode image
echo -e "${YELLOW}📸 Encoding image: ${DEFAULT_IMAGE}${NC}"
IMAGE_B64=$(base64 -w 0 "$DEFAULT_IMAGE" 2>/dev/null || base64 "$DEFAULT_IMAGE")
echo -e "${GREEN}✓ Image encoded (${#IMAGE_B64} chars)${NC}"
echo ""

# ============================================================================
# Function to run curl command
# ============================================================================
run_curl() {
    local name=$1
    local url=$2
    local payload=$3
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}▶ ${name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    echo ""
    echo -e "${YELLOW}cURL Command:${NC}"
    echo "curl -X POST '${url}' \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '${payload}' | jq '.'"
    echo ""
    
    echo -e "${YELLOW}Response:${NC}"
    curl -X POST "${url}" \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        --max-time 120 \
        -s | jq '.'
    
    echo ""
}

# ============================================================================
# Parse CLI arguments
# ============================================================================
case "${1:-all}" in
    "detection")
        echo -e "${GREEN}🎯 Testing: Object Detection${NC}"
        echo ""
        run_curl "Object Detection" "${LLM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"task\": \"detection\",
            \"categories\": [\"person\", \"cup\", \"laptop\", \"chair\"]
        }"
        ;;
    
    "pointing")
        echo -e "${GREEN}📍 Testing: Pointing${NC}"
        echo ""
        run_curl "Object Pointing" "${LLM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"task\": \"pointing\",
            \"categories\": [\"person\", \"laptop\", \"cup\"]
        }"
        ;;
    
    "sam")
        echo -e "${GREEN}✂️  Testing: SAM Segmentation${NC}"
        echo ""
        run_curl "SAM Segmentation" "${SAM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"categories\": [\"person\", \"laptop\", \"cup\"]
        }"
        ;;
    
    "grounding")
        echo -e "${GREEN}🔗 Testing: Phrase Grounding${NC}"
        echo ""
        run_curl "Phrase Grounding" "${GROUNDING_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"caption\": \"A person sitting at a table with a laptop and coffee cup\"
        }"
        ;;
    
    "keypoint")
        echo -e "${GREEN}🧍 Testing: Person Keypoint Detection${NC}"
        echo ""
        run_curl "Person Keypoint Detection" "${LLM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"task\": \"keypoint\",
            \"keypoint_type\": \"person\"
        }"
        ;;
    
    "visual-prompt")
        echo -e "${GREEN}👁️  Testing: Visual Prompting${NC}"
        echo ""
        run_curl "Visual Prompting" "${LLM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"task\": \"visual_prompting\",
            \"visual_prompt_boxes\": [[100, 100, 300, 300]],
            \"categories\": [\"similar objects\"]
        }"
        ;;
    
    "quick")
        echo -e "${GREEN}⚡ Quick Test - Detection Only${NC}"
        echo ""
        
        curl -X POST "${LLM_URL}" \
            -H "Content-Type: application/json" \
            -d "{
                \"image\": \"${IMAGE_B64}\",
                \"task\": \"detection\",
                \"categories\": [\"person\", \"cup\", \"laptop\"]
            }" \
            --max-time 120 | jq '.extracted_predictions'
        ;;
    
    "all")
        echo -e "${GREEN}🎯 Running All Examples${NC}"
        echo ""
        
        # 1. Detection
        run_curl "1. Object Detection" "${LLM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"task\": \"detection\",
            \"categories\": [\"person\", \"cup\", \"laptop\"]
        }"
        
        # 2. Pointing
        run_curl "2. Object Pointing" "${LLM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"task\": \"pointing\",
            \"categories\": [\"person\", \"cup\"]
        }"
        
        # 3. SAM
        run_curl "3. SAM Segmentation" "${SAM_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"categories\": [\"person\", \"cup\"]
        }"
        
        # 4. Grounding
        run_curl "4. Phrase Grounding" "${GROUNDING_URL}" "{
            \"image\": \"${IMAGE_B64}\",
            \"caption\": \"A person with a laptop and coffee cup\"
        }"
        
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║               All Tests Completed! ✓                        ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
        ;;
    
    "help"|*)
        echo -e "${YELLOW}Usage:${NC}"
        echo "  ./curl_examples.sh [command]"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo "  detection      - Object detection"
        echo "  pointing       - Object pointing (center points)"
        echo "  sam            - SAM segmentation"
        echo "  grounding      - Phrase grounding"
        echo "  keypoint       - Person keypoint detection"
        echo "  visual-prompt  - Visual prompting"
        echo "  quick          - Quick detection test"
        echo "  all            - Run all examples (default)"
        echo "  help           - Show this help"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo "  ./curl_examples.sh detection"
        echo "  ./curl_examples.sh sam"
        echo "  ./curl_examples.sh all"
        echo ""
        ;;
esac

echo ""
echo -e "${BLUE}💡 Tip: See CURL_COMMANDS.md for complete documentation${NC}"
echo ""

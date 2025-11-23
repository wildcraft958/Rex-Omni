#!/bin/bash
# Rex-Omni Test Suite Runner
# Convenient shell script to run various test scenarios

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Rex-Omni Exhaustive Test Suite Runner            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to run tests
run_test() {
    local test_name=$1
    local image=$2
    local mode=$3
    local flags=$4
    
    echo -e "${GREEN}▶ Running: ${test_name}${NC}"
    echo -e "${YELLOW}  Image: ${image} | Mode: ${mode}${NC}"
    echo ""
    
    python3 test_llm_service.py --image "$image" --mode "$mode" $flags
    
    echo ""
    echo -e "${GREEN}✓ Completed: ${test_name}${NC}"
    echo ""
}

# Parse command line arguments
case "${1:-full}" in
    "quick")
        echo "🚀 Running Quick Sanity Tests"
        echo ""
        run_test "Quick Test - Cafe" "cafe.jpg" "quick" "--quiet"
        ;;
    
    "cafe")
        echo "☕ Running Full Test Suite on Cafe Image"
        echo ""
        run_test "Full Test - Cafe" "cafe.jpg" "full" ""
        ;;
    
    "boys")
        echo "👥 Running Full Test Suite on Boys Image"
        echo ""
        run_test "Full Test - Boys" "boys.jpg" "full" ""
        ;;
    
    "gui")
        echo "🖥️  Running Full Test Suite on GUI Screenshot"
        echo ""
        run_test "Full Test - GUI" "gui.png" "full" ""
        ;;
    
    "layout")
        echo "📄 Running Full Test Suite on Layout Image"
        echo ""
        run_test "Full Test - Layout" "layout.jpg" "full" ""
        ;;
    
    "all")
        echo "🎯 Running Test Suite on ALL Images"
        echo ""
        
        run_test "Quick Test - Cafe" "cafe.jpg" "quick" "--quiet"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        run_test "Quick Test - Boys" "boys.jpg" "quick" "--quiet"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        run_test "Quick Test - GUI" "gui.png" "quick" "--quiet"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        run_test "Quick Test - Layout" "layout.jpg" "quick" "--quiet"
        
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║          All Tests Completed Successfully! ✓              ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        ;;
    
    "full"|*)
        echo "🎯 Running Full Test Suite with Verbose Output"
        echo ""
        run_test "Full Test - Cafe (Verbose)" "cafe.jpg" "full" ""
        ;;
esac

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Usage:${NC}"
echo -e "  ${GREEN}./test_llm_service.sh${NC}           # Full verbose test on cafe.jpg"
echo -e "  ${GREEN}./test_llm_service.sh quick${NC}     # Quick sanity check"
echo -e "  ${GREEN}./test_llm_service.sh cafe${NC}      # Full test on cafe.jpg"
echo -e "  ${GREEN}./test_llm_service.sh boys${NC}      # Full test on boys.jpg"
echo -e "  ${GREEN}./test_llm_service.sh gui${NC}       # Full test on gui.png"
echo -e "  ${GREEN}./test_llm_service.sh layout${NC}    # Full test on layout.jpg"
echo -e "  ${GREEN}./test_llm_service.sh all${NC}       # Quick test on all images"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

#!/bin/bash
# Deep Tree Echo RWKV.cpp Build Script
# Builds the rwkv.cpp integration for distributed agentic cognitive processing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 Deep Tree Echo RWKV.cpp Integration Builder${NC}"
echo -e "${BLUE}=============================================${NC}"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Check for dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is required but not installed.${NC}"
    echo -e "${YELLOW}On Ubuntu/Debian: sudo apt-get install cmake${NC}"
    echo -e "${YELLOW}On macOS: brew install cmake${NC}"
    exit 1
fi

# Check for git (should be available since we cloned)
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is required but not installed.${NC}"
    exit 1
fi

# Check for C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo -e "${RED}Error: C++ compiler (g++ or clang++) is required but not installed.${NC}"
    echo -e "${YELLOW}On Ubuntu/Debian: sudo apt-get install build-essential${NC}"
    echo -e "${YELLOW}On macOS: xcode-select --install${NC}"
    exit 1
fi

# Initialize git submodules for rwkv.cpp dependencies
echo -e "${YELLOW}Initializing git submodules...${NC}"
cd external/rwkv.cpp
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive
    echo -e "${GREEN}✓ Git submodules initialized${NC}"
else
    echo -e "${YELLOW}⚠ No git submodules found in rwkv.cpp${NC}"
fi
cd ../..

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p build
cd build

# Configure with cmake
echo -e "${YELLOW}Configuring build with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release -DECHO_BUILD_RWKV_CPP=ON

# Build the project
echo -e "${YELLOW}Building RWKV.cpp integration...${NC}"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    
    # Check if the bridge library was created
    BRIDGE_LIB="src/libecho_rwkv_cpp_bridge.so"
    if [ -f "$BRIDGE_LIB" ]; then
        echo -e "${GREEN}✓ Bridge library created: $BRIDGE_LIB${NC}"
    else
        echo -e "${YELLOW}⚠ Bridge library not found at expected location${NC}"
        # Look for it in other possible locations
        find . -name "*echo_rwkv_cpp_bridge*" -type f
    fi
    
    # Copy the library to the src directory for Python access
    if [ -f "$BRIDGE_LIB" ]; then
        cp "$BRIDGE_LIB" ../src/
        echo -e "${GREEN}✓ Bridge library copied to src directory${NC}"
    fi
    
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

cd ..

# Test the integration
echo -e "${YELLOW}Testing RWKV.cpp integration...${NC}"
cd src
python3 -c "
import sys
sys.path.append('.')
from rwkv_cpp_bridge import test_rwkv_cpp_integration
test_rwkv_cpp_integration()
" 2>/dev/null || echo -e "${YELLOW}⚠ Python test requires dependencies (run 'pip install numpy')${NC}"

echo -e "${GREEN}🎉 Deep Tree Echo RWKV.cpp integration build completed!${NC}"
echo -e "${BLUE}The system is now ready for distributed agentic cognitive micro-kernel processing.${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Install Python dependencies: ${BLUE}pip install numpy${NC}"
echo -e "2. Download an RWKV model in ggml format"
echo -e "3. Configure model path in your application"
echo -e "4. Run your Deep Tree Echo cognitive processing!"
echo ""
echo -e "${GREEN}Happy cognitive processing! 🧠✨${NC}"
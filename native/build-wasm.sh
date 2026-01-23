#!/bin/bash
# Build script for WASM target
# Requires Emscripten SDK to be installed and activated

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-wasm"
OUTPUT_DIR="${SCRIPT_DIR}/../dist/wasm"

# Check for emcc
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten (emcc) not found!"
    echo "Install Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

echo "Building Spitfire WASM module..."

# Create build directory
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${BUILD_DIR}"

# Configure with Emscripten
emcmake cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF

# Build
emmake make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Copy output to dist
if [ -f "llama-wasm.js" ]; then
    cp llama-wasm.js "${OUTPUT_DIR}/"
    cp llama-wasm.wasm "${OUTPUT_DIR}/"
    echo "Build complete! Output in ${OUTPUT_DIR}"
else
    echo "Build failed - output files not found"
    exit 1
fi

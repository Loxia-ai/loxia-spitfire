#!/bin/bash
# Direct WASM build using emcc (no CMake required)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../dist/wasm"

# Check for emcc
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten (emcc) not found!"
    echo "Run: source /tmp/emsdk/emsdk_env.sh"
    exit 1
fi

echo "Building Spitfire WASM module..."
echo "Script dir: ${SCRIPT_DIR}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Common include paths
INCLUDES="-I${SCRIPT_DIR}/ggml/include \
          -I${SCRIPT_DIR}/ggml/src \
          -I${SCRIPT_DIR}/ggml/src/ggml-cpu \
          -I${SCRIPT_DIR}/llama.cpp/include \
          -I${SCRIPT_DIR}/llama.cpp/src"

# Compiler flags
CFLAGS="-O3 -DNDEBUG -DGGML_USE_CPU"
CXXFLAGS="-O3 -DNDEBUG -DGGML_USE_CPU -std=c++17"
VERSION_FLAGS='-DGGML_VERSION="0.1.0" -DGGML_COMMIT="spitfire"'

echo "Compiling GGML core..."
emcc ${CFLAGS} ${VERSION_FLAGS} \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -I"${SCRIPT_DIR}/llama.cpp/include" \
    -c "${SCRIPT_DIR}/ggml/src/ggml.c" \
    -o /tmp/ggml.o 2>&1 || { echo "Failed to compile ggml.c"; exit 1; }
echo "✓ ggml.c"

echo "Compiling GGML alloc..."
emcc ${CFLAGS} \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-alloc.c" \
    -o /tmp/ggml-alloc.o 2>&1 || { echo "Failed to compile ggml-alloc.c"; exit 1; }
echo "✓ ggml-alloc.c"

echo "Compiling GGML quants..."
emcc ${CFLAGS} -msimd128 \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -I"${SCRIPT_DIR}/ggml/src/ggml-cpu" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-quants.c" \
    -o /tmp/ggml-quants.o 2>&1 || { echo "Failed to compile ggml-quants.c"; exit 1; }
echo "✓ ggml-quants.c"

echo "Compiling GGML backend..."
em++ ${CXXFLAGS} \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-backend.cpp" \
    -o /tmp/ggml-backend.o 2>&1 || { echo "Failed to compile ggml-backend.cpp"; exit 1; }
echo "✓ ggml-backend.cpp"

echo "Compiling GGML backend-reg..."
em++ ${CXXFLAGS} \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-backend-reg.cpp" \
    -o /tmp/ggml-backend-reg.o 2>&1 || { echo "Failed to compile ggml-backend-reg.cpp"; exit 1; }
echo "✓ ggml-backend-reg.cpp"

echo "Compiling GGML threading..."
em++ ${CXXFLAGS} \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-threading.cpp" \
    -o /tmp/ggml-threading.o 2>&1 || { echo "Failed to compile ggml-threading.cpp"; exit 1; }
echo "✓ ggml-threading.cpp"

echo "Compiling GGML CPU (C)..."
emcc ${CFLAGS} -msimd128 \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -I"${SCRIPT_DIR}/ggml/src/ggml-cpu" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-cpu/ggml-cpu.c" \
    -o /tmp/ggml-cpu.o 2>&1 || { echo "Failed to compile ggml-cpu.c"; exit 1; }
echo "✓ ggml-cpu.c"

echo "Compiling GGML CPU (C++)..."
em++ ${CXXFLAGS} -msimd128 \
    -I"${SCRIPT_DIR}/ggml/include" \
    -I"${SCRIPT_DIR}/ggml/src" \
    -I"${SCRIPT_DIR}/ggml/src/ggml-cpu" \
    -c "${SCRIPT_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp" \
    -o /tmp/ggml-cpu-cpp.o 2>&1 || { echo "Failed to compile ggml-cpu.cpp"; exit 1; }
echo "✓ ggml-cpu.cpp"

echo ""
echo "========================================="
echo "GGML core compiled successfully!"
echo "========================================="
echo ""

# Count compiled objects
echo "Compiled objects:"
ls -la /tmp/*.o 2>/dev/null | wc -l
echo "files in /tmp"

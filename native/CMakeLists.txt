cmake_minimum_required(VERSION 3.14)
project(spitfire-llama LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Detect if building for WebAssembly
if(EMSCRIPTEN)
    message(STATUS "Building for WebAssembly with Emscripten")
    set(BUILD_WASM ON)
else()
    message(STATUS "Building native bindings")
    set(BUILD_WASM OFF)
endif()

# Include directories
set(GGML_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include)
set(LLAMA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include)
set(LLAMA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src)
set(GGML_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src)

# GGML core sources
set(GGML_SOURCES
    ${GGML_SRC_DIR}/ggml.c
    ${GGML_SRC_DIR}/ggml-alloc.c
    ${GGML_SRC_DIR}/ggml-quants.c
    ${GGML_SRC_DIR}/ggml-backend.cpp
    ${GGML_SRC_DIR}/ggml-backend-reg.cpp
    ${GGML_SRC_DIR}/ggml-opt.cpp
    ${GGML_SRC_DIR}/ggml-threading.cpp
)

# GGML CPU backend sources
set(GGML_CPU_SOURCES
    ${GGML_SRC_DIR}/ggml-cpu/ggml-cpu.c
    ${GGML_SRC_DIR}/ggml-cpu/ggml-cpu.cpp
    ${GGML_SRC_DIR}/ggml-cpu/ops.cpp
    ${GGML_SRC_DIR}/ggml-cpu/vec.cpp
    ${GGML_SRC_DIR}/ggml-cpu/unary-ops.cpp
    ${GGML_SRC_DIR}/ggml-cpu/binary-ops.cpp
    ${GGML_SRC_DIR}/ggml-cpu/quants.c
    ${GGML_SRC_DIR}/ggml-cpu/repack.cpp
    ${GGML_SRC_DIR}/ggml-cpu/traits.cpp
    ${GGML_SRC_DIR}/ggml-cpu/hbm.cpp
    ${GGML_SRC_DIR}/ggml-cpu/llamafile/sgemm.cpp
)

# Llama.cpp sources
set(LLAMA_SOURCES
    ${LLAMA_SRC_DIR}/llama.cpp
    ${LLAMA_SRC_DIR}/llama-arch.cpp
    ${LLAMA_SRC_DIR}/llama-adapter.cpp
    ${LLAMA_SRC_DIR}/llama-batch.cpp
    ${LLAMA_SRC_DIR}/llama-chat.cpp
    ${LLAMA_SRC_DIR}/llama-context.cpp
    ${LLAMA_SRC_DIR}/llama-cparams.cpp
    ${LLAMA_SRC_DIR}/llama-grammar.cpp
    ${LLAMA_SRC_DIR}/llama-graph.cpp
    ${LLAMA_SRC_DIR}/llama-hparams.cpp
    ${LLAMA_SRC_DIR}/llama-impl.cpp
    ${LLAMA_SRC_DIR}/llama-io.cpp
    ${LLAMA_SRC_DIR}/llama-kv-cache.cpp
    ${LLAMA_SRC_DIR}/llama-kv-cache-iswa.cpp
    ${LLAMA_SRC_DIR}/llama-memory.cpp
    ${LLAMA_SRC_DIR}/llama-memory-hybrid.cpp
    ${LLAMA_SRC_DIR}/llama-memory-recurrent.cpp
    ${LLAMA_SRC_DIR}/llama-mmap.cpp
    ${LLAMA_SRC_DIR}/llama-model.cpp
    ${LLAMA_SRC_DIR}/llama-model-loader.cpp
    ${LLAMA_SRC_DIR}/llama-model-saver.cpp
    ${LLAMA_SRC_DIR}/llama-quant.cpp
    ${LLAMA_SRC_DIR}/llama-sampling.cpp
    ${LLAMA_SRC_DIR}/llama-vocab.cpp
    ${LLAMA_SRC_DIR}/unicode.cpp
    ${LLAMA_SRC_DIR}/unicode-data.cpp
)

# Common sources
set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/sampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common/json-schema-to-grammar.cpp
)

if(BUILD_WASM)
    # WebAssembly build
    add_executable(llama-wasm
        ${GGML_SOURCES}
        ${GGML_CPU_SOURCES}
        ${LLAMA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/wasm/bindings.cpp
    )

    target_include_directories(llama-wasm PRIVATE
        ${GGML_INCLUDE_DIR}
        ${LLAMA_INCLUDE_DIR}
        ${LLAMA_SRC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
    )

    target_compile_definitions(llama-wasm PRIVATE
        GGML_USE_CPU
        NDEBUG
    )

    # Emscripten flags
    set_target_properties(llama-wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
            -s WASM=1 \
            -s ALLOW_MEMORY_GROWTH=1 \
            -s MAXIMUM_MEMORY=4GB \
            -s STACK_SIZE=5MB \
            -s MODULARIZE=1 \
            -s EXPORT_ES6=1 \
            -s EXPORT_NAME='createLlamaModule' \
            -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','stringToUTF8','lengthBytesUTF8'] \
            -s EXPORTED_FUNCTIONS=['_malloc','_free','_llama_init','_llama_load_model','_llama_generate','_llama_free','_llama_tokenize','_llama_embedding'] \
            -s FILESYSTEM=1 \
            -s FORCE_FILESYSTEM=1 \
            -s SINGLE_FILE=0 \
            -s SIMD=1 \
            -msimd128 \
            -O3 \
            -flto \
        "
    )

    target_compile_options(llama-wasm PRIVATE
        -O3
        -msimd128
        -fno-exceptions
        -DNDEBUG
    )

else()
    # Native N-API build
    find_package(PkgConfig)

    add_library(llama-native SHARED
        ${GGML_SOURCES}
        ${GGML_CPU_SOURCES}
        ${LLAMA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/napi/bindings.cpp
    )

    target_include_directories(llama-native PRIVATE
        ${GGML_INCLUDE_DIR}
        ${LLAMA_INCLUDE_DIR}
        ${LLAMA_SRC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
        ${CMAKE_JS_INC}
    )

    target_compile_definitions(llama-native PRIVATE
        GGML_USE_CPU
        NAPI_VERSION=8
    )

    target_link_libraries(llama-native PRIVATE
        ${CMAKE_JS_LIB}
    )
endif()

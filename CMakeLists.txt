cmake_minimum_required(VERSION 3.18)
project(trtllm_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(TRTLLM_LIBS     /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs)
set(TOKENIZERS_DIR  ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp)
set(DROGON_LOCAL    ${CMAKE_SOURCE_DIR}/third_party/local)

include_directories(
    /app/tensorrt_llm/include
    /usr/local/cuda-13.0/targets/x86_64-linux/include
)

link_directories(
    ${TRTLLM_LIBS}
    /usr/local/tensorrt/targets/x86_64-linux-gnu/lib
    ${DROGON_LOCAL}/lib
)

# ── 公共依赖 ────────────────────────────────────────────────
set(TRTLLM_LINK_LIBS
    tensorrt_llm
    nvinfer_plugin_tensorrt_llm
    nvinfer
    pthread
    ${TOKENIZERS_DIR}/build/libtokenizers_cpp.a
    ${TOKENIZERS_DIR}/build/libtokenizers_c.a
)

set(COMMON_INCLUDE_DIRS
    ${TOKENIZERS_DIR}/include
    ${DROGON_LOCAL}/include
    /usr/include/jsoncpp
)

# ── benchmark 推理程序 ──────────────────────────────────────
add_executable(infer src/infer.cpp)
target_include_directories(infer PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(infer ${TRTLLM_LINK_LIBS})

# ── HTTP 推理服务 ───────────────────────────────────────────
add_executable(server src/server.cpp)
target_include_directories(server PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(server
    ${TRTLLM_LINK_LIBS}
    drogon
    trantor
    jsoncpp
    uuid
    ssl
    crypto
    z
    pthread
)
cmake_minimum_required(VERSION 3.18)
project(trtllm_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(TRTLLM_LIBS /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs)
set(TOKENIZERS_CPP_DIR ${CMAKE_SOURCE_DIR}/third_party/tokenizers-cpp)

include_directories(
    /app/tensorrt_llm/include
    /usr/local/cuda-13.0/targets/x86_64-linux/include
)

link_directories(
    ${TRTLLM_LIBS}
    /usr/local/tensorrt/targets/x86_64-linux-gnu/lib
)

add_executable(infer src/infer.cpp)

target_include_directories(infer PRIVATE
    ${TOKENIZERS_CPP_DIR}/include
)

target_link_libraries(infer
    tensorrt_llm
    nvinfer_plugin_tensorrt_llm
    nvinfer
    pthread
    ${TOKENIZERS_CPP_DIR}/build/libtokenizers_cpp.a
    ${TOKENIZERS_CPP_DIR}/build/libtokenizers_c.a
)
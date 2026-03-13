cmake_minimum_required(VERSION 3.18)
project(trtllm_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(TRTLLM_LIBS /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs)

include_directories(
    /app/tensorrt_llm/include
    /usr/local/cuda-13.0/targets/x86_64-linux/include
)

link_directories(
    ${TRTLLM_LIBS}
    /usr/local/tensorrt/targets/x86_64-linux-gnu/lib
)

add_executable(infer src/infer.cpp)

target_link_libraries(infer
    tensorrt_llm
    nvinfer_plugin_tensorrt_llm
    nvinfer
    pthread
)

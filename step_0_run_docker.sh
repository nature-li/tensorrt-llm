docker run -it --rm \
    --gpus all \
    --name trtllm_dev \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3 \
    bash
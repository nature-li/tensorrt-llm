docker run -it --rm \
    --gpus all \
    --name trtllm_dev \
    --user $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3 \
    bash
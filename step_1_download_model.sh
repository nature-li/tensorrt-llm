# 用镜像加速
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download \
    Qwen/Qwen2.5-3B-Instruct \
    --local-dir ./models/Qwen2.5-3B-Instruct
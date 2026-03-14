export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_MIRROR=https://hf-mirror.com
python /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir ./models/Qwen2.5-3B-Instruct \
    --output_dir ./checkpoints/qwen25_3b_int8 \
    --dtype float16 \
    --smoothquant 0.5 \
    --per_token \
    --per_channel
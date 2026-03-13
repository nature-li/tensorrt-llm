cd /workspace
python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir ./models/Qwen2.5-3B-Instruct \
    --output_dir ./checkpoints/qwen25_3b_fp16 \
    --dtype float16
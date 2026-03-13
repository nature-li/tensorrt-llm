python3 /app/tensorrt_llm/examples/run.py \
    --engine_dir ./engines/qwen25_3b_fp16 \
    --tokenizer_dir ./models/Qwen2.5-3B-Instruct \
    --max_output_len 50 \
    --input_text "Hello, who are you?"
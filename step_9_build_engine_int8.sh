trtllm-build \
    --checkpoint_dir ./checkpoints/qwen25_3b_int8 \
    --output_dir ./engines/qwen25_3b_int8 \
    --max_batch_size 64 \
    --max_input_len 1024 \
    --max_seq_len 1280 \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16
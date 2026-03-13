trtllm-build \
    --checkpoint_dir ./checkpoints/qwen25_3b_fp16 \
    --output_dir ./engines/qwen25_3b_fp16 \
    --max_batch_size 4 \
    --max_input_len 1024 \
    --max_seq_len 1280 \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16

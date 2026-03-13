export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH

./build/infer ./engines/qwen25_3b_fp16
# TensorRT-LLM C++ Executor 推理 Demo

基于 TensorRT-LLM C++ Executor API 实现的 LLM 推理 demo，覆盖模型下载、引擎构建、tokenizer 集成到 FP16/INT8 批量性能对比的完整流程。

**环境：** RTX 5060 Ti 16GB · CUDA 13.0 · TensorRT-LLM 1.2.0rc6 · Qwen2.5-3B-Instruct

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `src/infer.cpp` | C++ 推理主程序，支持 FP16/INT8 engine，输出 TTFT/TPOT/Throughput |
| `step_0_run_docker.sh` | 启动 TensorRT-LLM 容器 |
| `step_1_download_model.sh` | 下载 Qwen2.5-3B-Instruct 模型 |
| `step_2_transport.sh` | 转换 FP16 checkpoint |
| `step_3_build_engine.sh` | 构建 FP16 TensorRT engine |
| `step_4_verify_engine.sh` | 验证 engine 输出 |
| `step_5_check_cpp.sh` | 验证 C++ 编译环境 |
| `step_6_generate_cmake.sh` | 生成 CMakeLists.txt |
| `step_7_check_runtime.sh` | 检测运行环境 |
| `step_8_convert_int8.sh` | 转换 INT8 SmoothQuant checkpoint |
| `step_9_build_engine_int8.sh` | 构建 INT8 TensorRT engine |

---

## 构建流程

### 1. 启动容器

```bash
bash step_0_run_docker.sh
```

### 2. 下载模型

```bash
bash step_1_download_model.sh
```

### 3. 构建 FP16 engine

```bash
# 转换 checkpoint
bash step_2_transport.sh

# 构建 engine
bash step_3_build_engine.sh

# 验证
bash step_4_verify_engine.sh
```

### 4. 构建 INT8 engine（SmoothQuant）

```bash
bash step_8_convert_int8.sh
bash step_9_build_engine_int8.sh
```

### 5. 安装 tokenizers-cpp

```bash
cd /workspace
mkdir -p third_party
git clone https://github.com/mlc-ai/tokenizers-cpp.git third_party/tokenizers-cpp
cd third_party/tokenizers-cpp
git submodule update --init --recursive

# 安装 Rust（国内镜像）
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 编译
cd /workspace/third_party/tokenizers-cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 6. 安装 jsoncpp

```bash
cd /workspace/third_party/jsoncpp
git apply /workspace/patches/jsoncpp-cxx17.patch
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/workspace/third_party/local \
    -DBUILD_SHARED_LIBS=OFF \
    -DJSONCPP_WITH_TESTS=OFF \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-std=c++17 -DJSONCPP_USE_CPLUSPLUS17"
make -j
make install
```

### 7. 安装 Drogon

```bash
apt-get update && apt-get install -y uuid-dev

cd /workspace/third_party/drogon
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_CTL=OFF \
    -DCMAKE_PREFIX_PATH=/workspace/third_party/local \
    -DCMAKE_INSTALL_PREFIX=/workspace/third_party/local
make -j
make install
```

### 8. 编译推理程序

```bash
bash step_5_check_cpp.sh
bash step_6_generate_cmake.sh
bash step_7_check_runtime.sh

cd /workspace
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

---

## 运行

```bash
# FP16，batch_size=1
./infer ../engines/qwen25_3b_fp16_b64 1

# INT8，batch_size=32
./infer ../engines/qwen25_3b_int8_b64 32

# 开启 prefix cache（所有请求相同 prompt，触发 KV cache 命中）
./infer ../engines/qwen25_3b_fp16_b64 64 --reuse_kv
```

参数说明：

| 参数 | 说明 |
|---|---|
| `<engine_dir>` | engine 目录路径 |
| `<batch_size>` | 同时提交的请求数，范围 1~64 |
| `--reuse_kv` | 开启 prefix cache，所有请求使用相同 prompt |

---

## 性能基准

测试设备：RTX 5060 Ti 16GB · Qwen2.5-3B-Instruct · input=512 tokens · output=200 tokens

### FP16

| Batch | TTFT (ms) | TPOT (ms) | Throughput (t/s) |
|------:|----------:|----------:|-----------------:|
| 1     | 215       | 17.5      | 54               |
| 4     | 364       | 17.3      | 208              |
| 8     | 653       | 18.0      | 378              |
| 16    | 1193      | 19.1      | 640              |
| 32    | 1748      | 22.5      | 1026             |
| 64    | 2853      | 30.9      | 1416             |
| 64 + prefix cache | 327 | 21.9 | 2712        |

### INT8（SmoothQuant）

| Batch | TTFT (ms) | TPOT (ms) | Throughput (t/s) |
|------:|----------:|----------:|-----------------:|
| 1     | 714       | 11.5      | 66               |
| 16    | 732       | 12.5      | 991              |
| 32    | 728       | 14.8      | 1739             |
| 64    | 1319      | 20.5      | 2360             |
| 64 + prefix cache | 266 | 15.7 | **3742**    |

### FP16 vs INT8 结论

- **延迟敏感（单请求）→ FP16**：TTFT 215ms vs 714ms，低 3.3x；TPOT 相近但 FP16 略差
- **吞吐优先（高并发）→ INT8**：batch=32 时吞吐 1739 vs 1026 tokens/s，高 69%
- **INT8 + prefix cache 峰值 3742 tokens/s**，是 FP16 基线（无 cache）的 2.6x
- **INT8 engine 体积 3.9GB**，较 FP16（6.5GB）减少 40%，释放显存供 KV cache 使用
- INT8 在低 batch 时 TTFT 反而偏高（prefill 有固定量化开销），但对 batch size 不敏感（batch=1~32 TTFT 稳定在 ~730ms）

---

## 注意事项

- tokenizers-cpp 依赖 Rust，必须先安装 Rust 再编译
- 国内环境需要配置 Rust 镜像源，否则安装失败
- INT8 engine 使用 SmoothQuant 量化，转换时需要校准数据集，精度损失较小
- `--reuse_kv` 测试的是 100% prefix cache 命中的理论上界，实际场景收益取决于请求间共享前缀的比例

# TensorRT-LLM C++ Executor 推理 Demo

基于 TensorRT-LLM C++ Executor API 实现的 LLM 推理 demo，覆盖模型下载、引擎构建、tokenizer 集成到 FP16/INT8 批量性能对比的完整流程。

**环境：** RTX 5060 Ti · CUDA 13.0 · TensorRT-LLM 1.2.0rc6 · Qwen2.5-3B-Instruct

---

## 快速开始

### 1. 进入容器

```bash
bash step_0_run_docker.sh
```

### 2. 下载模型

```bash
bash step_1_download_model.sh
```

### 3. 转换 checkpoint

```bash
# FP16
bash step_2_transport.sh

# INT8（SmoothQuant）
bash step_8_convert_int8.sh
```

### 4. 构建 TensorRT Engine

```bash
# FP16
bash step_3_build_engine.sh

# INT8
bash step_9_build_int8_engine.sh
```

### 5. 验证 Engine

```bash
bash step_4_verify_engine.sh
```

### 6. 检查运行环境

```bash
bash step_7_check_runtime.sh
```

---

## 编译

### 安装 tokenizer 子模块

```bash
git submodule update --init --recursive
```

### 安装 Rust（使用国内镜像）

```bash
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 编译 tokenizers-cpp

```bash
cd /workspace/third_party/tokenizers-cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 编译推理程序

```bash
cd /workspace
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

---

## 运行

```bash
# FP16
./infer ../engines/qwen25_3b_fp16 <batch_size>

# INT8
./infer ../engines/qwen25_3b_int8 <batch_size>
```

---

## 推理性能基准

测试设备：RTX 5060 Ti · Qwen2.5-3B-Instruct · max_new_tokens=64

### FP16

| 批大小 | TTFT (ms) | TPOT (ms) | 吞吐量 (t/s) |
|-------:|----------:|----------:|-------------:|
| 1      | 167       | 17.1      | 51.5         |
| 2      | 160       | 17.3      | 102.2        |
| 4      | 164       | 17.2      | 205.5        |
| 8      | 793       | 26.8      | 206.3        |
| 16     | 1816      | 43.0      | 226.4        |

**饱和点：batch=4**，超过后吞吐不再增长，TTFT 暴涨。

### INT8（SmoothQuant）

| 批大小 | TTFT (ms) | TPOT (ms) | 吞吐量 (t/s) |
|-------:|----------:|----------:|-------------:|
| 4      | 474       | 12.8      | 199.7        |
| 8      | 171       | 11.9      | 555.9        |
| 12     | 171       | 11.0      | 892.0        |
| 14     | 169       | 11.0      | 1037.0       |
| 16     | 171       | 11.1      | **1179.7**   |
| 18     | 250       | 21.0      | 731.4        |
| 20     | 322       | 20.8      | 784.3        |

**饱和点：batch=16**，吞吐峰值 1179 t/s。

### FP16 vs INT8 对比（batch=4）

| 精度 | TTFT (ms) | TPOT (ms) | 吞吐量 (t/s) | 峰值吞吐 (t/s) | 饱和 batch |
|------|----------:|----------:|-------------:|---------------:|-----------:|
| FP16 | 164       | 17.2      | 205.5        | 226.4          | 4          |
| INT8 | 474       | 12.8      | 199.7        | **1179.7**     | 16         |

**结论：**
- INT8 峰值吞吐是 FP16 的 **5.2×**（1179 vs 226 t/s），饱和点从 batch=4 推迟到 batch=16
- INT8 的 TPOT 更低（11ms vs 17ms），decode 阶段更快，适合高并发场景
- INT8 在低 batch（=4）时 TTFT 反而更高（474ms vs 164ms），prefill 阶段有额外量化开销
- 延迟敏感场景（单请求）用 FP16；吞吐优先场景（高并发服务）用 INT8

## 进入容器
bash step_0_run_docker.sh

## 在容器内下载模型
bash step_1_download_model.sh

## 在容器内转换
bash step_2_transport.sh

## 在容器内 build engine
bash step_3_build_engine.sh

## 在容器内验证 engine
bash step_4_verify_engine.sh

## 在容器内检测运行环境
bash step_7_check_runtime.sh

## 安装 tokenizer 模块
git submodule update --init --recursive

## 安装 rust
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustc --version

## 编译 tokenizer
cd /workspace/third_party/tokenizers-cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

## 编译
mkdir build
cd build
cmake ..
make -j

## 运行
./infer ../engines/qwen25_3b_fp16

## 推理性能基准
![批量推理基准](docs/basic_benchmark.svg)


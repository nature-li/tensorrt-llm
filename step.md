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

## 在容器内验证 c++ 环境
bash step_5_check_cpp.sh

## 在容器内生成 CMakeLists.txt
bash step_6_generate_cmake.sh

## 在容器内检测运行环境
bash step_7_check_runtime.sh

## 安装 tokenizer 模块
cd /workspace
mkdir -p third_party
git clone https://github.com/mlc-ai/tokenizers-cpp.git third_party/tokenizers-cpp
cd third_party/tokenizers-cpp
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

## 再 build 一个 INT8 engine
bash step_8_convert_int8.sh
bash step_9_build_engine_int8.sh

## 编译 jsoncpp
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/workspace/third_party/local \
    -DBUILD_SHARED_LIBS=OFF \
    -DJSONCPP_WITH_TESTS=OFF


## 编译安装 dragon
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_CTL=OFF \
    -DCMAKE_PREFIX_PATH=/workspace/third_party/local \
    -DCMAKE_INSTALL_PREFIX=/workspace/third_party/local
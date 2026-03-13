mkdir -p /workspace/src
cat > /workspace/src/infer.cpp << 'EOF'
// 先写最简单的骨架，确认能编译
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>
#include <iostream>
#include <filesystem>

using namespace tensorrt_llm::executor;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_dir>\n";
        return 1;
    }

    initTrtLlmPlugins();

    ExecutorConfig config(/*maxBeamWidth=*/1);
    Executor executor(argv[1], ModelType::kDECODER_ONLY, config);

    std::cout << "Engine loaded successfully!\n";
    return 0;
}
EOF
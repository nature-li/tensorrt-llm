#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

#include <chrono>
#include <filesystem>
#include <iostream>

using namespace tensorrt_llm::executor;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <engine_dir>\n";
    return 1;
  }

  initTrtLlmPlugins();

  /**
   * 创建 Executor 对象
   */
  ExecutorConfig executor_config(1);
  Executor executor(argv[1], ModelType::kENCODER_ONLY, executor_config);

  /**
   * 测试 prompt
   * "Hello, who are you?" 的 Qwen2.5 tokenization
   */
  VecTokens input_ids = {9707, 11, 889, 553, 498, 30};
  SamplingConfig sampling(1);
  sampling.setTopK(50);
  sampling.setTopP(0.9f);
  sampling.setTemperature(0.8f);
  const int32_t max_new_tokens = 64;

  /**
   * 提交请求
   */
  auto t0 = std::chrono::high_resolution_clock::now();
  Request req(input_ids, max_new_tokens, true, sampling);
  auto req_id = executor.enqueueRequest(req);

  /**
   * 收集结果
   */
  VecTokens output_tokens;
  bool done = false;
  while (!done) {
    auto response =
        executor.awaitResponses(req_id, std::chrono::milliseconds(200));
    for (const auto& r : response) {
      if (r.hasError()) {
        std::cerr << "Error: " << r.getErrorMsg() << std::endl;
        return 1;
      }

      const auto& res = r.getResult();
      if (!res.outputTokenIds.empty()) {
        const auto& beam0 = res.outputTokenIds[0];
        output_tokens.insert(output_tokens.end(), beam0.begin(), beam0.end());
      }

      done = res.isFinal;
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  /**
   * 打印结果
   */
  std::cout << "Generated " << output_tokens.size() << " tokens\n";
  std::cout << "Time: " << ms << " ms\n";
  std::cout << "Throughput: " << output_tokens.size() * 1000.0 / ms
            << " tokens/s\n";
  std::cout << "Token ids: ";
  for (const auto& t : output_tokens) {
    std::cout << t << " ";
  }
  std::cout << "\n";
  return 0;
}
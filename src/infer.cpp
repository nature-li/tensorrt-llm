#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>
#include <tokenizers_cpp.h>

#include <chrono>
#include <filesystem>
#include <fstream>
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
   * 加载 tokenizer
   */
  std::ifstream f("/workspace/models/Qwen2.5-3B-Instruct/tokenizer.json");
  std::string json((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json);

  /**
   * 测试 prompt
   */
  VecTokens input_ids = tokenizer->Encode("Hello, who are you?");
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
  bool first_token = false;
  std::chrono::high_resolution_clock::time_point t_first;

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
        // 记录第一个 token 的时间
        if (!first_token && !beam0.empty()) {
          t_first = std::chrono::high_resolution_clock::now();
          first_token = true;
        }
        output_tokens.insert(output_tokens.end(), beam0.begin(), beam0.end());
      }

      done = res.isFinal;
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  // time to first token
  auto ttfs_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_first - t0)
          .count();
  auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  // time per output token
  auto tpot_ms = output_tokens.size() > 1
                     ? (total_ms - ttfs_ms) / (double)(output_tokens.size() - 1)
                     : 0.0;

  /**
   * decode
   */
  auto text = tokenizer->Decode(output_tokens);

  /**
   * 打印结果
   */
  std::cout << "TTFT: " << ttfs_ms << " ms\n";
  std::cout << "TPOT: " << tpot_ms << " ms\n";
  std::cout << "Throughput: " << output_tokens.size() * 1000.0 / total_ms
            << " tokens/s\n";
  std::cout << "Generated " << output_tokens.size() << " tokens\n";
  std::cout << "Time: " << total_ms << " ms\n";
  std::cout << "Token ids: ";
  for (const auto& t : output_tokens) {
    std::cout << t << " ";
  }
  std::cout << "Token decode: " << text << "\n";
  std::cout << "\n";
  return 0;
}

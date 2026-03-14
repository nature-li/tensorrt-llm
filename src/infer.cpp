#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>
#include <tokenizers_cpp.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace tensorrt_llm::executor;

struct RequestResult {
  VecTokens output_tokens;
  std::chrono::high_resolution_clock::time_point t_first;
  bool first_token_received = false;
  bool done = false;
};

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_dir> <batch_size>\n";
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
   * 同时提交 batch_size 个请求
   */
  int batch_size = std::stoi(argv[2]);
  if (batch_size <= 0) {
    batch_size = 1;
  }
  auto t0 = std::chrono::high_resolution_clock::now();

  std::map<IdType, RequestResult> results;
  for (int i = 0; i < batch_size; i++) {
    Request req(input_ids, max_new_tokens, true, sampling);
    auto req_id = executor.enqueueRequest(req);
    results[req_id] = RequestResult{};
  }

  /**
   * 等待所有请求完成
   */
  int done_count = 0;
  while (done_count < batch_size) {
    auto responses = executor.awaitResponses(std::chrono::milliseconds(200));
    for (const auto& r : responses) {
      auto req_id = r.getRequestId();
      auto& res_state = results[req_id];

      if (r.hasError()) {
        std::cerr << "Error on req " << req_id << ": " << r.getErrorMsg()
                  << "\n";
        return 1;
      }

      const auto& res = r.getResult();
      if (!res.outputTokenIds.empty()) {
        const auto& beam0 = res.outputTokenIds[0];
        if (!res_state.first_token_received && !beam0.empty()) {
          res_state.t_first = std::chrono::high_resolution_clock::now();
          res_state.first_token_received = true;
        }

        res_state.output_tokens.insert(res_state.output_tokens.end(),
                                       beam0.begin(), beam0.end());
      }

      if (res.isFinal) {
        res_state.done = true;
        done_count++;
      }
    }
  }

  /**
   * 计算时间
   */
  auto t1 = std::chrono::high_resolution_clock::now();
  auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  double avg_ttft = 0;
  double avg_tpot = 0;
  int total_tokens = 0;
  for (auto& [req_id, res_state] : results) {
    auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       res_state.t_first - t0)
                       .count();
    auto tpot_ms = res_state.output_tokens.size() > 1
                       ? (total_ms - ttft_ms) /
                             (double)(res_state.output_tokens.size() - 1)
                       : 0.0;

    avg_ttft += ttft_ms;
    avg_tpot += tpot_ms;
    total_tokens += res_state.output_tokens.size();
  }
  avg_ttft /= batch_size;
  avg_tpot /= batch_size;

  std::cout << "=== Batch Size: " << batch_size << " ===\n";
  std::cout << "Avg TTFT:   " << avg_ttft << " ms\n";
  std::cout << "Avg TPOT:   " << avg_tpot << " ms/token\n";
  std::cout << "Throughput: " << total_tokens * 1000.0 / total_ms
            << " tokens/s\n";
  std::cout << "Total time: " << total_ms << " ms\n";
  std::cout << "Total tokens: " << total_tokens << "\n";

  // 打印每条请求的解码结果
  for (auto& [req_id, res_state] : results) {
    auto text = tokenizer->Decode(res_state.output_tokens);
    std::cout << "[req " << req_id << "] " << text << "\n";
  }

  return 0;
}

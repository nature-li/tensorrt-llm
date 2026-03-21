#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>

#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <string>

using namespace tensorrt_llm::executor;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::milliseconds;

// ── CLI helpers ──────────────────────────────────────────────────────────────
bool hasFlag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i)
    if (argv[i] == key) return true;
  return false;
}

// ── 随机 prompt
VecTokens make_random_prompt(int length, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(100, 100000);
  VecTokens tokens(length);
  for (auto& t : tokens) t = dist(rng);
  return tokens;
}

struct RequestResult {
  Clock::time_point t_enqueue;
  Clock::time_point t_first;
  Clock::time_point t_end;
  VecTokens output_tokens;
  bool first_token_received = false;
  bool done = false;
};

int main(int argc, char** argv) {
  // Usage: infer <engine_dir> <batch_size> [--reuse_kv]
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_dir> <batch_size> [--reuse_kv]\n";
    return 1;
  }

  const int  batch_size = std::max(1, std::stoi(argv[2]));
  const bool reuse_kv   = hasFlag(argc, argv, "--reuse_kv");
  const int  input_len  = 512;
  const int  output_len = 200;

  initTrtLlmPlugins();

  // ── Executor ──────────────────────────────────────────────────────────────
  KvCacheConfig kv_cache_config;
  kv_cache_config.setEnableBlockReuse(reuse_kv);

  ExecutorConfig executor_config(1);
  executor_config.setKvCacheConfig(kv_cache_config);
  Executor executor(argv[1], ModelType::kDECODER_ONLY, executor_config);

  std::cout << "engine     : " << argv[1] << "\n";
  std::cout << "batch_size : " << batch_size << "\n";
  std::cout << "input_len  : " << input_len << "\n";
  std::cout << "output_len : " << output_len << "\n";
  std::cout << "reuse_kv   : " << (reuse_kv ? "on" : "off") << "\n\n";

  // ── Sampling ──────────────────────────────────────────────────────────────
  SamplingConfig sampling(1);
  sampling.setTopK(50);
  sampling.setTopP(0.9f);
  sampling.setTemperature(0.8f);

  // ── Enqueue ───────────────────────────────────────────────────────────────
  std::map<IdType, RequestResult> results;
  Clock::time_point t_wall_start;

  for (int i = 0; i < batch_size; i++) {
    // reuse_kv=on：固定 seed，所有请求相同 prompt，触发 prefix cache 命中
    // reuse_kv=off：seed=i，每个请求不同 prompt
    unsigned seed = reuse_kv ? 42 : (unsigned)i;
    VecTokens prompt = make_random_prompt(input_len, seed);

    Request req(prompt, output_len, true, sampling);
    auto t_enq = Clock::now();
    if (i == 0) t_wall_start = t_enq;  // wall time 从第一个请求开始
    auto req_id = executor.enqueueRequest(req);
    results[req_id].t_enqueue = t_enq;
  }

  // ── 收集响应 ──────────────────────────────────────────────────────────────
  int done_count = 0;
  while (done_count < batch_size) {
    auto responses = executor.awaitResponses(Ms(200));
    for (const auto& r : responses) {
      auto req_id = r.getRequestId();
      auto& rs    = results[req_id];

      if (r.hasError()) {
        std::cerr << "Error on req " << req_id << ": "
                  << r.getErrorMsg() << "\n";
        return 1;
      }

      const auto& res = r.getResult();
      if (!res.outputTokenIds.empty()) {
        const auto& beam0 = res.outputTokenIds[0];
        if (!rs.first_token_received && !beam0.empty()) {
          rs.t_first = Clock::now();
          rs.first_token_received = true;
        }
        rs.output_tokens.insert(rs.output_tokens.end(),
                                beam0.begin(), beam0.end());
      }

      if (res.isFinal) {
        rs.t_end = Clock::now();
        rs.done  = true;
        done_count++;
      }
    }
  }

  long wall_ms = std::chrono::duration_cast<Ms>(
                     Clock::now() - t_wall_start).count();

  // ── 统计 ──────────────────────────────────────────────────────────────────
  double sum_ttft = 0, sum_tpot = 0;
  int total_tokens = 0;

  for (auto& [req_id, rs] : results) {
    long ttft = std::chrono::duration_cast<Ms>(
                    rs.t_first - rs.t_enqueue).count();
    long e2e  = std::chrono::duration_cast<Ms>(
                    rs.t_end   - rs.t_enqueue).count();
    int  n    = (int)rs.output_tokens.size();
    double tpot = n > 1 ? (double)(e2e - ttft) / (n - 1) : 0.0;

    sum_ttft     += ttft;
    sum_tpot     += tpot;
    total_tokens += n;
  }

  double avg_ttft   = sum_ttft / batch_size;
  double avg_tpot   = sum_tpot / batch_size;
  double throughput = wall_ms ? total_tokens * 1000.0 / wall_ms : 0.0;

  std::cout << "=== Results ===\n";
  std::cout << "Avg TTFT   : " << avg_ttft   << " ms\n";
  std::cout << "Avg TPOT   : " << avg_tpot   << " ms/token\n";
  std::cout << "Throughput : " << throughput  << " tokens/s\n";
  std::cout << "Wall time  : " << wall_ms     << " ms\n";
  std::cout << "Total tokens: " << total_tokens << "\n";

  return 0;
}

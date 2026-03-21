#include <drogon/drogon.h>
#include <json/writer.h>
#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/plugins/api/tllmPlugin.h>
#include <tokenizers_cpp.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

using namespace tensorrt_llm::executor;
using namespace drogon;
using Callback = std::function<void(const Response&)>;

/**
 * 后台轮循
 */
class ExecutorWrapper {
 public:
  ExecutorWrapper(const std::string& engine_dir)
      : executor_(engine_dir, ModelType::kDECODER_ONLY, ExecutorConfig(1)) {
    poll_thread_ = std::thread([this]() { poolLoop(); });
  }

  ~ExecutorWrapper() {
    stop_ = true;
    poll_thread_.join();
  }

  IdType enqueue(const Request& req, Callback cb) {
    auto req_id = executor_.enqueueRequest(req);
    std::lock_guard<std::mutex> lock(mutex_);
    callback_[req_id] = std::move(cb);
    return req_id;
  }

 private:
  void poolLoop() {
    while (!stop_) {
      auto response = executor_.awaitResponses(std::chrono::milliseconds(10));
      for (const auto& r : response) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = callback_.find(r.getRequestId());
        if (it != callback_.end()) {
          it->second(r);
          if (r.getResult().isFinal) {
            callback_.erase(it);
          }
        }
      }
    }
  }

 private:
  Executor executor_;
  std::unordered_map<IdType, Callback> callback_;
  std::mutex mutex_;
  std::thread poll_thread_;
  std::atomic<bool> stop_{false};
};

/**
 * 增量 decode，解决中文跨 token 乱码
 */
class IncrementalDecoder {
 public:
  explicit IncrementalDecoder(tokenizers::Tokenizer* tok) : tok_(tok) {}

  std::string decode(const VecTokens& all_tokens) {
    std::string full = tok_->Decode(all_tokens);
    std::string delta = full.substr(prev_text_.size());
    prev_text_ = full;
    return delta;
  }

 private:
  tokenizers::Tokenizer* tok_;
  std::string prev_text_;
};

static std::string to_json(const Json::Value& v) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  return Json::writeString(builder, v);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <engine_dir> <tokenizer_json> [port]\n";
    return 1;
  }

  const std::string engine_dir = argv[1];
  const std::string tokenizer_json = argv[2];
  const int port = argc >= 4 ? std::stoi(argv[3]) : 8080;

  initTrtLlmPlugins();

  /**
   * 加载 tokenizer
   */
  std::ifstream f(tokenizer_json);
  std::string json((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json);

  /**
   * 启动后台轮询
   */
  auto wrapper = std::make_shared<ExecutorWrapper>(engine_dir);

  /**
   * 测试时同源策略
   */
  app().registerSyncAdvice([](const HttpRequestPtr& req) -> HttpResponsePtr {
    if (req->method() == Options) {
      auto resp = HttpResponse::newHttpResponse();
      resp->addHeader("Access-Control-Allow-Origin", "*");
      resp->addHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
      resp->addHeader("Access-Control-Allow-Headers", "Content-Type");
      resp->setStatusCode(k204NoContent);
      return resp;
    }
    // 不拦截，继续往下走
    return nullptr;
  });

  /**
   * GET /health
   */
  app().registerHandler(
      "/health",
      [](const HttpRequestPtr& req,
         std::function<void(const HttpResponsePtr&)>&& callback) {
        Json::Value json;
        json["status"] = "ok";
        auto resp = HttpResponse::newHttpJsonResponse(json);
        resp->addHeader("Access-Control-Allow-Origin", "*");
        callback(resp);
      },
      {Get});

  /**
   * POST /generate（流式 SSE）
   */
  app().registerHandler(
      "/generate",
      [wrapper, &tokenizer](
          const HttpRequestPtr& req,
          std::function<void(const HttpResponsePtr&)>&& callback) {
        // 解析请求体
        auto body_json = req->getJsonObject();
        if (!body_json) {
          auto resp = HttpResponse::newHttpResponse();
          resp->setStatusCode(k400BadRequest);
          resp->setBody("invalid json");
          callback(resp);
          return;
        }

        std::string prompt = (*body_json)["prompt"].asString();
        int max_new_tokens = body_json->get("max_new_tokens", 64).asInt();
        float temperature = body_json->get("temperature", 0.8f).asFloat();
        int top_k = body_json->get("top_k", 50).asInt();
        float top_p = body_json->get("top_p", 0.9f).asFloat();

        if (prompt.empty()) {
          auto resp = HttpResponse::newHttpResponse();
          resp->setStatusCode(k400BadRequest);
          resp->setBody("prompt is required");
          callback(resp);
          return;
        }

        // encode prompt
        VecTokens input_ids = tokenizer->Encode(prompt);

        SamplingConfig sampling(1);
        sampling.setTopK(top_k);
        sampling.setTopP(top_p);
        sampling.setTemperature(temperature);

        Request llm_req(input_ids, max_new_tokens, true, sampling);

        // 计时
        auto t0 =
            std::make_shared<std::chrono::high_resolution_clock::time_point>(
                std::chrono::high_resolution_clock::now());
        auto t_first =
            std::make_shared<std::chrono::high_resolution_clock::time_point>();
        auto first_token = std::make_shared<bool>(false);
        auto all_tokens = std::make_shared<VecTokens>();
        auto decoder = std::make_shared<IncrementalDecoder>(tokenizer.get());

        /**
         * 创建 SSE 异步流
         * Drogon 的 AsyncStreamResponse 做到不阻塞 HTTP 线程：
         * callback 立即返回，后台回调线程负责持续写入 stream
         */
        auto resp = HttpResponse::newAsyncStreamResponse(
            [wrapper, llm_req, t0, t_first, first_token, all_tokens,
             decoder](ResponseStreamPtr stream) mutable {
              auto shared_stream =
                  std::make_shared<ResponseStreamPtr>(std::move(stream));
              wrapper->enqueue(llm_req, [shared_stream, t0, t_first,
                                         first_token, all_tokens,
                                         decoder](const Response& r) {
                auto& stream = *shared_stream;
                if (r.hasError()) {
                  Json::Value err;
                  err["error"] = r.getErrorMsg();
                  //   stream->send("data: " + err.toStyledString() + "\n\n");
                  stream->send("data: " + to_json(err) + "\n\n");
                  stream->close();
                  return;
                }

                const auto& result = r.getResult();

                if (!result.outputTokenIds.empty()) {
                  const auto& beam0 = result.outputTokenIds[0];

                  // 记录首个 token 时间
                  if (!*first_token && !beam0.empty()) {
                    *t_first = std::chrono::high_resolution_clock::now();
                    *first_token = true;
                  }

                  // 增量 decode
                  all_tokens->insert(all_tokens->end(), beam0.begin(),
                                     beam0.end());
                  std::string delta = decoder->decode(*all_tokens);
                  if (!delta.empty()) {
                    std::cout << "delta: " << delta << std::endl;
                    Json::Value chunk;
                    chunk["delta"] = delta;
                    // 每生成一段文字立刻推给客户端
                    // stream->send("data: " + chunk.toStyledString() + "\n\n");
                    stream->send("data: " + to_json(chunk) + "\n\n");
                  }

                  if (!result.outputTokenIds.empty()) {
                    const auto& beam0 = result.outputTokenIds[0];
                  }
                }

                if (result.isFinal) {
                  auto t1 = std::chrono::high_resolution_clock::now();
                  auto total_ms =
                      std::chrono::duration_cast<std::chrono::milliseconds>(t1 -
                                                                            *t0)
                          .count();
                  auto ttft_ms =
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          *t_first - *t0)
                          .count();
                  double tpot_ms = all_tokens->size() > 1
                                       ? (total_ms - ttft_ms) /
                                             (double)(all_tokens->size() - 1)
                                       : 0.0;

                  Json::Value done;
                  done["done"] = true;
                  done["ttft_ms"] = (int)ttft_ms;
                  done["tpot_ms"] = tpot_ms;
                  done["total_ms"] = (int)total_ms;
                  done["num_tokens"] = (int)all_tokens->size();
                  done["throughput"] = all_tokens->size() * 1000.0 / total_ms;
                  //   stream->send("data: " + done.toStyledString() + "\n\n");
                  stream->send("data: " + to_json(done) + "\n\n");
                  stream->close();
                }
              });
            });

        resp->setContentTypeString("text/event-stream");
        resp->addHeader("Cache-Control", "no-cache");
        resp->addHeader("Connection", "keep-alive");
        resp->addHeader("Access-Control-Allow-Origin", "*");

        /**
         * 立即返回，不阻塞 HTTP 线程
         * 后台回调线程持续向 stream 写入数据
         */
        callback(resp);
      },
      {Post});

  std::cout << "Server listening on http://0.0.0.0:" << port << "\n";
  std::cout << "Engine:    " << engine_dir << "\n";
  std::cout << "Tokenizer: " << tokenizer_json << "\n";

  app().addListener("0.0.0.0", port).setThreadNum(4).run();

  return 0;
}
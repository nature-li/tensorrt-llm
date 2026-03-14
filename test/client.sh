curl -N -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "用中文介绍一下你自己", "max_new_tokens": 64}'
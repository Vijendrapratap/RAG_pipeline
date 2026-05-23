#!/usr/bin/env bash
set -euo pipefail
docker exec ollama ollama pull bge-m3
docker exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker exec ollama ollama pull deepseek-r1:7b-qwen-distill-q4_K_M
echo "✅ All models pulled. Ollama list:"
docker exec ollama ollama list

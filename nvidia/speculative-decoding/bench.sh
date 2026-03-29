#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Speculative Decoding Benchmark
#
# Runs a TRT-LLM server in one of three modes (baseline, eagle3, draft-target),
# waits for it to become ready, sends benchmark prompts, and reports results.
#
# Models are cached on the host so they survive container restarts.
#
# Usage:
#   ./bench.sh eagle3          # EAGLE-3 with GPT-OSS-120B
#   ./bench.sh draft-target    # Draft-Target with Llama-3.3-70B + 8B draft
#   ./bench.sh baseline-eagle  # GPT-OSS-120B without speculation
#   ./bench.sh baseline-draft  # Llama-3.3-70B without speculation
#
# Requires: HF_TOKEN environment variable set
# =============================================================================

MODE="${1:-}"
PORT=8000
CONTAINER_NAME="spec-decode-bench"
IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"
HF_CACHE="$HOME/.cache/huggingface"
SPEC_CACHE="$HOME/.cache/speculative-models"
MAX_TOKENS=200
NUM_RUNS=3
STARTUP_TIMEOUT=600  # seconds to wait for server

# -- Prompts used for benchmarking (deterministic with temperature=0) ----------
PROMPTS=(
  "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves."
  "Write a step-by-step guide to implementing a hash table from scratch in C, including collision handling."
  "Solve the following problem step by step. If a train travels 180 km in 3 hours, and then slows down by 20% for the next 2 hours, what is the total distance traveled? Show all intermediate calculations and provide a final numeric answer."
)

# -- Validation ----------------------------------------------------------------
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 {eagle3|draft-target|baseline-eagle|baseline-draft}"
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN is not set. Export it first:"
  echo "  export HF_TOKEN=hf_..."
  exit 1
fi

# -- Ensure cache directories exist --------------------------------------------
mkdir -p "$HF_CACHE" "$SPEC_CACHE"

# -- Stop any previous benchmark container -------------------------------------
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
  echo "Stopping existing benchmark container..."
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  sleep 2
fi

# -- Build docker run command based on mode ------------------------------------
DOCKER_COMMON=(
  docker run
  --name "$CONTAINER_NAME"
  -e "HF_TOKEN=$HF_TOKEN"
  -v "$HF_CACHE:/root/.cache/huggingface/"
  -v "$SPEC_CACHE:/opt/speculative-models/"
  --rm -d
  --ulimit memlock=-1 --ulimit stack=67108864
  --gpus=all --ipc=host --network host
  "$IMAGE"
)

case "$MODE" in
  eagle3)
    MODEL="openai/gpt-oss-120b"
    SPEC_DIR="/opt/speculative-models/gpt-oss-120b-Eagle3"
    SCRIPT=$(cat <<'INNEREOF'
      set -e
      hf download openai/gpt-oss-120b
      hf download nvidia/gpt-oss-120b-Eagle3-long-context \
          --local-dir /opt/speculative-models/gpt-oss-120b-Eagle3/
      cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/speculative-models/gpt-oss-120b-Eagle3/
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
      export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs"
      mkdir -p $TIKTOKEN_ENCODINGS_BASE
      wget -q -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
      wget -q -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
      trtllm-serve openai/gpt-oss-120b \
        --backend pytorch --tp_size 1 \
        --max_batch_size 1 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml
INNEREOF
    )
    ;;

  baseline-eagle)
    MODEL="openai/gpt-oss-120b"
    SCRIPT=$(cat <<'INNEREOF'
      set -e
      hf download openai/gpt-oss-120b
      export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs"
      mkdir -p $TIKTOKEN_ENCODINGS_BASE
      wget -q -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
      wget -q -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
      cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
      trtllm-serve openai/gpt-oss-120b \
        --backend pytorch --tp_size 1 \
        --max_batch_size 1 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml
INNEREOF
    )
    ;;

  draft-target)
    MODEL="nvidia/Llama-3.3-70B-Instruct-FP4"
    SCRIPT=$(cat <<'INNEREOF'
      set -e
      hf download nvidia/Llama-3.3-70B-Instruct-FP4
      hf download nvidia/Llama-3.1-8B-Instruct-FP4 \
          --local-dir /opt/speculative-models/Llama-3.1-8B-Instruct-FP4/
      cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
disable_overlap_scheduler: true
speculative_config:
  decoding_type: DraftTarget
  max_draft_len: 4
  speculative_model_dir: /opt/speculative-models/Llama-3.1-8B-Instruct-FP4/
kv_cache_config:
  enable_block_reuse: false
EOF
      trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \
        --backend pytorch --tp_size 1 \
        --max_batch_size 1 \
        --kv_cache_free_gpu_memory_fraction 0.9 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml
INNEREOF
    )
    ;;

  baseline-draft)
    MODEL="nvidia/Llama-3.3-70B-Instruct-FP4"
    SCRIPT=$(cat <<'INNEREOF'
      set -e
      hf download nvidia/Llama-3.3-70B-Instruct-FP4
      cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
disable_overlap_scheduler: true
kv_cache_config:
  enable_block_reuse: false
EOF
      trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \
        --backend pytorch --tp_size 1 \
        --max_batch_size 1 \
        --kv_cache_free_gpu_memory_fraction 0.9 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml
INNEREOF
    )
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {eagle3|draft-target|baseline-eagle|baseline-draft}"
    exit 1
    ;;
esac

# -- Launch the container ------------------------------------------------------
echo "=== Starting server in '$MODE' mode ==="
echo "Model: $MODEL"
echo "Container: $CONTAINER_NAME"
echo ""

"${DOCKER_COMMON[@]}" bash -c "$SCRIPT"

# -- Wait for server to be ready -----------------------------------------------
echo "Waiting for server on port $PORT (timeout: ${STARTUP_TIMEOUT}s)..."
echo "  (Model download + loading can take a while on first run)"
echo "  Tail logs with: docker logs -f $CONTAINER_NAME"
echo ""

elapsed=0
while ! curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    echo "Error: Container exited. Check logs:"
    echo "  docker logs $CONTAINER_NAME"
    exit 1
  fi
  if (( elapsed >= STARTUP_TIMEOUT )); then
    echo "Error: Server did not become ready within ${STARTUP_TIMEOUT}s"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  # Print progress every 30 seconds
  if (( elapsed % 30 == 0 )); then
    echo "  Still waiting... (${elapsed}s elapsed)"
  fi
done

echo "Server is ready! (took ${elapsed}s)"
echo ""

# -- Warmup --------------------------------------------------------------------
echo "=== Sending warmup request (discarded) ==="
curl -s -o /dev/null -X POST "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg model "$MODEL" \
    '{model: $model, prompt: "Hello", max_tokens: 10, temperature: 0}'
  )"
echo "Warmup complete."
echo ""

# -- Run benchmarks ------------------------------------------------------------
echo "=== Running benchmark: $NUM_RUNS runs x ${#PROMPTS[@]} prompts, max_tokens=$MAX_TOKENS ==="
echo ""

RESULTS_FILE=$(mktemp)
echo "mode,prompt_idx,run,tokens,wall_time_s,tokens_per_sec,ttft_s,avg_decoded_per_iter" > "$RESULTS_FILE"

for prompt_idx in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$prompt_idx]}"
  short_prompt="${prompt:0:60}..."
  echo "Prompt $((prompt_idx+1))/${#PROMPTS[@]}: \"$short_prompt\""

  for run in $(seq 1 "$NUM_RUNS"); do
    # Use curl timing to get TTFT and total time
    timing=$(curl -s -o /tmp/bench_response.json -w "%{time_starttransfer} %{time_total}" \
      -X POST "http://localhost:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d "$(jq -n \
        --arg model "$MODEL" \
        --arg prompt "$prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        '{model: $model, prompt: $prompt, max_tokens: $max_tokens, temperature: 0}'
      )")

    ttft=$(echo "$timing" | awk '{print $1}')
    total=$(echo "$timing" | awk '{print $2}')

    # Extract metrics from response
    tokens=$(jq -r '.usage.completion_tokens // 0' /tmp/bench_response.json)
    avg_per_iter=$(jq -r '.choices[0].avg_decoded_tokens_per_iter // 1.0' /tmp/bench_response.json)
    tps=$(echo "$tokens $total" | awk '{if ($2 > 0) printf "%.2f", $1/$2; else print "0"}')

    echo "  Run $run: ${tokens} tokens in ${total}s (${tps} tok/s, TTFT=${ttft}s, avg_per_iter=${avg_per_iter})"
    echo "$MODE,$prompt_idx,$run,$tokens,$total,$tps,$ttft,$avg_per_iter" >> "$RESULTS_FILE"
  done
  echo ""
done

# -- Summary -------------------------------------------------------------------
echo "=== Summary for mode: $MODE ==="
echo ""

# Calculate averages using awk
awk -F',' 'NR>1 {
  n++
  tok += $4; wall += $5; tps += $6; ttft += $7; api += $8
}
END {
  if (n > 0) {
    printf "  Runs:                    %d\n", n
    printf "  Avg completion tokens:   %.0f\n", tok/n
    printf "  Avg wall time:           %.2fs\n", wall/n
    printf "  Avg tokens/sec:          %.2f\n", tps/n
    printf "  Avg TTFT:                %.3fs\n", ttft/n
    printf "  Avg decoded tokens/iter: %.2f\n", api/n
  }
}' "$RESULTS_FILE"

echo ""
echo "Raw results saved to: $RESULTS_FILE"
echo ""

# -- Cleanup prompt ------------------------------------------------------------
echo "Server is still running. When done benchmarking:"
echo "  docker stop $CONTAINER_NAME"

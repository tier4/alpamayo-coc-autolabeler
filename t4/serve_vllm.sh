#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 TIER IV, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Source this file, then call start_vllm / stop_vllm. start_vllm starts REPLICAS OpenAI-compatible
# vLLM servers of MODEL on GPUS GPUs (each replica on GPUS/REPLICAS of them) on ports 8100,
# 8101, ... of 127.0.0.1, waits until all are healthy and sets the arrays `pids` and `ports`.
# It returns non-zero if a server exits or is not healthy within 45 minutes.
#
# Inputs (environment):
#   VPY            python with vLLM installed (default: python3)
#   MODEL          model path or Hugging Face id (required)
#   SERVED         name the servers report (default: basename of MODEL)
#   LOGDIR         where r<i>.log go (required)
#   GPUS=8 REPLICAS=2 MAX_MODEL_LEN=262144 IMAGE_LIMIT=160 THINKING=1
#   PRESET         "dense" (default): plain tensor parallel.
#                  "flash-next": the vLLM recipe flags for Qwen3.8-Flash-Next-FP8 (expert
#                  parallel, Triton MoE, MTP speculative decoding, FP8 KV cache).
#   EXTRA_VLLM_ARGS  more server flags, appended as is.
#
# --interleave-mm-strings keeps each frame next to its timestamp caption: without it vLLM moves
# every image ahead of the text for string-format chat templates, and the model loses which
# time each frame shows. --reasoning-parser qwen3 separates the thinking from the JSON answer.

VPY=${VPY:-python3}
GPUS=${GPUS:-8}
REPLICAS=${REPLICAS:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
IMAGE_LIMIT=${IMAGE_LIMIT:-160}
PRESET=${PRESET:-dense}
SERVED=${SERVED:-$(basename "${MODEL:-model}")}
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_$(id -u)}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$(id -u)}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# flashinfer compiles kernels at startup and needs an nvcc that knows its flags: prefer the
# toolkit of torch's CUDA major, then the newest installed one.
if [ -z "${CUDA_HOME:-}" ]; then
  cuda_major=$($VPY -c 'import torch; print(torch.version.cuda.split(".")[0])')
  for _cuda in $(ls -d /usr/local/cuda-"$cuda_major"* 2>/dev/null | sort -rV) \
      $(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -rV) /usr/local/cuda; do
    if [ -x "$_cuda/bin/nvcc" ]; then export CUDA_HOME="$_cuda"; break; fi
  done
fi
# Cached JIT builds made with another nvcc (or a failed build) are rebuilt.
for _ninja in "$HOME"/.cache/flashinfer/*/*/cached_ops/*/build.ninja; do
  [ -e "$_ninja" ] || continue
  grep -q "$CUDA_HOME/bin/nvcc" "$_ninja" || rm -rf "$(dirname "$_ninja")"
done

start_vllm() {
  : "${MODEL:?set MODEL}" "${LOGDIR:?set LOGDIR}"
  local tp=$((GPUS / REPLICAS)) r port devices flags=()
  mkdir -p "$LOGDIR"
  case "$PRESET" in
    dense) ;;
    flash-next)
      flags=(--enable-expert-parallel --moe-backend triton --max-num-seqs 256
             --enable-prefix-caching --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8
             --speculative-config '{"method":"mtp","num_speculative_tokens":3}') ;;
    *) echo "unknown PRESET $PRESET"; return 1 ;;
  esac
  [ "${THINKING:-1}" = 0 ] && flags+=(--default-chat-template-kwargs '{"enable_thinking": false}')
  # shellcheck disable=SC2206
  flags+=(${EXTRA_VLLM_ARGS:-})
  pids=() ports=()
  echo "starting $REPLICAS server(s) of $SERVED, TP$tp each, preset $PRESET, CUDA_HOME=${CUDA_HOME:-}"
  for ((r = 0; r < REPLICAS; r++)); do
    port=$((8100 + r))
    devices=$(seq -s, $((r * tp)) $((r * tp + tp - 1)))
    CUDA_VISIBLE_DEVICES=$devices OMP_NUM_THREADS=8 $VPY -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size "$tp" \
      --max-model-len "$MAX_MODEL_LEN" --limit-mm-per-prompt "{\"image\": $IMAGE_LIMIT}" \
      --interleave-mm-strings --reasoning-parser qwen3 \
      --host 127.0.0.1 --port "$port" "${flags[@]}" > "$LOGDIR/r$r.log" 2>&1 &
    pids+=($!) ports+=("$port")
  done
  local begin=$SECONDS pid
  for port in "${ports[@]}"; do
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
      for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || { echo "a vLLM server exited during startup; see $LOGDIR"; return 1; }
      done
      [ $((SECONDS - begin)) -lt 2700 ] || { echo "vLLM not healthy after 45 min"; return 1; }
      sleep 15
    done
  done
  echo "$REPLICAS server(s) healthy after $((SECONDS - begin)) s on ports ${ports[*]}"
}

stop_vllm() {
  kill "${pids[@]}" 2>/dev/null
  wait "${pids[@]}" 2>/dev/null
  pids=()
}

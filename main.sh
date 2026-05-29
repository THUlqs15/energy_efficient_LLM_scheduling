#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# ENERGY-EFFICIENT LLM SCHEDULING — master runner
#
# Usage:  bash main.sh
#
# Edit the USER KNOBS section below to change parameters before running.
# ==============================================================================

# ==============================================================================
# USER KNOBS — change these to control the experiment
# ==============================================================================

TAG="demo"
# Folder name under results/ where outputs are stored.
# Change this each time to keep old results (e.g. TAG="beta01_run").

MODE="both"
# Which scheduler to run: "default" (baseline), "custom" (energy-aware),
# or "both" (default then custom, same workload).

# ----- vLLM server settings ---------------------------------------------------
VLLM_DIR="/home/ubuntu/lqs/vllm"
# Path to the local vLLM source tree (editable install).

MODEL_DIR="/home/ubuntu/lqs/LLM_model"
# Path to the model weights (Qwen3-14B).

MODEL_NAME="default"
# Name served to clients (used in --model flag of workload sender).

PORT=8000
# HTTP port for the vLLM OpenAI-compatible API server.

GPU_INDEX=0
# Which GPU to use (CUDA device index).

MAX_MODEL_LEN=8192
# Maximum sequence length (input + output tokens).

DEFAULT_MAX_NUM_SEQS=256
# Active cap for default scheduler.

CUSTOM_MAX_NUM_SEQS=400
# Active cap for custom scheduler (larger to give solver more candidates).

GPU_MEM_UTIL=0.95
# Fraction of GPU memory allocated for KV cache (0.0–1.0).

# ----- Workload settings ------------------------------------------------------
NUM_REQUESTS=400
# Number of requests to send in this run.
# Must not exceed the number of lines in trace.jsonl.

RATE_QPS=2.0
# Request arrival rate (requests per second).
# Determines how fast the workload is replayed from trace.jsonl.

MIN_OUT_TOK=1024
# Minimum number of output tokens per request.

MAX_OUT_TOK=1024
# Maximum number of output tokens per request.

TRACE_SEED=42
# Random seed for trace generation (used by prepare_dataset.py).

# ----- Scheduler hyper-parameters (custom mode only) --------------------------
BETA=4.0
# Energy-utility trade-off parameter. Larger β → solver prioritises energy
# saving over SLO attainment, tends to pick lower GPU frequencies.
# The energy term is β × Watts × seconds (Joules).
# Typical range: 0.001 (mild energy saving) to 1.0 (aggressive energy saving).

W_TTFT=2000.0
# Weight for TTFT in the per-request priority calculation.
# Higher w_TTFT makes the solver more sensitive to TTFT deadlines.

W_TPOT=1.0
# Weight for TPOT in the per-request priority calculation.
# Higher w_TPOT makes the solver more sensitive to TPOT deadlines.

ETA_MS=200
# Per-iteration time budget η (milliseconds). Default 1e9 is effectively
# infinite, so the time constraint (eq. 6) is inactive. Lower this to cap
# worst-case iteration latency (e.g. 500 for a 500ms cap).

VLLM_MAX_BATCHED_TOKENS=8192
# Passed to vLLM --max-num-batched-tokens. Must be >= MAX_MODEL_LEN.
# 0 means let vLLM pick its default.

SOLVER_LMAX=8192
# Only for custom solver. Maximum tokens per batch in the solver's greedy fill.
# Can be smaller than MAX_MODEL_LEN.

FREQ_STRIDE=3
# Stride for frequency candidate subsampling. A800 has 82 supported SM clocks;
# with stride=4, the solver evaluates every 4th clock (ceil(82/4) = 21 candidates).
# Larger = faster solving but coarser frequency search.
# Typical range: 1 (exhaustive) to 8 (very fast).

MAX_BATCH_SIZE=256
# Maximum requests per iteration (batch cap).
# Independent of MAX_NUM_SEQS (active cap = max concurrent KV-holding requests).

SOLUTION_MODE=3
# Solver heuristic:
#   1 = Heuristic 2 (freq-independent priority, single admission, joint prefix×freq)
#   2 = Heuristic 3 (freq-dependent priority, per-frequency admission and prefix)
#   3 = Heuristic 4 (freq-dependent priority, stop when q_n(f) <= 0, no prefix)

IS_CHUNKED_PREFILL=1
# 0 = non-chunked prefill (original logic, full prompt in one iteration)
# 1 = chunked prefill (prompt may be split across multiple iterations;
#     partial-prefill requests use TTFT deadline and cross-term b_p·l_q·l_kv)

# ----- Power logging ----------------------------------------------------------
POWER_INTERVAL_S=0.1
# Interval between GPU power samples (seconds). Smaller gives finer-grained
# energy integration but produces more CSV data.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- Conda activation ---------------------------------------------------
for conda_sh in \
    ~/miniconda3/etc/profile.d/conda.sh \
    /home/ubuntu/miniconda3/etc/profile.d/conda.sh \
    ~/anaconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh; do
    if [[ -f "$conda_sh" ]]; then
        source "$conda_sh"
        break
    fi
done
conda activate myvllm

echo "[main] TAG=$TAG  MODE=$MODE  NUM_REQUESTS=$NUM_REQUESTS  RATE_QPS=$RATE_QPS"

# ---------- 0. Apply patch -----------------------------------------------------
echo "[main] Applying vLLM energy scheduler patch ..."
bash "${SCRIPT_DIR}/vllm_patches/apply_patch.sh" "$VLLM_DIR"

# ---------- 1. Prepare trace ---------------------------------------------------
if [[ -f "${SCRIPT_DIR}/trace.jsonl" ]]; then
    echo "[main] trace.jsonl already exists — skipping (delete it to re-generate)"
else
    echo "[main] Preparing trace.jsonl ..."
    python "${SCRIPT_DIR}/scripts/prepare_dataset.py"
fi

mkdir -p "${SCRIPT_DIR}/results/${TAG}"

reset_gpu_clocks() {
    python -c "
import os, sys
sys.path.insert(0, '${SCRIPT_DIR}/vllm_patches')
from frequency_controller import FrequencyController
ctl = FrequencyController($GPU_INDEX)
ctl.reset()
print('[main] GPU clocks reset')
" 2>/dev/null || { nvidia-smi -i $GPU_INDEX -rgc 2>/dev/null; nvidia-smi -i $GPU_INDEX -rmc 2>/dev/null; } || true
}

# ---------- Helper: run one sub-experiment -------------------------------------
run_experiment() {
    local sched_label="$1"   # "default" or "custom"
    local tag_dir="${SCRIPT_DIR}/results/${TAG}"
    local label="default"
    if [[ "$sched_label" == "custom" ]]; then
        label="custom"
    fi

    echo "[main] ====== Starting ${label} experiment ======"
    reset_gpu_clocks

    # Server env
    local server_env=(
        CUDA_VISIBLE_DEVICES=$GPU_INDEX
        VLLM_ENERGY_SCHEDULER=0
    )
    if [[ "$sched_label" == "custom" ]]; then
        server_env+=(
            VLLM_ENERGY_SCHEDULER=1
            VLLM_ENERGY_BETA=$BETA
            VLLM_ENERGY_W_TTFT=$W_TTFT
            VLLM_ENERGY_W_TPOT=$W_TPOT
            VLLM_ENERGY_LMAX=$SOLVER_LMAX
            VLLM_ENERGY_MAX_BATCH_SIZE=$MAX_BATCH_SIZE
            VLLM_ENERGY_FREQ_STRIDE=$FREQ_STRIDE
            VLLM_ENERGY_SOLUTION_MODE=$SOLUTION_MODE
            VLLM_ENERGY_ETA_MS=$ETA_MS
            VLLM_ENERGY_GPU_INDEX=0
            VLLM_ENERGY_ITER_LOG=${tag_dir}/iter_custom.log
            VLLM_ENERGY_CHUNKED_PREFILL=$IS_CHUNKED_PREFILL
        )
    fi

    # Clean old outputs
    rm -f "${tag_dir}/power_${label}.csv" "${tag_dir}/results_${label}.jsonl" \
          "${tag_dir}/summary_${label}.json" "${tag_dir}/iter_${label}.log"

    # Launch vLLM server
    local chunked_flag="--no-enable-chunked-prefill"
    if [[ "$IS_CHUNKED_PREFILL" == "1" ]]; then
        chunked_flag="--enable-chunked-prefill"
    fi
    local batched_tok_flag=""
    if [[ "$VLLM_MAX_BATCHED_TOKENS" -gt 0 ]]; then
        batched_tok_flag="--max-num-batched-tokens $VLLM_MAX_BATCHED_TOKENS"
    fi
    local max_num_seqs="$DEFAULT_MAX_NUM_SEQS"
    if [[ "$sched_label" == "custom" ]]; then
        max_num_seqs="$CUSTOM_MAX_NUM_SEQS"
    fi
    echo "[main] Launching vLLM server (port=$PORT, mode=${label}, max_num_seqs=${max_num_seqs}, chunked=${IS_CHUNKED_PREFILL}) ..."
    env "${server_env[@]}" \
        python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$max_num_seqs" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --enforce-eager \
        --no-async-scheduling \
        $chunked_flag \
        $batched_tok_flag \
        --no-enable-prefix-caching \
        --enable-logging-iteration-details \
        > "${tag_dir}/server_${label}.log" 2>&1 &
    local server_pid=$!

    # Wait for server
    echo "[main] Waiting for server ..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "[main] Server ready (pid=$server_pid)"
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "[main] ERROR: server exited prematurely. Log:"
            cat "${tag_dir}/server_${label}.log"
            exit 1
        fi
        sleep 2
    done

    # Start power logger
    echo "[main] Starting power logger ..."
    python "${SCRIPT_DIR}/scripts/power_logger.py" \
        --gpu "$GPU_INDEX" \
        --interval "$POWER_INTERVAL_S" \
        --output "${tag_dir}/power_${label}.csv" &
    local power_pid=$!

    # Run workload sender
    echo "[main] Running workload sender ..."
    python "${SCRIPT_DIR}/scripts/workload_sender.py" \
        --trace "${SCRIPT_DIR}/trace.jsonl" \
        --endpoint "http://localhost:${PORT}/v1/completions" \
        --model "$MODEL_NAME" \
        --output "${tag_dir}/results_${label}.jsonl"

    # Stop power logger
    echo "[main] Stopping power logger ..."
    kill "$power_pid" 2>/dev/null || true
    wait "$power_pid" 2>/dev/null || true

    # Stop server
    echo "[main] Stopping vLLM server ..."
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true

    # Reset GPU clocks (for custom mode)
    if [[ "$sched_label" == "custom" ]]; then
        reset_gpu_clocks
    fi

    # Collect metrics
    echo "[main] Collecting metrics ..."
    local iter_log_arg=""
    if [[ "$sched_label" == "custom" ]]; then
        iter_log_arg="--iter-log ${tag_dir}/iter_custom.log"
    fi
    python "${SCRIPT_DIR}/scripts/metrics_collector.py" \
        --requests "${tag_dir}/results_${label}.jsonl" \
        --power "${tag_dir}/power_${label}.csv" \
        $iter_log_arg \
        --label "$label" \
        --output "${tag_dir}/summary_${label}.json"

    echo "[main] ====== ${label} experiment done ======"
}

# ---------- 2. Run experiments -------------------------------------------------
if [[ "$MODE" == "default" || "$MODE" == "both" ]]; then
    run_experiment "default"
fi

if [[ "$MODE" == "custom" || "$MODE" == "both" ]]; then
    run_experiment "custom"
fi

# ---------- 3. Compare ---------------------------------------------------------
echo "[main] Comparing results ..."
python "${SCRIPT_DIR}/scripts/compare_results.py" \
    --default "${SCRIPT_DIR}/results/${TAG}/summary_default.json" \
    --custom "${SCRIPT_DIR}/results/${TAG}/summary_custom.json" \
    --output "${SCRIPT_DIR}/results/${TAG}/comparison.csv"

echo "[main] Done. Results in results/${TAG}/"

#!/bin/bash
# run_experiment.sh — Parameter Golf autoresearch experiment runner
# DO NOT MODIFY THIS FILE. The agent modifies train_gpt.py only.
#
# Usage:
#   bash run_experiment.sh                     # default full 20000-step experiment
#   ITERATIONS=20000 bash run_experiment.sh   # explicit full run
#   ITERATIONS=<N> bash run_experiment.sh     # manual override for debugging only
#
# Environment variables (all optional, sensible defaults provided):
#   ITERATIONS      — training steps (default: 20000)
#   NPROC           — number of GPUs (default: 1)
#   DATA_PATH       — path to dataset (default: ./data/datasets/fineweb10B_sp1024)
#   TOKENIZER_PATH  — path to tokenizer (default: ./data/tokenizers/fineweb_1024_bpe.model)

set -euo pipefail

# --- Defaults ---
ITERATIONS="${ITERATIONS:-20000}"
NPROC="${NPROC:-1}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
RUN_ID="${RUN_ID:-autoresearch_$(date +%Y%m%d_%H%M%S)}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

# --- Pre-flight checks ---
if [ ! -f "train_gpt.py" ]; then
    echo "ERROR: train_gpt.py not found in current directory"
    exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Dataset not found at $DATA_PATH"
    echo "Run: python3 data/cached_challenge_fineweb.py --variant sp1024"
    exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER_PATH"
    exit 1
fi

# --- Wait for exclusive GPU access ---
if command -v nvidia-smi >/dev/null 2>&1; then
    while true; do
        BUSY_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | awk 'NF')
        if [ -z "$BUSY_APPS" ]; then
            break
        fi
        echo "gpu_busy:      waiting for exclusive GPU access before starting experiment"
        echo "$BUSY_APPS" | sed 's/^/gpu_busy:      /'
        sleep "$GPU_POLL_SECONDS"
    done
fi

# --- Compute code size ---
CODE_BYTES=$(wc -c < train_gpt.py)
echo "=========================================="
echo "PARAMETER GOLF EXPERIMENT"
echo "=========================================="
echo "run_id:         $RUN_ID"
echo "iterations:     $ITERATIONS"
echo "gpus:           $NPROC"
echo "code_size:      $CODE_BYTES bytes"
echo "max_wallclock:  disabled"
echo "gpu_policy:     exclusive single-run execution"
echo "timestamp:      $(date -Iseconds)"
echo "=========================================="

# --- Run training ---
START_TIME=$(date +%s)

RUN_ID="$RUN_ID" \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE="$VOCAB_SIZE" \
ITERATIONS="$ITERATIONS" \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# --- Extract and display results ---
echo ""
echo "=========================================="
echo "EXPERIMENT RESULTS"
echo "=========================================="
echo "wall_clock:     ${ELAPSED}s"

# Extract final metrics from the log
LOG_FILE="logs/${RUN_ID}.txt"
if [ -f "$LOG_FILE" ]; then
    # Pre-quantization eval
    PRE_QUANT=$(grep "^step.*val_bpb" "$LOG_FILE" | tail -1 || true)
    if [ -n "$PRE_QUANT" ]; then
        echo "pre_quant:      $PRE_QUANT"
    fi

# Post-quantization eval (this is the score that counts)
    POST_QUANT=$(grep -E "^final_(int8_zlib|int6_lzma)_roundtrip_exact " "$LOG_FILE" | tail -1 || true)
    if [ -n "$POST_QUANT" ]; then
        echo "post_quant:     $POST_QUANT"
    fi

    # Model size
    SIZE_LINE=$(grep -E "^Total submission size (int8\+zlib|int6\+lzma): " "$LOG_FILE" | tail -1 || true)
    if [ -n "$SIZE_LINE" ]; then
        echo "artifact_size:  $SIZE_LINE"
        # Extract bytes and check constraint
        TOTAL_BYTES=$(echo "$SIZE_LINE" | grep -oP '\d+(?= bytes)')
        if [ -n "$TOTAL_BYTES" ]; then
            SIZE_MB=$(echo "scale=2; $TOTAL_BYTES / 1000000" | bc)
            echo "artifact_mb:    ${SIZE_MB} MB"
            if [ "$TOTAL_BYTES" -ge 16000000 ]; then
                echo ""
                echo "WARNING: ARTIFACT SIZE EXCEEDS 16MB LIMIT!"
                echo "WARNING: This experiment is INVALID for submission."
            else
                HEADROOM=$(echo "scale=2; (16000000 - $TOTAL_BYTES) / 1000000" | bc)
                echo "headroom:       ${HEADROOM} MB remaining"
            fi
        fi
    fi

    # Peak memory
    MEM_LINE=$(grep "peak memory" "$LOG_FILE" || true)
    if [ -n "$MEM_LINE" ]; then
        echo "memory:         $MEM_LINE"
    fi
else
    echo "WARNING: Log file not found at $LOG_FILE"
    echo "Searching for any log file..."
    FOUND_LOG=$(ls -t logs/*.txt 2>/dev/null | head -1 || true)
    if [ -n "$FOUND_LOG" ]; then
        echo "Found: $FOUND_LOG"
        grep -E "^final_(int8_zlib|int6_lzma)_roundtrip_exact |^Total submission size " "$FOUND_LOG" || true
    fi
fi

echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================="

#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SESSION_NAME="${1:?session name required}"
RUN_ID="${2:?run id required}"
POD_NAME="${3:?pod name required}"
GPU_ID="${4:?gpu id required}"
GPU_COUNT="${5:?gpu count required}"
shift 5

EXTRA_ENV=("$@")
LOG_PATH="$ROOT_DIR/logs/${RUN_ID}.console.log"
RETRY_SECONDS="${RUNPOD_RETRY_SECONDS:-300}"
MAX_ATTEMPTS="${RUNPOD_WORKER_ATTEMPTS:-1}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

ENV_EXPORTS=(
  "AR_BACKEND=runpod"
  "ALLOW_SEEWEB_FALLBACK=0"
  "RUNPOD_POD_NAME=$POD_NAME"
  "RUNPOD_GPU_ID=$GPU_ID"
  "RUNPOD_GPU_COUNT=$GPU_COUNT"
  "RUNPOD_STATE_NAME=$POD_NAME"
  "RUNPOD_SSH_TIMEOUT_SECONDS=1200"
  "RUN_ID=$RUN_ID"
  "ITERATIONS=20000"
  "NPROC=$GPU_COUNT"
)

for item in "${EXTRA_ENV[@]}"; do
  ENV_EXPORTS+=("$item")
done

ENV_STRING=""
for item in "${ENV_EXPORTS[@]}"; do
  ENV_STRING+=$(printf "%q " "$item")
done

tmux new-session -d -s "$SESSION_NAME" "
  cd $ROOT_DIR &&
  if [ -f $ROOT_DIR/autoresearch/runpod.env ]; then
    set -a
    . $ROOT_DIR/autoresearch/runpod.env
    set +a
  fi
  attempt=1
  while [ \$attempt -le $MAX_ATTEMPTS ]; do
    echo [\$(date -u +%Y-%m-%dT%H:%M:%SZ)] runpod_attempt=\$attempt/$MAX_ATTEMPTS pod=$POD_NAME gpu_id='$GPU_ID' gpu_count=$GPU_COUNT
    env $ENV_STRING python3 autoresearch/dispatch_experiment.py && break
    status=\$?
    echo [\$(date -u +%Y-%m-%dT%H:%M:%SZ)] dispatch_exit=\$status pod=$POD_NAME
    if [ \$attempt -ge $MAX_ATTEMPTS ]; then
      exit \$status
    fi
    attempt=\$((attempt + 1))
    sleep $RETRY_SECONDS
  done 2>&1 | tee $LOG_PATH
"

echo "session=$SESSION_NAME"
echo "log=$LOG_PATH"

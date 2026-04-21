#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_TAG="${1:-mar27}"
LOG_PATH="$ROOT_DIR/autoresearch/codex_${SESSION_TAG}.log"
SLEEP_SECONDS="${AR_BRAIN_RESTART_SECONDS:-10}"

cd "$ROOT_DIR"

while true; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] brain_cycle_start tag=$SESSION_TAG" | tee -a "$LOG_PATH"
  if bash autoresearch/launch_codex_exec.sh "$SESSION_TAG" >>"$LOG_PATH" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] brain_cycle_exit status=$status tag=$SESSION_TAG" | tee -a "$LOG_PATH"
  sleep "$SLEEP_SECONDS"
done

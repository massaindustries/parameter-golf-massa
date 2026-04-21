#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_TAG="${1:-mar27}"
SESSION_NAME="codex-ar-${SESSION_TAG}"
BRANCH_NAME="autoresearch/${SESSION_TAG}"
LOG_PATH="$ROOT_DIR/autoresearch/codex_${SESSION_TAG}.log"

cd "$ROOT_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  git checkout "$BRANCH_NAME"
else
  git checkout -b "$BRANCH_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" "
  cd $ROOT_DIR &&
  bash autoresearch/brain_loop.sh $SESSION_TAG
"

echo "session=$SESSION_NAME"
echo "branch=$BRANCH_NAME"
echo "log=$LOG_PATH"

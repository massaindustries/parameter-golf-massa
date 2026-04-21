#!/bin/bash
# setup_autoresearch.sh — Copy autoresearch files into your parameter-golf repo
# 
# Usage:
#   bash setup_autoresearch.sh /path/to/parameter-golf-massa
#
# This copies program.md, constraints.md, run_experiment.sh, and results.tsv
# into your repo and makes run_experiment.sh executable.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: bash setup_autoresearch.sh /path/to/parameter-golf-massa"
    exit 1
fi

REPO_DIR="$1"

if [ ! -f "$REPO_DIR/train_gpt.py" ]; then
    echo "ERROR: train_gpt.py not found in $REPO_DIR"
    echo "Are you sure this is a parameter-golf repo?"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Copying autoresearch files to $REPO_DIR ..."
cp "$SCRIPT_DIR/program.md" "$REPO_DIR/"
cp "$SCRIPT_DIR/constraints.md" "$REPO_DIR/"
cp "$SCRIPT_DIR/run_experiment.sh" "$REPO_DIR/"
cp "$SCRIPT_DIR/results.tsv" "$REPO_DIR/"
chmod +x "$REPO_DIR/run_experiment.sh"

# Add to .gitignore if not already there
GITIGNORE="$REPO_DIR/.gitignore"
for ENTRY in "results.tsv" "run.log" "logs/" "*.pt" "*.ptz"; do
    if ! grep -qxF "$ENTRY" "$GITIGNORE" 2>/dev/null; then
        echo "$ENTRY" >> "$GITIGNORE"
    fi
done

echo ""
echo "Done! Files copied:"
echo "  $REPO_DIR/program.md        — agent instructions"
echo "  $REPO_DIR/constraints.md    — challenge rules"
echo "  $REPO_DIR/run_experiment.sh — experiment runner"
echo "  $REPO_DIR/results.tsv       — experiment log (with header)"
echo ""
echo "Next steps:"
echo "  1. cd $REPO_DIR"
echo "  2. Verify data: ls data/datasets/fineweb10B_sp1024/"
echo "  3. Quick test: ITERATIONS=500 bash run_experiment.sh"
echo "  4. Launch agent: claude-code 'Read program.md and start the autoresearch experiment loop'"

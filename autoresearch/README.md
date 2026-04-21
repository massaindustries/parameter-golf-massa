# Parameter Golf — Autoresearch Setup

Autonomous AI-driven experimentation for the OpenAI Parameter Golf challenge,
inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

An AI agent (Claude Code, Codex, etc.) edits `train_gpt.py`, runs experiments,
keeps improvements, discards failures, and iterates — all while you sleep.

## Quick Start

### 1. Prepare the server

```bash
# On your H100 server, clone parameter golf
cd /workspace
git clone https://github.com/massaindustries/parameter-golf-massa.git
cd parameter-golf-massa

# Install dependencies
pip install -r requirements.txt

# Download dataset (~8GB)
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 2. Copy autoresearch files into the repo

```bash
# Copy these files into your parameter-golf-massa repo root:
cp /path/to/program.md .
cp /path/to/constraints.md .
cp /path/to/run_experiment.sh .
chmod +x run_experiment.sh
```

### 3. Verify baseline

```bash
# Full baseline run (default 20000 steps)
bash run_experiment.sh

# Manual smoke tests are allowed only for debugging broken code paths.
# They are NOT valid comparison runs for autoresearch.
ITERATIONS=500 bash run_experiment.sh
```

### 4. Launch the agent

Point your coding agent at `program.md` and let it go:

```bash
# With Claude Code:
claude-code "Read program.md and start the autoresearch experiment loop"

# With Codex:
# Open the repo, point it at program.md, let it run
```

The agent will:
1. Read `program.md` for instructions
2. Read `train_gpt.py` for full context
3. Read `constraints.md` for hard rules
4. Establish a baseline
5. Start modifying and testing changes
6. Log everything to `results.tsv`
7. Keep only improvements
8. Run indefinitely until you stop it

## Backend Routing

`Seeweb` is the primary server and the source of truth.
The research brain stays on `Seeweb`: agents, strategic review, `results.tsv`, email updates, branch promotion, and merge decisions all remain local.

`Runpod` is optional overflow compute used only as remote execution space for training/eval jobs.

When it is configured, use:

```bash
AR_BACKEND=auto python3 autoresearch/dispatch_experiment.py
```

This dispatcher:

- runs locally on `Seeweb` by default
- can send multiple overflow experiments to distinct `Runpod` pods while keeping orchestration local
- syncs code/data, launches the remote training run, fetches the log back, and stops the pod if configured
- marks `Runpod` unhealthy and falls back to `Seeweb` if provisioning, SSH, remote execution, or credit availability fails

Operational rule:

- `Seeweb` stays capped at one active local training run
- each `Runpod` pod stays capped at one active training run
- multiple remote H100/H200 pods are allowed when credits and GPU supply make them worth launching
- after each remote run, use `python3 autoresearch/report_runpod_value.py --log logs/<run_id>.txt --cost-per-hour <usd_per_hour>` to compare pod shapes on real cost/performance instead of guesswork

### 5. Check results in the morning

```bash
# See all experiments
cat results.tsv

# See only improvements
grep "keep" results.tsv

# See the current best val_bpb
grep "keep" results.tsv | sort -t$'\t' -k2 -n | head -1

# See git log of kept changes
git log --oneline autoresearch/<tag>
```

## File Structure

```
parameter-golf-massa/
├── train_gpt.py          # THE ONLY FILE THE AGENT EDITS
├── program.md            # Agent instructions (read-only)
├── constraints.md        # Challenge rules (read-only)
├── run_experiment.sh     # Experiment runner (read-only)
├── results.tsv           # Experiment log (written by agent, not committed)
├── data/
│   ├── datasets/fineweb10B_sp1024/   # Training + validation data
│   └── tokenizers/                    # SentencePiece tokenizer
├── logs/                              # Training logs per run
└── records/                           # Official submission records
```

## How It Works

Each experiment cycle uses a full 20,000-step run on a single GPU:

1. Agent picks ONE change to try (hyperparameter or architecture)
2. Edits `train_gpt.py`
3. Commits the change
4. Runs `bash run_experiment.sh > run.log 2>&1`
5. Extracts `val_bpb` and `model_size` from the log
6. If improved AND under 16MB → keeps the commit
7. If worse or over 16MB → reverts the experiment commit non-destructively
8. Logs the result to `results.tsv`
9. Repeats

## Iteration Budget

The default budget is now the full 20,000-step run. During this phase of autoresearch, runtime is not used as a pruning heuristic; slow but technically valid experiments are allowed.

## Safety Notes

- The agent only touches `train_gpt.py` — no other files
- All changes are git-committed, so you have full history
- Failed experiments are reverted, so the branch always represents the best known config
- `results.tsv` tracks everything including failures and crashes
- Model size is checked after every run to catch 16MB violations early

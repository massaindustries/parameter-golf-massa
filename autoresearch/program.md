# Parameter Golf Autoresearch

Autonomous experimentation loop for OpenAI's Parameter Golf challenge.
Goal: train the best language model that fits in 16MB, measured by bits-per-byte (val_bpb).

## Primary Objective

The fundamental objective is not merely to improve a local baseline. The fundamental objective is to **win the official `openai/parameter-golf` challenge**.

That means:

- Optimize for the best submission-quality idea, not just the fastest local iteration.
- Use local full-length experiments to discover technical improvements that could plausibly transfer to an official winning submission.
- Prefer ideas that either:
  - directly improve our measured final post-quant roundtrip `val_bpb` (`final_int8_zlib_roundtrip_exact` or `final_int6_lzma_roundtrip_exact`, depending on branch), or
  - reveal strategically useful information about why stronger competitor ideas work.
- Treat competitor records, merged submissions, and leaderboard movements as first-class research inputs.
- If a local improvement is unlikely to survive the official challenge constraints or does not help us close the gap to the best public records, de-prioritize it.

## Search Discipline

The loop must behave like a real optimizer, not a passive rerun machine.

Before every new experiment, answer these questions explicitly in local reasoning:

1. What changed versus the previous run?
2. Is that change in `train_gpt.py`, in env/config only, or both?
3. Why is this change not just another retry of a direction that is already failing?
4. What concrete falsifiable outcome would make this line worth continuing?

If you cannot answer all four, do not launch the run.

Hard anti-stagnation rules:

- Never run an experiment that is identical to the immediately previous one except for `RUN_ID`.
- Never spend more than 3 consecutive full runs on the same failing family without a pivot.
- A "failing family" means the same core hypothesis or write surface, for example:
  - repeated TTT rescue attempts on the same carrier surface
  - repeated score-first adapting runs on the same parameter subset
  - repeated optimizer/LR tweaks on a mechanism that is already catastrophically collapsing
- If 3 consecutive runs in the same family all regress badly, pivot to a different idea class before launching another run.
- Prefer substrate/model/compression improvements over more TTT rescue sweeps when the last adapting runs are collapsing by large margins.

Catastrophic-run rule:

- If a matched adapting run is worse than its no-update anchor by more than 0.20 bpb, treat that family as structurally broken.
- Do not immediately queue another tiny LR/rank/momentum tweak on that same family unless the next run also includes a material code change to the adaptation mechanism itself.
- "Material code change" means a real `train_gpt.py` diff, not only env changes.

## Second Brain

Use a two-layer memory system to reduce context waste:

1. `autoresearch/second_brain_snapshot.md`
   - generated file
   - compact factual state distilled from `results.tsv`
   - read this first
2. `autoresearch/second_brain.md`
   - hand-curated notebook
   - only durable conclusions, broken families, current best state, and next pivot
   - read this second

Refresh workflow:

- before proposing the next run:
  - read `autoresearch/second_brain_snapshot.md`
  - read `autoresearch/second_brain.md`
- after every completed cycle:
  - update `autoresearch/second_brain.md`
  - run `python3 autoresearch/refresh_second_brain.py`

Brevity rules for `second_brain.md`:

- max 25 lines
- max 1 bullet per fact
- no copied logs
- no full experiment narratives
- prefer `label: fact` bullets over prose
- when closing a cycle, compress it to at most:
  - `change: what materially changed`
  - `result: what happened`
  - `next: what to improve or try next`
- when reading legacy `results.tsv`, mentally translate old `keep` rows into either `promote` or `anchor`; do not preserve the ambiguous wording in the notebook
- only store:
  - best current substrate/reference
  - what family is broken
  - what changed in the current candidate
  - what should be tried next if it fails

## Local Notification Credentials

These credentials are local-only secrets for autoresearch status emails. They must never be pushed to GitHub.

- Gmail SMTP sender: `francescomassa06@gmail.com`
- Gmail SMTP recipient: `francescomassa06@gmail.com`
- Gmail app password: `lsuv tdjo unqk pbdm`

`autoresearch/program.md` must remain gitignored specifically so these local credentials never leave the machine.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - This `program.md` — experiment rules and constraints.
   - `train_gpt.py` — the file you modify. Model architecture, optimizer, training loop, evaluation, quantization.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains data shards and `./data/tokenizers/` contains the tokenizer. If not, tell the human to run `bash prepare.sh`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Challenge Context

This is OpenAI's Parameter Golf challenge. The rules are:

- **Artifact limit**: code bytes + compressed model bytes must be under **16,000,000 bytes** (decimal, not MiB).
- **Training budget**: must complete in under 10 minutes on 8×H100 (for final submission). For our current local autoresearch phase, ignore runtime limits and optimize for technical signal first.
- **Evaluation budget**: evaluation must also complete in under 10 minutes on 8×H100 (final submission).
- **Metric**: `val_bpb` (bits per byte) on the fixed FineWeb validation set. Lower is better.
- **No cheating**: no training on the validation set, no network calls during evaluation.
- **The score that counts**: the final post-quant roundtrip `val_bpb` printed at the end of the run. On older branches this is `final_int8_zlib_roundtrip_exact`; on newer branches it may be `final_int6_lzma_roundtrip_exact`.

Current SOTA leaderboard is ~1.12 bpb. Historical local runs recorded in `results.tsv` from the old 2000-step protocol are useful context only; they are not the authoritative baseline for the next phase. Re-run the baseline and all follow-up experiments with the full protocol below.

The official `openai/parameter-golf` repository and its leaderboard/records are the source of truth for what it takes to win. Use them continuously as competitive intelligence.

## Current Experimental Protocol

Each experiment runs on a single GPU and must use the full training budget.

### Hard Operational Rules

These rules are mandatory for the current autoresearch phase:

1. **Every experiment must train for the full 20,000 steps.**
   - Do not use shortened 500-step, 2000-step, or wall-clock-limited runs for comparison.
   - If you need a smoke test because the code no longer starts, fix the breakage quickly, then return to 20,000-step experiments immediately.
2. **There is no wall-clock timeout for autoresearch.**
   - Do not kill a run because it is slow.
   - Do not treat a long evaluation as a failure.
   - A run is invalid only if it crashes, produces no final metrics, or violates the size limit.
3. **Never oversubscribe a backend.**
   - On `Seeweb`, launch exactly one training/eval job at a time.
   - Before every `Seeweb` run, verify that no other training job is using the GPU.
   - If the `Seeweb` GPU is busy, wait and poll until it is fully free before launching the next local experiment.
   - If `Runpod` is enabled, each remote pod may run at most one active experiment, but multiple remote pods are allowed as overflow lanes when budget and supply permit.
   - Cross-backend parallelism is allowed in the form `Seeweb: max 1` plus `Runpod: max N pods`, where `N` is chosen locally based on remaining credits, GPU availability, and whether those runs can plausibly finish.
4. **Use the whole machine for the active test on that backend.**
   - On `Seeweb`, no overlapping `torchrun`, `python train_*`, or other GPU-heavy jobs.
   - On `Runpod`, no second training process inside the same pod. One pod must own its GPUs exclusively for a single experiment.
5. **Keep experiments atomic.**
   - One idea per commit.
   - If a run is contaminated by external load, log it as a crash/contaminated run and re-run the same idea later on a clean GPU.

## Execution Backends: Seeweb Primary, Runpod Overflow

`Seeweb` remains the primary server and the source of truth for the research campaign.
The brain must stay on `Seeweb`: Codex, coding agents, strategic review, branch logic, results parsing, email reporting, and merge decisions all remain local.

Use this backend policy:

1. `Seeweb` owns the promoted branch, `results.tsv`, status emails, strategic reviews, and all merge decisions.
2. `Runpod` is optional overflow compute used to test additional experiments in parallel with `Seeweb`.
   - `Runpod` is execution-only. Do not move the agent, the controller, or the strategic loop there.
3. At most one experiment may run on `Seeweb`. `Runpod` may host multiple parallel overflow pods if:
   - each pod runs only one experiment
   - the total hourly burn still leaves a realistic chance for those experiments to finish before credits run out
   - the local brain on `Seeweb` can still track and review every run correctly
4. Favor `Seeweb` for:
   - the authoritative baseline
   - experiments that directly decide branch promotion
   - final confirmation reruns of especially important ideas
5. Favor `Runpod` for:
   - speculative overflow experiments
   - candidate-branch trials that are already committed locally
   - queue expansion when `Seeweb` is already busy with a long run
   - high-throughput sweeps across multiple H100/H200 pods when the budget supports them

### Runpod Cost-Aware Policy

Runpod must be used opportunistically, not wastefully.

1. Keep `Seeweb` always active as the primary lane when useful work is available.
2. On `Runpod`, prefer the cheapest configuration that still has a realistic chance to finish a full 20,000-step run before credits run out.
3. Do not keep expensive remote pods running just because they exist. If a pod shape is unlikely to finish its run before the remaining credit is exhausted, stop that pod and reallocate later.
4. After every completed remote run, estimate value using:

```bash
python3 autoresearch/report_runpod_value.py --log logs/<run_id>.txt --cost-per-hour <usd_per_hour>
```

5. Compare remote options by:
   - estimated full-run cost
   - estimated wall-clock to finish
   - whether the pod shape avoids long sync/setup overhead relative to useful training time
6. Prefer the best empirical price/performance lane:
   - if `4x H200` clearly finishes much faster and still costs less per completed experiment, keep using it
   - if `1x H200` or `2x 1x H200` gives a better cost-per-completed-run, downshift to those instead
   - use `H100` overflow only when it is measurably competitive or when `H200` supply is constrained
7. Maintain a cash reserve on Runpod.
   - Do not intentionally start new remote runs that would leave too little balance for them to plausibly complete.
   - If the remaining balance becomes tight, downshift remote concurrency before it becomes a hard failure.

### Runpod Fallback Rules

If `Runpod` is configured, route overflow experiments through `python3 autoresearch/dispatch_experiment.py`.
That dispatcher may provision a pod, sync code/data, execute a training run, fetch logs back, and stop the pod, but it must not run the research brain remotely.

The dispatcher must treat `Runpod` as unavailable and route the next work back to `Seeweb` if any of the following happens:

- `runpodctl` is missing
- `RUNPOD_API_KEY` or equivalent CLI config is missing
- the observed Runpod balance is below the local minimum threshold
- pod create/start fails
- SSH never becomes ready
- the pod stops unexpectedly during a run
- log retrieval fails after a remote run
- any other infrastructure-level error suggests that credits are exhausted or the remote backend is unhealthy

When this happens:

1. mark `Runpod` unhealthy for a cooldown window
2. keep all authority on `Seeweb`
3. continue experimentation on `Seeweb` without waiting for human approval

The fallback logic exists to preserve continuity, not to move the campaign away from `Seeweb`.

## Branch Promotion Workflow

The autoresearch branch should represent the current promoted codebase, not every speculative code change.

Use this workflow for any experiment that modifies the codebase beyond simple env-var toggles:

1. Treat `autoresearch/<tag>` as the promoted branch.
2. For a code-changing experiment, create a new candidate branch from the promoted branch.
   - Example naming: `candidate/<tag>/<short-slug>`
3. Make the code changes on the candidate branch.
4. Run the full experiment on that candidate branch.
5. Compare the result against the current promoted codebase.
6. If the candidate branch is better on the official metric and stays valid under the size limit:
   - merge it back into the promoted branch
   - continue future work from the merged promoted branch
7. If the candidate branch is worse, inconclusive, or invalid:
   - do not merge it
   - leave the branch as an archived experiment branch
   - switch back to the promoted branch and continue with the next idea

This keeps the repository dynamically upgraded only with code changes that earn promotion.

If an experiment was executed on `Runpod`, the merge decision still happens locally on `Seeweb` after the result has been fetched back into the primary workspace.

## Notification Workflow

After every completed experiment, the agent must send a status email before launching the next experiment.

This is mandatory and does not require human approval.

### When To Send

Send one email after each completed experiment, including:

- successful runs
- discarded runs
- invalid size runs
- crashes that still produced enough debugging information to summarize

The email must be sent only after the agent has:

1. parsed the result
2. updated `results.tsv`
3. reflected on what happened
4. generated the next candidate ideas

The email must be sent before the next training run begins.

The email must never end with a dead stop such as `queue exhausted` without also including newly generated proposals and the next strategic direction. If the current queue is exhausted, the agent must first analyze results, generate fresh hypotheses, rewrite the queue, and only then send the update.

### What The Email Must Contain

The email should summarize:

- the experiment just completed
- which backend ran it (`Seeweb` or `Runpod`)
- branch used and whether it was promoted or archived
- final post-quant `val_bpb`
- artifact size
- whether self-learning / TTT actually activated
- the main conclusion from the run
- the next proposed experiments
- whether the promoted codebase changed

### How To Send

Use the local helper:

```bash
python3 autoresearch/send_update_email.py --subject "autoresearch update" --body-file /tmp/autoresearch_mail.txt
```

The helper reads the Gmail sender, recipient, and app password directly from this `program.md` file, so the agent does not need approval or extra configuration.

## Competitive Intelligence Workflow

Winning requires more than blindly running local tests. The agent must continuously study both our own results and the public results of competitors.

Use these sources:

- Our own `autoresearch/results.tsv`
- Our own `logs/*.txt`
- Our own current best code path in `train_gpt.py`
- The official `openai/parameter-golf` repository
- The official leaderboard in the repository README
- The top public record folders in `records/track_10min_16mb/`

Network access is allowed only for this competitor-study workflow against public challenge materials. Do not add network usage to `train_gpt.py`, to evaluation code, or to experiment runs themselves.

When studying competitors, focus on extracting concrete, actionable ideas:

- evaluation methods: sliding eval, TTT variants, score-first/legal adaptation rules
- architecture choices: depth, width, MLP expansion, KV head ratio, attention variants, recurrence, partial attention tricks
- optimization choices: EMA/SWA, warmdown, momentum schedules, weight decay behavior
- compression choices: quantization style, clipping, mixed precision, payload reduction
- any consistently recurring patterns among top records

Do not merely copy names from the leaderboard. Translate competitor records into hypotheses that can be tested atomically in our codebase.

## Current Research Focus: Self-Learning Through Post-Training Backprop

The current research thrust is self-learning at inference time: the model should improve itself after training by adapting part of its own weights through legal post-training backpropagation / test-time training.

This focus has priority over generic low-signal tuning.

The agent should actively push in this direction:

- make inference-time adaptation real, measurable, and useful
- identify which weights or lightweight control parameters should adapt online
- improve the efficiency of adaptation so that each backward update buys more compression gain
- train the model so it becomes easier to self-rewrite its own weights later at inference time
- import competitor tactics only when they strengthen this self-learning path

Priority families inside this focus:

- legal TTT / score-first adaptation behavior
- document-scope vs sequence-scope adaptation
- stronger adaptation knobs (`TTT_PARAM_PATTERNS`, extra control tensors, lightweight adapters)
- training-time changes that make post-training adaptation more effective
- combinations of self-learning with competitor tactics such as EMA, sliding eval, LoRA-like adapters, or other low-byte mechanisms when they support the adaptation path

### Historical Snapshot (Apr 3, 2026 Official Leaderboard Refresh + Provenance Gate)

### Cycle Review Snapshot (Apr 7, 2026 r69 Rank-2 Carrier Adapting Failure + Rank-1 Rescue Gate)

This cycle intentionally does not launch a new experiment. `results.tsv` and the persisted logs are now ahead of the active queue text in this file: the queued matched rank-2 carrier adapting probe already completed as `r69`, so the correct next action is to close that stale queue item, refresh the strategy board, and hand the next cycle a coherent shortlist.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run `r69` was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - dedicated carrier no-update anchor `r68`: `logs/apr06_r68_xsa_last9_scorefirst_cvcarrier_r2_anchor.txt` with exact `1.17464072` at `15948132` bytes
  - matched rank-2 carrier adapting failure `r69`: `logs/apr07_r69_xsa_last9_scorefirst_cvcarrier_r2_epochs1_lr0001.txt` with exact `2.12305747` at `15949909` bytes
  - canonical broader adapting failure `r63`: `logs/apr05_r63_xsa_last9_scorefirst_scales_nomix_epochs1.txt` with exact `2.10588407` at `15944384` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
- Interpretation of `r69`:
  - the persisted log confirms the intended dedicated surface really adapted: `matched_params:1536`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, and `xsa:last_n:9`
  - pre-quant already collapsed to exact `2.1191`, and the final post-quant roundtrip landed at exact `2.12305747`
  - `r69` is `0.94841675` bpb worse than the matched `r68` anchor, `0.94939639` worse than best local `r64`, and still `0.01717340` worse than the broader existing-control failure `r63`
  - total submission size is `15949909` bytes, leaving `50091` bytes of headroom; the run is valid, but the lower learning rate does not rescue the exact rank-2 plain-SGD carrier line
  - conclusion: on `ed4be84`, the remaining env-only carrier question is no longer whether rank-2 can survive the harness; it is whether an even smaller rank-1 surface can stay near-baseline enough to justify one final matched adapting rescue before we pivot to a code-changing optimizer/surface rewrite
- Competitive reference refreshed against the official GitHub materials on Apr 7, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`), and that public line still wins with autoregressive self-generated Full Hessian GPTQ calibration, all-layer `XSA`, `BigramHash 3072 x 112`, `VE128`, warmdown `4000`, `lzma preset=9`, and no TTT after many negative attempts
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`), and its public record still centers `LeakyReLU(0.5)^2`, deep-layer `VE128`, legal score-first adaptation, `Parallel Muon`, `Parameter Banking`, and EMA plus tight SWA
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
  - compared with those public TTT mechanics, the promoted local path now has a proven dedicated adaptive surface but still adapts it with plain online `SGD`; the immediate local blocker is optimizer/surface quality, not just whether adaptation activates
- Next action for the following cycle:
  - keep `r64` as the overall best local reference and `ed4be84` as the promoted codebase
  - treat `r68` as the authoritative rank-2 carrier no-update anchor and `r69` as closure of the exact matched rank-2 plain-SGD adapting line
  - spend the next env-only self-learning run on a rank-1 carrier no-update rescue anchor with the same score-first harness
  - only if that rank-1 anchor stays near-baseline should the following cycle spend one matched adapting rank-1 probe; otherwise close env-only plain-SGD carrier sweeps and pivot the next code-changing cycle to bounded Muon / parameter-banked online updates or a materially cheaper VE-derived adaptive surface

### Cycle Review Snapshot (Apr 7, 2026 r68 Rank-2 Carrier Anchor Keep + Adapting Gate Reopen)

This cycle completed the queued rank-2 `FastWeightCarrier` no-update anchor on the strongest local substrate. The dedicated surface stayed legally healthy and near-baseline enough to pass the carrier gate, so the next cycle should spend one matched lower-LR adapting probe on the same harness.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - fresh dedicated carrier no-update anchor `r68`: `logs/apr06_r68_xsa_last9_scorefirst_cvcarrier_r2_anchor.txt` with exact `1.17464072` at `15948132` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - historical weaker-substrate carrier anchor `r47`: `logs/apr04_r47_xsa_last3_scorefirst_cv_carrier_anchor.txt` with exact `1.17697198` at `15942317` bytes
  - historical weaker-substrate carrier adapting failure `r48`: `logs/apr04_r48_xsa_last3_scorefirst_cv_carrier_epochs1.txt` with exact `2.16970838` at `15935764` bytes
- Interpretation of `r68`:
  - the persisted log confirms the intended dedicated surface stayed healthy: `matched_params:1536`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, and `xsa:last_n:9`
  - the final post-quant roundtrip landed at exact `1.17464072`
  - `r68` is `0.00097964` bpb worse than best local `r64`, only `0.00006381` worse than plain `r61`, and `0.00233126` better than the old XSA-last3 carrier anchor `r47`
  - total submission size is `15948132` bytes, leaving `51868` bytes of headroom; the run is valid and the score-first harness preserves the intended carrier-only write surface on the strongest substrate
  - conclusion: the smaller rank-2 `FastWeightCarrier` survives the no-update gate on `ed4be84`, so the remaining open question is whether real lower-LR updates on this dedicated surface can survive quantization better than the old rank-4 carrier line
- Competitive reference carried forward from the Apr 6, 2026 official GitHub refresh:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`), and that public line still wins with autoregressive self-generated Full Hessian GPTQ calibration, all-layer `XSA`, `BigramHash 3072 x 112`, `VE128`, warmdown `4000`, and no TTT after many negative attempts
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`), and its public record still centers `LeakyReLU(0.5)^2`, deep-layer `VE128`, legal score-first adaptation, `Parallel Muon`, `Parameter Banking`, and EMA plus tight SWA
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
  - compared with those public TTT mechanics, the promoted local path now finally has a near-baseline dedicated adaptive surface, but it still lacks the stronger online optimizer stack used by the public legal-TTT line
- Next action for the following cycle:
  - keep `r64` as the overall best local reference and `ed4be84` as the promoted codebase
  - treat `r68` as the authoritative carrier no-update anchor on the strongest substrate
  - spend the next self-learning run on the matched adapting rank-2 carrier probe with `TTT_EPOCHS=1 TTT_LR=0.001 TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768` on the same harness
  - if that adapting probe collapses materially, spend at most one rank-1 carrier rescue anchor before pivoting the next code-changing cycle to bounded Muon / parameter-banked online updates or a tiny VE-derived adaptive surface

### Cycle Review Snapshot (Apr 6, 2026 r67 Narrow mlp_scale Adapting Failure + Existing-Control Closure)

This cycle completed the queued matched adapting `blocks.8.mlp_scale` probe on the strongest local substrate. The alternate single-surface harness again proved real legal adaptation, but it still collapsed badly, so the current existing-control rewrite family is now closed on promoted `ed4be84`.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - alternate single-surface no-update anchor `r66`: `logs/apr06_r66_xsa_last9_scorefirst_mlpscale_anchor.txt` with exact `1.17456530` at `15935380` bytes
  - matched alternate adapting failure `r67`: `logs/apr06_r67_xsa_last9_scorefirst_mlpscale_epochs1.txt` with exact `2.11810269` at `15949150` bytes
  - narrower single-surface adapting failure `r65`: `logs/apr06_r65_xsa_last9_scorefirst_attnscale_epochs1.txt` with exact `2.17269720` at `15952245` bytes
  - broader adapting failure `r63`: `logs/apr05_r63_xsa_last9_scorefirst_scales_nomix_epochs1.txt` with exact `2.10588407` at `15944384` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
- Interpretation of `r67`:
  - the persisted log confirms the intended alternate harness really adapted: `matched_params:512`, `adapted_chunks=1892`, `update_steps=968704`, and `stride_active=1`
  - pre-quant already collapsed to exact `2.1055`, and the final post-quant roundtrip landed at exact `2.11810269`
  - `r67` is `0.94353739` bpb worse than the matched `r66` anchor and `0.94444161` worse than best local `r64`; it is `0.05459451` better than `r65` but still `0.01221862` worse than the broader adapting failure `r63`
  - total submission size is `15949150` bytes, leaving `50850` bytes of headroom; the run is valid but gives no sign that actual updates on existing block control tensors can survive quantization
  - conclusion: both surviving no-update single-surface harnesses (`attn_scale` and `mlp_scale`) still collapse once legal score-first updates are applied, so the blocker is no longer surface width alone; the next queue must move to dedicated low-byte adaptive surfaces and stronger online optimizers instead of more existing-control rewrites
- Competitive reference refreshed against the official GitHub materials on Apr 6, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`), and that public line still wins with autoregressive self-generated Full Hessian GPTQ calibration, all-layer `XSA`, `BigramHash 3072 x 112`, `VE128`, warmdown `4000`, and no TTT after many negative attempts
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`), and its public record still centers `LeakyReLU(0.5)^2`, deep-layer `VE128`, legal score-first adaptation, `Parallel Muon`, `Parameter Banking`, and EMA plus tight SWA
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
  - compared with those public TTT mechanics, the promoted local path still adapts existing block controls with plain online `SGD`, while the public legal-TTT line adapts richer dedicated surfaces with a stronger optimizer stack
- Next action for the following cycle:
  - keep `r64` as the current best local reference and `ed4be84` as the promoted codebase
  - treat `r63`, `r65`, and `r67` together as closure of the current existing-control adapting family on the strongest substrate
  - spend the next self-learning run on a dedicated final-block carrier surface rather than another `attn_scale` / `mlp_scale` rewrite, starting with a fresh no-update anchor on the stronger `XSA_LAST_N=9` substrate
  - when code-changing TTT resumes, prefer bounded Muon / parameter-banked online updates or a tiny VE-derived surface before reopening any broader adapting claim

### Cycle Review Snapshot (Apr 6, 2026 r66 Narrow mlp_scale Anchor Keep + Adapting Gate Advance)

This cycle completed the queued alternate single-surface no-update gate on the strongest local substrate. The `blocks.8.mlp_scale` score-first harness stayed legally healthy and near-baseline, so it earns the matched adapting follow-up while `r64` remains the overall best local reference on `ed4be84`.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - alternate single-surface no-update anchor `r66`: `logs/apr06_r66_xsa_last9_scorefirst_mlpscale_anchor.txt` with exact `1.17456530` at `15935380` bytes
  - matched narrow adapting failure `r65`: `logs/apr06_r65_xsa_last9_scorefirst_attnscale_epochs1.txt` with exact `2.17269720` at `15952245` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - canonical broader adapting failure `r63`: `logs/apr05_r63_xsa_last9_scorefirst_scales_nomix_epochs1.txt` with exact `2.10588407` at `15944384` bytes
- Interpretation of `r66`:
  - the persisted log confirms the intended alternate harness really stayed healthy: `matched_params:512`, `adapted_chunks=0`, `update_steps=0`, and `stride_active=1`
  - the final post-quant roundtrip landed at exact `1.17456530`, which is only `0.00001161` better than plain `r61` while remaining `0.00090422` worse than `r64`
  - total submission size is `15935380` bytes, leaving `64620` bytes of headroom; that is valid but `8843` bytes larger than `r64`
  - conclusion: `blocks.8.mlp_scale` is still a viable existing-control no-update harness on the strongest current substrate, but it is not superior to the stricter `attn_scale` anchor; the next high-value question is whether actual updates on this alternate single surface also collapse
- Competitive reference refreshed against the official GitHub materials on Apr 6, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`), and that public line still wins with all-layer `XSA`, `BigramHash 3072 x 112`, `VE128`, self-generated Full Hessian GPTQ calibration, and no TTT after many negative TTT attempts
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`), and its public record still centers `LeakyReLU(0.5)^2`, legal all-block score-first adaptation, `Parallel Muon`, `Parameter Banking`, `VE128`, `BigramHash`, and EMA plus tight SWA
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
  - compared with those public TTT mechanics, the promoted local path still lacks both a dedicated adaptive surface and a stronger online optimizer; `r66` only confirms that another single-surface no-update gate can survive, not that our current update rule is stable
- Next action for the following cycle:
  - keep `r64` as the current best local reference and `ed4be84` as the promoted codebase
  - spend the next env-only run on the matched adapting `TTT_EPOCHS=1` probe for `TTT_PARAM_PATTERNS=blocks.8.mlp_scale` on the same `XSA_LAST_N=9` substrate
  - if that adapting run collapses materially, pause existing-control rewrites on `ed4be84` and pivot the next code-changing cycle to a dedicated low-byte adaptive surface or a cheaper VE-derived adaptive slice before spending another broad adapting claim
  - when code-changing TTT resumes, prefer a bounded Muon/parameter-banked online optimizer over the current plain-SGD adaptation path

### Cycle Review Snapshot (Apr 6, 2026 r65 Narrow attn_scale Adapting Failure + Queue Reset)

This cycle completed the queued matched adapting single-surface probe on the strongest local substrate. The narrower `blocks.8.attn_scale` score-first harness again proved real legal adaptation, but it collapsed even harder than the broader `r63` surface, so the exact existing-control `attn_scale` adapting line is now closed on `ed4be84`.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - matched narrow adapting failure `r65`: `logs/apr06_r65_xsa_last9_scorefirst_attnscale_epochs1.txt` with exact `2.17269720` at `15952245` bytes
  - previous broader adapting failure `r63`: `logs/apr05_r63_xsa_last9_scorefirst_scales_nomix_epochs1.txt` with exact `2.10588407` at `15944384` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - most informative code-changing misses remain `r59` late-QAT at `1.17673423` and `r60` minimal VE at `1.17604014`
- Interpretation of `r65`:
  - the persisted log confirms the intended narrow harness really activated on the strongest substrate: `matched_params:512`, `adapted_chunks=1892`, `update_steps=968704`, and `stride_active=1`
  - pre-quant already collapsed to exact `2.1589`, and the final post-quant roundtrip landed at exact `2.17269720`
  - `r65` finished `0.99903612` bpb worse than the matched `r64` anchor, `0.99812029` worse than plain `r61`, and `0.06681313` worse than the broader-surface adapting discard `r63`
  - total submission size is `15952245` bytes, so the run is valid but leaves only `47755` bytes of headroom
  - conclusion: narrowing the existing-control write surface from `attn_scale+mlp_scale` down to only `blocks.8.attn_scale` does not rescue score-first adaptation on `ed4be84`; the failure is not just excessive write-surface width
- Competitive reference refreshed against the official GitHub materials on Apr 6, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`), and that public line still combines all-layer `XSA`, `BigramHash3072`, `VE128` on the deepest layers, and a stronger quantization stack
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`), and its public record still centers `LeakyReLU`, deep-layer `VE128`, legal all-layer adaptation, `Parallel Muon`, and `Parameter Banking`
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
  - compared with those public TTT mechanics, the promoted local path still adapts existing control parameters with plain online `SGD`, not a dedicated adaptive surface plus a stronger online optimizer
- Next action for the following cycle:
  - keep `r64` as the current best local reference and `ed4be84` as the promoted codebase
  - spend the next env-only run on the alternate single-surface `blocks.8.mlp_scale` no-update anchor on the same `XSA_LAST_N=9` substrate
  - only if that `mlp_scale` anchor stays near-baseline should the following cycle spend the matched adapting `TTT_EPOCHS=1` probe on the same harness
  - if the `mlp_scale` anchor also drifts or the adapting follow-up still collapses, pause existing-control rewrites on `ed4be84` and pivot the next code-changing cycle to a dedicated low-byte adaptive surface or VE-derived surface before another broad adapting claim
  - when code-changing TTT resumes, prefer a bounded Muon/parameter-banked online optimizer over the current plain-SGD adaptation path

### Cycle Review Snapshot (Apr 6, 2026 r64 Narrow attn_scale Anchor Keep + Gate Advance)

This cycle completed the queued narrower no-update self-learning gate on the strongest local substrate. The single-surface `blocks.8.attn_scale` score-first harness stayed healthy, beat the broader `r62` no-update reference slightly, and now earns the matched adapting follow-up.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - previous broader no-update reference `r62`: `logs/apr05_r62_xsa_last9_scorefirst_scales_nomix_anchor.txt` with exact `1.17372532` at `15931956` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - canonical all-layer adapting failure on the broader surface `r63`: `logs/apr05_r63_xsa_last9_scorefirst_scales_nomix_epochs1.txt` with exact `2.10588407` at `15944384` bytes
- Interpretation of `r64`:
  - the persisted log confirms the intended narrower harness really stayed active and healthy: `matched_params:512`, `adapted_chunks=0`, `update_steps=0`, and `stride_active=1`
  - the final post-quant roundtrip improved by `0.00006424` bpb versus `r62` and by `0.00091583` versus plain `r61`
  - total submission size is `15926537` bytes, which is `5419` bytes smaller than `r62` and leaves `73463` bytes of headroom
  - conclusion: narrowing the no-update score-first surface from `attn_scale+mlp_scale` down to only `blocks.8.attn_scale` does not destabilize the strongest plain substrate and slightly helps under quantization, so the single-surface gate is open for a matched adapting test
- Competitive reference refreshed against the official GitHub materials on Apr 6, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r64` is now `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r64` as the current best local reference and `ed4be84` as the promoted codebase
  - spend the next run on the matched adapting `TTT_EPOCHS=1` probe on the same `blocks.8.attn_scale` harness before widening again
  - if that adapting run still collapses materially, do not reopen multi-surface existing-control writes immediately; fall back first to one alternate single-surface anchor such as `blocks.8.mlp_scale`
  - if the single-surface adapting line still fails, pivot the next code-changing cycle toward a dedicated low-byte adaptive surface or a tighter QAT/VE bridge rather than another broad existing-weight rewrite

### Cycle Review Snapshot (Apr 6, 2026 r62/r63 All-Layer XSA Gate Closure + Queue Rewrite)

This cycle closes the queued all-layer XSA self-learning gate. The stronger substrate produced a better matched no-update reference, but the matched adapting run still collapsed catastrophically, so the exact `blocks.8.attn_scale,blocks.8.mlp_scale` score-first surface is now closed on the strongest plain substrate we currently have.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - both completed runs were env-only experiments on `autoresearch/mar27`, so no candidate branch or merge was needed
  - both runs were local on Seeweb only, with exclusive GPU ownership throughout the experiments
- Authoritative local references are now:
  - best kept local run `r62`: `logs/apr05_r62_xsa_last9_scorefirst_scales_nomix_anchor.txt` with exact `1.17372532` at `15931956` bytes
  - best plain keep `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - matched adapting all-layer XSA failure `r63`: `logs/apr05_r63_xsa_last9_scorefirst_scales_nomix_epochs1.txt` with exact `2.10588407` at `15944384` bytes
  - older weaker-substrate matched reference `r57`: `logs/apr04_r57_xsa_last4_scorefirst_scales_nomix_anchor.txt` with exact `1.17630787` at `15948610` bytes
  - older weaker-substrate matched failure `r58`: `logs/apr04_r58_xsa_last4_scorefirst_scales_nomix_epochs1.txt` with exact `2.11475299` at `15951830` bytes
- Interpretation of the completed gate:
  - `r62` kept the stronger all-layer XSA substrate and the score-first harness healthy: the persisted log shows `xsa:last_n:9`, `adapted_chunks=0`, `update_steps=0`, and `stride_active=1`
  - `r62` improved the final post-quant roundtrip by `0.00085159` bpb versus plain `r61` while staying valid at `15931956` bytes, leaving `68044` bytes of headroom
  - `r63` proves legal adaptation really happened on the same harness: `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`
  - despite that real adaptation, `r63` finished `0.93215875` bpb worse than `r62` and `0.93130716` worse than `r61`
  - conclusion: stronger all-layer XSA improves the no-update substrate, but it still does not rescue the exact minimal existing-control adapting surface; future self-learning probes must either narrow the write surface further or move to a dedicated low-byte adaptive surface rather than rerunning `r63`
- Competitive reference refreshed against the official GitHub materials on Apr 6, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r62` is now `0.05899023` bpb to the public leader and `0.05434565` to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r62` as the current best local reference and `ed4be84` as the promoted codebase
  - do not rerun the exact `r63` harness unchanged
  - first narrow the self-learning write surface on the same substrate, starting with a fresh matched `TTT_EPOCHS=0` anchor on `XSA_LAST_N=9` using only `TTT_PARAM_PATTERNS=blocks.8.attn_scale`
  - only if that narrower anchor stays near-baseline should the next cycle spend a matched adapting run on the same single-surface harness; otherwise pivot to a cheaper VE-derived adaptive surface or a more targeted QAT bridge on top of `r62`

### Cycle Review Snapshot (Apr 5, 2026 r61 All-Layer XSA Plain Keep + Self-Learning Gate Reset)

This cycle completed the queued zero-byte deeper-context probe on the promoted substrate. The run was valid, all-layer XSA really activated, and the gain survived quantization, so the promoted plain reference improves without any code change.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept plain run `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - previous plain keep `r56`: `logs/apr04_r56_xsa_last4_nonttt.txt` with exact `1.17594018` at `15927044` bytes
  - matched no-update XSA-last4 anchor `r57`: `logs/apr04_r57_xsa_last4_scorefirst_scales_nomix_anchor.txt` with exact `1.17630787` at `15948610` bytes
  - matched adapting XSA-last4 failure `r58`: `logs/apr04_r58_xsa_last4_scorefirst_scales_nomix_epochs1.txt` with exact `2.11475299` at `15951830` bytes
  - activation-proven plain late-QAT bridge `r59`: `logs/apr05_r59_xsa_last4_lateqat015_nonttt.txt` with exact `1.17673423` at `15931333` bytes
  - byte-aware minimal VE bridge `r60`: `logs/apr05_r60_xsa_last4_ve24_plain.txt` with exact `1.17604014` at `15992214` bytes
- Interpretation of `r61`:
  - the persisted log proves the intended mechanism really activated: `xsa:last_n:9 layers:[0, 1, 2, 3, 4, 5, 6, 7, 8]`
  - the sliding pre-quant metric improved to exact `1.1666` versus `1.1686` on `r56`
  - the final post-quant roundtrip improved by `0.00136327` bpb versus `r56`, by `0.00215732` versus `r59`, and by `0.00146323` versus `r60`
  - total submission size is `15927210` bytes, only `166` bytes larger than `r56`, leaving `72790` bytes of headroom
  - conclusion: deeper-context XSA still transfers locally when extended to all 9 layers of the promoted 9-layer model, and it is now the strongest zero-byte plain substrate we have
- Competitive reference carried forward from the Apr 5, 2026 official-board refresh:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r61` therefore narrows to `0.05984182` bpb to the public leader and `0.05519724` bpb to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r61` / `ed4be84` as the promoted plain reference
  - rebuild a fresh matched `TTT_EPOCHS=0` no-update score-first anchor on the stronger `XSA_LAST_N=9` substrate before making any new adapting claim
  - start that anchor from the smallest existing-control write surface that previously stayed near-baseline, namely `TTT_PARAM_PATTERNS=blocks.8.attn_scale,blocks.8.mlp_scale` with `TTT_CV_CARRIER_RANK=0`
  - only if the new anchor stays near-baseline should the next run spend a matched adapting `TTT_EPOCHS=1` probe on the same harness; otherwise pivot back to a cheaper VE slice or a more targeted QAT bridge on top of `r61`

### Cycle Review Snapshot (Apr 5, 2026 r60 Minimal-VE Plain Gate Failure + Plateau Review)

This cycle completed the queued minimal `VE` bridge on a fresh candidate branch and then hit the plateau clause again. The run was valid and nearly neutral before quantization, but it still failed to beat the promoted plain XSA-last4 keep and spent almost all remaining artifact headroom.

- Promoted branch/root state:
  - `HEAD` remains the promoted XSA-last4 keep `ed4be84`
  - the tested code change lives on archived candidate branch `candidate/mar27/ve-minimal` at commit `88ee64c`
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept plain run `r56`: `logs/apr04_r56_xsa_last4_nonttt.txt` with exact `1.17594018` at `15927044` bytes
  - matched no-update XSA-last4 anchor `r57`: `logs/apr04_r57_xsa_last4_scorefirst_scales_nomix_anchor.txt` with exact `1.17630787` at `15948610` bytes
  - matched adapting XSA-last4 failure `r58`: `logs/apr04_r58_xsa_last4_scorefirst_scales_nomix_epochs1.txt` with exact `2.11475299` at `15951830` bytes
  - activation-proven plain late-QAT bridge `r59`: `logs/apr05_r59_xsa_last4_lateqat015_nonttt.txt` with exact `1.17673423` at `15931333` bytes
  - byte-aware minimal VE bridge `r60`: `logs/apr05_r60_xsa_last4_ve24_plain.txt` with exact `1.17604014` at `15992214` bytes
- Interpretation of `r60`:
  - the printed sliding pre-quant metric matched `r56` to the logged precision (`val_bpb:1.1686`), so the plain model quality did not obviously collapse
  - the final post-quant roundtrip still regressed by `0.00009996` bpb versus `r56`
  - total submission size rose to `15992214` bytes, leaving only `7786` bytes of headroom
  - conclusion: the exact `VE_ENABLED=1 VE_DIM=24 VE_LAYERS=7,8` surface is too byte-expensive for its current gain on the int8+zlib stack; future VE work must get materially cheaper or supply a dedicated self-learning knob that justifies the bytes
- Cycle trigger:
  - `r57`, `r58`, `r59`, and `r60` are now four consecutive valid experiments after `r56` that failed to improve the current best run, so the plateau clause triggers a mandatory strategic review and queue rewrite
- Competitive reference refreshed against the official GitHub materials on Apr 5, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r56` therefore remains `0.06120509` bpb to the public leader and `0.05656051` bpb to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r56` / `ed4be84` as the promoted plain reference
  - do not promote or rerun the exact `VE_DIM=24 VE_LAYERS=7,8` surface unchanged
  - prioritize a zero-byte deeper-context bridge next, starting with `XSA_LAST_N=9` and TTT disabled on the promoted substrate
  - only if a stronger plain keep appears should the next run rebuild a matched no-update score-first anchor there before another adapting claim

### Cycle Review Snapshot (Apr 5, 2026 r59 Late-QAT Plain Gate Failure + Plateau Review)

This cycle completed the queued plain late-QAT bridge on a fresh candidate branch and then hit the plateau clause. The candidate was valid, the new bridge really activated, but it still failed to beat the promoted XSA-last4 keep.

- Promoted branch/root state:
  - `HEAD` remains the promoted XSA-last4 keep `ed4be84`
  - the tested code change lives on archived candidate branch `candidate/mar27/xsa-last4-late-qat` at commit `cf8313d`
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept plain run `r56`: `logs/apr04_r56_xsa_last4_nonttt.txt` with exact `1.17594018` at `15927044` bytes
  - matched no-update XSA-last4 anchor `r57`: `logs/apr04_r57_xsa_last4_scorefirst_scales_nomix_anchor.txt` with exact `1.17630787` at `15948610` bytes
  - matched adapting XSA-last4 failure `r58`: `logs/apr04_r58_xsa_last4_scorefirst_scales_nomix_epochs1.txt` with exact `2.11475299` at `15951830` bytes
  - activation-proven plain late-QAT bridge `r59`: `logs/apr05_r59_xsa_last4_lateqat015_nonttt.txt` with exact `1.17673423` at `15931333` bytes
- Interpretation of `r59`:
  - the persisted log proves the intended mechanism really activated: `late_qat:enabled step:19476 scale:0.1497`
  - artifact size stayed valid at `15931333` bytes, leaving `68667` bytes of headroom
  - pre-quant drifted to exact `1.1695` versus `1.1686` on `r56`, and final post-quant regressed by `0.00079405` bpb versus `r56`
  - conclusion: a broad all-matrix late int8 QAT tail at threshold `0.15` does not strengthen the promoted XSA-last4 plain substrate enough to justify promotion; any future QAT retry must be more targeted or ride on a stronger auxiliary-context bridge
- Cycle trigger:
  - `r57`, `r58`, and `r59` are now three consecutive valid experiments after `r56` that failed to improve the current best run, so the plateau clause triggers a mandatory strategic review and queue rewrite
- Competitive reference refreshed against the official GitHub materials on Apr 5, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r56` therefore remains `0.06120509` bpb to the public leader and `0.05656051` bpb to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r56` / `ed4be84` as the promoted plain reference
  - do not rerun the exact broad `LATE_QAT_THRESHOLD=0.15` bridge on the same substrate
  - prioritize a byte-aware minimal `VE` bridge on a fresh candidate branch with TTT disabled before reopening any new adapting claim
  - if a future plain keep is neutral or better, rebuild a matched no-update score-first anchor there before spending another adapting run

### Cycle Review Snapshot (Apr 4, 2026 r55 Smaller-Write-Surface Gate Failure + Plateau Review)

This cycle completed the queued smaller-write-surface anchor on the promoted `ed4be84` substrate and then stopped for the mandatory end-of-cycle strategic review. The run was valid, the score-first harness stayed healthy, but the exact no-update surface drifted far enough that it does not earn a matched adapting follow-up.

- Promoted branch/root state:
  - `HEAD` remains the promoted XSA-lite keep `ed4be84`
  - `git diff --stat -- train_gpt.py` is still empty, so the promoted root workspace remains clean
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept run `r46`: `logs/apr03_r46_xsa_last3_nonttt.txt` with exact `1.17673157` at `15931124` bytes
  - direct-carrier no-update anchor `r47`: `logs/apr04_r47_xsa_last3_scorefirst_cv_carrier_anchor.txt` with exact `1.17697198` at `15942317` bytes
  - direct-carrier adapting failure `r48`: `logs/apr04_r48_xsa_last3_scorefirst_cv_carrier_epochs1.txt` with exact `2.16970838` at `15935764` bytes
  - smaller existing-control no-update anchor `r55`: `logs/apr04_r55_xsa_last3_scorefirst_scales_anchor.txt` with exact `1.17905724` at `15947552` bytes
- Interpretation of `r55`:
  - the logs persist the required score-first evidence twice: `adapted_chunks=0`, `update_steps=0`, `stride_active=1`
  - pre-quant stayed reasonable at exact `1.1704`, but the final post-quant roundtrip finished `0.00208526` bpb worse than `r47` and `0.00232567` worse than `r46`
  - artifact size remains valid at `15947552` bytes, leaving only `52448` bytes of headroom
  - conclusion: the exact `blocks.8.attn_scale,blocks.8.mlp_scale,blocks.8.resid_mix` no-carrier harness is not near-baseline on `ed4be84`, so it does not earn the queued matched adapting run
- Cycle trigger:
  - `r47`, `r48`, and `r55` are now three consecutive valid experiments after `r46` that failed to improve the current best run, so the plateau clause triggers a mandatory strategic review and queue rewrite
- Competitive reference refreshed against the official GitHub README on Apr 4, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r46` therefore remains `0.06199648` bpb to the public leader and `0.05735190` bpb to the best public legal-TTT line
- Next action for the following cycle:
  - do not spend the matched adapting run on the exact `r55` scales+mix harness
  - prioritize a stronger plain substrate first, starting with a plain `XSA_LAST_N=4` comparison with TTT disabled on a fresh candidate branch
  - if the stronger plain keep is valid, re-establish a matched no-update anchor there before reopening any adapting claim
  - keep any QAT bridge gated on a persisted `late_qat:enabled` marker

### Cycle Review Snapshot (Apr 4, 2026 XSA Carrier Gate Closure + Queue Reset)

This cycle intentionally did not launch a new experiment. The promoted XSA-last3 substrate has already consumed its queued self-learning gate: `r47` and `r48` both completed, so the required next action is a strategic review and queue rewrite rather than another blind run.

- Promoted branch/root state:
  - `HEAD` remains the promoted XSA-lite keep `ed4be84`
  - `git diff --stat -- train_gpt.py` is empty, so the promoted root workspace is clean
  - `nvidia-smi` showed `0 MiB` in use and no local `torchrun` / `train_gpt.py` / `dispatch_experiment.py` process was active, so Seeweb is ready for the next single-run cycle
- Authoritative local references are now:
  - best kept run `r46`: `logs/apr03_r46_xsa_last3_nonttt.txt` with exact `1.17673157` at `15931124` bytes
  - matched no-update XSA carrier anchor `r47`: `logs/apr04_r47_xsa_last3_scorefirst_cv_carrier_anchor.txt` with exact `1.17697198` at `15942317` bytes
  - matched adapting XSA carrier probe `r48`: `logs/apr04_r48_xsa_last3_scorefirst_cv_carrier_epochs1.txt` with exact `2.16970838` at `15935764` bytes
- Interpretation of the completed XSA gate:
  - `r47` stayed close enough to `r46` to validate the score-first harness on `ed4be84`: it finished only `0.00024041` bpb worse while preserving `adapted_chunks=0`, `update_steps=0`, and `stride_active=1`
  - `r48` proves real legal adaptation on the same substrate: `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`
  - despite that real adaptation, `r48` finished `0.99273640` bpb worse than `r47` and `0.99297681` worse than `r46`, so deeper-context XSA alone does not rescue direct rank-4 `blocks.8.attn.c_v_carrier.` score-first TTT
  - conclusion: the direct carrier line is now closed on `ed4be84` unless a genuinely new stabilizer or write surface is added
- Resource check:
  - `r46` leaves `68876` bytes of headroom, `r47` leaves `57683`, and even failed `r48` still leaves `64236`
  - plain keeps / no-update anchors on the promoted substrate take about `2.22` to `2.24` hours end-to-end on Seeweb, while a real adapting run like `r48` takes about `4.76` hours, so future adapting spend must isolate one high-value question
- Competitive reference refreshed against the official GitHub README on Apr 4, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r46` is therefore `0.06199648` bpb to the public leader and `0.05735190` bpb to the best public legal-TTT line
- Next action for the following cycle:
  - do not spend another run on the same rank-4 carrier harness on `ed4be84`
  - reset the queue toward a smaller TTT write surface or a stronger plain substrate before reopening any direct adaptation claim
  - require any new QAT bridge to prove activation with a persisted `late_qat:enabled` marker

### Cycle Review Snapshot (Apr 3, 2026 Clean r45 Closure + Next-Bridge Reset)

This cycle again intentionally did not launch a new experiment. The required review target changed: the promoted workspace is now already clean, the dirty bundled prototype has already been archived, and the missing state was the strategy rewrite after the clean `r45` rerun actually finished.

- The promoted branch/root state is now reconciled:
  - `HEAD` is still the promoted GPTQ-lite keep `9689d36`
  - `git status --short --untracked-files=no` is empty, so the tracked workspace matches the promoted source
  - the earlier bundled prototype is now archived on `candidate/mar27/dirty-root-prototype-archive`, so provenance is no longer the blocker
- The authoritative local references are now:
  - best kept non-TTT run `r43`: `logs/apr03_r43_gptqlite_rowclip_nonttt.txt` with exact `1.17685468` at `15937725` bytes
  - matched no-update score-first anchor `r44`: `logs/apr03_r44_gptqlite_scorefirst_cv_carrier_anchor.txt` with exact `1.17719201` at `15936229` bytes
  - clean zero-momentum rerun `r45`: `logs/apr03_r45rerun_clean_gptqlite_scorefirst_cv_carrier_mom000.txt` with exact `2.15711753` at `15940694` bytes
- `r45` is now a valid discard, not a provenance question and not an infrastructure crash:
  - the persisted log contains the full score-first summary twice
  - legal adaptation is confirmed: `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`
  - despite real adaptation, final post-quant bpb is `0.97992552` worse than `r44` and `0.98026285` worse than `r43`
  - conclusion: pure carrier momentum stabilizers are closed on the promoted `9689d36` substrate
- The official board was rechecked against upstream GitHub README:
  - public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r43` remains `0.06211959` to the public leader and `0.05747501` to the best public legal-TTT line
- Public-record takeaway for the next branch:
  - stronger substrate still dominates the leaderboard delta
  - the cheapest portable deeper-context bridge remains minimal XSA on the deepest blocks
  - any QAT follow-up must prove activation with a persisted `late_qat:enabled` marker
- Resource check:
  - `r43` leaves `62275` bytes of headroom, `r44` leaves `63771`, and the clean `r45` discard still leaves `59306`
  - plain keeps / no-update anchors are about `2.2h` end-to-end on Seeweb, while a real adapting run like clean `r45` takes about `4.6h`, so the next adapting spend should happen only after a stronger plain substrate exists
- Next action for the following cycle:
  - create a fresh candidate branch from `9689d36`
  - test a minimal XSA-lite / deeper-context bridge with TTT disabled first
  - only if that plain keep is valid and competitive, run its matched `TTT_EPOCHS=0` anchor before any new adapting claim

This cycle did not launch a new experiment. Instead, it closed the gating review required before the next run: verify the promoted source of truth, verify whether `r45` contains usable final evidence, and refresh the competitive reference against the current official `openai/parameter-golf` leaderboard rather than the stale local checkout alone.

The key clarification is still provenance. `HEAD` remains the promoted GPTQ-lite keep `9689d36`, but the root working copy `train_gpt.py` has a large uncommitted prototype layered on top of it. `git diff --stat -- train_gpt.py` shows `394` changed lines (`329` insertions, `65` deletions), so this is not harmless env drift. The prototype bundles int6/lzma export, bigram/XSA/VE paths, extra QAT controls, weight decay plumbing, and wider defaults into one dirty tree with no clean branch/result lineage. Under the current protocol, the next action is still "resolve provenance first" rather than "rerun `r45` from root."

- The promoted code commit remains `9689d36` on `autoresearch/mar27`; `git rev-parse --short HEAD` still matches the promoted `r43` substrate.
- The best kept local run remains `logs/apr03_r43_gptqlite_rowclip_nonttt.txt`: `final_int8_zlib_roundtrip_exact val_bpb=1.17685468` at `15937725` bytes, leaving `62275` bytes of headroom.
- The best matched self-learning anchor remains `logs/apr03_r44_gptqlite_scorefirst_cv_carrier_anchor.txt`: `1.17719201` at `15936229` bytes, leaving `63771` bytes of headroom.
- `r45` remains an incomplete crash, not TTT evidence:
  - log file: `logs/apr03_r45_gptqlite_scorefirst_cv_carrier_mom000.txt`
  - it reaches `step:20000/20000 train_loss:2.0388 train_time:7199683ms`
  - it contains no persisted `ttt_eval` summary, no final post-quant metric, and no final size line
  - conclusion: zero-momentum carrier stabilization is still unresolved empirically
- The blocker to the next run is provenance, not local capacity:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` returned empty during this review
  - no local `torchrun`/`train_gpt.py` job was active
  - Seeweb is currently free enough to run once the workspace is reconciled
- The official competitive reference is ahead of the local checkout:
  - the local `README.md` in this repo still tops out at `1.1228` from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - the official GitHub README on Apr 3, 2026 lists `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` at `1.1147` as the current public leader, with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at `1.1194`
  - therefore our real public gap is `0.06215468` bpb, not the smaller stale gap implied by the local README
- Official competitor study sharpened the missing deltas:
  - the `2026-03-25` leader drops TTT entirely on that stronger stack and instead wins with AR self-generated Full Hessian GPTQ calibration, XSA on all 11 layers, BigramHash `3072 x 112`, warmdown `4000`, LZMA preset `9`, and the PR #549 substrate; its README explicitly says TTT was neutral or negative across `25` failed attempts on that stack
  - the `2026-03-23` legal TTT record still matters because it shows a real post-TTT gain (`~ -0.0025` bpb), but it does so on a much stronger substrate: LeakyReLU(0.5)^2, XSA on the last 4 layers, BigramHash, VE, EMA/SWA, Parameter Banking, and Parallel Muon
  - the `2026-03-22` GPTQ-lite bridge remains our closest portable quantization reference: about `-0.0006` from clip search, `-0.0006` from EMA, `-0.0002` from warmdown `3500`, plus a small but real QAT gain when activation is actually proven
  - we have already imported `EVAL_STRIDE=64`, `EMA_ENABLED=1`, `WARMDOWN_ITERS=3500`, Partial RoPE, and per-row clip search, so the largest remaining portable deltas are substrate-side (deeper-context/XSA/bigram/VE family) plus a training-side quantization bridge with explicit activation proof

Immediate implications:

- keep `9689d36` as the promoted code branch and `r43` as the current best local result
- keep `r44` as the authoritative matched no-update anchor for post-bridge carrier probes
- keep `r45` logged as `crash`/incomplete; do not reinterpret it as weakly negative TTT signal
- treat this invocation itself as a valid no-run strategic-review cycle; the local wrapper should relaunch into the provenance gate rather than assume an experiment was skipped by mistake
- do not launch another run from the current dirty root tree until it is either archived on a named candidate branch, split into atomic candidate work, or removed from the rerun path
- once provenance is resolved, rerun the exact `r45` idea once on a clean promoted workspace before making any claim about `TTT_MOMENTUM=0.0`
- if that clean rerun again fails to emit final metrics, treat the next cycle as instrumentation/persistence work before spending another adapting TTT run
- if that clean rerun completes but lands far from `r44`, close pure momentum stabilizers and pivot the next code-changing cycle to a stronger substrate bridge rather than another carrier-only stabilizer

### Fresh Hypotheses For The Next Cycle

Generate the next queue from these hypotheses, ordered from lowest-risk signal to highest-upside rewrite:

1. Low-risk workflow gate: resolve workspace provenance first. If the current dirty `train_gpt.py` is intentional, capture or split it onto named candidate branch(es); otherwise restore the exact promoted `9689d36` source locally before any run.
2. Low-risk self-learning follow-up: once the root workspace again matches the promoted `r43/r44` substrate, rerun `r45` exactly with the same `r44` harness and `TTT_MOMENTUM=0.0`, insisting on a complete persisted log before drawing any conclusion.
3. Low-risk self-learning follow-up: only if that clean `r45` rerun finishes within roughly `0.01` bpb of `r44`, test one lower-LR follow-up (`TTT_LR=0.001`, `TTT_MOMENTUM=0.0`) before closing pure optimizer-persistence stabilizers.
4. Medium-risk workflow-plus-model follow-up: if a second clean `r45` rerun again truncates before final metrics, isolate that as an instrumentation candidate and patch persistence/log flushing before spending another adapting run.
5. Medium-risk competitor adaptation: prototype a minimal XSA-lite / deeper-context bridge on `9689d36`, and judge it first with TTT disabled before spending another adaptation cycle.
6. Medium-risk competitor adaptation: add an int8-compatible late fake-quant or row-aware training bridge on `9689d36`, prove that it really activates, and require a plain non-TTT keep before any TTT pairing.
7. Medium-risk competitor adaptation: if the dirty root prototype is intentional, do not bless it as one bundled experiment; split or archive it into atomic candidate work and start with a plain non-TTT keep on the first coherent substrate slice.
8. Aggressive self-rewriting idea: add a learned scalar or per-channel plasticity gate on top of `blocks.8.attn.c_v_carrier.` so training can explicitly bound inference-time update magnitude.

### Cycle Review Snapshot (Apr 3, 2026 Dirty Prototype Audit + Official Record Delta)

This cycle again intentionally did not launch a new experiment. The blocker is still provenance: the root working tree does not match the promoted `9689d36` substrate, so a rerun of `r45` from this tree would not be a clean comparison.

- `HEAD` is still the promoted row-clip keep `9689d36`, and `Seeweb` is currently free:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` returned empty
  - no local `torchrun` / `train_gpt.py` process was active
- The working tree is materially dirty:
  - `git diff --stat -- .gitignore train_gpt.py` shows `394` changed lines in `train_gpt.py` plus local ignore updates in `.gitignore`
  - the dirty `train_gpt.py` is not a small rerun tweak; it bundles `int6+lzma` export, `BigramHash`, `XSA`, `VE`, `late_qat`, weight decay plumbing, and wider/default substrate changes in one unattributed prototype
  - conclusion: do not treat the dirty root tree as a valid rerun surface for `r45`, and do not bless it as a single experiment branch
- The promoted local references remain:
  - best kept run: `logs/apr03_r43_gptqlite_rowclip_nonttt.txt` at `1.17685468` and `15937725` bytes
  - matched no-update anchor: `logs/apr03_r44_gptqlite_scorefirst_cv_carrier_anchor.txt` at `1.17719201` and `15936229` bytes
  - `r44` therefore trails `r43` by only `0.00033733` bpb while preserving `adapted_chunks=0`, `update_steps=0`, and `stride_active=1`
- `r45` still cannot be interpreted as a negative TTT result:
  - `logs/apr03_r45_gptqlite_scorefirst_cv_carrier_mom000.txt` ends at `step:20000/20000 train_loss:2.0388 train_time:7199683ms`
  - it contains no persisted `ttt_eval` summary, no final post-quant line, and no final size line
  - there is no traceback or OOM marker in the persisted file, so the current evidence points to incomplete persistence / termination rather than a completed bad run
- Official competitive delta was refreshed from the upstream leaderboard:
  - current public leader is `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` at exact `1.11473509` bpb, so our gap from `r43` is `0.06211959` bpb
  - the strongest legal-TTT public reference remains `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at exact `1.11937967` bpb, so our gap from `r43` is `0.05747501` bpb
  - the best public TTT gain is still on the order of `-0.0025` bpb, while our carrier adaptation line remains catastrophic; the missing difference is therefore substrate quality plus adaptation robustness, not simply “enable score-first”
- The strongest portable gaps versus the public records are still substrate-side:
  - XSA/deeper-context attention beyond the current promoted line
  - BigramHash and VE-style auxiliary embeddings
  - a real training-side quantization bridge such as late fake-quant/QAT with activation proof
  - stronger post-training quantization than the current GPTQ-lite row-clip path

Immediate next-test order for the following cycle:

1. Resolve provenance first: archive or split the dirty root prototype onto named candidate branch(es), then restore the root workspace to the exact promoted `9689d36` source before any rerun.
2. Once provenance is clean, rerun `r45` exactly on `9689d36` with the `r44` harness plus `TTT_EPOCHS=1` and `TTT_MOMENTUM=0.0`.
3. If that clean rerun again omits final metrics, treat the next code-changing cycle as a logging/persistence instrumentation branch rather than another TTT sweep.
4. If the clean rerun completes but still collapses far from `r44`, close pure carrier optimizer stabilizers and pivot the next candidate branch to a minimal non-TTT XSA-lite / deeper-context bridge with its own keep and matched score-first anchor.
5. If the dirty prototype is revisited later, split it into atomic substrate slices starting with non-TTT bridges, not as one bundled experiment.

### Cycle Review Snapshot (Apr 3, 2026 Candidate-Branch Gate Before Any Clean Rerun)

This cycle again intentionally did not launch a new experiment. The objective was to decide whether the queued `r45` rerun is actually runnable from the current workspace. The answer is still no: the root tree remains a bundled, unnamed prototype rather than the promoted `9689d36` substrate or a clean candidate branch.

- Local execution capacity is not the blocker:
  - `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` returned empty
  - `ps -eo pid,etimes,cmd | grep -E 'torchrun|train_gpt.py|dispatch_experiment.py'` found no active local training/eval job
- The provenance problem is still material:
  - `git diff --stat -- .gitignore train_gpt.py` still shows `394` changed lines in `train_gpt.py` plus local `.gitignore` edits
  - the dirty tree is not a single tweak to `r45`; the diff visibly imports `lzma`/mixed-int6 export, weight decay plumbing, QAT gates, `BigramHash`, `XSA`, `VE`, and wider promoted defaults together
  - existing `candidate/mar27/*` branches cover earlier atomic experiments, but none names or captures this bundled prototype, so it is still not a valid rerun surface
- The local result stack is unchanged:
  - best kept run remains `r43` (`logs/apr03_r43_gptqlite_rowclip_nonttt.txt`) at exact `1.17685468` and `15937725` bytes
  - matched no-update anchor remains `r44` (`logs/apr03_r44_gptqlite_scorefirst_cv_carrier_anchor.txt`) at exact `1.17719201` and `15936229` bytes
  - the most informative valid failure remains `r42` (`logs/apr02_r42_scorefirst_cv_carrier_rank2_epochs1_lr0001.txt`): real legal adaptation persisted (`adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`) yet final post-quant collapsed to `2.15885029`
  - `r45` still ends immediately after `step:20000/20000 train_loss:2.0388 ...` with no persisted `ttt_eval`, no final post-quant line, and no final size line
- The official competitive reference remains the same after refresh:
  - public leader is still exact `1.11473509`
  - best public legal-TTT result is still exact `1.11937967`
  - our exact gaps from `r43` therefore remain `0.06211959` bpb to the public leader and `0.05747501` bpb to the public legal-TTT leader

Immediate implications:

- treat this invocation as a valid no-run strategic-review cycle
- do not launch `r45` or any other run from the current dirty root tree
- the next cycle must start by capturing the dirty prototype onto a named candidate branch or restoring the root workspace to the exact promoted source
- only after that provenance gate is closed should the loop spend a full 20,000-step run on the clean `r45` rerun
- if the clean rerun still omits final metrics, the next code-changing branch is logging/persistence instrumentation, not another carrier or momentum sweep

## Self-Learning Verification Rules

Do not claim self-learning is working unless the logs show evidence.

For any self-learning / TTT experiment, verify all of the following:

1. `eval_mode` indicates a TTT path rather than plain standard evaluation
2. the log shows non-zero adaptation activity such as:
   - `adapted_docs > 0` or adapted windows > 0
   - `update_steps > 0`
3. the final post-quant metric improves versus the comparable non-TTT baseline or the best prior TTT baseline
4. if new adaptation parameters are added, the matched parameter set for TTT actually grows in a controlled, size-aware way
5. compare against the closest non-TTT baseline with the same context protocol; if the TTT branch also changes document isolation, stride, or chunk scoring, run or reference the matching non-adaptation ablation before attributing the delta to TTT
6. if the experiment intends to preserve sliding evaluation, the persisted evidence must confirm that the active score-first path kept stride active; for score-first global runs, prefer an explicit `stride_active:1` summary over generic startup logging
7. a generic line such as `eval_stride_ignored_due_to_ttt` from older prefix-TTT logging is not by itself proof that stride was lost or preserved on the score-first path; rely on the score-first summary or matched code path instead
8. for score-first global experiments, the run log must persist the adaptation summary (`adapted_chunks`, `update_steps`, `stride_active`, and schedule details) into `logs/<run_id>.txt`, not only console stdout; otherwise treat the run as instrumentation-incomplete for promotion purposes

If TTT is enabled but the logs show no real adaptation activity, treat that experiment as non-working, even if the configuration nominally enabled TTT.

If a self-learning idea improves only pre-quant performance but not the final post-quant metric, do not count it as a win.

If a TTT candidate drops `EVAL_STRIDE=64`, treat it as a protocol diagnostic or archived branch unless it has an explicit matched-context justification.

For score-first global experiments on an unchanged harness, remember that `TTT_EPOCHS=0` never executes adaptation steps, so a no-update run can be reused across pure `TTT_PARAM_PATTERNS` sweeps. Re-run the no-update reference only when the score-first scoring path or other context-setting behavior changes.

Launch command:
```bash
RUN_ID=autoresearch \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=500 \
VAL_LOSS_EVERY=0 \
bash autoresearch/run_experiment.sh > run.log 2>&1
```

**What you CAN modify** (all inside `train_gpt.py`):

- Model architecture: number of layers, model dimension, number of heads, number of KV heads, MLP multiplier
- Optimizer hyperparameters: learning rates (embed_lr, matrix_lr, scalar_lr, tied_embed_lr), momentum, warmup/warmdown steps, beta1/beta2
- Training batch size and sequence length
- Activation functions (relu², gelu, swiglu, etc.)
- Attention variants (GQA ratios, window patterns, sliding window)
- Normalization strategies
- Embedding strategy (tied vs untied, init std)
- Skip connection patterns
- Adding bias terms to linear layers
- Logit softcap value
- RoPE base frequency
- QK gain init
- Any architectural innovation that stays within a single file

**What you CANNOT modify**:

- The data files or data paths (they are fixed)
- The tokenizer or vocabulary size (1024 BPE, fixed)
- The evaluation formula (eval_val function logic for bpb calculation)
- The quantization format (int8 + zlib) — you can tune percentile clipping but not the scheme
- The output format (the script must still print `final_int8_zlib_roundtrip val_bpb:X.XXXX`)
- Do NOT add external package dependencies
- Do NOT add network calls
- Do NOT modify data loading to touch validation files during training

**Hard constraints to check after every run**:

1. `val_bpb` must be a valid number (not nan, not inf)
2. The final `Total submission size ...` line must be **under 16,000,000 bytes**
3. The run must complete without crashing

If the submission size exceeds 16MB, the experiment is **invalid** regardless of val_bpb. Log it as `discard` with a note about size violation.

## Output format

After the run finishes, the script prints several lines. The key metrics are:

```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXXms
final_int{8_zlib|6_lzma}_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
Total submission size {int8+zlib|int6+lzma}: XXXXXXX bytes
```

Extract them with:
```bash
grep -E "^final_(int8_zlib|int6_lzma)_roundtrip_exact |^Total submission size (int8\\+zlib|int6\\+lzma): " run.log
```

The primary metric is the `val_bpb` from the final post-quant roundtrip line.
The size constraint is from the matching final `Total submission size ...` line.

If neither line appears, the run crashed. Run `tail -n 50 run.log` to debug.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	val_bpb	size_bytes	size_ok	status	param_changes	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (from the final post-quant roundtrip line) — use 0.000000 for crashes
3. total submission size in bytes (from the final `Total submission size ...` line) — use 0 for crashes
4. size_ok: `yes` if under 16000000, `no` if over — use `n/a` for crashes
5. status: `promote`, `anchor`, `discard`, or `crash`
6. param_changes: concise list of env vars or code changes (e.g. "NUM_LAYERS=11 MODEL_DIM=448")
7. short text description of what this experiment tried and the reasoning

Status semantics are strict:

- `promote`: a true promoted improvement or a candidate branch that earned merge/promotion
- `anchor`: a useful control/reference/no-update run that is valid but not a promoted improvement
- `discard`: a valid run that should not influence the promoted codebase
- `crash`: no valid final metric

Do not label near-baseline anchors or references as `promote`. The TSV must let a human see real progress at a glance.

If a run is contaminated by external server load or accidentally launched alongside another GPU job, log it as `crash` with a clear note such as `contaminated parallel GPU usage`, then re-run that same experiment later on a clean GPU.

Example:

```
commit	val_bpb	size_bytes	size_ok	status	param_changes	description
a1b2c3d	1.2244	15863489	yes	anchor	baseline	baseline run with default hyperparameters
b2c3d4e	1.2180	15901234	yes	promote	NUM_LAYERS=11 MODEL_DIM=448	more layers with smaller dim to stay under 16MB
c3d4e5f	1.2350	14500000	yes	discard	MLP_MULT=3	triple MLP width - too many params wasted on MLP
d4e5f6g	1.1900	17200000	no	discard	NUM_LAYERS=13 MODEL_DIM=512	better bpb but over 16MB size limit
e5f6g7h	0.0000	0	n/a	crash	NUM_HEADS=16	OOM error with 16 attention heads
```

## Mandatory Queue Status

The original mandatory rerun queue is complete on the promoted baseline. These are now historical anchor points, not the next actions:

1. Baseline full rerun: `1.21803700`
2. Document TTT rerun: `1.70763809`
3. Sequence TTT rerun: `1.83743383`
4. Document TTT with `TTT_STEPS=2`: `1.70855981`
5. Document TTT with `TTT_LR=0.003`: `1.70814995`
6. Document TTT with `TTT_LR=0.03`: `1.70634470`
7. Document TTT with wider param patterns: `1.71203171`
8. Sliding evaluation rerun with `EVAL_STRIDE=64`: `1.18390334`
9. EMA rerun: `1.21728762`
10. Quantization clip rerun with `INT8_CLIP_PERCENTILE=99.999`: `1.21845211`

Do not repeat the full mandatory queue unless the promoted branch changes materially enough that these anchor points stop being comparable.

## Self-Learning Buildout Queue

The alternate single-surface fallback gate and the dedicated-carrier rescue gate are now complete on the promoted `ed4be84` substrate: `r64` remains the best local no-update reference, `r65` and `r67` close the exact single-surface existing-control adapting probes, `r68`/`r69` close the rank-2 carrier line, and `r70`/`r71` close the rank-1 carrier rescue. Env-only plain-`SGD` carrier sweeps are therefore closed on the current promoted codebase. Continue sequentially, one experiment at a time, using `EVAL_STRIDE=64 EMA_ENABLED=1 WARMDOWN_ITERS=3500` as the default base unless a queue item explicitly overrides it. Any real code change still requires a fresh candidate branch.

1. Historical closure remains in force for the old carrier-era substrate: `r35` through `r45` already close pure existing-weight rewrites, rank sweeps, and momentum-only carrier stabilizers on the pre-XSA lines. Do not reopen those exact old boards unchanged.
2. Current best local reference: `r64` is the best kept local run at `1.17366108` and the authoritative no-update score-first reference on `ed4be84`; the promoted codebase itself still remains `ed4be84`.
3. Current plain substrate reference: `r61` is the best plain non-TTT keep on `ed4be84` at `1.17457691` and remains the correct plain comparator for future code-changing branches.
4. Canonical broader-surface adapting failure: `r63` is the canonical proof that the exact minimal existing-control surface `blocks.8.attn_scale,blocks.8.mlp_scale` still adapts legally on the strongest current substrate (`adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`) yet collapses by `0.93215875` bpb versus `r62`.
5. Existing-control closure on the strongest substrate remains in force: `r65` closes the exact single-surface `blocks.8.attn_scale` adapting harness and `r67` closes the exact single-surface `blocks.8.mlp_scale` adapting harness. Do not spend another run on the exact current `attn_scale`, `mlp_scale`, or `attn_scale+mlp_scale` adapting surfaces unchanged on `ed4be84`.
6. Fresh carrier references on the strongest substrate:
   - `r68` is the authoritative rank-2 carrier no-update anchor on the strongest substrate, with exact `1.17464072`, valid size `15948132`, `matched_params=1536`, and persisted `adapted_chunks=0`, `update_steps=0`, `stride_active=1`
   - `r69` is the authoritative matched rank-2 adapting failure on the same harness, with exact `2.12305747`, valid size `15949909`, and persisted `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`
   - `r70` is the authoritative rank-1 carrier no-update anchor on the same harness, with exact `1.17495154`, valid size `15946190`, `matched_params=768`, and persisted `adapted_chunks=0`, `update_steps=0`, `stride_active=1`
   - `r71` is the authoritative matched rank-1 adapting failure on the same harness, with exact `2.11417331`, valid size `15944673`, and persisted `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`
7. Env-only plain-`SGD` carrier closure is now in force on `ed4be84`: do not spend another full run on the exact `FastWeightCarrier` rank-1 or rank-2 adapting sweeps unless the optimizer, reset logic, or adaptive surface changes materially.
8. Archived weaker-substrate references stay closed: `r57`/`r58` remain useful context only; do not reopen the exact XSA-last4 gate while the stronger all-layer substrate is available.
9. Archived code-changing failures also stay closed unchanged: do not rerun the exact broad late-QAT bridge from `r59` or the exact `VE_DIM=24 VE_LAYERS=7,8` bridge from `r60`.
10. Next active self-learning experiment: open a fresh candidate branch for bounded Muon online adaptation on the rank-1 carrier surface while keeping the current score-first harness and `XSA_LAST_N=9`.
11. Gate rule for item 10: if the scoring path, adaptive surface, or TTT state-reset semantics change materially, the candidate branch must first emit a matched `TTT_EPOCHS=0` no-update anchor before any adapting claim.
12. Gate rule for item 10: the first adapting Muon run must still persist `matched_params`, `adapted_chunks`, `update_steps`, and `stride_active`, and it only earns follow-up work if it materially beats `r71` without losing the near-baseline no-update behavior or violating size.
13. Second code-changing self-learning branch after item 10: replace the current plain-`SGD` update path with a parameter-banked or banked-momentum online optimizer, again only for dedicated adaptive matrices such as the carrier or a future VE surface.
14. Third code-changing adaptive branch after item 10: mine the archived prototype for a materially cheaper deepest-layer VE slice or VE-scales-only surface, starting smaller and later than `VE_DIM=24 VE_LAYERS=7,8`, and pair its first adapting claim with a matched `TTT_EPOCHS=0` anchor.
15. Next competitor-inspired plain bridge after the carrier closure: if substrate work is needed before another TTT claim, revisit late QAT only on a deepest-XSA-aligned bounded matrix subset, and still require a persisted `late_qat:enabled` marker before counting the run as valid signal.
16. BigramHash remains downstream: do not spend a BigramHash run on the current int8+zlib stack unless either plain headroom improves materially first or the slice is made small enough to preserve a realistic size margin.
17. Aggressive self-learning follow-up: add a tiny carrier-derived or VE-derived adaptive surface that learns its own update scale or clamp during training so inference-time backprop can move it without destabilizing the core block controls.
18. Optional last-resort diagnostic: if one more surgical env-only clarification is still desired before the optimizer rewrite, isolate `blocks.8.attn.c_v_carrier.up` versus `.down` with matched no-update/adapting pairs and stop there.

Once the queue above is exhausted, continue autonomous exploration, still sequentially and still at full 20,000-step runs only.

## End-Of-Cycle Strategic Review

At the end of every test cycle, perform a mandatory strategic review before launching the next cycle.

### Definition Of A Cycle

A cycle is any one of the following:

- completion of the mandatory rerun queue
- completion of the self-learning buildout queue
- completion of 4 valid full experiments after the queues are exhausted
- any plateau where 3 consecutive valid experiments fail to improve the current best run

### End-Of-Cycle Tasks

At the end of each cycle, do all of the following in order:

1. **Deeply analyze our results**
   - Read `autoresearch/results.tsv`
   - Read the logs for the best run, the worst run, and the most informative failed runs
   - Identify which changes helped, hurt, or were inconclusive
   - Separate true model-signal wins from backend/infrastructure issues
   - Note artifact-size headroom and where capacity might still fit
2. **Study the competitors**
   - Read the current official leaderboard in `openai/parameter-golf`
   - Inspect the most relevant top public records in `records/track_10min_16mb/`
   - Extract the exact mechanisms those records use
   - Compare those mechanisms to what we have and have not tested
   - Specifically ask: which public tactics could make our self-learning / TTT path stronger?
3. **Generate fresh hypotheses**
   - Produce at least 5 new candidate experiments
   - Include at least:
     - 2 low-risk self-learning follow-ups
     - 2 medium-risk adaptations of competitor ideas that strengthen self-learning
     - 1 aggressive/high-upside self-rewriting or inference-backprop idea
4. **Rewrite the program before the next cycle**
   - Update the strategy sections inside `autoresearch/program.md`
   - Refresh the experiment queue ordering
   - Replace stale hypotheses with newer ones
   - Preserve the core hard rules and constraints
5. **Launch the next cycle**
   - Resume sequential, single-run experimentation using the updated program

This strategic review is mandatory. The agent must not keep running blind local experiments forever without pausing to analyze and adapt.

## Mandatory Post-Experiment Micro-Review

In addition to the deeper end-of-cycle review, every single completed experiment must trigger a micro-review.

After each experiment, before the next run starts, the agent must:

1. interpret what the result means
2. generate new candidate ideas in light of that result
3. update the immediate next-test shortlist
4. include those fresh proposals in the status email
5. if the run was remote, record or refresh the remote value estimate so future pod-shape decisions use measured price/performance

This is mandatory even if the current predefined queue is exhausted. Queue exhaustion is not a stopping condition; it is a trigger to think, rewrite the plan, and continue.

## Controller Policy

Helper scripts are allowed for:

- launching runs
- parsing logs
- sending emails
- simple branch housekeeping

But the agent must not replace the strategic loop with a finite static controller that only executes a hardcoded queue and then idles.

If a helper/controller is used, it must still hand control back to the agent after every experiment so that the agent can:

- analyze the result
- study competitors
- generate new hypotheses
- rewrite the program/queue
- decide the next run

A controller that ends with `queue exhausted -> idle wait loop` is incorrect and must not be used as the main orchestration mechanism.

## Self-Rewriting Rules For `program.md`

`autoresearch/program.md` is allowed to evolve, but only in a controlled way.

### Core Rules That Must Never Be Removed

- primary objective: win the official challenge
- full 20,000-step runs only for comparison
- no wall-clock timeout pruning during this phase
- at most one active training/eval job on `Seeweb`; on `Runpod`, one active job per pod with concurrency chosen by cost-aware local policy
- `Seeweb` GPU must be exclusive before each local run
- one atomic idea per experiment
- promoted-branch / candidate-branch workflow for code-changing experiments
- size limit and final post-quant metric remain mandatory
- end-of-cycle strategic review remains mandatory
- self-learning / inference-backprop focus remains a top research priority until explicitly changed by the human
- `Seeweb` remains the source of truth for promotions, results, and strategic control

### Sections The Agent May Rewrite Between Cycles

- the active experiment queue
- the competitor-insight notes
- the hypothesis backlog
- the prioritization of next experiments
- the specific self-learning verification checklist, if it becomes more informative

### Sections The Agent Must Not Rewrite Casually

- hard challenge constraints
- logging format
- artifact validity rules
- the requirement to study competitors and rewrite strategy after each cycle

## Active Strategy Board

This section is meant to be rewritten at the end of each cycle.

### Cycle Review Snapshot (Apr 13, 2026 r91 Recur-Gate Anchor Promotion)

This cycle completed the planned recurrence pivot on `candidate/mar27/recur-gate-selflearn`. The tiny late-layer recurrence gate stayed fully legal under the score-first no-update harness, learned a strong non-zero activation marker during training, and improved the final post-quant metric enough to become the new promoted reference.

- Promoted branch/root state:
  - `HEAD` is now promoted `41cc8a3` on `autoresearch/mar27`
  - the completed run was a code-changing experiment on `candidate/mar27/recur-gate-selflearn`, and it was merged locally back into the promoted branch after beating `r90`
  - the run stayed local on Seeweb with exclusive GPU ownership throughout training and both score-first eval passes
- Authoritative local references are now:
  - previous promoted best `r90`: `logs/apr13_r90_xsa_last9_ve12_l8_qgain_lanemix_anchor.txt` with exact `1.17093579` at `15975950` bytes
  - fresh promoted recurrence anchor `r91`: `logs/apr13_r91_xsa_last9_ve12_l8_recur_anchor.txt` with exact `1.17002954` at `15986864` bytes
  - strongest plain comparator `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
- Interpretation of `r91`:
  - startup confirms the intended dedicated surface is live on the correct substrate: `patterns:recur`, `matched_params=513`, `recur:enabled=True requested_layers:8 active_layers:[8]`, `lanemix:enabled=True requested_layers:8 active_layers:[8]`, and `xsa:last_n:9`
  - the score-first no-update gate stayed legal twice: both eval passes print `adapted_chunks=0`, `update_steps=0`, and `stride_active=1`
  - the recurrence branch genuinely learned during training: the final marker lands at `mean_abs=0.6335`, `max_abs=1.0000`, and `gain=2.0749`
  - pre-quant lands at exact `1.1635`, the final post-quant int8+zlib roundtrip lands at exact `1.17002954`, and the artifact stays valid at `15986864` bytes with `13136` bytes of headroom
  - `r91` improves promoted `r90` by `0.00090625` bpb despite the tighter size margin, so the recurrence gate is not merely a gate-passing anchor; it is a real promoted substrate improvement
- Competitive refresh from the official public board on Apr 13, 2026:
  - the public leader remains `2026-04-13_SP8192_3-Layer Recurrence + Parallel Residuals + Legal TTT` at exact `1.0810`
  - the public `2026-04-11_SP8192_3-Layer Recurrence + Parallel Residuals` record at exact `1.0816` and `2026-04-08_SP8192_Parallel Residuals + Score-First TTT` at exact `1.0822` reinforce the same message: recurrence plus richer routing are still the frontier mechanisms
  - the local gap from promoted `r91` is therefore about `+0.0890` bpb

### Cycle Review Snapshot (Apr 13, 2026 r92 Recur-Gate Adapting Failure + Family Closure)

This cycle completed the queued matched adapting recurrence probe on promoted `41cc8a3`. The dedicated `recur` surface really adapts under the legal score-first harness, but both pre-quant and post-quant metrics collapse badly enough that the exact low-LR recurrence adapting line is now closed.

- Promoted branch/root state:
  - `HEAD` stays promoted `41cc8a3` on `autoresearch/mar27`
  - the completed run was an env-only experiment on the promoted branch, so no candidate branch or merge was involved
  - the run stayed local on Seeweb with exclusive GPU ownership throughout training and both score-first eval passes
- Authoritative local references are now:
  - promoted best `r91`: `logs/apr13_r91_xsa_last9_ve12_l8_recur_anchor.txt` with exact `1.17002954` at `15986864` bytes
  - matched recurrence adapting failure `r92`: `logs/apr13_r92_xsa_last9_ve12_l8_recur_epochs1_lr0005_clip1.txt` with exact `2.10705140` at `15990957` bytes
- Interpretation of `r92`:
  - the persisted logs confirm real legal adaptation on the intended surface: `matched_params=513`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, `xsa:last_n:9`, and `recur:final active_layers:[8] summary:8:mean_abs=0.6389|max_abs=1.0000|gain=1.8943`
  - pre-quant already collapses to exact `2.0911`, and the final post-quant int8+zlib roundtrip lands at exact `2.10705140`
  - `r92` is `+0.93702186` bpb versus promoted `r91` and leaves only `9043` bytes of headroom at `15990957` bytes
  - conclusion: the exact low-LR plain-SGD recurrence adapting line is closed on promoted `41cc8a3`; the next cycle must pivot away from env-only recurrence adaptation to a code-changing plain bridge
- Competitive refresh from the official public board on Apr 13, 2026:
  - the public leader remains `2026-04-13_SP8192_3-Layer Recurrence + Parallel Residuals + Legal TTT` at exact `1.0810`
  - that public line still couples recurrence with stronger routing and a much larger `QK-Gain 5.25`, so the next local pivot should borrow the plain q-gain signal before spending another adapting claim

### Current Competitor Insights

- The official README was rechecked on Apr 13, 2026. The public leader is still exact `1.0810` on `2026-04-13_SP8192_3-Layer Recurrence + Parallel Residuals + Legal TTT`; local strategy should keep targeting that recurrence-heavy board.
- The next official records at `1.0816` and `1.0822` still center recurrence or parallel-residual routing, not another tiny env-only clarification on a legacy surface.
- Local evidence now matches the board only on the anchor side: the recurrence gate beats the widened no-update `q_gain+lane_mix` anchor, but real score-first updates on that surface collapse badly.
- The promoted `train_gpt.py` now exposes a dedicated `recur` surface on top of VE/XSA plus the existing late `lane_mix`/`q_gain` substrate.
- Runtime remains expensive on Seeweb: `r92` took about `8.36h` end-to-end, with each adapting eval pass taking about `3.04h`, so future adapting claims need a stronger reason than another exact rerun.

### Current Hypothesis Backlog

- Highest-priority next run: open a fresh candidate branch for a bounded late-layer `q_gain` plain bridge on promoted `41cc8a3`, targeting a public-style gain scale closer to `5.25` while keeping TTT disabled on the first full run.
- Second priority: if that bridge is near-baseline but not yet a best, spend one no-update score-first anchor on the bridged `q_gain` surface before any new adapting claim.
- Tertiary fallback only: keep parked env-only adapting `q_gain+lane_mix` only after a plain bridge improves the substrate; do not reopen exact `recur` adaptation unchanged.
- Instrumentation rule: any `q_gain` bridge must log its late-layer gain marker together with `xsa:last_n:9`, `lanemix`, and `recur` summaries so the substrate stays attributable.
- Anti-rerun rule: do not rerun the exact `r92` harness, and do not reopen old scalar-only VE, carrier, exact BLM, exact `lane_mix`, or exact `recur` adapting surfaces without a material code change.

### Current Queue Update Rule

- At the end of every cycle, rewrite the next queue in this file before running the next experiments.
- `r91` is the promoted best run: `1.17002954` at `15986864` bytes on `41cc8a3`.
- `r92` is the authoritative matched recurrence adapting failure: `2.10705140` at `15990957` bytes on `41cc8a3`.
- `r90` is the authoritative pre-recurrence routing reference at `1.17093579`; `r61` remains the strongest plain non-TTT comparator at `1.17457691`.
- `r68`/`r69`, `r70`/`r71`, `r72`, `r74`, and `r75` together close the carrier family under plain-SGD, bounded Muon, chunk reset, and state-banked carry. Do not reopen those exact carrier lines without a materially new adaptive surface.
- `r77`, `r79`, `r81`, `r83`, and `r85` together close the scalar-only VE adapting family. `r86`/`r87` close the exact bank-local matrix line. `r89` closes exact plain-SGD `lane_mix` adaptation. `r92` now closes the exact low-LR plain-SGD recurrence adapting line. Do not spend another env-only full run on those exact surfaces.
- The next full run should open a fresh candidate branch from promoted `41cc8a3` for a bounded late-layer `q_gain` plain bridge, not another env-only adapting rerun:
  - keep the promoted VE/XSA + `lane_mix` + `recur` substrate fixed
  - add a bounded or log-parameterized late-layer `q_gain` bridge on block `8`, aimed at recovering some of the public `QK-Gain 5.25` signal during training
  - keep the first full run plain (`TTT_ENABLE=0`) with `XSA_LAST_N=9 EVAL_STRIDE=64 EMA_ENABLED=1 WARMDOWN_ITERS=3500`
- Exact goal for that plain bridge: stay under 16,000,000 bytes, recover some of the public q-gain signal without opening online adaptation, and either beat `r91` or at least stay close enough to justify one later no-update score-first anchor.
- If the bounded `q_gain` bridge drifts badly or breaks size, pivot to a broader plain routing/recurrence bridge before reopening any new adapting claim.
- Before every future adapting follow-up, require a new plain or no-update gate on that substrate first; do not spend another adapting full run on a surface that has not cleared a near-baseline anchor.
- Competitive study should keep using the official README and `records/track_10min_16mb/` before each cycle. The public target remains exact `1.0810`.

### Cycle Review Snapshot (Apr 9, 2026 r76 VE Scale-Only Anchor Gate Pass)

This cycle completed the first full run on `candidate/mar27/ve-scales-l8-d12`. The branch is valid, the minimal VE surface really activates, and the no-update anchor stays close enough to best local to earn one matched adapting follow-up, but it is not good enough to promote.

- Promoted branch/root state:
  - `HEAD` has been returned to the promoted codebase `ed4be84` on `autoresearch/mar27`
  - the completed run was launched from candidate branch `candidate/mar27/ve-scales-l8-d12` at commit `f8005bc`
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout training and both score-first eval passes
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - plain all-layer XSA comparator `r61`: `logs/apr05_r61_xsa_last9_nonttt.txt` with exact `1.17457691` at `15927210` bytes
  - older VE bridge reference `r60`: `logs/apr05_r60_xsa_last4_ve24_plain.txt` with exact `1.17604014` at `15992214` bytes
  - fresh VE scale-only anchor `r76`: `logs/apr09_r76_xsa_last9_ve12_l8_scales_anchor.txt` with exact `1.17394549` at `15960289` bytes
- Interpretation of `r76`:
  - the persisted log confirms the intended minimal VE surface is live on the right substrate: `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, `matched_params=2`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, and `xsa:last_n:9`
  - pre-quant landed at exact `1.1663`, and the final post-quant roundtrip landed at exact `1.17394549`
  - `r76` is `+0.00028441` bpb versus best local `r64`, `-0.00063142` versus plain `r61`, and `-0.00209465` versus the older VE bridge `r60`
  - total submission size is `15960289` bytes, leaving `39711` bytes of headroom; the branch is valid but under noticeably tighter byte pressure than `r64`
  - conclusion: the minimal VE scale-only slice is the first code-changing post-carrier surface that passes the no-update gate on the strongest substrate; it stays unmerged for now because it does not beat `r64`, but it earns one matched adapting run
- Competitive reference rechecked against the official GitHub materials on Apr 9, 2026:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - both public reference lines still point the same local way: add late-layer capacity first, then test legal adaptation only on the richer surface that survives the no-update gate
- Next action for the following cycle:
  - keep promoted `ed4be84` unchanged and keep `candidate/mar27/ve-scales-l8-d12` as the active unmatched branch
  - spend one matched adapting VE-scales-only run on the exact `r76` surface
  - require `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, `matched_params:2`, `adapted_chunks>0`, `update_steps>0`, and a post-quant gain versus `r76` before keeping the VE family open

### Cycle Review Snapshot (Apr 9, 2026 Strategic Review + VE Candidate Prep)

This cycle stops at strategic prep rather than launching a new full experiment. The carrier family is already closed, and the useful work for this invocation was to prepare the next clean VE branch so the following cycle can spend its full budget on the anchor run itself.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84` on `autoresearch/mar27`
  - Seeweb is idle and uncontaminated at the end of this review
  - the fresh candidate branch `candidate/mar27/ve-scales-l8-d12` now exists at `f8005bc`
- What changed this cycle:
  - cherry-picked the old atomic VE bridge (`88ee64c`) onto a fresh candidate branch, producing `f8005bc`
  - verified the branch still compiles with `python3 -m py_compile train_gpt.py`
  - kept the extraction narrow: `ValueEmbedding` plus late-layer injection and optimizer wiring only, with no BigramHash/LN/QAT bundle
- Why no run this cycle:
  - `r71`, `r72`, and `r74` are already three full adapting failures on the same carrier-side family, and `r75` fails the no-update gate before adaptation
  - the correct pivot was branch surgery plus queue rewrite, not another carrier-side launch
  - the wrapper will relaunch the next cycle, which can now spend the whole local budget on the prepared VE anchor instead of mixing a long run with code extraction
- Competitive refresh carried forward:
  - public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - both public reference lines still reinforce the same local pivot: add late-layer capacity first, then test legal adaptation on that richer surface
- Next action for the following cycle:
  - launch `candidate/mar27/ve-scales-l8-d12` with the exact no-update anchor env from the queue above
  - require `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, `matched_params:2`, `adapted_chunks:0`, `update_steps:0`, and `stride_active:1` before trusting the result
  - only if that anchor stays near-baseline and under `16000000` bytes should the following cycle spend the matched adapting VE-scales-only run

### Cycle Review Snapshot (Apr 9, 2026 r75 Banked-State Carrier Anchor Gate Failure)

This cycle completed the first full run on `candidate/mar27/banked-momentum-carry-r1`. The branch is valid and the new state-carry marker really activates, but the no-update anchor drifts too far to justify spending the matched adapting run.

- Promoted branch/root state:
  - `HEAD` has been returned to the promoted codebase `ed4be84` on `autoresearch/mar27`
  - the completed run was launched from candidate branch `candidate/mar27/banked-momentum-carry-r1` at commit `9c30c99`
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout training and both long score-first eval passes
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - chunk-reset no-update anchor `r73`: `logs/apr08_r73_xsa_last9_scorefirst_cvcarrier_r1_chunkreset_anchor_lr0001.txt` with exact `1.17553922` at `15943958` bytes
  - plain rank-1 carrier anchor `r70`: `logs/apr07_r70_xsa_last9_scorefirst_cvcarrier_r1_anchor_lr0001.txt` with exact `1.17495154` at `15946190` bytes
  - fresh state-banked carry anchor `r75`: `logs/apr09_r75_xsa_last9_scorefirst_cvcarrier_r1_bankedstate8_anchor_lr0001.txt` with exact `1.17679366` at `15947144` bytes
- Interpretation of `r75`:
  - the persisted log confirms the intended new mechanism is live on the right substrate: `matched_params=768`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, `opt_state_banks=8`, `state_carry:banked`, and `xsa:last_n:9`
  - pre-quant landed at exact `1.1688`, and the final post-quant roundtrip landed at exact `1.17679366`
  - `r75` is `+0.00125444` bpb versus the matched `r73` anchor, `+0.00184212` versus the plain rank-1 anchor `r70`, and `+0.00313258` versus best local `r64`
  - total submission size is `15947144` bytes, leaving `52856` bytes of headroom; the branch is valid but `3186` bytes larger than `r73`
  - conclusion: banked optimizer-state carry alone does not preserve a strong enough no-update carrier anchor, so the exact branch is closed before adaptation and stays archived
- Competitive reference carried forward from the official GitHub materials:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - the strongest transferable gap still looks like adaptive-surface choice plus richer parameter banking, not just optimizer-state routing on the existing carrier
- Next action for the following cycle:
  - keep promoted `ed4be84` unchanged and archive `candidate/mar27/banked-momentum-carry-r1`
  - open the next candidate on a true parameter-banked carrier surface or pivot to a tiny VE / VE-scales-only adaptive surface
  - do not spend the matched adapting run on the exact `r75` branch unchanged

### Cycle Review Snapshot (Apr 9, 2026 r74 Chunk-Reset Carrier Closure)

This cycle completed the queued matched adapting probe on `candidate/mar27/banked-momentum-reset-r1`. The run is valid, the adaptation is real, and the result is decisively bad enough to close the exact reset-only carrier rewrite.

- Promoted branch/root state:
  - `HEAD` has been returned to the promoted codebase `ed4be84` on `autoresearch/mar27`
  - the completed run was launched from candidate branch `candidate/mar27/banked-momentum-reset-r1` at commit `5430230`
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout training and both long score-first eval passes
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - chunk-reset no-update anchor `r73`: `logs/apr08_r73_xsa_last9_scorefirst_cvcarrier_r1_chunkreset_anchor_lr0001.txt` with exact `1.17553922` at `15943958` bytes
  - fresh chunk-reset adapting result `r74`: `logs/apr09_r74_xsa_last9_scorefirst_cvcarrier_r1_chunkreset_epochs1_lr0001.txt` with exact `2.23678186` at `15937900` bytes
  - prior code-changing carrier misses remain `r71` plain-SGD rank-1 at `2.11417331` and `r72` bounded Muon at `2.25739178`
- Interpretation of `r74`:
  - the persisted log confirms the intended adaptation really happened on the right substrate: `matched_params=768`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, `reset_opt_state_per_chunk:1`, and `xsa:last_n:9`
  - pre-quant already collapsed to exact `2.2190`, and the final post-quant roundtrip landed at exact `2.23678186`
  - `r74` is `+1.06124264` bpb versus the matched `r73` anchor, `+1.06312078` versus best local `r64`, and `+0.12260855` worse than the plain-SGD rank-1 failure `r71`; it is only `0.02060992` better than bounded Muon `r72`
  - total submission size is `15937900` bytes, leaving `62100` bytes of headroom; the branch remains valid but not competitive
  - conclusion: resetting optimizer state before each score-first chunk does not rescue the rank-1 carrier path; the exact reset-only carrier rewrite is closed and must not be rerun unchanged
- Competitive reference carried forward from the official GitHub materials:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - the strongest transferable gap still looks like optimizer/state-carry quality plus adaptive-surface choice: our local path now has three independent carrier-side adapting failures (`r71`, `r72`, `r74`) despite real legal adaptation
- Next action for the following cycle:
  - keep promoted `ed4be84` unchanged and archive `candidate/mar27/banked-momentum-reset-r1`
  - open the next candidate on a materially different carrier update rule such as parameter banking or true banked-momentum state carry
  - if that rewrite cannot hold a near-baseline no-update anchor, pivot immediately to a tiny VE or VE-scales-only adaptive surface instead of another carrier-side rerun

### Cycle Review Snapshot (Apr 9, 2026 Strategic Review + r74 Launch Hygiene)

This cycle intentionally does not complete a new full experiment. The next queued adapting probe on `candidate/mar27/banked-momentum-reset-r1` remains the right experiment, but the cycle closes as a strategy refresh after verifying the public board, re-reading the decisive local evidence, and catching an early startup mislaunch before it consumed a multi-hour invalid comparison.

- Promoted branch/root state:
  - promoted source of truth remains `autoresearch/mar27` at `ed4be84`
  - the active candidate remains `candidate/mar27/banked-momentum-reset-r1` at `5430230`
  - Seeweb ends this review fully idle and uncontaminated
- Launch-hygiene conclusion:
  - an Apr 9 startup attempt for the matched adapting probe was aborted immediately after the log printed `xsa:last_n:0`; that means the intended all-layer XSA substrate was missing, so the attempt is not a valid experiment and must not be logged in `results.tsv`
  - the next valid full run on this candidate must explicitly export `XSA_LAST_N=9` and confirm that marker before the run is allowed to continue
- Local evidence carried forward:
  - best local keep remains `r64` at exact `1.17366108` and `15926537` bytes
  - chunk-reset no-update gate remains `r73` at exact `1.17553922` and `15943958` bytes with persisted `reset_opt_state_per_chunk:1`
  - plain-SGD rank-1 carrier failure remains `r71` at exact `2.11417331`; bounded Muon remains worse at exact `2.25739178` in `r72`
  - interpretation: resetting score-first optimizer state is still the only unspent carrier-side code change with a near-baseline no-update anchor; if its matched adapting run fails, the branch family should close
- Competitive refresh from the official GitHub materials on Apr 9, 2026:
  - public leader unchanged at exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result unchanged at exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - the strongest transferable gap remains optimizer/surface quality: public legal-TTT still pairs richer late-layer capacity with Parameter Banking and Parallel Muon, while our local path still lacks a surviving adaptive update rule
- Next action for the following cycle:
  - launch the matched adapting `r74` probe on `candidate/mar27/banked-momentum-reset-r1` with explicit `XSA_LAST_N=9`
  - if `r74` still collapses materially, pivot immediately to parameter-banked carrier updates or a tiny VE-derived adaptive surface
  - keep the carrier up/down split only as a last diagnostic, not as the main branch of search

### Cycle Review Snapshot (Apr 8, 2026 r73 Chunk-Reset Carrier Anchor Gate)

This cycle completed the first full run on a new code-changing candidate branch after the bounded-Muon failure. The branch does not improve the promoted best, but its no-update anchor stays close enough to baseline that one matched adapting test is now justified.

- Promoted branch/root state:
  - `HEAD` has been returned to the promoted codebase `ed4be84`
  - the completed run was launched from candidate branch `candidate/mar27/banked-momentum-reset-r1` at commit `5430230`
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - matched rank-1 carrier no-update anchor `r70`: `logs/apr07_r70_xsa_last9_scorefirst_cvcarrier_r1_anchor_lr0001.txt` with exact `1.17495154` at `15946190` bytes
  - bounded-Muon adapting failure `r72`: `logs/apr08_r72_xsa_last9_scorefirst_cvcarrier_r1_muon_bound010_lr00025.txt` with exact `2.25739178` at `15942837` bytes
  - fresh chunk-reset no-update anchor `r73`: `logs/apr08_r73_xsa_last9_scorefirst_cvcarrier_r1_chunkreset_anchor_lr0001.txt` with exact `1.17553922` at `15943958` bytes
- Interpretation of `r73`:
  - the persisted log confirms the intended branch-specific knob is active: `reset_opt_state_per_chunk:1`
  - the no-update anchor also preserves the expected score-first evidence: `matched_params=768`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, and `xsa:last_n:9`
  - pre-quant landed at exact `1.1676`, and the final post-quant roundtrip landed at exact `1.17553922`
  - `r73` is `+0.00058768` bpb versus the matched `r70` anchor, `+0.00187814` versus best local `r64`, and `+0.00096231` versus plain `r61`
  - total submission size is `15943958` bytes, leaving `56042` bytes of headroom and shrinking the artifact by `2232` bytes versus `r70`
  - conclusion: resetting optimizer state before each score-first chunk does not by itself improve the promoted baseline, but it does preserve a near-baseline no-update carrier anchor after the `r72` Muon failure, so the branch earns one matched adapting run before we consider the rewrite closed
- Competitive reference carried forward from the official GitHub materials:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r64` as the overall best local reference and keep promoted `ed4be84` unchanged
  - keep `r73` as the authoritative no-update gate for `candidate/mar27/banked-momentum-reset-r1`
  - spend the next run on the matched adapting probe with the same branch/harness and `TTT_EPOCHS=1`
  - if that adapting probe still collapses materially, pivot away from this exact carrier rewrite toward parameter-banked updates or a smaller VE-derived adaptive surface

### Cycle Review Snapshot (Apr 8, 2026 r71 Rank-1 Carrier Adapting Closure + Optimizer Pivot)

This cycle completed the queued matched adapting rank-1 `FastWeightCarrier` rescue on the strongest local score-first substrate. The result resolves the last scheduled env-only plain-SGD carrier question on promoted `ed4be84`.

- Promoted branch/root state:
  - `HEAD` remains the promoted codebase `ed4be84`
  - the completed run was an env-only experiment on `autoresearch/mar27`, so no candidate branch or merge was needed
  - the completed run was local on Seeweb only, with exclusive GPU ownership throughout the experiment
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - matched rank-1 carrier no-update anchor `r70`: `logs/apr07_r70_xsa_last9_scorefirst_cvcarrier_r1_anchor_lr0001.txt` with exact `1.17495154` at `15946190` bytes
  - matched rank-1 carrier adapting result `r71`: `logs/apr07_r71_xsa_last9_scorefirst_cvcarrier_r1_epochs1_lr0001.txt` with exact `2.11417331` at `15944673` bytes
- Interpretation of `r71`:
  - the persisted log confirms the intended dedicated surface really adapted: `matched_params=768`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, and `xsa:last_n:9`
  - pre-quant landed at exact `2.1010` and the final post-quant roundtrip landed at exact `2.11417331`
  - `r71` is `+0.93922177` bpb versus the matched `r70` anchor and `+0.94051223` versus best local `r64`
  - total submission size is `15944673` bytes, leaving `55327` bytes of headroom; the run is valid
  - conclusion: Rank-1 FastWeightCarrier adaptation still collapses under plain score-first SGD: final post-quant val_bpb 2.11417331, which is +0.93922177 versus the matched r70 anchor and +0.94051223 versus best local r64.
- Competitive reference carried forward from the official GitHub materials:
  - current public leader remains exact `1.11473509` (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`)
  - best public legal-TTT result remains exact `1.11937967` (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`)
  - our exact gap from `r64` therefore remains `0.05892599` bpb to the public leader and `0.05428141` to the best public legal-TTT line
- Next action for the following cycle:
  - keep `r64` as the overall best local reference unless a better rerun appears
  - keep `r70` as the authoritative rank-1 no-update anchor and `r71` as closure of the matched rank-1 adapting line
  - open the next code-changing cycle on a candidate branch for bounded Muon / parameter-banked online updates on the rank-1 carrier surface
  - if one more env-only diagnostic is still desired before the optimizer rewrite, isolate `blocks.8.attn.c_v_carrier.up` versus `.down` with matched no-update/adapting pairs and stop there

### Cycle Review Snapshot (Apr 10, 2026 r77 VE Scale-Only Adapting Closure)

This cycle completed the queued matched adapting VE scale-only probe on `candidate/mar27/ve-scales-l8-d12`. The run proves the minimal VE surface really adapts under the legal score-first harness, but it collapses by nearly `+0.94` bpb versus its matched anchor and does not earn promotion.

- Promoted branch/root state:
  - `HEAD` returns to the promoted codebase `ed4be84`
  - the completed run launched from candidate branch `candidate/mar27/ve-scales-l8-d12` at `f8005bc`
  - the branch stays archived and unmerged; Seeweb remained the only active lane
- Authoritative local references are now:
  - best kept local run `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - matched VE no-update anchor `r76`: `logs/apr09_r76_xsa_last9_ve12_l8_scales_anchor.txt` with exact `1.17394549` at `15960289` bytes
  - matched VE adapting result `r77`: `logs/apr09_r77_xsa_last9_ve12_l8_scales_epochs1.txt` with exact `2.11311309` at `15976965` bytes
- Interpretation of `r77`:
  - the persisted log confirms real legal adaptation on the intended tiny surface: `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, `matched_params=2`, `adapted_chunks=1892`, `update_steps=968704`, and `stride_active=1`
  - pre-quant already lands at exact `2.1050`, and the final post-quant roundtrip lands at exact `2.11311309`
  - `r77` is `0.93916760` bpb worse than matched `r76`, `0.93945201` worse than best local `r64`, and `0.93853618` worse than plain `r61`
  - the artifact remains valid at `15976965` bytes, but that leaves only `23035` bytes of headroom
  - score-first eval remains operationally expensive even on this two-scalar surface: `10486139 ms` pre-quant and `10529323 ms` post-quant
  - conclusion: the exact VE12 scale-only score-first adapting line is closed on `f8005bc`; the surface is real, but the current VE surface/update-rule pairing is structurally bad
- Next action for the following cycle:
  - keep promoted `ed4be84` unchanged and keep `r64` as the overall best local reference
  - keep `r76` as the VE no-update evidence and `r77` as closure of the exact VE12 adapting line
  - reopen VE only through a code-changing rewrite: either a smaller or more local VE adaptive surface, or a materially constrained VE update rule, each with a fresh no-update anchor before any adapting claim
  - if the VE rewrite is not ready, pivot the next candidate branch to true parameter-banked carrier weights or bank-local adaptive matrices rather than another env-only VE or carrier rerun

### Cycle Review Snapshot (Apr 10, 2026 r78 VE Local-Only Layer-Scale Promotion)

This cycle completed the first full run on `candidate/mar27/ve-layer-scale-only-l8-d12`. The branch is valid, the intended one-parameter local VE surface is real, and the no-update anchor beats the previous global best, so the candidate is promoted.

- Promoted branch/root state:
  - `HEAD` fast-forwards from `ed4be84` to `356fc32` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/ve-layer-scale-only-l8-d12` at `356fc32`
  - the branch is now merged because it improved the best global result under the size limit
- Authoritative local references are now:
  - new best local run `r78`: `logs/apr10_r78_xsa_last9_ve12_l8_layerscaleonly_anchor.txt` with exact `1.17353119` at `15967622` bytes
  - previous best keep `r64`: `logs/apr06_r64_xsa_last9_scorefirst_attnscale_anchor.txt` with exact `1.17366108` at `15926537` bytes
  - previous VE anchor `r76`: `logs/apr09_r76_xsa_last9_ve12_l8_scales_anchor.txt` with exact `1.17394549` at `15960289` bytes
- Interpretation of `r78`:
  - the persisted log confirms the intended local-only surface is active on the right substrate: `matched_params=1`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, `shared_scale:fixed0.1001`, and `xsa:last_n:9`
  - pre-quant landed at exact `1.1665`, and the final post-quant roundtrip landed at exact `1.17353119`
  - `r78` is `-0.00012989` bpb versus best local `r64` and `-0.00041430` versus the older two-scale VE anchor `r76`
  - total submission size is `15967622` bytes, leaving `32378` bytes of headroom; the run is valid but tighter on bytes than `r64`
  - conclusion: fixing the shared VE bridge amplitude and leaving only local `ve_layer_scales` as the score-first surface improves the no-update VE substrate enough to become the new promoted codebase
- Next action for the following cycle:
  - keep promoted `356fc32` and `r78` as the new authoritative reference
  - spend one matched adapting run on the promoted branch with `TTT_PARAM_PATTERNS=ve_layer_scales` and the same score-first harness
  - if that one-parameter adapting run lands more than `+0.20` bpb worse than `r78`, close plain-SGD local-only VE and pivot immediately to clipped/log-parameterized VE updates or another bank-local adaptive surface

### Cycle Review Snapshot (Apr 10, 2026 r79 VE Local-Only Layer-Scale Adapting Closure)

This cycle completed the queued matched adapting run on promoted `356fc32`. The adaptation is real, the artifact stays valid, but the exact plain-SGD local-only VE line collapses badly enough to close it.

- Promoted branch/root state:
  - `HEAD` remains promoted `356fc32` on `autoresearch/mar27`
  - the completed run launched on the promoted branch as an env-only Seeweb experiment, so no candidate branch or merge was needed
  - Seeweb remained the only active lane with exclusive GPU ownership throughout the cycle
- Authoritative local references are now:
  - best local run `r78`: `logs/apr10_r78_xsa_last9_ve12_l8_layerscaleonly_anchor.txt` with exact `1.17353119` at `15967622` bytes
  - matched adapting local-only VE result `r79`: `logs/apr10_r79_xsa_last9_ve12_l8_layerscaleonly_epochs1.txt` with exact `2.12725907` at `15975943` bytes
  - older two-scale VE adapting closure `r77`: `logs/apr09_r77_xsa_last9_ve12_l8_scales_epochs1.txt` with exact `2.11311309` at `15976965` bytes
- Interpretation of `r79`:
  - the persisted logs confirm real legal adaptation on the intended one-parameter surface: `matched_params:1`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, `shared_scale:fixed0.1001`, and `xsa:last_n:9`
  - pre-quant already lands at exact `2.1168`, and the final post-quant roundtrip lands at exact `2.12725907`
  - `r79` is `0.95372788` bpb worse than matched `r78`, and the artifact stays valid at `15975943` bytes with only `24057` bytes of headroom
  - the run is also operationally expensive: pre-quant score-first eval took `10523098 ms`, and the final post-quant score-first eval took `10552201 ms`
  - conclusion: exact plain-SGD local-only `ve_layer_scales` adaptation is closed on promoted `356fc32`; do not rerun it unchanged
- Competitive reference refreshed against the official GitHub materials on Apr 10, 2026:
  - current public leader verified in the official README is `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`)
  - the new public legal-TTT leader couples Legal TTT with 3-layer recurrence, parallel residuals, and `QK-Gain 5.25`, so our best local gap from `r78` is now about `0.0925` bpb
  - compared with that public line, our immediate blocker remains update-rule quality on a tiny adaptive surface, not proof that adaptation activates
- Next action for the following cycle:
  - keep promoted `356fc32` and `r78` as the authoritative reference
  - close env-only plain-SGD local-only VE on the promoted branch
  - open a fresh candidate branch for a materially constrained update rule on the same `ve_layer_scales` surface, preferably log-parameterized or delta-clipped, and require a matched `TTT_EPOCHS=0` anchor before the next adapting claim
  - if that rewrite is not ready, pivot to bank-local adaptive matrices or parameter-banked carrier weights rather than another env-only VE rerun

### Cycle Review Snapshot (Apr 10, 2026 r80 Delta-Clipped VE Local-Only Promotion)

This cycle completed the required no-update anchor for a materially constrained update-rule rewrite on the same one-parameter `ve_layer_scales` surface. The candidate stays fully valid, matches the expected runtime profile, and improves the promoted best enough to earn promotion immediately.

- Promoted branch/root state:
  - `HEAD` fast-forwards from `356fc32` to `1bd7c9b` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/ve-layerscale-deltaclip` at `1bd7c9b`
  - the branch is now merged because it improved the best global result under the size limit
- Authoritative local references are now:
  - new best local run `r80`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - previous best local run `r78`: `logs/apr10_r78_xsa_last9_ve12_l8_layerscaleonly_anchor.txt` with exact `1.17353119` at `15967622` bytes
  - plain adapting closure `r79`: `logs/apr10_r79_xsa_last9_ve12_l8_layerscaleonly_epochs1.txt` with exact `2.12725907` at `15975943` bytes
- Interpretation of `r80`:
  - the persisted logs confirm the intended rewrite is active without changing the legal no-update harness: `matched_params:1`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, `delta_budget_ratio:0.100`, `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, and `xsa:last_n:9`
  - pre-quant landed at exact `1.1664`, and the final post-quant roundtrip landed at exact `1.17309950`
  - `r80` is `-0.00043169` bpb versus promoted `r78` and `-0.00056158` versus older best `r64`
  - total submission size is `15968741` bytes, leaving `31259` bytes of headroom; the run is valid and only `1119` bytes larger than `r78`
  - runtime stays effectively neutral versus the previous anchor: pre-quant score-first eval took `832316 ms` and the final post-quant score-first eval took `828223 ms`
  - conclusion: constraining cumulative TTT delta on the same one-parameter local VE surface improves the no-update substrate enough to become the new promoted codebase
- Competitive reference refreshed against the official GitHub materials on Apr 10, 2026:
  - current public leader verified in the official README remains `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`)
  - the local gap from promoted `r80` is now about `0.0921` bpb
  - compared with that public line, our immediate blocker is still whether the constrained update rule stabilizes real adaptation, not substrate strength
- Next action for the following cycle:
  - keep promoted `1bd7c9b` and `r80` as the authoritative reference
  - spend one matched adapting run on the promoted branch with the same score-first harness plus `TTT_EPOCHS=1 GRAD_CLIP_NORM=1.0 TTT_DELTA_BUDGET_RATIO=0.10`
  - count that adapting run as valid only if the persisted log again shows `matched_params:1`, `adapted_chunks>0`, `update_steps>0`, `stride_active:1`, and `delta_budget_ratio:0.100`
  - if that adapting delta-clipped run lands more than `+0.20` bpb worse than `r80`, close this exact delta-clipped VE line and pivot immediately to log-parameterized VE updates or bank-local adaptive matrices

### Cycle Review Snapshot (Apr 11, 2026 r81 Delta-Clipped VE Adapting Closure)

This cycle closes the queued matched adapting run on promoted `1bd7c9b`. The adaptation is real and the artifact stays valid, but the exact delta-clipped env-only VE line still collapses badly enough that the family is now closed until there is a material code change.

- Promoted branch/root state:
  - `HEAD` remains promoted `1bd7c9b` on `autoresearch/mar27`
  - the completed run launched on the promoted branch as an env-only Seeweb experiment, so no candidate branch or merge was needed
  - the completed run used exclusive local GPU ownership and emitted final roundtrip metrics in `logs/apr10_r81_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_epochs1.txt`
- Authoritative local references are now:
  - promoted no-update anchor `r80`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - matched adapting closure `r81`: `logs/apr10_r81_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_epochs1.txt` with exact `2.13337643` at `15980955` bytes
  - previous plain local-only VE adapting closure `r79`: `logs/apr10_r79_xsa_last9_ve12_l8_layerscaleonly_epochs1.txt` with exact `2.12725907` at `15975943` bytes
- Interpretation of `r81`:
  - the persisted logs confirm the intended legal score-first path really adapted: `matched_params:1`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, `delta_budget_ratio:0.100`, `ve:enabled=True dim:12 requested_layers:8 active_layers:[8]`, and `xsa:last_n:9`
  - pre-quant already collapsed to exact `2.1225`, and the final post-quant int8+zlib roundtrip landed at exact `2.13337643`
  - `r81` is `+0.96027693` bpb versus the matched `r80` anchor, slightly worse than the earlier plain local-only VE failure `r79`, and decisively beyond the `+0.20` structural-break rule
  - total submission size is `15980955` bytes, leaving only `19045` bytes of headroom; the run is valid on size but not competitive on metric
  - conclusion: delta clipping improves the no-update VE substrate, but it does not rescue real env-only adaptation on the exact 1-parameter local VE surface; this exact env-only line is now closed
- Competitive reference carried forward from the Apr 10, 2026 official refresh:
  - current public leader remains `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`)
  - the local gap from promoted `r80` stays about `0.0921` bpb, and `r81` confirms that our blocker is still update-rule quality under real adaptation rather than the anchor substrate itself
- Next action for the following cycle:
  - keep promoted `1bd7c9b` and `r80` as the authoritative reference
  - do not rerun `r81` unchanged or spend another env-only VE adapting retry on the same surface
  - open a fresh candidate branch for a material rewrite such as log-parameterized VE updates or a learned update-scale/clamp on the same 1-parameter surface, and require a matched `TTT_EPOCHS=0` anchor before the next adapting claim
  - if that rewrite is not ready, pivot to bank-local adaptive matrices or another materially different low-byte adaptive surface rather than reopening carrier or VE env-only reruns

### Cycle Review Snapshot (Apr 11, 2026 r82 Log-Param VE Anchor Keep + Adapting Gate)

This cycle spent the required no-update anchor on a fresh code-changing VE rewrite. The branch is valid, the intended log-space rewrite is active, and the anchor stays close enough to the promoted best to justify one matched adapting follow-up, but it does not beat the current promoted codebase.

- Promoted branch/root state:
  - `HEAD` returns to promoted `1bd7c9b` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/ve-logscale-l8-d12` at `69e8be8`
  - the branch stays unmerged because the final post-quant metric does not beat promoted `r80`
- Authoritative local references are now:
  - promoted no-update anchor `r80`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - fresh log-param no-update anchor `r82`: `logs/apr11_r82_xsa_last9_ve12_l8_logscale_anchor.txt` with exact `1.17333784` at `15966165` bytes
  - matched failure on the previous exact promoted line `r81`: `logs/apr10_r81_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_epochs1.txt` with exact `2.13337643` at `15980955` bytes
- Interpretation of `r82`:
  - the persisted logs confirm the intended rewrite is really active on the same tiny surface: `scale_mode:log_exp`, `matched_params:1`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, `delta_budget_ratio:0.100`, `active_layers:[8]`, and `xsa:last_n:9`
  - pre-quant improves to exact `1.1661`, and the final post-quant int8+zlib roundtrip lands at exact `1.17333784`
  - `r82` is `+0.00023834` bpb versus promoted `r80`, but the artifact shrinks by `2576` bytes to `15966165`, leaving `33835` bytes of headroom
  - conclusion: log-parameterizing the same 1-parameter local VE surface improves the unquantized anchor and byte headroom, but it gives back enough under quantization that promotion is not yet justified
- Competitive reference refreshed against the official README on Apr 11, 2026:
  - current public leader remains `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`)
  - the local gap from promoted `r80` therefore remains about `0.0921` bpb
- Next action for the following cycle:
  - keep promoted `1bd7c9b` and `r80` as the authoritative reference
  - spend one matched adapting run on `candidate/mar27/ve-logscale-l8-d12` with the same score-first harness plus `TTT_EPOCHS=1 GRAD_CLIP_NORM=1.0`
  - if that adapting run lands more than `+0.20` bpb worse than `r82`, close exact log-param VE and pivot to a learned update-scale/clamp or another materially different low-byte adaptive surface


### Cycle Review Snapshot (Apr 11, 2026 r83 Log-Param VE Adapting Failure + Family Closure)

This cycle completed the queued matched adapting follow-up on the log-parameterized VE candidate branch. The intended legal adaptation is real, the artifact stays under 16,000,000 bytes, but the exact log-param local-only VE line still collapses badly enough that the branch stays archived and this env-only family is now closed until a material code change.

- Promoted branch/root state:
  - `HEAD` returns to promoted `1bd7c9b` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/ve-logscale-l8-d12` at `69e8be8`
  - the branch stays unmerged because the final post-quant metric does not beat promoted `r80`
- Authoritative local references are now:
  - promoted best `r80`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - matched log-param anchor `r82`: `logs/apr11_r82_xsa_last9_ve12_l8_logscale_anchor.txt` with exact `1.17333784` at `15966165` bytes
  - matched log-param adapting failure `r83`: `logs/apr11_r83_xsa_last9_ve12_l8_logscale_epochs1_clip1.txt` with exact `2.10794132` at `15975850` bytes
- Interpretation of `r83`:
  - the persisted logs confirm the intended rewrite really adapted on the same tiny surface: `scale_mode:log_exp`, `matched_params:1`, `adapted_chunks=1892`, `update_steps=968704`, `stride_active=1`, `delta_budget_ratio:0.100`, `active_layers:[8]`, and `xsa:last_n:9`
  - pre-quant already collapses to exact `2.0941`, and the final post-quant int8+zlib roundtrip lands at exact `2.10794132`
  - `r83` is `+0.93460348` bpb versus matched `r82` and `+0.93484182` versus promoted `r80`
  - total submission size is `15975850` bytes, leaving `24150` bytes of headroom; the run is valid on size but not competitive on metric
  - conclusion: log-parameterizing the same 1-parameter local VE surface does not rescue real env-only adaptation; the exact log-param VE line is now closed and the next cycle must pivot to a materially different update rule or adaptive surface
- Competitive reference carried forward from the Apr 11, 2026 official refresh:
  - current public leader remains `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`)
  - the local gap from promoted `r80` therefore stays about `0.0921` bpb, and `r83` confirms the blocker is still update-rule/surface quality under real adaptation rather than no-update anchor quality
- Next action for the following cycle:
  - keep promoted `1bd7c9b` and `r80` as the authoritative reference
  - archive `candidate/mar27/ve-logscale-l8-d12` without merge
  - do not rerun exact log-param local-only VE adaptation unchanged
  - pivot the next code-changing cycle to a learned update-scale/clamp on the VE-local surface or a tiny bank-local adaptive matrix, with a fresh `TTT_EPOCHS=0` anchor before any new adapting claim

### Cycle Review Snapshot (Apr 11, 2026 r84 Learned-Gain VE Anchor Keep + Adapting Gate)

This cycle spent the required fresh no-update anchor on a material learned update-scale rewrite for the local VE surface. The learned bounded gain really trains and the artifact stays valid under 16,000,000 bytes, but the final post-quant anchor still does not beat the promoted codebase, so the candidate remains archived.

- Promoted branch/root state:
  - `HEAD` returns to promoted `1bd7c9b` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/ve-learned-updategain-l8-d12` at `96ad3a5`
  - the branch stays unmerged because the final post-quant metric does not beat promoted `r80`
- Authoritative local references are now:
  - promoted best `r80`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - strongest archived no-update alternate `r82`: `logs/apr11_r82_xsa_last9_ve12_l8_logscale_anchor.txt` with exact `1.17333784` at `15966165` bytes
  - fresh learned-gain anchor `r84`: `logs/apr11_r84_xsa_last9_ve12_l8_updategain_anchor.txt` with exact `1.17430575` at `15966809` bytes
- Interpretation of `r84`:
  - the persisted logs confirm the intended rewrite is really active on the same tiny surface: `update_gain_mode:bounded_sigmoid`, `ve:update_gains_final:1.5643`, `matched_params:1`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, `delta_budget_ratio:0.100`, `active_layers:[8]`, and `xsa:last_n:9`
  - pre-quant lands at exact `1.1666`, and the final post-quant int8+zlib roundtrip lands at exact `1.17430575`
  - `r84` is `+0.00120625` bpb versus promoted `r80` and `+0.00096791` versus archived log-param anchor `r82`
  - total submission size is `15966809` bytes, leaving `33191` bytes of headroom; the run is valid on size and stays near-baseline, but not good enough to promote
  - conclusion: a learned bounded update gain can be trained on the 1-parameter VE surface and stays operationally clean, but quantization retention still loses enough that the anchor remains below both `r80` and `r82`
- Competitive reference carried forward from the Apr 11, 2026 official refresh:
  - current public leader remains `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`), which combines legal score-first TTT with parallel residuals and deeper recurrence
  - the local gap from promoted `r80` therefore stays about `0.0921` bpb, and `r84` suggests that future self-learning gains likely need a stronger adaptive surface or more competitor-like routing than a scalar-only rewrite
- Next action for the following cycle:
  - keep promoted `1bd7c9b` and `r80` as the authoritative reference
  - archive `candidate/mar27/ve-learned-updategain-l8-d12` without merge
  - spend one matched adapting run on `candidate/mar27/ve-learned-updategain-l8-d12` with the same harness plus `TTT_EPOCHS=1 GRAD_CLIP_NORM=1.0`
  - if that adapting run lands more than `+0.20` bpb worse than `r84`, close learned-gain VE and pivot to a tiny bank-local adaptive matrix or a competitor-inspired parallel-residual/lane-merge self-learning surface

### Cycle Review Snapshot (Apr 11, 2026 r85 Learned-Gain VE Adapting Failure + Family Closure)

This cycle completed the queued matched adapting follow-up on `candidate/mar27/ve-learned-updategain-l8-d12`. The learned-gain rewrite really adapts and slightly improves the catastrophic result versus the prior log-param failure, but it still collapses by `+0.92463529` versus its matched `r84` anchor, so the exact scalar-only learned-gain VE family is now closed.

- Promoted branch/root state:
  - `HEAD` returns to promoted `1bd7c9b` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/ve-learned-updategain-l8-d12` at `96ad3a5`
  - the branch stays unmerged and archived because the final post-quant metric is catastrophically worse than the promoted codebase
- Authoritative local references are now:
  - promoted best `r80`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - matched learned-gain anchor `r84`: `logs/apr11_r84_xsa_last9_ve12_l8_updategain_anchor.txt` with exact `1.17430575` at `15966809` bytes
  - matched learned-gain adapting failure `r85`: `logs/apr11_r85_xsa_last9_ve12_l8_updategain_epochs1_clip1.txt` with exact `2.09894104` at `15977526` bytes
- Interpretation of `r85`:
  - the persisted logs confirm real legal adaptation on the intended surface: `update_gain_mode:bounded_sigmoid`, `ve:update_gains_final:1.5281`, `matched_params:1`, `adapted_chunks:1892`, `update_steps:968704`, `stride_active:1`, `delta_budget_ratio:0.100`, and `xsa:last_n:9`
  - pre-quant already collapses to exact `2.0861`, and the final post-quant int8+zlib roundtrip lands at exact `2.09894104`
  - `r85` is `+0.92463529` bpb versus matched `r84`, `+0.92584154` versus promoted `r80`, and only `-0.00900028` better than the earlier log-param failure `r83`
  - total submission size is `15977526` bytes, leaving `22474` bytes of headroom; the run is valid on size but nowhere near competitive
  - conclusion: learned bounded update gain does not rescue real adaptation on the exact 1-parameter VE surface; this exact scalar-only learned-gain line is now closed
- Competitive reference refreshed against the official README on Apr 11, 2026:
  - the public leader remains `1.0810` on `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`
  - the strongest public line still combines SP8192, multi-layer recurrence, parallel residuals, QK-gain `5.25`, and legal score-first TTT
  - the immediate local blocker remains adaptive-surface quality: repeated scalar-only VE rewrites keep producing real adaptation with catastrophically bad metrics
- Next action for the following cycle:
  - keep promoted `1bd7c9b` and `r80` as the authoritative reference
  - archive `candidate/mar27/ve-learned-updategain-l8-d12` without merge and do not rerun scalar-only VE adaptation unchanged
  - priority 1: open `candidate/mar27/tiny-bank-local-matrix-l8` for a tiny bank-local adaptive matrix on the latest layer; the first run must be a matched `TTT_EPOCHS=0` anchor
  - priority 2: if the matrix slice is not ready, open `candidate/mar27/parallel-residual-lanemix` for a small parallel-residual lane mixer whose no-update anchor must first clear the near-baseline gate
  - priority 3: if a richer lane surface is cheaper than a matrix, open `candidate/mar27/recur-gate-selflearn` and expose only a tiny recurrence or lane-gate parameter to score-first TTT, again anchor first
  - priority 4: if bytes get tight, prefer a 2-parameter or 4-parameter late-lane mix over reopening carrier or another scalar-only VE rewrite
  - priority 5: do not reopen carrier, plain local-only VE, delta-clipped VE, log-param VE, or learned-gain VE env-only reruns without another material code change


## The Experiment Loop

LOOP FOREVER:

1. Look at the git state and results.tsv to understand what has been tried and what worked.
2. Keep the orchestration brain on `Seeweb`.
   - All planning, analysis, competitor study, email sending, branch promotion, and queue rewriting happen locally.
   - `Runpod` may only execute training/eval jobs and return artifacts/logs.
3. Choose the backend for the next run.
   - Default to `Seeweb`.
   - If `Runpod` is configured, healthy, and the remaining balance is sufficient, it may take a subordinate overflow experiment.
   - Never exceed one active experiment on `Seeweb`.
   - On `Runpod`, use only the number of remote pods that the cost-aware policy currently permits.
4. Before launching any `Seeweb` experiment, ensure the local GPU is fully free and no other local training job is active. If busy, wait and poll.
5. **Think about what to try next.** Consider:
   - What improved so far? Try variations of winning ideas.
   - What hasn't been explored yet?
   - Is there a size budget to spare? (if current model is 14MB, you have 2MB to add capacity)
   - Is there a size problem? (if over 16MB, reduce capacity)
   - Is this experiment helping the self-learning / inference-backprop direction or clarifying why it is failing?
6. If the experiment changes only env vars or other non-structural settings, you may stay on the promoted branch.
7. If the experiment changes the codebase itself, create a candidate branch first, make the changes there, and test from that branch.
8. Edit `train_gpt.py` with your experimental change. Keep changes **minimal and isolated** — one idea per experiment so you know what caused the improvement.
9. `git add train_gpt.py && git commit -m "description of change"`
10. Run the experiment:
   ```bash
   AR_BACKEND=auto \
   RUN_ID=autoresearch \
   ITERATIONS=20000 \
   MAX_WALLCLOCK_SECONDS=0 \
   TRAIN_LOG_EVERY=500 \
   VAL_LOSS_EVERY=0 \
   python3 autoresearch/dispatch_experiment.py > run.log 2>&1
   ```
11. Extract results:
   ```bash
   grep -E "^final_(int8_zlib|int6_lzma)_roundtrip_exact |^Total submission size (int8\\+zlib|int6\\+lzma): " run.log
   ```
12. If grep output is empty, the run crashed or the final metrics were never produced. Run `tail -n 50 run.log` to debug. If the cause is contamination from shared GPU usage, log it as `crash`, revert or abandon the candidate branch, wait for a clean GPU, and re-run the same experiment later.
13. Check both val_bpb AND size. A run with great bpb but over 16MB is invalid.
14. For self-learning experiments, also check the adaptation evidence in the logs before deciding that the idea truly worked.
15. Record results in results.tsv.
16. If the result improves over the currently promoted codebase and stays valid:
    - keep it
    - if it came from a candidate branch, merge that branch into the promoted branch
17. If the result is equal/worse or invalid:
    - do not merge it
    - if it came from a candidate branch, leave the candidate branch as an archived experiment branch and switch back to the promoted branch
    - if it was a promoted-branch env-only experiment, revert it non-destructively
18. Reflect on the result, generate the next candidate ideas, and send the mandatory status email from `Seeweb`.
19. At the end of each cycle, stop and execute the full End-of-Cycle Strategic Review before continuing.

## Experiment Ideas (prioritized)

After the mandatory queue and second-wave queue are done, explore further in roughly this order:

### Tier 1: Architecture shape (high impact, easy to test)
- Number of layers: try 10, 11, 12, 13 (more layers = more capacity but more size)
- Model dimension: try 448, 480, 512, 576 (wider = more capacity per layer)
- NUM_KV_HEADS: try 2, 4, 8 (fewer KV heads = less size, might hurt quality)
- MLP_MULT: try 1, 2, 3, 4 (controls MLP width relative to model dim)
- Find the sweet spot: maximize capacity while staying under 16MB compressed

### Tier 2: Training dynamics (medium impact)
- Learning rates: try different matrix_lr (0.02, 0.04, 0.08), scalar_lr, embed_lr
- Warmdown iterations: try 800, 1200, 2000
- Muon momentum: try 0.90, 0.95, 0.99
- Muon backend steps: try 3, 5, 7
- Batch size: try 262144, 524288, 1048576 (TRAIN_BATCH_TOKENS)
- Sequence length: try 512, 1024, 2048 (TRAIN_SEQ_LEN)

### Tier 3: Architecture innovations (medium impact, more complex)
- Add bias to linear layers (small param cost, might help)
- Change activation: try LeakyReLU², GeLU, SwiGLU instead of relu²
- Gated attention (add a gate parameter to attention output)
- Value Residual connections
- Change logit_softcap: try 20, 30, 50
- RoPE base: try 5000, 10000, 50000
- EMA (exponential moving average) of weights for evaluation

### Tier 4: Compression optimization (small but free gains)
- Tune INT8_CLIP_PERCENTILE (currently 99.99984)
- Try per-channel vs per-tensor quantization thresholds

### Tier 5: Combine winners
- Once you have individual improvements, combine them

## Important principles

- **One change at a time.** If you change 3 things and the result improves, you don't know which one helped. Keep experiments atomic.
- **One brain, local only.** The only legitimate orchestrator is the Seeweb-resident Codex session. `local_autoresearch_queue.py` is deprecated and must not be used as the main controller.
- **One run per device lane, not one static controller.** Full GPU exclusivity is mandatory on `Seeweb`; remote pods are allowed only as subordinate execution lanes chosen by the local brain.
- **The brain stays on Seeweb.** `Runpod` is only an execution lane for training/eval jobs; analysis, branching, promotion, and strategy stay local.
- **Full runs only.** For comparison runs, 20,000 steps is the minimum acceptable budget.
- **Size awareness.** Always check the submission size. The best bpb in the world is useless at 17MB.
- **Diminishing returns.** If you're getting 0.0001 improvements, try a more radical change direction.
- **Do not use runtime as a pruning heuristic in this phase.** Slow but technically valid runs are acceptable.
- **Log everything.** The human will read results.tsv to understand what worked. Write clear descriptions.
- **Runpod is budget-gated.** If the remaining remote balance is too low for a full run to plausibly finish, do not force a remote launch. Park Runpod and continue on Seeweb until credits improve or a cheaper viable shape is identified.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. Run experiments indefinitely until manually interrupted. If you run out of ideas, re-read the code, try combining near-misses, try more radical changes.

### Cycle Review Snapshot (Apr 13, 2026 r93 Bounded Q-Gain Plain Bridge Archive)

This cycle completed the queued bounded late-layer `q_gain` plain bridge on `candidate/mar27/qk-gain-bridge-l8`. The bridge learned a strong public-style gain signal on block 8 and stayed valid under the size limit, but the final post-quant score still missed promoted `r91`, so the branch stays archived.

- Promoted branch/root state:
  - `HEAD` returns to promoted `41cc8a3` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/qk-gain-bridge-l8` at `4090692`
  - the branch stays unmerged because the final post-quant metric does not beat promoted `r91`
- Authoritative local references are now:
  - promoted best `r91`: `logs/apr13_r91_xsa_last9_ve12_l8_recur_anchor.txt` with exact `1.17002954` at `15986864` bytes
  - fresh bounded `q_gain` plain bridge `r93`: `logs/apr13_r93_xsa_last9_ve12_l8_recur_qkbridge_plain.txt` with exact `1.17219573` at `15998527` bytes
  - matched recurrence adapting failure `r92`: `logs/apr13_r92_xsa_last9_ve12_l8_recur_epochs1_lr0005_clip1.txt` with exact `2.10705140` at `15990957` bytes
- Interpretation of `r93`:
  - the persisted logs confirm the intended bridge really trained on the correct substrate: `qgain_bridge:final active_layers:[8] summary:8:mix=0.7616|base_mean=2.0762|eff_mean=4.4933|eff_min=4.1288|eff_max=4.9780`, together with the expected `lanemix` and `recur` final markers
  - pre-quant lands at exact `1.1647`, and the final post-quant int8+zlib roundtrip lands at exact `1.17219573`
  - `r93` is `+0.00216619` bpb versus promoted `r91`, while the artifact remains valid at `15998527` bytes and leaves only `1473` bytes of headroom
  - conclusion: the bounded `QK-Gain 5.25` bridge is technically plausible and really trains, but this exact plain configuration gives back too much post-quant score and leaves too little size margin to promote unchanged
- Competitive reference refreshed against the official GitHub materials on Apr 13, 2026:
  - the public leader remains exact `1.0810` on `2026-04-13_SP8192_3-Layer Recurrence + Parallel Residuals + Legal TTT`
  - the next public records at exact `1.0816` and `1.0822` still center recurrence and parallel-residual routing rather than another plain scalar bridge
- Next action for the following cycle:
  - keep promoted `41cc8a3` and `r91` as the authoritative reference
  - archive `candidate/mar27/qk-gain-bridge-l8` without merge and do not rerun the exact plain bridge unchanged
  - spend the next cycle on a matched score-first no-update anchor on the same bridged branch with `TTT_ENABLE=1 TTT_PROTOCOL=score_first TTT_SCOPE=global TTT_EPOCHS=0 TTT_PARAM_PATTERNS=q_gain`, preserving the `qgain_bridge`, `lanemix`, `recur`, and `xsa:last_n:9` markers
  - if that anchor drifts materially or burns the remaining size margin, pivot the following code-changing cycle to a tighter late-layer-only `q_gain` surface or a smaller target/init before any adapting claim

### Current Queue Update Rule

- `r91` remains the promoted best run: `1.17002954` at `15986864` bytes on `41cc8a3`.
- `r92` remains the authoritative matched recurrence adapting failure: `2.10705140` at `15990957` bytes on `41cc8a3`.
- `r93` is now the authoritative bounded late-layer `q_gain` plain bridge archive: `1.17219573` at `15998527` bytes on `4090692`, with a trained final bridge marker `mix=0.7616` and `eff_mean=4.4933`.
- The next full run should stay on `candidate/mar27/qk-gain-bridge-l8` for a matched score-first no-update anchor rather than opening another plain rerun:
  - keep the exact plain bridge code from `4090692`
  - add the no-update score-first harness with `TTT_PARAM_PATTERNS=q_gain`
  - require persisted `matched_params`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, and the final `qgain_bridge` marker before counting the run as valid
- If the no-update anchor stays near `r93` and under the size limit, it earns one later adapting follow-up only after a fresh local review.
- If the no-update anchor drifts badly or leaves no realistic size margin, close the exact broad `q_gain` anchor path and pivot to a materially tighter late-layer-only `q_gain` surface before any adapting claim.

### Cycle Review Snapshot (Apr 14, 2026 r94 Q-Gain No-Update Anchor Size Closure)

This cycle completed the queued matched score-first no-update anchor on `candidate/mar27/qk-gain-bridge-l8`. The intended score-first harness stayed legal and slightly improved the plain `q_gain` bridge score, but the exact branch is now closed because the final artifact crosses the 16,000,000-byte limit.

- Promoted branch/root state:
  - `HEAD` returns to promoted `41cc8a3` on `autoresearch/mar27`
  - the completed run launched from candidate branch `candidate/mar27/qk-gain-bridge-l8` at `4090692`
  - the branch stays unmerged because the final artifact is invalid at `16001342` bytes
- Authoritative local references are now:
  - promoted best `r91`: `logs/apr13_r91_xsa_last9_ve12_l8_recur_anchor.txt` with exact `1.17002954` at `15986864` bytes
  - prior plain `q_gain` bridge `r93`: `logs/apr13_r93_xsa_last9_ve12_l8_recur_qkbridge_plain.txt` with exact `1.17219573` at `15998527` bytes
  - fresh matched no-update `q_gain` anchor `r94`: `logs/apr14_r94_xsa_last9_ve12_l8_recur_qkbridge_qgain_anchor.txt` with exact `1.17182780` at `16001342` bytes
- Interpretation of `r94`:
  - the persisted logs confirm the intended no-update harness stayed legal: `matched_params=73`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, and the expected `qgain_bridge`, `lanemix`, `recur`, and `xsa:last_n:9` markers all persisted
  - pre-quant lands at exact `1.1645`, and the final post-quant int8+zlib roundtrip lands at exact `1.17182780`
  - `r94` improves the plain bridge `r93` by `0.00036793` bpb, but still trails promoted `r91` by `0.00179826`
  - the decisive blocker is size: the final artifact is `16001342` bytes, which is `1342` bytes over the limit, so the exact broad `q_gain` anchor path cannot earn a matched adapting follow-up
  - conclusion: the branch shows the `q_gain` bridge can survive the score-first no-update harness and recover a little post-quant score, but it does so only by spending more bytes than the challenge allows
- Next action for the following cycle:
  - keep promoted `41cc8a3` and `r91` as the authoritative reference
  - archive `candidate/mar27/qk-gain-bridge-l8` without merge and do not rerun the exact broad `q_gain` bridge unchanged
  - spend the next cycle on a tighter `q_gain` variant that explicitly tries to claw back at least `1342` bytes before any adapting claim, for example a smaller `QK_BRIDGE_TARGET`, a colder `QK_BRIDGE_INIT`, or a materially narrower late-layer `q_gain` surface
  - if a tighter `q_gain` variant still cannot get back under the limit while staying near `r94`, pivot the following code-changing cycle away from broad `q_gain` and back toward a cheaper recurrence/routing bridge

### Cycle Review Snapshot (Apr 16, 2026 r95 Tight-Target Q-Gain Anchor Validity Without Score Recovery)

This cycle completed the queued tighter `q_gain` no-update anchor on `candidate/mar27/qk-gain-bridge-l8`. Lowering the bridge target to `4.75` succeeds at bringing the artifact back under the 16,000,000-byte limit, but the final post-quant score falls behind both the invalid stronger anchor `r94` and the plain bridge `r93`, so the broad `q_gain` family is valid again without earning promotion.

- Promoted branch/root state:
  - `HEAD` returns to promoted `41cc8a3` on `autoresearch/mar27`
  - the completed run launched from archived candidate branch `candidate/mar27/qk-gain-bridge-l8` at `4090692`
  - the branch stays unmerged because the valid tightened anchor still misses promoted `r91`
- Authoritative local references are now:
  - promoted best `r91`: `logs/apr13_r91_xsa_last9_ve12_l8_recur_anchor.txt` with exact `1.17002954` at `15986864` bytes
  - best-scoring broad `q_gain` anchor `r94`: `logs/apr14_r94_xsa_last9_ve12_l8_recur_qkbridge_qgain_anchor.txt` with exact `1.17182780` at `16001342` bytes (invalid)
  - plain `q_gain` bridge `r93`: `logs/apr13_r93_xsa_last9_ve12_l8_recur_qkbridge_plain.txt` with exact `1.17219573` at `15998527` bytes
  - fresh valid tightened anchor `r95`: `logs/apr16_r95_xsa_last9_ve12_l8_recur_qkbridge_qgain_t475_anchor.txt` with exact `1.17298435` at `15994452` bytes
- Interpretation of `r95`:
  - the persisted logs confirm the intended no-update harness stayed legal: `matched_params=73`, `adapted_chunks=0`, `update_steps=0`, `stride_active=1`, and the expected `qgain_bridge`, `lanemix`, `recur`, and `xsa:last_n:9` markers all persisted
  - pre-quant lands at exact `1.1649`, and the final post-quant int8+zlib roundtrip lands at exact `1.17298435`
  - `r95` recovers validity with a final artifact of `15994452` bytes, buying back `6890` bytes versus invalid `r94` and `4075` versus `r93`, leaving `5548` bytes of headroom
  - the score trade is material: `r95` is `+0.00115655` bpb worse than `r94`, `+0.00078862` worse than `r93`, and `+0.00295481` worse than promoted `r91`
  - the final bridge marker still stays aggressive (`mix=0.8242|base_mean=2.7534|eff_mean=4.3990`), so target-only tightening changed the byte footprint more than it improved the final score frontier
  - conclusion: broad `q_gain` is no longer size-closed, but a simple target drop to `4.75` overshoots and does not recover the best no-update tradeoff
- Next action for the following cycle:
  - keep promoted `41cc8a3` and `r91` as the authoritative reference
  - archive `candidate/mar27/qk-gain-bridge-l8` without merge and do not rerun either exact `r94` or exact `r95` unchanged
  - spend at most one more env-only interpolation anchor on the same branch, preferably `QK_BRIDGE_TARGET≈5.0` and/or a colder `QK_BRIDGE_INIT`, to search between invalid higher-score `r94` and valid lower-score `r95`
  - if that interpolation anchor still cannot stay near `r94` while remaining valid, pivot the following cycle to a code-changing narrower `q_gain` surface or back toward a cheaper recurrence/routing bridge

### Current Queue Update Rule (Updated Apr 16, 2026)

- `r91` remains the promoted best run: `1.17002954` at `15986864` bytes on `41cc8a3`.
- `r94` remains the best-scoring broad `q_gain` anchor, but it is invalid at `16001342` bytes on `4090692`.
- `r95` is the authoritative valid broad `q_gain` anchor result: `1.17298435` at `15994452` bytes on `4090692`, with `matched_params=73`, `adapted_chunks=0`, `update_steps=0`, and final bridge marker `mix=0.8242|base_mean=2.7534|eff_mean=4.3990`.
- Do not spend the next cycle on a matched adapting run for broad `q_gain` yet, because the valid anchor is still materially weaker than both `r94` and promoted `r91`.
- The next full run should spend at most one interpolation-style broad `q_gain` anchor before a structural pivot:
  - first preference: move toward `QK_BRIDGE_TARGET≈5.0` and/or a colder `QK_BRIDGE_INIT` to probe the score/size frontier between `r94` and `r95`
  - second preference: if that single interpolation anchor still misses, open a code-changing branch that narrows the `q_gain` write surface itself before re-running any anchor
  - only after a clearly stronger under-limit anchor exists should the queue spend another adapting `TTT_EPOCHS=1` claim on `q_gain`

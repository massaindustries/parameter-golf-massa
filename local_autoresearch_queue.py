#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEPRECATION_MESSAGE = (
    "local_autoresearch_queue.py is deprecated and must not be used as the main orchestrator. "
    "Use the Seeweb Codex tmux brain session instead."
)

if __name__ == "__main__":
    raise SystemExit(DEPRECATION_MESSAGE)


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "autoresearch" / "results.tsv"
MAIL_PATH = Path("/tmp/autoresearch_mail.txt")
STATE_PATH = ROOT / ".autoresearch_queue_state.json"
MAX_SIZE_BYTES = 16_000_000
POLL_SECONDS = 60
PROMOTED_BRANCH = "autoresearch/mar25"


@dataclass
class Experiment:
    run_id: str
    param_changes: str
    description: str
    env: dict[str, str] = field(default_factory=dict)
    branch: str = PROMOTED_BRANCH
    candidate_branch: bool = False
    promote_on_improve: bool = False


MANDATORY_QUEUE: list[Experiment] = [
    Experiment(
        run_id="mar25_r01_baseline",
        param_changes="baseline",
        description="full 20000-step baseline rerun on promoted HEAD with standard evaluation",
    ),
    Experiment(
        run_id="mar25_r02_ttt_doc_s1",
        param_changes="TTT_ENABLE=1 TTT_SCOPE=document TTT_STEPS=1 TTT_PREFIX_TOKENS=128",
        description="document-scope self-learning rerun to verify real adaptation and post-quant gain",
        env={
            "TTT_ENABLE": "1",
            "TTT_SCOPE": "document",
            "TTT_STEPS": "1",
            "TTT_PREFIX_TOKENS": "128",
        },
    ),
    Experiment(
        run_id="mar25_r03_ttt_seq_s1",
        param_changes="TTT_ENABLE=1 TTT_SCOPE=sequence TTT_STEPS=1 TTT_PREFIX_TOKENS=128",
        description="sequence-scope self-learning rerun to compare against document-scope adaptation",
        env={
            "TTT_ENABLE": "1",
            "TTT_SCOPE": "sequence",
            "TTT_STEPS": "1",
            "TTT_PREFIX_TOKENS": "128",
        },
    ),
    Experiment(
        run_id="mar25_r04_ttt_doc_s2",
        param_changes="TTT_ENABLE=1 TTT_SCOPE=document TTT_STEPS=2 TTT_PREFIX_TOKENS=128",
        description="document-scope self-learning rerun with two online updates per adapted window",
        env={
            "TTT_ENABLE": "1",
            "TTT_SCOPE": "document",
            "TTT_STEPS": "2",
            "TTT_PREFIX_TOKENS": "128",
        },
    ),
    Experiment(
        run_id="mar25_r05_ttt_doc_lr003",
        param_changes="TTT_ENABLE=1 TTT_SCOPE=document TTT_LR=0.003 TTT_PREFIX_TOKENS=128",
        description="document-scope self-learning rerun with lower adaptation learning rate",
        env={
            "TTT_ENABLE": "1",
            "TTT_SCOPE": "document",
            "TTT_LR": "0.003",
            "TTT_PREFIX_TOKENS": "128",
        },
    ),
    Experiment(
        run_id="mar25_r06_ttt_doc_lr03",
        param_changes="TTT_ENABLE=1 TTT_SCOPE=document TTT_LR=0.03 TTT_PREFIX_TOKENS=128",
        description="document-scope self-learning rerun with higher adaptation learning rate",
        env={
            "TTT_ENABLE": "1",
            "TTT_SCOPE": "document",
            "TTT_LR": "0.03",
            "TTT_PREFIX_TOKENS": "128",
        },
    ),
    Experiment(
        run_id="mar25_r07_ttt_doc_extpatterns",
        param_changes="TTT_ENABLE=1 TTT_SCOPE=document TTT_PARAM_PATTERNS=attn_scale,mlp_scale,resid_mix,q_gain,skip_weight",
        description="document-scope self-learning rerun with a wider matched parameter set for adaptation",
        env={
            "TTT_ENABLE": "1",
            "TTT_SCOPE": "document",
            "TTT_PARAM_PATTERNS": "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight",
        },
    ),
    Experiment(
        run_id="mar25_r08_eval_stride64",
        param_changes="EVAL_STRIDE=64",
        description="baseline rerun with sliding evaluation stride 64 to assess standalone utility",
        env={"EVAL_STRIDE": "64"},
    ),
    Experiment(
        run_id="mar25_r09_ema",
        param_changes="EMA_ENABLED=1",
        description="baseline rerun with EMA enabled to test compatibility with the self-learning path",
        env={"EMA_ENABLED": "1"},
    ),
    Experiment(
        run_id="mar25_r10_clip99999",
        param_changes="INT8_CLIP_PERCENTILE=99.999",
        description="baseline rerun with a lower int8 clip percentile using a dedicated candidate branch",
        env={"INT8_CLIP_PERCENTILE": "99.999"},
        branch="candidate/mar25/clip99999",
        candidate_branch=True,
        promote_on_improve=True,
    ),
]


def run(cmd: list[str], *, check: bool = True, capture: bool = True, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
        env=env,
    )
    return result.stdout.strip() if capture else ""


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp} UTC] {msg}", flush=True)


def current_branch() -> str:
    return run(["git", "branch", "--show-current"])


def head_commit() -> str:
    return run(["git", "rev-parse", "--short", "HEAD"])


def ensure_results_header() -> None:
    header = "commit\tval_bpb\tsize_bytes\tsize_ok\tstatus\tparam_changes\tdescription\n"
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(header, encoding="utf-8")
        return
    text = RESULTS_PATH.read_text(encoding="utf-8")
    if not text.startswith("commit\tval_bpb\t"):
        RESULTS_PATH.write_text(header + text, encoding="utf-8")


def load_state() -> dict[str, object]:
    if not STATE_PATH.exists():
        return {
            "phase": "mandatory",
            "completed": [],
            "best_valid_bpb": None,
            "best_valid_run": None,
            "best_valid_branch": PROMOTED_BRANCH,
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, object]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def active_train_process() -> bool:
    result = subprocess.run(
        [
            "bash",
            "-lc",
            "ps -eo pid=,args= | grep -E 'torchrun --standalone --nproc_per_node=1 train_gpt.py|python3 -u train_gpt.py' | grep -v grep || true",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip())


def parse_ttt_activity(log_text: str) -> tuple[bool, str]:
    eval_mode_match = re.findall(r"eval_mode:(ttt:\w+)", log_text)
    eval_mode = eval_mode_match[-1] if eval_mode_match else "standard"
    doc_hits = [
        (int(adapted), int(steps))
        for adapted, steps in re.findall(r"ttt_eval scope:document .*adapted_docs:(\d+).*update_steps:(\d+)", log_text)
    ]
    seq_hits = [
        (int(adapted), int(steps))
        for adapted, steps in re.findall(r"ttt_eval scope:sequence adapted_seqs:(\d+).*update_steps:(\d+)", log_text)
    ]
    active = any(adapted > 0 and steps > 0 for adapted, steps in doc_hits + seq_hits)
    if doc_hits:
        adapted, steps = doc_hits[-1]
        return active, f"{eval_mode}; adapted_docs={adapted}; update_steps={steps}"
    if seq_hits:
        adapted, steps = seq_hits[-1]
        return active, f"{eval_mode}; adapted_seqs={adapted}; update_steps={steps}"
    return active, eval_mode


def parse_run_result(run_id: str) -> dict[str, object]:
    log_path = ROOT / "logs" / f"{run_id}.txt"
    if not log_path.exists():
        return {
            "status": "crash",
            "val_bpb": 0.0,
            "size_bytes": 0,
            "size_ok": "n/a",
            "ttt_active": False,
            "ttt_summary": "log missing",
            "log_path": str(log_path),
        }
    text = log_path.read_text(encoding="utf-8")
    bpb_matches = re.findall(
        r"^final_(?:int8_zlib|int6_lzma)_roundtrip_exact val_loss:\S+ val_bpb:(\S+)$",
        text,
        flags=re.MULTILINE,
    )
    size_matches = re.findall(
        r"^Total submission size (?:int8\+zlib|int6\+lzma): (\d+) bytes$",
        text,
        flags=re.MULTILINE,
    )
    ttt_active, ttt_summary = parse_ttt_activity(text)
    if not bpb_matches or not size_matches:
        return {
            "status": "crash",
            "val_bpb": 0.0,
            "size_bytes": 0,
            "size_ok": "n/a",
            "ttt_active": ttt_active,
            "ttt_summary": ttt_summary,
            "log_path": str(log_path),
        }
    val_bpb = float(bpb_matches[-1])
    size_bytes = int(size_matches[-1])
    size_ok = "yes" if size_bytes < MAX_SIZE_BYTES else "no"
    return {
        "status": "ok",
        "val_bpb": val_bpb,
        "size_bytes": size_bytes,
        "size_ok": size_ok,
        "ttt_active": ttt_active,
        "ttt_summary": ttt_summary,
        "log_path": str(log_path),
    }


def checkout_branch(branch: str) -> None:
    if current_branch() == branch:
        return
    run(["git", "checkout", branch], capture=False)


def append_result(commit: str, result: dict[str, object], status: str, exp: Experiment, description: str) -> None:
    row = (
        f"{commit}\t{result['val_bpb']:.8f}\t{result['size_bytes']}\t{result['size_ok']}\t"
        f"{status}\t{exp.param_changes}\t{description}\n"
    )
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(row)


def send_email(exp: Experiment, branch: str, commit: str, result: dict[str, object], status: str, note: str, next_items: list[Experiment]) -> None:
    next_runs = ", ".join(item.run_id for item in next_items[:3]) if next_items else "queue exhausted"
    body = "\n".join(
        [
            f"Experiment: {exp.run_id}",
            f"Branch used: {branch}",
            f"Commit: {commit}",
            f"Promotion outcome: {'promoted into autoresearch/mar25' if note == 'promoted' else note}",
            f"Post-quant val_bpb: {result['val_bpb']:.8f}",
            f"Artifact size bytes: {result['size_bytes']}",
            f"Result status: {status}",
            f"Self-learning active: {'yes' if result['ttt_active'] else 'no'}",
            f"Self-learning evidence: {result['ttt_summary']}",
            f"Main conclusion: {note}",
            f"Promoted codebase changed: {'yes' if note == 'promoted' else 'no'}",
            f"Next proposed experiments: {next_runs}",
            f"Log file: {result['log_path']}",
        ]
    )
    MAIL_PATH.write_text(body + "\n", encoding="utf-8")
    subprocess.run(
        [
            "python3",
            "autoresearch/send_update_email.py",
            "--subject",
            "autoresearch update",
            "--body-file",
            str(MAIL_PATH),
        ],
        cwd=ROOT,
        check=True,
    )


def decide_status(state: dict[str, object], result: dict[str, object], exp: Experiment) -> tuple[str, str]:
    if result["status"] != "ok":
        return "crash", "run crashed or did not emit final metrics"
    if result["size_ok"] != "yes":
        return "discard", "size limit violated"
    if exp.env.get("TTT_ENABLE") == "1" and not result["ttt_active"]:
        return "discard", "TTT was configured but logs show no real adaptation activity"
    best_valid_bpb = state.get("best_valid_bpb")
    if best_valid_bpb is None or result["val_bpb"] < float(best_valid_bpb):
        return "keep", "new best valid result"
    return "discard", "valid run but did not beat the current best valid result"


def wait_for_existing_run(run_id: str) -> dict[str, object]:
    missing_final_polls = 0
    while True:
        result = parse_run_result(run_id)
        if result["status"] == "ok":
            return result
        if active_train_process():
            missing_final_polls = 0
        else:
            missing_final_polls += 1
            if missing_final_polls >= 3:
                return result
        time.sleep(POLL_SECONDS)


def launch_and_wait(exp: Experiment) -> dict[str, object]:
    env = os.environ.copy()
    env["RUN_ID"] = exp.run_id
    env["ITERATIONS"] = "20000"
    env.update(exp.env)
    cmd = ["bash", "autoresearch/run_experiment.sh"]
    log(f"launching {exp.run_id} on {current_branch()} with {exp.param_changes}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, check=False)
    if proc.returncode != 0:
        log(f"{exp.run_id} exited with return code {proc.returncode}")
    return wait_for_existing_run(exp.run_id)


def maybe_promote(exp: Experiment, status: str) -> str:
    if not exp.candidate_branch:
        checkout_branch(PROMOTED_BRANCH)
        return "promoted codebase unchanged"
    if status != "keep" or not exp.promote_on_improve:
        checkout_branch(PROMOTED_BRANCH)
        return "candidate archived without promotion"
    checkout_branch(PROMOTED_BRANCH)
    run(["git", "merge", "--no-ff", exp.branch, "-m", f"Promote {exp.run_id}"], capture=False)
    return "promoted"


def ensure_branch_preconditions(exp: Experiment) -> None:
    status = run(["git", "status", "--porcelain", "--", "train_gpt.py"])
    if status:
        raise RuntimeError("train_gpt.py has uncommitted changes; refusing to switch experiment branches")
    checkout_branch(exp.branch)


def run_queue() -> None:
    ensure_results_header()
    state = load_state()
    completed = set(state.get("completed", []))
    queue = [exp for exp in MANDATORY_QUEUE if exp.run_id not in completed]
    if not queue:
        log("mandatory queue already completed; entering idle wait loop")
        while True:
            time.sleep(POLL_SECONDS)

    for idx, exp in enumerate(queue):
        ensure_branch_preconditions(exp)
        run_branch = current_branch()
        result = launch_and_wait(exp)
        commit = head_commit()
        status, note = decide_status(state, result, exp)
        promotion_note = maybe_promote(exp, status)
        full_note = note if promotion_note == "promoted codebase unchanged" else f"{note}; {promotion_note}"
        append_result(commit, result, status, exp, exp.description)
        if status == "keep":
            state["best_valid_bpb"] = result["val_bpb"]
            state["best_valid_run"] = exp.run_id
            state["best_valid_branch"] = current_branch()
        completed.add(exp.run_id)
        state["completed"] = sorted(completed)
        save_state(state)
        next_items = queue[idx + 1 :]
        send_email(exp, run_branch, commit, result, status, full_note, next_items)
        log(
            f"completed {exp.run_id}: status={status} val_bpb={result['val_bpb']:.8f} "
            f"size={result['size_bytes']} note={full_note}"
        )

    log("mandatory queue complete; entering idle wait loop")
    while True:
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        run_queue()
    except Exception as exc:
        log(f"controller failed: {exc}")
        raise

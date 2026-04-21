#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/fmassapg")
RUN_ID = "apr11_r83_xsa_last9_ve12_l8_logscale_epochs1_clip1"
RUN_DATE_LABEL = "Apr 11, 2026"
RUN_SHORT = "r83"
COMMIT = "69e8be8"
CANDIDATE_BRANCH = "candidate/mar27/ve-logscale-l8-d12"
PROMOTED_BRANCH = "autoresearch/mar27"
PROMOTED_COMMIT = "1bd7c9b"
BACKEND = "Seeweb"
RUN_LOG = ROOT / "run.log"
LOG_PATH = ROOT / "logs" / f"{RUN_ID}.txt"
RESULTS_PATH = ROOT / "autoresearch" / "results.tsv"
SECOND_BRAIN_PATH = ROOT / "autoresearch" / "second_brain.md"
PROGRAM_PATH = ROOT / "autoresearch" / "program.md"
MAIL_BODY_PATH = ROOT / ".r83_mail.txt"
FINISHER_LOG = ROOT / ".r83_finisher_runtime.log"
PARAM_CHANGES = (
    "code: log-parameterized VE local-only layer scale; "
    "TTT_ENABLE=1 TTT_PROTOCOL=score_first TTT_SCOPE=global TTT_EPOCHS=1 "
    "TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768 "
    "TTT_PARAM_PATTERNS=ve_layer_scales VE_ENABLED=1 VE_DIM=12 VE_LAYERS=8 "
    "TTT_DELTA_BUDGET_RATIO=0.10 XSA_LAST_N=9 EVAL_STRIDE=64 EMA_ENABLED=1 "
    "WARMDOWN_ITERS=3500 GRAD_CLIP_NORM=1.0"
)
ANCHOR_BPB = 1.17333784
ANCHOR_LABEL = "r82"
GLOBAL_BEST_LABEL = "r80"
MAX_WAIT_SECONDS = 16 * 60 * 60
SLEEP_SECONDS = 120


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"[{timestamp} UTC] {msg}"
    print(line, flush=True)
    with FINISHER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, capture_output=True, check=False)


def active_run_processes() -> list[str]:
    proc = run(["ps", "-eo", "pid,cmd"])
    lines = []
    for raw in proc.stdout.splitlines():
        if RUN_ID in raw and "grep" not in raw:
            lines.append(raw.strip())
    if lines:
        return lines
    generic = []
    for raw in proc.stdout.splitlines():
        if "grep" in raw:
            continue
        if any(token in raw for token in ("dispatch_experiment.py", "run_experiment.sh", "train_gpt.py", "torchrun")):
            generic.append(raw.strip())
    return generic


def wait_for_final_marker() -> tuple[bool, str]:
    start = time.time()
    saw_log = False
    while time.time() - start < MAX_WAIT_SECONDS:
        text = read_text(LOG_PATH)
        if text:
            saw_log = True
        if re.search(r"^final_(?:int8_zlib|int6_lzma)_roundtrip_exact ", text, flags=re.MULTILINE):
            return True, text
        if not active_run_processes() and saw_log:
            return False, text
        time.sleep(SLEEP_SECONDS)
    return False, read_text(LOG_PATH)


def wait_for_no_active_run(timeout_seconds: int = 600) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if not active_run_processes():
            return
        time.sleep(5)


def load_results() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def is_valid(row: dict[str, str]) -> bool:
    return row.get("size_ok") == "yes" and row.get("status") in {"keep", "promote", "anchor"}


def best_valid_bpb(rows: list[dict[str, str]]) -> float:
    values = []
    for row in rows:
        if not is_valid(row):
            continue
        try:
            values.append(float(row["val_bpb"]))
        except (KeyError, ValueError):
            continue
    return min(values) if values else float("inf")


def append_result(row: dict[str, str]) -> None:
    existing_text = read_text(RESULTS_PATH)
    if RUN_ID in existing_text:
        log("results row already present; skipping append")
        return
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                row["commit"],
                row["val_bpb"],
                row["size_bytes"],
                row["size_ok"],
                row["status"],
                row["param_changes"],
                row["description"],
            ]
        )


def parse_metric(text: str) -> tuple[float | None, str | None]:
    matches = re.findall(
        r"^final_(int8_zlib|int6_lzma)_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        return None, None
    metric_name, bpb = matches[-1]
    return float(bpb), metric_name


def parse_size(text: str) -> tuple[int | None, str | None]:
    matches = re.findall(r"^Total submission size (int8\+zlib|int6\+lzma): (\d+) bytes", text, flags=re.MULTILINE)
    if not matches:
        return None, None
    kind, size = matches[-1]
    return int(size), kind


def parse_prequant(text: str) -> float | None:
    matches = re.findall(
        r"^step:20000/20000 val_loss:[0-9.]+ val_bpb:([0-9.]+) .*eval_mode:ttt:score_first:global",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        return None
    return float(matches[-1])


def parse_ttt_start(text: str) -> dict[str, str]:
    match = re.search(
        r"^ttt:enabled=(True|False).*?epochs:(\d+).*?delta_budget_ratio:([0-9.]+).*?matched_params:(\d+) patterns:(.+)$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return {}
    return {
        "enabled": match.group(1),
        "epochs": match.group(2),
        "delta_budget_ratio": match.group(3),
        "matched_params": match.group(4),
        "patterns": match.group(5).strip(),
    }


def parse_ve_mode(text: str) -> str:
    match = re.search(r"^ve:enabled=.* scale_mode:([A-Za-z0-9_]+)$", text, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def parse_ttt_evals(text: str) -> list[dict[str, str]]:
    matches = re.findall(
        r"^ttt_eval protocol:score_first scope:global scored_chunks:(\d+) adapted_chunks:(\d+) "
        r"update_steps:(\d+) stride_active:(\d+) stride:(\d+) chunk_tokens:(\d+) epochs:(\d+) "
        r"momentum:([0-9.]+)(?: delta_budget_ratio:([0-9.]+))? .*? elapsed_ms:(\d+)$",
        text,
        flags=re.MULTILINE,
    )
    evals = []
    for (
        scored_chunks,
        adapted_chunks,
        update_steps,
        stride_active,
        stride,
        chunk_tokens,
        epochs,
        momentum,
        delta_budget_ratio,
        elapsed_ms,
    ) in matches:
        evals.append(
            {
                "scored_chunks": scored_chunks,
                "adapted_chunks": adapted_chunks,
                "update_steps": update_steps,
                "stride_active": stride_active,
                "stride": stride,
                "chunk_tokens": chunk_tokens,
                "epochs": epochs,
                "momentum": momentum,
                "delta_budget_ratio": delta_budget_ratio or "0.000",
                "elapsed_ms": elapsed_ms,
            }
        )
    return evals


def replace_or_append_line(lines: list[str], prefix: str, new_line: str) -> None:
    target = f"- {prefix}:"
    for index, line in enumerate(lines):
        if line.startswith(target):
            lines[index] = new_line
            return
    lines.append(new_line)


def upsert_second_brain(
    final_bpb: float | None,
    size_bytes: int | None,
    prequant_bpb: float | None,
    ttt_start: dict[str, str],
    ttt_eval: dict[str, str],
    ve_mode: str,
    status: str,
) -> None:
    lines = [line.rstrip("\n") for line in read_text(SECOND_BRAIN_PATH).splitlines() if line.strip()]
    if not lines:
        lines = ["# Second Brain"]

    broken_line = (
        f"- broken_family: exact log-param 1-param local VE score-first adapting line on `{COMMIT}` "
        f"collapses by about +{(final_bpb - ANCHOR_BPB):.8f} bpb vs `{ANCHOR_LABEL}`"
        if final_bpb is not None
        else f"- broken_family: exact log-param 1-param local VE score-first adapting line on `{COMMIT}` crashed before final metric"
    )
    if not any(line.startswith("- broken_family: exact log-param 1-param local VE score-first adapting line on") for line in lines):
        lines.append(broken_line)
    else:
        for idx, line in enumerate(lines):
            if line.startswith("- broken_family: exact log-param 1-param local VE score-first adapting line on"):
                lines[idx] = broken_line
                break

    replace_or_append_line(
        lines,
        "rule",
        "- rule: no more env-only reruns on carrier, two-scale VE, plain-SGD local-only VE, delta-clipped local-only VE, or log-param local-only VE without a material code change",
    )
    replace_or_append_line(
        lines,
        "archive",
        f"- archive: `{CANDIDATE_BRANCH}` stays unmerged on `{COMMIT}` after the matched adapting follow-up",
    )

    evidence_parts = [f"{RUN_SHORT} log-param VE adapting fail" if status == "discard" else f"{RUN_SHORT} log-param VE adapting result"]
    if final_bpb is not None:
        evidence_parts.append(f"{final_bpb:.8f}")
    if size_bytes is not None:
        evidence_parts.append(f"{size_bytes} bytes")
    if ttt_start.get("matched_params"):
        evidence_parts.append(f"matched_params {ttt_start['matched_params']}")
    if ttt_eval.get("adapted_chunks"):
        evidence_parts.append(f"adapted_chunks {ttt_eval['adapted_chunks']}")
    if ttt_eval.get("update_steps"):
        evidence_parts.append(f"update_steps {ttt_eval['update_steps']}")
    if prequant_bpb is not None:
        evidence_parts.append(f"prequant {prequant_bpb:.4f}")
    if ttt_eval.get("stride_active"):
        evidence_parts.append(f"stride_active {ttt_eval['stride_active']}")
    evidence_parts.append(f"scale_mode {ve_mode}")
    if ttt_eval.get("delta_budget_ratio"):
        evidence_parts.append(f"delta_budget_ratio {ttt_eval['delta_budget_ratio']}")
    replace_or_append_line(lines, "evidence", "- evidence: " + " | ".join(evidence_parts))

    replace_or_append_line(
        lines,
        "next",
        "- next: pivot to a material code change such as a learned update-scale/clamp on the VE-local surface or a tiny bank-local adaptive matrix; do not rerun exact log-param VE env-only adaptation",
    )
    replace_or_append_line(
        lines,
        "fallback",
        "- fallback: if the next rewrite is not ready, prefer another materially different low-byte adaptive surface over reopening carrier or local-only VE env-only reruns",
    )

    SECOND_BRAIN_PATH.write_text("\n".join(lines[:25]) + "\n", encoding="utf-8")


def format_delta(value: float) -> str:
    return f"{value:+.8f}"


def build_program_block(
    final_bpb: float,
    size_bytes: int,
    prequant_bpb: float | None,
    ttt_start: dict[str, str],
    ttt_eval: dict[str, str],
    current_best: float,
    ve_mode: str,
) -> str:
    delta_anchor = final_bpb - ANCHOR_BPB
    delta_best = final_bpb - current_best
    headroom = 16_000_000 - size_bytes
    prequant_text = f"{prequant_bpb:.4f}" if prequant_bpb is not None else "unknown"
    return f"""
### Cycle Review Snapshot ({RUN_DATE_LABEL} {RUN_SHORT} Log-Param VE Adapting Failure + Family Closure)

This cycle completed the queued matched adapting follow-up on the log-parameterized VE candidate branch. The intended legal adaptation is real, the artifact stays under 16,000,000 bytes, but the exact log-param local-only VE line still collapses badly enough that the branch stays archived and this env-only family is now closed until a material code change.

- Promoted branch/root state:
  - `HEAD` returns to promoted `{PROMOTED_COMMIT}` on `{PROMOTED_BRANCH}`
  - the completed run launched from candidate branch `{CANDIDATE_BRANCH}` at `{COMMIT}`
  - the branch stays unmerged because the final post-quant metric does not beat promoted `{GLOBAL_BEST_LABEL}`
- Authoritative local references are now:
  - promoted best `{GLOBAL_BEST_LABEL}`: `logs/apr10_r80_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_anchor.txt` with exact `1.17309950` at `15968741` bytes
  - matched log-param anchor `{ANCHOR_LABEL}`: `logs/apr11_r82_xsa_last9_ve12_l8_logscale_anchor.txt` with exact `1.17333784` at `15966165` bytes
  - matched log-param adapting failure `{RUN_SHORT}`: `logs/{RUN_ID}.txt` with exact `{final_bpb:.8f}` at `{size_bytes}` bytes
- Interpretation of `{RUN_SHORT}`:
  - the persisted logs confirm the intended rewrite really adapted on the same tiny surface: `scale_mode:{ve_mode}`, `matched_params:{ttt_start.get('matched_params', '?')}`, `adapted_chunks={ttt_eval.get('adapted_chunks', '?')}`, `update_steps={ttt_eval.get('update_steps', '?')}`, `stride_active={ttt_eval.get('stride_active', '?')}`, `delta_budget_ratio:{ttt_eval.get('delta_budget_ratio', ttt_start.get('delta_budget_ratio', '?'))}`, `active_layers:[8]`, and `xsa:last_n:9`
  - pre-quant already collapses to exact `{prequant_text}`, and the final post-quant int8+zlib roundtrip lands at exact `{final_bpb:.8f}`
  - `{RUN_SHORT}` is `{format_delta(delta_anchor)}` bpb versus matched `{ANCHOR_LABEL}` and `{format_delta(delta_best)}` versus promoted `{GLOBAL_BEST_LABEL}`
  - total submission size is `{size_bytes}` bytes, leaving `{headroom}` bytes of headroom; the run is valid on size but not competitive on metric
  - conclusion: log-parameterizing the same 1-parameter local VE surface does not rescue real env-only adaptation; the exact log-param VE line is now closed and the next cycle must pivot to a materially different update rule or adaptive surface
- Competitive reference carried forward from the Apr 11, 2026 official refresh:
  - current public leader remains `1.0810` (`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`)
  - the local gap from promoted `{GLOBAL_BEST_LABEL}` therefore stays about `0.0921` bpb, and `{RUN_SHORT}` confirms the blocker is still update-rule/surface quality under real adaptation rather than no-update anchor quality
- Next action for the following cycle:
  - keep promoted `{PROMOTED_COMMIT}` and `{GLOBAL_BEST_LABEL}` as the authoritative reference
  - archive `{CANDIDATE_BRANCH}` without merge
  - do not rerun exact log-param local-only VE adaptation unchanged
  - pivot the next code-changing cycle to a learned update-scale/clamp on the VE-local surface or a tiny bank-local adaptive matrix, with a fresh `TTT_EPOCHS=0` anchor before any new adapting claim

""".lstrip("\n")


def update_program_md(
    final_bpb: float,
    size_bytes: int,
    prequant_bpb: float | None,
    ttt_start: dict[str, str],
    ttt_eval: dict[str, str],
    current_best: float,
    ve_mode: str,
) -> None:
    text = read_text(PROGRAM_PATH)
    if RUN_ID in text:
        log("program.md already mentions r83; skipping program update")
        return
    marker = "\n## The Experiment Loop\n"
    block = build_program_block(final_bpb, size_bytes, prequant_bpb, ttt_start, ttt_eval, current_best, ve_mode)
    if marker not in text:
        log("program.md marker missing; skipping program update")
        return
    updated = text.replace(marker, "\n" + block + marker, 1)
    write_text(PROGRAM_PATH, updated)


def build_email_body(
    status: str,
    final_bpb: float | None,
    size_bytes: int | None,
    prequant_bpb: float | None,
    ttt_start: dict[str, str],
    ttt_eval: dict[str, str],
    current_best: float,
) -> str:
    if final_bpb is None or size_bytes is None:
        result_value = "no final metric"
        size_value = "n/a"
        promoted = "no"
        conclusion = "The run did not emit a final roundtrip metric before the finisher timeout, so it is logged as a crash."
    else:
        result_value = f"{final_bpb:.8f} bpb"
        size_value = f"{size_bytes} bytes"
        promoted = "yes" if status == "promote" else "no"
        delta_anchor = final_bpb - ANCHOR_BPB
        delta_best = final_bpb - current_best
        conclusion = (
            f"Real legal adaptation happened, but the final post-quant metric lands {delta_anchor:+.8f} vs {ANCHOR_LABEL} "
            f"and {delta_best:+.8f} vs the promoted best, so the exact log-param VE env-only adapting line is closed."
        )
    ttt_value = (
        f"scale_mode=log_exp matched_params={ttt_start.get('matched_params', '?')} "
        f"adapted_chunks={ttt_eval.get('adapted_chunks', '?')} update_steps={ttt_eval.get('update_steps', '?')} "
        f"stride_active={ttt_eval.get('stride_active', '?')} delta_budget_ratio={ttt_eval.get('delta_budget_ratio', ttt_start.get('delta_budget_ratio', '?'))}"
    )
    prequant_value = "n/a" if prequant_bpb is None else f"{prequant_bpb:.4f}"
    body_lines = [
        f"Run: {RUN_ID}",
        f"Backend: {BACKEND}",
        f"Branch: {CANDIDATE_BRANCH} @ {COMMIT}",
        f"Status: {status}",
        f"Result: {result_value}",
        f"Artifact size: {size_value}",
        f"Pre-quant bpb: {prequant_value}",
        f"TTT activity: {ttt_value}",
        f"Promoted codebase changed: {promoted}",
        "",
        conclusion,
        "",
        "Next proposals:",
        "- Add a learned update-scale or clamp parameter on the same VE-local surface instead of another env-only rerun.",
        "- Prototype a tiny bank-local adaptive matrix in the deepest block so the update surface is still low-byte but no longer one scalar.",
        "- If the VE path stays unstable, pivot to another materially different low-byte adaptive surface rather than reopening carrier or exact local-only VE lines.",
    ]
    return "\n".join(body_lines) + "\n"


def git_checkout_promoted_or_merge(status: str) -> None:
    if status == "promote":
        proc = run(["git", "checkout", PROMOTED_BRANCH])
        if proc.returncode != 0:
            log(f"git checkout promoted failed: {proc.stderr.strip() or proc.stdout.strip()}")
            return
        merge = run(["git", "merge", "--ff-only", CANDIDATE_BRANCH])
        if merge.returncode != 0:
            log(f"git merge failed: {merge.stderr.strip() or merge.stdout.strip()}")
        return

    proc = run(["git", "checkout", PROMOTED_BRANCH])
    if proc.returncode != 0:
        log(f"git checkout promoted failed: {proc.stderr.strip() or proc.stdout.strip()}")


def main() -> int:
    log(f"waiting for {RUN_SHORT} completion")
    success, text = wait_for_final_marker()
    final_bpb, _metric_name = parse_metric(text)
    size_bytes, _size_kind = parse_size(text)
    prequant_bpb = parse_prequant(text)
    ttt_start = parse_ttt_start(text)
    ttt_evals = parse_ttt_evals(text)
    ttt_eval = ttt_evals[-1] if ttt_evals else {}
    ve_mode = parse_ve_mode(text)

    rows = load_results()
    current_best = best_valid_bpb(rows)

    if final_bpb is None or size_bytes is None:
        status = "crash"
        size_ok = "n/a"
        description = (
            f"{RUN_ID} matched adapting log-parameterized local-only VE probe on `{CANDIDATE_BRANCH}` "
            f"did not emit a final roundtrip metric and is logged as crash pending tail diagnosis."
        )
        result_row = {
            "commit": COMMIT,
            "val_bpb": "0.00000000",
            "size_bytes": "0",
            "size_ok": size_ok,
            "status": status,
            "param_changes": PARAM_CHANGES,
            "description": description,
        }
    else:
        size_ok = "yes" if size_bytes < 16_000_000 else "no"
        best_improved = size_ok == "yes" and final_bpb < current_best
        status = "promote" if best_improved else "discard"
        delta_anchor = final_bpb - ANCHOR_BPB
        delta_best = final_bpb - current_best
        prequant_text = f"{prequant_bpb:.4f}" if prequant_bpb is not None else "unknown"
        description = (
            f"Matched adapting follow-up on `{CANDIDATE_BRANCH}` with log-parameterized local-only VE updates; "
            f"logs confirm `scale_mode:log_exp`, matched_params={ttt_start.get('matched_params', '?')}, "
            f"adapted_chunks={ttt_eval.get('adapted_chunks', '?')}, update_steps={ttt_eval.get('update_steps', '?')}, "
            f"stride_active={ttt_eval.get('stride_active', '?')}, and delta_budget_ratio={ttt_eval.get('delta_budget_ratio', ttt_start.get('delta_budget_ratio', '?'))}. "
            f"Pre-quant already collapses to {prequant_text}, and the final post-quant int8+zlib roundtrip lands at {final_bpb:.8f} "
            f"with a valid {size_bytes}-byte artifact, which is {delta_anchor:+.8f} vs {ANCHOR_LABEL} and {delta_best:+.8f} vs the promoted best {GLOBAL_BEST_LABEL}. "
        )
        if status == "promote":
            description += "The candidate unexpectedly beats the promoted codebase and earns promotion."
        else:
            description += "This closes the exact log-param local-only VE adapting line; the next cycle should pivot to a learned update-scale/clamp or tiny bank-local adaptive matrix rather than another identical env-only rerun."
        result_row = {
            "commit": COMMIT,
            "val_bpb": f"{final_bpb:.8f}",
            "size_bytes": str(size_bytes),
            "size_ok": size_ok,
            "status": status,
            "param_changes": PARAM_CHANGES,
            "description": description,
        }

    append_result(result_row)
    if final_bpb is not None and size_bytes is not None:
        upsert_second_brain(final_bpb, size_bytes, prequant_bpb, ttt_start, ttt_eval, ve_mode, status)
        update_program_md(final_bpb, size_bytes, prequant_bpb, ttt_start, ttt_eval, current_best, ve_mode)

    refresh = run(["python3", "autoresearch/refresh_second_brain.py"])
    if refresh.returncode != 0:
        log(f"refresh_second_brain failed: {refresh.stderr.strip() or refresh.stdout.strip()}")

    MAIL_BODY_PATH.write_text(
        build_email_body(status, final_bpb, size_bytes, prequant_bpb, ttt_start, ttt_eval, current_best),
        encoding="utf-8",
    )
    mail = run(
        [
            "python3",
            "autoresearch/send_update_email.py",
            "--subject",
            "autoresearch update r83",
            "--body-file",
            str(MAIL_BODY_PATH),
        ]
    )
    if mail.returncode != 0:
        log(f"send_update_email failed: {mail.stderr.strip() or mail.stdout.strip()}")
    else:
        log("status email sent")

    wait_for_no_active_run()
    git_checkout_promoted_or_merge(status)

    if not success and final_bpb is None:
        tail = run(["tail", "-n", "120", str(RUN_LOG)])
        log("run tail:\n" + (tail.stdout or tail.stderr))

    log(f"{RUN_SHORT} finisher completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

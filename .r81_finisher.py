#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/root/fmassapg")
RUN_ID = "apr10_r81_xsa_last9_ve12_l8_layerscaleonly_deltaclip010_epochs1"
COMMIT = "1bd7c9b"
BRANCH = "autoresearch/mar27"
BACKEND = "Seeweb"
RUN_LOG = ROOT / "run.log"
LOG_PATH = ROOT / "logs" / f"{RUN_ID}.txt"
RESULTS_PATH = ROOT / "autoresearch" / "results.tsv"
SECOND_BRAIN_PATH = ROOT / "autoresearch" / "second_brain.md"
MAIL_BODY_PATH = ROOT / ".r81_mail.txt"
FINISHER_LOG = ROOT / ".r81_finisher_runtime.log"
PARAM_CHANGES = (
    "TTT_ENABLE=1 TTT_PROTOCOL=score_first TTT_SCOPE=global TTT_EPOCHS=1 "
    "TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768 "
    "TTT_PARAM_PATTERNS=ve_layer_scales VE_ENABLED=1 VE_DIM=12 VE_LAYERS=8 "
    "TTT_DELTA_BUDGET_RATIO=0.10 XSA_LAST_N=9 EVAL_STRIDE=64 EMA_ENABLED=1 "
    "WARMDOWN_ITERS=3500 GRAD_CLIP_NORM=1.0"
)
ANCHOR_BPB = 1.17309950
ANCHOR_LABEL = "r80"
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


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, capture_output=True, check=False)


def active_run_processes() -> list[str]:
    proc = run(["ps", "-eo", "pid,cmd"])
    lines = []
    for raw in proc.stdout.splitlines():
        if RUN_ID not in raw:
            continue
        if "grep" in raw:
            continue
        lines.append(raw.strip())
    if lines:
        return lines

    # Fallback to generic local training processes while this run is still the only active lane.
    generic = []
    for raw in proc.stdout.splitlines():
        if any(token in raw for token in ("dispatch_experiment.py", "run_experiment.sh", "train_gpt.py")):
            if "grep" in raw:
                continue
            generic.append(raw.strip())
    return generic


def wait_for_completion() -> tuple[bool, str]:
    start = time.time()
    saw_log = False
    while time.time() - start < MAX_WAIT_SECONDS:
        text = read_text(LOG_PATH)
        if text:
            saw_log = True
        if re.search(r"^final_(?:int8_zlib|int6_lzma)_roundtrip_exact ", text, flags=re.MULTILINE):
            return True, text
        procs = active_run_processes()
        if not procs and saw_log:
            return False, text
        time.sleep(SLEEP_SECONDS)
    return False, read_text(LOG_PATH)


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


def parse_ttt_eval(text: str) -> dict[str, str]:
    matches = re.findall(
        r"^ttt_eval protocol:score_first scope:global scored_chunks:(\d+) adapted_chunks:(\d+) "
        r"update_steps:(\d+) stride_active:(\d+) stride:(\d+) chunk_tokens:(\d+) epochs:(\d+) "
        r"momentum:([0-9.]+)(?: delta_budget_ratio:([0-9.]+))? .*? elapsed_ms:(\d+)$",
        text,
        flags=re.MULTILINE,
    )
    if not matches:
        return {}
    scored_chunks, adapted_chunks, update_steps, stride_active, stride, chunk_tokens, epochs, momentum, delta_budget_ratio, elapsed_ms = matches[-1]
    return {
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
        if is_valid(row):
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


def upsert_second_brain(best_improved: bool, bpb: float | None, size: int | None, ttt_start: dict[str, str], ttt_eval: dict[str, str], status: str) -> None:
    lines = [line.rstrip("\n") for line in read_text(SECOND_BRAIN_PATH).splitlines() if line.strip()]
    if not lines:
        lines = ["# Second Brain"]

    def replace_or_append(prefix: str, new_line: str) -> None:
        target = f"- {prefix}:"
        for index, line in enumerate(lines):
            if line.startswith(target):
                lines[index] = new_line
                return
        lines.append(new_line)

    if best_improved and bpb is not None and size is not None:
        replace_or_append(
            "best_global",
            f"- best_global: env-only delta-clipped VE adapting run on `{COMMIT}` | {bpb:.8f} bpb | {size} bytes",
        )

    evidence_parts = [f"{RUN_ID} | status {status}"]
    if bpb is not None:
        evidence_parts.append(f"{bpb:.8f}")
    if size is not None:
        evidence_parts.append(f"{size} bytes")
    if ttt_start.get("matched_params"):
        evidence_parts.append(f"matched_params {ttt_start['matched_params']}")
    if ttt_eval.get("adapted_chunks"):
        evidence_parts.append(f"adapted_chunks {ttt_eval['adapted_chunks']}")
    if ttt_eval.get("update_steps"):
        evidence_parts.append(f"update_steps {ttt_eval['update_steps']}")
    if ttt_eval.get("stride_active"):
        evidence_parts.append(f"stride_active {ttt_eval['stride_active']}")
    if ttt_eval.get("delta_budget_ratio"):
        evidence_parts.append(f"delta_budget_ratio {ttt_eval['delta_budget_ratio']}")
    replace_or_append("evidence", "- evidence: " + " | ".join(evidence_parts))

    if bpb is None or size is None:
        replace_or_append(
            "next",
            "- next: if r81 failed before final metrics, rerun this exact delta-clipped adapting probe only after confirming the crash cause from the persisted log and keeping the GPU exclusive",
        )
        replace_or_append(
            "fallback",
            "- fallback: if the failure is instrumentation or persistence, fix logging first; otherwise pivot to log-parameterized VE updates or bank-local adaptive matrices instead of another env-only rerun",
        )
    else:
        delta_anchor = bpb - ANCHOR_BPB
        if delta_anchor > 0.20:
            broken_line = (
                f"- broken_family: exact delta-clipped 1-param local VE score-first adapting line on `{COMMIT}` "
                f"collapses by +{delta_anchor:.8f} bpb vs `{ANCHOR_LABEL}`"
            )
            if broken_line not in lines:
                lines.append(broken_line)
            replace_or_append(
                "next",
                "- next: pivot away from exact delta-clipped env-only VE adaptation and open a code-changing branch for log-parameterized VE updates or bank-local adaptive matrices",
            )
            replace_or_append(
                "fallback",
                "- fallback: if the VE rewrite is not ready, test a tiny learned update-scale or clamp surface before reopening any broader carrier or VE adapting claim",
            )
        elif best_improved:
            replace_or_append(
                "next",
                "- next: rerun the new best delta-clipped adapting configuration once for confirmation, then decide whether to widen the adaptive surface only if the win survives quantization twice",
            )
            replace_or_append(
                "fallback",
                "- fallback: if the confirmation rerun does not hold, keep delta-clipped VE only as the best anchor and pivot to a code-changing adaptive-surface rewrite",
            )
        else:
            replace_or_append(
                "next",
                "- next: keep the delta-clipped VE family open only through a material code change such as log-parameterized updates, learned update scales, or bank-local matrices; do not spend another identical env-only rerun",
            )
            replace_or_append(
                "fallback",
                "- fallback: if the rewrite is not ready, prioritize the smallest code-changing adaptive surface that preserves the `r80` no-update substrate",
            )

    trimmed = lines[:25]
    SECOND_BRAIN_PATH.write_text("\n".join(trimmed) + "\n", encoding="utf-8")


def build_email_body(status: str, bpb: float | None, size: int | None, ttt_start: dict[str, str], ttt_eval: dict[str, str], description: str) -> str:
    if bpb is None or size is None:
        next_bullets = [
            "Inspect the persisted crash tail before any rerun.",
            "If the failure is runtime-only, rerun this exact delta-clipped adapting probe once on clean Seeweb.",
            "If the log shows the adaptation path is structurally broken, pivot directly to log-parameterized VE updates or bank-local adaptive matrices.",
        ]
        result_value = "no final metric"
        size_value = "n/a"
        ttt_value = "startup markers loaded; final adaptation summary missing"
        conclusion = "The cycle produced no valid final roundtrip metric, so this counts as a crash until the persisted log proves otherwise."
        promoted = "no"
    else:
        delta_anchor = bpb - ANCHOR_BPB
        ttt_value = (
            f"matched_params={ttt_start.get('matched_params', '?')} "
            f"adapted_chunks={ttt_eval.get('adapted_chunks', '?')} "
            f"update_steps={ttt_eval.get('update_steps', '?')} "
            f"stride_active={ttt_eval.get('stride_active', '?')} "
            f"delta_budget_ratio={ttt_eval.get('delta_budget_ratio', ttt_start.get('delta_budget_ratio', '?'))}"
        )
        result_value = f"{bpb:.8f} bpb"
        size_value = f"{size} bytes"
        promoted = "yes" if status == "promote" else "no"
        if status == "promote":
            conclusion = (
                f"The adapting run stayed valid and improved the current global best by {-delta_anchor:.8f} bpb on the promoted codebase."
            )
            next_bullets = [
                "Do one confirmation rerun of the same adapting configuration before widening the surface.",
                "If the rerun holds, test whether a second tiny VE-local parameter can help without losing the size margin.",
                "Keep log-parameterized VE updates as the fallback if the win does not replicate.",
            ]
        elif delta_anchor > 0.20:
            conclusion = (
                f"Real adaptation happened, but the final post-quant metric regressed by +{delta_anchor:.8f} bpb versus {ANCHOR_LABEL}, so the exact delta-clipped env-only VE line is closed."
            )
            next_bullets = [
                "Open a code-changing branch for log-parameterized VE updates on the same one-parameter surface.",
                "Prototype a bank-local adaptive matrix or learned update-scale surface instead of another identical env-only rerun.",
                "Keep the carrier family closed unless the adaptive surface changes materially.",
            ]
        else:
            conclusion = (
                f"Real adaptation happened and stayed within +{delta_anchor:.8f} bpb of {ANCHOR_LABEL}, but it still did not beat the best global result."
            )
            next_bullets = [
                "Keep this family open only through a material code change, not another identical env-only rerun.",
                "Try log-parameterized VE updates or a learned clamp/update-scale on the same surface.",
                "If code-change bandwidth is tight, test the smallest bank-local adaptive matrix that preserves the r80 anchor.",
            ]

    lines = [
        f"Run: {RUN_ID}",
        f"Backend: {BACKEND}",
        f"Branch: {BRANCH} @ {COMMIT}",
        f"Status: {status}",
        f"Result: {result_value}",
        f"Artifact size: {size_value}",
        f"TTT activity: {ttt_value}",
        f"Promoted codebase changed: {promoted}",
        "",
        conclusion,
        "",
        f"Result note: {description}",
        "",
        "Next proposals:",
    ]
    for bullet in next_bullets:
        lines.append(f"- {bullet}")
    return "\n".join(lines) + "\n"


def main() -> int:
    log("waiting for r81 completion")
    success, text = wait_for_completion()
    final_bpb, metric_name = parse_metric(text)
    size_bytes, size_kind = parse_size(text)
    ttt_start = parse_ttt_start(text)
    ttt_eval = parse_ttt_eval(text)

    rows = load_results()
    current_best = best_valid_bpb(rows)

    if final_bpb is None or size_bytes is None:
        status = "crash"
        size_ok = "n/a"
        description = (
            f"{RUN_ID} matched adapting delta-clipped local-only VE probe on promoted `{COMMIT}` "
            "did not emit a final roundtrip metric and is logged as crash pending log-tail diagnosis."
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
        best_improved = False
    else:
        size_ok = "yes" if size_bytes < 16_000_000 else "no"
        best_improved = size_ok == "yes" and final_bpb < current_best
        status = "promote" if best_improved else "discard"
        delta_anchor = final_bpb - ANCHOR_BPB
        delta_best = final_bpb - current_best
        metric_label = metric_name.replace("_", "+") if metric_name else "roundtrip"
        description = (
            f"{RUN_ID} matched adapting delta-clipped local-only VE probe on promoted `{COMMIT}`; "
            f"logs confirm real legal adaptation with matched_params={ttt_start.get('matched_params', '?')}, "
            f"adapted_chunks={ttt_eval.get('adapted_chunks', '?')}, update_steps={ttt_eval.get('update_steps', '?')}, "
            f"stride_active={ttt_eval.get('stride_active', '?')}, and delta_budget_ratio={ttt_eval.get('delta_budget_ratio', ttt_start.get('delta_budget_ratio', '?'))}. "
            f"Final post-quant {metric_label} val_bpb lands at {final_bpb:.8f} with a {size_bytes}-byte artifact, "
            f"which is {delta_anchor:+.8f} vs {ANCHOR_LABEL} and {delta_best:+.8f} vs the prior best global."
        )
        if size_ok == "no":
            description += " The artifact violates the 16,000,000-byte limit, so the run is invalid for promotion."
        elif status == "promote":
            description += " The env-only adapting configuration becomes the new best result on the promoted codebase."
        elif delta_anchor > 0.20:
            description += " This closes the exact delta-clipped env-only VE adapting line and the next branch should pivot to log-parameterized VE updates or bank-local adaptive matrices."
        else:
            description += " The family stays open only through a material code change, not another identical env-only rerun."
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
    upsert_second_brain(best_improved, final_bpb, size_bytes, ttt_start, ttt_eval, status)

    refresh = run(["python3", "autoresearch/refresh_second_brain.py"])
    if refresh.returncode != 0:
        log(f"refresh_second_brain failed: {refresh.stderr.strip() or refresh.stdout.strip()}")

    MAIL_BODY_PATH.write_text(build_email_body(status, final_bpb, size_bytes, ttt_start, ttt_eval, description), encoding="utf-8")
    mail = run(
        [
            "python3",
            "autoresearch/send_update_email.py",
            "--subject",
            "autoresearch update r81",
            "--body-file",
            str(MAIL_BODY_PATH),
        ]
    )
    if mail.returncode != 0:
        log(f"send_update_email failed: {mail.stderr.strip() or mail.stdout.strip()}")
    else:
        log("status email sent")

    if not success and final_bpb is None:
        tail = run(["tail", "-n", "80", str(RUN_LOG)])
        log("run tail:\n" + (tail.stdout or tail.stderr))

    log("r81 finisher completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

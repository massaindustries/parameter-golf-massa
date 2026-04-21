#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH_DIR = ROOT / "autoresearch"
LOGS_DIR = ROOT / "logs"

DEFAULT_RUNPOD_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
DEFAULT_REMOTE_ROOT = "/workspace/parameter-golf"
DEFAULT_POD_NAME = "parameter-golf-autoresearch"
DEFAULT_MIN_BALANCE = 5.0
DEFAULT_COOLDOWN_MINUTES = 180


class RunpodUnavailable(RuntimeError):
    pass


@dataclass
class PodInfo:
    pod_id: str
    pod_name: str
    ssh_ip: str
    ssh_port: int
    ssh_key_path: str | None


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "default"


def runpod_state_name() -> str:
    explicit = os.environ.get("RUNPOD_STATE_NAME")
    if explicit:
        return sanitize_name(explicit)
    pod_name = os.environ.get("RUNPOD_POD_NAME")
    if pod_name:
        return sanitize_name(pod_name)
    return "default"


def state_path() -> Path:
    name = runpod_state_name()
    if name == "default":
        return AUTORESEARCH_DIR / "runpod_state.json"
    return AUTORESEARCH_DIR / f"runpod_state_{name}.json"


def lock_path() -> Path:
    name = runpod_state_name()
    if name == "default":
        return AUTORESEARCH_DIR / "runpod.lock"
    return AUTORESEARCH_DIR / f"runpod_{name}.lock"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def log(msg: str) -> None:
    ts = utc_now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin: Any = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        text=True,
        check=check,
        capture_output=capture,
        stdin=stdin,
    )


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(state: dict[str, Any]) -> None:
    state_path().write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def runpod_enabled() -> bool:
    return os.environ.get("RUNPOD_ENABLED", "0") == "1"


def cooldown_active(state: dict[str, Any]) -> bool:
    until = parse_iso8601(state.get("unhealthy_until"))
    return bool(until and utc_now() < until)


def mark_unhealthy(state: dict[str, Any], reason: str) -> None:
    minutes = int(os.environ.get("RUNPOD_COOLDOWN_MINUTES", str(DEFAULT_COOLDOWN_MINUTES)))
    until = utc_now() + timedelta(minutes=minutes)
    state["last_failure_reason"] = reason
    state["last_failure_at"] = utc_now().isoformat()
    state["unhealthy_until"] = until.isoformat()
    save_state(state)
    log(f"runpod_marked_unhealthy: {reason}; cooldown_until={until.isoformat()}")


def clear_unhealthy(state: dict[str, Any]) -> None:
    changed = False
    for key in ("last_failure_reason", "unhealthy_until"):
        if key in state:
            state.pop(key, None)
            changed = True
    if changed:
        save_state(state)


def parse_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def extract_balance(data: Any) -> float | None:
    if isinstance(data, dict):
        for key in ("balance", "creditBalance", "credits", "accountBalance", "availableBalance"):
            if key in data:
                parsed = parse_float(data[key])
                if parsed is not None:
                    return parsed
        for value in data.values():
            parsed = extract_balance(value)
            if parsed is not None:
                return parsed
    elif isinstance(data, list):
        for item in data:
            parsed = extract_balance(item)
            if parsed is not None:
                return parsed
    return None


def runpod_json(args: list[str], *, check: bool = True) -> Any:
    proc = run(["runpodctl", *args, "--output=json"], check=False, capture=True)
    payload = proc.stdout.strip() or proc.stderr.strip()
    if proc.returncode != 0:
        if check:
            raise RunpodUnavailable(f"runpodctl {' '.join(args)} failed: {payload or proc.returncode}")
        return None
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        if check:
            raise RunpodUnavailable(f"runpodctl {' '.join(args)} returned non-json output: {exc}")
        return None


def ensure_runpod_config(state: dict[str, Any]) -> None:
    if shutil.which("runpodctl") is None:
        raise RunpodUnavailable("runpodctl is not installed on Seeweb")
    config_candidates = [
        Path.home() / ".runpod" / "config.toml",
        Path.home() / ".runpod" / ".runpod.yaml",
        Path.home() / ".runpod.yaml",
    ]
    has_config_file = any(path.exists() for path in config_candidates)
    if not os.environ.get("RUNPOD_API_KEY") and not has_config_file:
        raise RunpodUnavailable("RUNPOD_API_KEY or ~/.runpod/config.toml is missing")
    if not os.environ.get("RUNPOD_GPU_ID"):
        raise RunpodUnavailable("RUNPOD_GPU_ID is not configured")
    if os.environ.get("RUNPOD_CHECK_BALANCE", "1") != "0":
        user_data = runpod_json(["user"], check=False)
        balance = extract_balance(user_data)
        if balance is not None:
            state["last_seen_balance_usd"] = balance
            save_state(state)
            min_balance = float(os.environ.get("RUNPOD_MIN_BALANCE_USD", str(DEFAULT_MIN_BALANCE)))
            if balance < min_balance:
                raise RunpodUnavailable(f"Runpod balance too low ({balance:.2f} < {min_balance:.2f})")


def pick_backend(requested: str, state: dict[str, Any]) -> str:
    if requested in {"seeweb", "runpod"}:
        return requested
    if not runpod_enabled():
        return "seeweb"
    if cooldown_active(state):
        return "seeweb"
    return "runpod"


def find_existing_pod(state: dict[str, Any], pod_name: str) -> dict[str, Any] | None:
    pods = runpod_json(["pod", "list", "--all"], check=False)
    if not isinstance(pods, list):
        return None
    wanted_id = state.get("pod_id")
    for pod in pods:
        if wanted_id and pod.get("id") == wanted_id:
            return pod
    for pod in pods:
        if pod.get("name") == pod_name:
            return pod
    return None


def ensure_pod_running(pod_id: str) -> None:
    attempts = int(os.environ.get("RUNPOD_START_ATTEMPTS", "3"))
    sleep_seconds = int(os.environ.get("RUNPOD_START_RETRY_SECONDS", "20"))
    for attempt in range(1, attempts + 1):
        proc = run(["runpodctl", "pod", "start", pod_id], check=False, capture=True)
        if proc.returncode == 0:
            return
        details = (proc.stderr or proc.stdout or "").strip()
        if attempt == attempts:
            raise RunpodUnavailable(
                f"runpodctl pod start {pod_id} failed after {attempts} attempts: {details or proc.returncode}"
            )
        log(
            f"runpod_start_retry: pod_id={pod_id} attempt={attempt}/{attempts} "
            f"details={details or proc.returncode}"
        )
        time.sleep(sleep_seconds)


def create_or_reuse_pod(state: dict[str, Any]) -> str:
    pod_name = os.environ.get("RUNPOD_POD_NAME", DEFAULT_POD_NAME)
    existing = find_existing_pod(state, pod_name)
    if existing:
        pod_id = existing["id"]
        desired = str(existing.get("desiredStatus", "")).upper()
        if desired != "RUNNING":
            log(f"runpod_start: pod_id={pod_id}")
            ensure_pod_running(pod_id)
        state["pod_id"] = pod_id
        state["pod_name"] = pod_name
        save_state(state)
        return pod_id

    cmd = [
        "runpodctl",
        "pod",
        "create",
        "--name",
        pod_name,
        "--image",
        os.environ.get("RUNPOD_IMAGE", DEFAULT_RUNPOD_IMAGE),
        "--gpu-id",
        os.environ["RUNPOD_GPU_ID"],
        "--container-disk-in-gb",
        os.environ.get("RUNPOD_CONTAINER_DISK_GB", "40"),
        "--ports",
        "22/tcp",
        "--ssh=true",
        "--cloud-type",
        os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE"),
    ]
    if os.environ.get("RUNPOD_GPU_COUNT"):
        cmd.extend(["--gpu-count", os.environ["RUNPOD_GPU_COUNT"]])
    if os.environ.get("RUNPOD_VOLUME_GB"):
        cmd.extend(["--volume-in-gb", os.environ["RUNPOD_VOLUME_GB"]])
    if os.environ.get("RUNPOD_NETWORK_VOLUME_ID"):
        cmd.extend(["--network-volume-id", os.environ["RUNPOD_NETWORK_VOLUME_ID"]])
    if os.environ.get("RUNPOD_DATA_CENTER_IDS"):
        cmd.extend(["--data-center-ids", os.environ["RUNPOD_DATA_CENTER_IDS"]])
    if os.environ.get("RUNPOD_PUBLIC_IP") == "1":
        cmd.extend(["--public-ip=true"])
    if os.environ.get("RUNPOD_CREATE_ENV_JSON"):
        cmd.extend(["--env", os.environ["RUNPOD_CREATE_ENV_JSON"]])

    log(f"runpod_create: name={pod_name}")
    created = runpod_json(cmd[1:], check=True)
    pod_id = created.get("id")
    if not pod_id:
        raise RunpodUnavailable("runpod pod create returned no pod id")
    state["pod_id"] = pod_id
    state["pod_name"] = pod_name
    save_state(state)
    return pod_id


def wait_for_ssh_info(pod_id: str) -> PodInfo:
    timeout_seconds = int(os.environ.get("RUNPOD_SSH_TIMEOUT_SECONDS", "600"))
    poll_seconds = int(os.environ.get("RUNPOD_SSH_POLL_SECONDS", "15"))
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        info = runpod_json(["ssh", "info", pod_id], check=False)
        if isinstance(info, dict) and info.get("ip") and info.get("port"):
            key_path = None
            ssh_key = info.get("ssh_key")
            if isinstance(ssh_key, dict):
                key_path = ssh_key.get("path")
            return PodInfo(
                pod_id=pod_id,
                pod_name=str(info.get("name") or pod_id),
                ssh_ip=str(info["ip"]),
                ssh_port=int(info["port"]),
                ssh_key_path=key_path,
            )
        time.sleep(poll_seconds)
    raise RunpodUnavailable(f"timed out waiting for ssh readiness on pod {pod_id}")


def ssh_base_args(pod: PodInfo) -> list[str]:
    args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=10",
    ]
    if pod.ssh_key_path:
        args.extend(["-i", pod.ssh_key_path])
    args.extend(["-p", str(pod.ssh_port), f"root@{pod.ssh_ip}"])
    return args


def ssh_run(
    pod: PodInfo,
    remote_cmd: str,
    *,
    check: bool = True,
    capture: bool = True,
    stdin: Any = None,
) -> subprocess.CompletedProcess[str]:
    cmd = ssh_base_args(pod) + [remote_cmd]
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        check=check,
        capture_output=capture,
        stdin=stdin,
    )


def sync_tree_to_remote(pod: PodInfo) -> None:
    excludes = [
        ".git",
        ".venv",
        ".mypy_cache",
        "__pycache__",
        "logs",
        "data/datasets",
        "final_model.pt",
        "final_model.int8.ptz",
        ".autoresearch_queue_state.json",
        "autoresearch/program.md",
        "autoresearch/results.tsv",
        "autoresearch/codex_mar24.log",
        "autoresearch/codex_mar25.log",
        "autoresearch/runpod_state.json",
        "autoresearch/runpod.lock",
    ]
    tar_cmd = ["tar"]
    for pattern in excludes:
        tar_cmd.extend(["--exclude", pattern])
    tar_cmd.extend(["-cf", "-", "."])

    remote_root = shlex.quote(os.environ.get("RUNPOD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    remote_cmd = f"mkdir -p {remote_root} && tar -C {remote_root} -xf -"
    log("runpod_sync_code: streaming repo snapshot")
    tar_proc = subprocess.Popen(tar_cmd, cwd=str(ROOT), stdout=subprocess.PIPE)
    try:
        ssh_proc = ssh_run(pod, remote_cmd, check=False, capture=True, stdin=tar_proc.stdout)
    finally:
        if tar_proc.stdout:
            tar_proc.stdout.close()
        tar_proc.wait()
    if tar_proc.returncode != 0 or ssh_proc.returncode != 0:
        raise RunpodUnavailable(
            f"code sync failed (tar={tar_proc.returncode}, ssh={ssh_proc.returncode}): "
            f"{ssh_proc.stderr.strip() or ssh_proc.stdout.strip()}"
        )


def ensure_remote_data(pod: PodInfo) -> None:
    remote_root = os.environ.get("RUNPOD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT)
    remote_dataset = f"{remote_root}/data/datasets/fineweb10B_sp1024"
    remote_tokenizer = f"{remote_root}/data/tokenizers/fineweb_1024_bpe.model"
    probe = ssh_run(
        pod,
        (
            f"test -d {shlex.quote(remote_dataset)} && "
            f"test -f {shlex.quote(remote_dataset + '/fineweb_train_000000.bin')} && "
            f"test -f {shlex.quote(remote_dataset + '/fineweb_val_000000.bin')} && "
            f"test -f {shlex.quote(remote_tokenizer)}"
        ),
        check=False,
        capture=True,
    )
    if probe.returncode == 0:
        return
    if os.environ.get("RUNPOD_SKIP_DATA_SYNC", "0") == "1":
        raise RunpodUnavailable("remote dataset/tokenizer missing and RUNPOD_SKIP_DATA_SYNC=1")
    log("runpod_sync_data: streaming dataset and tokenizer to pod")
    tar_cmd = [
        "tar",
        "-C",
        str(ROOT),
        "-cf",
        "-",
        "data/datasets/fineweb10B_sp1024",
        "data/tokenizers",
    ]
    remote_cmd = f"mkdir -p {shlex.quote(remote_root)} && tar -C {shlex.quote(remote_root)} -xf -"
    tar_proc = subprocess.Popen(tar_cmd, cwd=str(ROOT), stdout=subprocess.PIPE)
    try:
        ssh_proc = ssh_run(pod, remote_cmd, check=False, capture=True, stdin=tar_proc.stdout)
    finally:
        if tar_proc.stdout:
            tar_proc.stdout.close()
        tar_proc.wait()
    if tar_proc.returncode != 0 or ssh_proc.returncode != 0:
        raise RunpodUnavailable(
            f"dataset sync failed (tar={tar_proc.returncode}, ssh={ssh_proc.returncode}): "
            f"{ssh_proc.stderr.strip() or ssh_proc.stdout.strip()}"
        )


def ensure_remote_bootstrap(pod: PodInfo) -> None:
    remote_root = os.environ.get("RUNPOD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT)
    remote_cmd = (
        f"cd {shlex.quote(remote_root)} && "
        f"if [ ! -f .bootstrap_complete ]; then "
        f"python3 -m pip install --break-system-packages -r requirements.txt && touch .bootstrap_complete; "
        f"fi"
    )
    log("runpod_bootstrap: checking python deps on pod")
    proc = ssh_run(pod, remote_cmd, check=False, capture=True)
    if proc.returncode != 0:
        raise RunpodUnavailable(f"remote bootstrap failed: {proc.stderr.strip() or proc.stdout.strip()}")


def experiment_env() -> dict[str, str]:
    exact = {
        "RUN_ID",
        "ITERATIONS",
        "NPROC",
        "DATA_PATH",
        "TOKENIZER_PATH",
        "VOCAB_SIZE",
        "GPU_POLL_SECONDS",
        "MAX_WALLCLOCK_SECONDS",
        "TRAIN_LOG_EVERY",
        "VAL_LOSS_EVERY",
    }
    prefixes = (
        "TTT_",
        "EMA_",
        "EVAL_",
        "INT8_",
        "NUM_",
        "MODEL_",
        "MLP_",
        "LOGIT_",
        "ROPE_",
        "QK_",
        "TRAIN_",
        "VAL_",
        "WARM",
        "MUON_",
        "MATRIX_",
        "SCALAR_",
        "EMBED_",
        "TIED_",
        "BETA",
        "MOMENTUM",
        "BATCH",
        "SEQ",
        "ACTIVATION",
        "NORM_",
        "SELF_",
        "ADAPT_",
        "SKIP_",
        "RESID_",
        "ATTN_",
        "KV_",
        "FFN_",
        "BIAS_",
        "SWA_",
        "LORA_",
        "CLIP_",
    )
    env = {}
    for key, value in os.environ.items():
        if key in exact or key.startswith(prefixes):
            env[key] = value
    env.setdefault("RUN_ID", f"autoresearch_{int(time.time())}")
    env.setdefault("ITERATIONS", "20000")
    env.setdefault("NPROC", "1")
    return env


def run_remote_experiment(pod: PodInfo) -> int:
    env = experiment_env()
    remote_root = os.environ.get("RUNPOD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT)
    env_assignment = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    remote_cmd = f"cd {shlex.quote(remote_root)} && {env_assignment} bash autoresearch/run_experiment.sh"
    log(f"runpod_launch: run_id={env['RUN_ID']} pod_id={pod.pod_id}")
    proc = subprocess.run(ssh_base_args(pod) + [remote_cmd], cwd=str(ROOT), text=True)
    return proc.returncode


def fetch_remote_log(pod: PodInfo, run_id: str) -> bool:
    remote_root = os.environ.get("RUNPOD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT)
    remote_log = f"{remote_root}/logs/{run_id}.txt"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    target = LOGS_DIR / f"{run_id}.txt"
    proc = subprocess.run(
        ssh_base_args(pod) + [f"cat {shlex.quote(remote_log)}"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return False
    target.write_text(proc.stdout, encoding="utf-8")
    return True


def stop_pod_if_requested(pod_id: str) -> None:
    if os.environ.get("RUNPOD_STOP_AFTER_RUN", "1") != "1":
        return
    run(["runpodctl", "pod", "stop", pod_id], check=False, capture=True)


def run_local() -> int:
    log("backend=seeweb")
    proc = subprocess.run(["bash", "autoresearch/run_experiment.sh"], cwd=str(ROOT), text=True)
    return proc.returncode


def run_runpod(state: dict[str, Any]) -> int:
    ensure_runpod_config(state)
    pod_id = create_or_reuse_pod(state)
    pod = wait_for_ssh_info(pod_id)
    sync_tree_to_remote(pod)
    ensure_remote_data(pod)
    ensure_remote_bootstrap(pod)
    run_id = experiment_env()["RUN_ID"]
    exit_code = run_remote_experiment(pod)
    fetched = fetch_remote_log(pod, run_id)
    stop_pod_if_requested(pod.pod_id)
    state["last_success_at"] = utc_now().isoformat()
    state["pod_id"] = pod.pod_id
    state["pod_name"] = pod.pod_name
    save_state(state)
    if fetched:
        log(f"runpod_log_fetched: {LOGS_DIR / f'{run_id}.txt'}")
        return exit_code
    raise RunpodUnavailable(f"remote run completed but log fetch failed for {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch Parameter Golf experiments to Seeweb or Runpod.")
    parser.add_argument("--backend", choices=["auto", "seeweb", "runpod"], default=os.environ.get("AR_BACKEND", "auto"))
    parser.add_argument(
        "--allow-seeweb-fallback",
        action="store_true",
        default=os.environ.get("ALLOW_SEEWEB_FALLBACK", "1") == "1",
        help="If Runpod is unavailable, retry on Seeweb instead of failing hard.",
    )
    args = parser.parse_args()

    state = load_state()
    backend = pick_backend(args.backend, state)
    if backend == "seeweb":
        return run_local()

    with lock_path().open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            reason = "another Runpod experiment is already active"
            if args.allow_seeweb_fallback:
                log(f"runpod_unavailable: {reason}; falling back to Seeweb")
                return run_local()
            raise RunpodUnavailable(reason) from exc

        try:
            log("backend=runpod")
            clear_unhealthy(state)
            return run_runpod(state)
        except RunpodUnavailable as exc:
            mark_unhealthy(state, str(exc))
            if args.allow_seeweb_fallback:
                log(f"runpod_unavailable: {exc}; falling back to Seeweb")
                return run_local()
            raise
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunpodUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

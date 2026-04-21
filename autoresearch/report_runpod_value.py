#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"step:(\d+)/(\d+)\s+.*?step_avg:(\d+(?:\.\d+)?)ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate price/performance from a training log.")
    parser.add_argument("--log", required=True, help="Path to a training log file.")
    parser.add_argument("--cost-per-hour", type=float, required=True, help="Hourly pod cost in USD.")
    parser.add_argument("--label", default="", help="Optional label for the report.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    matches = STEP_RE.findall(text)
    if not matches:
        raise SystemExit(f"no step_avg lines found in {args.log}")

    step, total_steps, step_avg_ms = matches[-1]
    step_i = int(step)
    total_steps_i = int(total_steps)
    step_avg_ms_f = float(step_avg_ms)

    total_train_hours = (step_avg_ms_f * total_steps_i) / 3_600_000.0
    total_train_cost_usd = total_train_hours * args.cost_per_hour
    cost_per_1k_steps = total_train_cost_usd / max(total_steps_i / 1000.0, 1e-9)

    report = {
        "label": args.label,
        "log": str(Path(args.log).resolve()),
        "observed_step": step_i,
        "total_steps": total_steps_i,
        "step_avg_ms": step_avg_ms_f,
        "cost_per_hour_usd": args.cost_per_hour,
        "estimated_train_hours": total_train_hours,
        "estimated_train_cost_usd": total_train_cost_usd,
        "estimated_cost_per_1k_steps_usd": cost_per_1k_steps,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    label = f"{args.label}: " if args.label else ""
    print(f"{label}{report['log']}")
    print(f"observed_step={step_i}/{total_steps_i}")
    print(f"step_avg_ms={step_avg_ms_f:.2f}")
    print(f"cost_per_hour_usd={args.cost_per_hour:.2f}")
    print(f"estimated_train_hours={total_train_hours:.3f}")
    print(f"estimated_train_cost_usd={total_train_cost_usd:.3f}")
    print(f"estimated_cost_per_1k_steps_usd={cost_per_1k_steps:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.tsv"
OUT_PATH = ROOT / "second_brain_snapshot.md"
RECENT_LIMIT = 8
BEST_LIMIT = 4
VALID_STATUSES = {"keep", "promote", "anchor"}


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def load_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("inf")


def is_valid(row: dict[str, str]) -> bool:
    return row.get("size_ok") == "yes" and row.get("status") in VALID_STATUSES


def is_adapting_ttt(row: dict[str, str]) -> bool:
    params = row.get("param_changes", "")
    return "TTT_ENABLE=1" in params and "TTT_EPOCHS=1" in params


def normalized_status(row: dict[str, str], best_bpb: float) -> str:
    status = row.get("status", "")
    if status != "keep":
        return status
    bpb = parse_float(row.get("val_bpb", ""))
    if bpb != float("inf") and best_bpb != float("inf") and abs(bpb - best_bpb) < 1e-12:
        return "promote"
    return "anchor"


def format_run_line(row: dict[str, str], best_bpb: float) -> str:
    bpb = parse_float(row["val_bpb"])
    delta = bpb - best_bpb if best_bpb != float("inf") and bpb != float("inf") else float("inf")
    delta_str = "NA" if delta == float("inf") else f"+{delta:.6f}"
    return (
        f"- {normalized_status(row, best_bpb)} | bpb {row['val_bpb']} | d_best {delta_str} | "
        f"size {row['size_bytes']} | change {shorten(row['param_changes'], 76)}"
    )


def build_snapshot(rows: list[dict[str, str]]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if not rows:
        return (
            "# Second Brain Snapshot\n\n"
            f"Updated: {now}\n\n"
            "No results available yet.\n"
        )

    valid_rows = [row for row in rows if is_valid(row)]
    valid_rows_sorted = sorted(valid_rows, key=lambda row: parse_float(row["val_bpb"]))
    best_row = valid_rows_sorted[0] if valid_rows_sorted else None
    best_bpb = parse_float(best_row["val_bpb"]) if best_row else float("inf")

    plain_rows = [
        row
        for row in valid_rows
        if "TTT_ENABLE=1" not in row.get("param_changes", "") or "TTT_EPOCHS=0" in row.get("param_changes", "")
    ]
    best_plain = min(plain_rows, key=lambda row: parse_float(row["val_bpb"])) if plain_rows else None

    recent_rows = rows[-RECENT_LIMIT:]
    top_rows = valid_rows_sorted[:BEST_LIMIT]
    recent_adapting = [row for row in rows[::-1] if is_adapting_ttt(row)]
    last_adapting = recent_adapting[:4]
    adapt_line = "No adapting TTT runs yet."
    pressure_line = "No automatic pivot signal yet."
    if last_adapting:
        deltas = []
        for row in last_adapting:
            bpb = parse_float(row["val_bpb"])
            if bpb != float("inf") and best_bpb != float("inf"):
                deltas.append(bpb - best_bpb)
        if deltas:
            adapt_line = (
                f"last {len(deltas)} adapting TTT runs valid="
                f"{'yes' if all(row['status'] != 'crash' for row in last_adapting) else 'no'} | "
                f"delta_vs_best +{min(deltas):.6f}..+{max(deltas):.6f}"
            )
            if min(deltas) > 0.20:
                pressure_line = (
                    f"pivot: recent adapting TTT family is broken "
                    f"(all last {len(deltas)} runs > +0.20 bpb vs best)"
                )
            else:
                pressure_line = "recent adapting TTT is not auto-broken by the +0.20 rule"

    lines = [
        "# Second Brain Snapshot",
        "",
        f"Updated: {now}",
        "",
    ]

    if best_row:
        lines.extend(
            [
                "## Best Global",
                format_run_line(best_row, best_bpb),
                f"- note {shorten(best_row['description'], 110)}",
                "",
            ]
        )

    if best_plain and best_plain is not best_row:
        lines.extend(
            [
                "## Best Plain Or No-Update",
                format_run_line(best_plain, best_bpb),
                f"- note {shorten(best_plain['description'], 110)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Trend",
            f"- valid runs {len(valid_rows)} / total {len(rows)}",
            f"- {adapt_line}",
            f"- {pressure_line}",
            "",
        ]
    )

    lines.append("## Recent Runs")
    for row in recent_rows:
        lines.append(format_run_line(row, best_bpb))
    lines.append("")

    if top_rows:
        lines.append("## Top Valid")
        for row in top_rows:
            lines.append(format_run_line(row, best_bpb))
        lines.append("")

    lines.extend(
        [
            "## Compression Rules",
            "- Read this snapshot first, then read second_brain.md, then inspect raw logs only if needed.",
            "- Do not paste log walls into second_brain.md.",
            "- Preserve only the smallest set of facts needed to choose the next run.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    OUT_PATH.write_text(build_snapshot(rows), encoding="utf-8")


if __name__ == "__main__":
    main()

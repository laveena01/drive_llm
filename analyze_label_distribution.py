"""
A1 — Label distribution analysis.

Loads a run's driving_qa_data.json, extracts target field distributions
(brake_pct, accelerator_pct, steering, risk_level), and writes both a
machine-readable JSON and a human-readable markdown summary.

Purpose: quantify the trivial-baseline accuracy floor caused by label
imbalance. If, say, 91% of action targets have brake=0%, a constant
predictor scores 91% — the saturation isn't reasoning, it's class
imbalance.

Usage:
    python analyze_label_distribution.py --run <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional


# Regex patterns matching the canonical target string format from
# `_paper_target` in datasets_builder.py.
RE_BRAKE = re.compile(r"Brake pedal:\s*(\d+)%", re.IGNORECASE)
RE_ACCEL = re.compile(r"Accelerator pedal:\s*(\d+)%", re.IGNORECASE)
RE_STEER = re.compile(r"Steering:\s*(left|straight|right)", re.IGNORECASE)
RE_RISKLVL = re.compile(
    r"Risk level:\s*(CRITICAL|HIGH|MODERATE|LOW|MINIMAL)", re.IGNORECASE
)


def _extract_int(pattern: re.Pattern, text: str) -> Optional[int]:
    m = pattern.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_str(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text or "")
    if not m:
        return None
    return m.group(1).strip().lower()


def _resolve_run_id(arg_run: Optional[str]) -> str:
    if arg_run:
        return arg_run
    env_run = os.environ.get("RUN_ID")
    if env_run:
        return env_run
    pointer = os.path.join("runs", ".latest_build_run_id")
    if os.path.isfile(pointer):
        with open(pointer, "r") as f:
            return f.read().strip()
    sys.exit(
        "[A1] No --run argument, no RUN_ID env var, no .latest_build_run_id pointer."
    )


def _percent(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * n / total):.1f}%"


def _counter_table(counter: Counter, total: int, key_order: Optional[List] = None) -> str:
    if not counter:
        return "_(no data)_\n"
    if key_order is None:
        # Sort by key (numeric if possible, else string).
        try:
            keys = sorted(counter.keys(), key=lambda x: (int(x), x))
        except Exception:
            keys = sorted(counter.keys(), key=lambda x: str(x))
    else:
        # Use the provided order, but include any extra keys at the end.
        seen = set(key_order)
        keys = list(key_order) + [k for k in counter.keys() if k not in seen]

    lines = ["| Value | Count | % |", "|---|---:|---:|"]
    for k in keys:
        count = counter.get(k, 0)
        lines.append(f"| `{k}` | {count} | {_percent(count, total)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="A1 — Label distribution analysis")
    p.add_argument("--run", default=None, help="Run ID (folder under runs/)")
    args = p.parse_args()

    run_id = _resolve_run_id(args.run)
    qa_path = os.path.join("runs", run_id, "data", "driving_qa_data.json")
    out_json = os.path.join("runs", run_id, "data", "label_distributions.json")
    out_md = os.path.join("runs", run_id, "data", "label_distributions.md")

    if not os.path.isfile(qa_path):
        sys.exit(f"[A1] QA data not found at {qa_path}")

    print(f"[A1] Loading QA data from {qa_path}")
    with open(qa_path, "r") as f:
        data: List[Dict] = json.load(f)
    print(f"[A1] Loaded {len(data)} samples")

    # Per-question-type counters.
    qtype_counter: Counter = Counter()

    # action + action_future (action-like) — extract brake/accel/steer.
    brake_by_qtype: Dict[str, Counter] = {"action": Counter(), "action_future": Counter()}
    accel_by_qtype: Dict[str, Counter] = {"action": Counter(), "action_future": Counter()}
    steer_by_qtype: Dict[str, Counter] = {"action": Counter(), "action_future": Counter()}

    # risk samples — extract risk_level from target.
    risk_target_counter: Counter = Counter()

    # Sample-level fields (present on every sample regardless of qtype).
    risk_level_counter: Counter = Counter()
    policy_label_counter: Counter = Counter()
    risk_level_future_counter: Counter = Counter()
    brake_required_future_counter: Counter = Counter()

    for s in data:
        qtype = (s.get("question_type") or "action").strip().lower()
        qtype_counter[qtype] += 1

        target = s.get("target", "") or ""

        if qtype in ("action", "action_future"):
            b = _extract_int(RE_BRAKE, target)
            a = _extract_int(RE_ACCEL, target)
            st = _extract_str(RE_STEER, target)
            if b is not None:
                brake_by_qtype[qtype][b] += 1
            if a is not None:
                accel_by_qtype[qtype][a] += 1
            if st is not None:
                steer_by_qtype[qtype][st] += 1
        elif qtype == "risk":
            rl = _extract_str(RE_RISKLVL, target)
            if rl is not None:
                risk_target_counter[rl.upper()] += 1

        rl_field = (s.get("risk_level") or "").upper().strip()
        if rl_field:
            risk_level_counter[rl_field] += 1

        pl_field = (s.get("policy_label") or "").upper().strip()
        if pl_field:
            policy_label_counter[pl_field] += 1

        rlf_field = (s.get("risk_level_future") or "").upper().strip()
        if rlf_field:
            risk_level_future_counter[rlf_field] += 1

        brf_field = s.get("brake_required_future")
        if brf_field is not None:
            brake_required_future_counter[bool(brf_field)] += 1

    total = len(data)
    n_action = qtype_counter.get("action", 0)
    n_action_future = qtype_counter.get("action_future", 0)
    n_risk = qtype_counter.get("risk", 0)

    # --- Write JSON ---
    out = {
        "run": run_id,
        "total_samples": total,
        "question_type_counts": dict(qtype_counter),
        "brake_pct_by_qtype": {
            qt: dict(c) for qt, c in brake_by_qtype.items()
        },
        "accelerator_pct_by_qtype": {
            qt: dict(c) for qt, c in accel_by_qtype.items()
        },
        "steering_by_qtype": {
            qt: dict(c) for qt, c in steer_by_qtype.items()
        },
        "risk_level_target_counts": dict(risk_target_counter),
        "risk_level_current_frame_counts": dict(risk_level_counter),
        "policy_label_counts": dict(policy_label_counter),
        "risk_level_future_counts": dict(risk_level_future_counter),
        "brake_required_future_counts": {
            str(k): v for k, v in brake_required_future_counter.items()
        },
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A1] Wrote machine-readable JSON: {out_json}")

    # --- Write markdown ---
    risk_order = ["MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]

    md_lines: List[str] = []
    md_lines.append(f"# Label distributions for run `{run_id}`\n")
    md_lines.append(f"**Total samples**: {total}\n\n")

    md_lines.append("## Question type breakdown\n")
    md_lines.append("| question_type | Count | % |")
    md_lines.append("|---|---:|---:|")
    for qt in ("action", "action_future", "risk"):
        n = qtype_counter.get(qt, 0)
        md_lines.append(f"| `{qt}` | {n} | {_percent(n, total)} |")
    md_lines.append("")

    md_lines.append("## Brake pedal % — `action` samples (per-frame target)")
    md_lines.append(f"Total `action` rows: {n_action}\n")
    md_lines.append(_counter_table(brake_by_qtype["action"], n_action))

    md_lines.append("## Brake pedal % — `action_future` samples (anticipation target)")
    md_lines.append(f"Total `action_future` rows: {n_action_future}\n")
    md_lines.append(_counter_table(brake_by_qtype["action_future"], n_action_future))

    md_lines.append("## Accelerator pedal % — `action` samples")
    md_lines.append(_counter_table(accel_by_qtype["action"], n_action))

    md_lines.append("## Accelerator pedal % — `action_future` samples")
    md_lines.append(_counter_table(accel_by_qtype["action_future"], n_action_future))

    md_lines.append("## Steering — `action` samples")
    md_lines.append(_counter_table(
        steer_by_qtype["action"], n_action,
        key_order=["straight", "left", "right"],
    ))

    md_lines.append("## Steering — `action_future` samples")
    md_lines.append(_counter_table(
        steer_by_qtype["action_future"], n_action_future,
        key_order=["straight", "left", "right"],
    ))

    md_lines.append("## Risk level — extracted from `risk` question targets")
    md_lines.append(f"Total `risk` rows: {n_risk}\n")
    md_lines.append(_counter_table(risk_target_counter, n_risk, key_order=risk_order))

    md_lines.append("## Risk level — sample's current-frame `risk_level` field (all sample types)")
    md_lines.append(_counter_table(risk_level_counter, total, key_order=risk_order))

    md_lines.append("## Policy label (current frame)")
    md_lines.append(_counter_table(policy_label_counter, total))

    md_lines.append("## `risk_level_future` — max severity over H=4 lookahead (all sample types)")
    md_lines.append(_counter_table(risk_level_future_counter, total, key_order=risk_order))

    md_lines.append("## `brake_required_future` — anticipation slice")
    md_lines.append(_counter_table(
        brake_required_future_counter,
        sum(brake_required_future_counter.values()),
        key_order=[True, False],
    ))

    # Trivial-baseline note.
    md_lines.append("## Trivial-baseline floor\n")
    if brake_by_qtype["action"]:
        top_brake, top_count = brake_by_qtype["action"].most_common(1)[0]
        pct = 100.0 * top_count / max(n_action, 1)
        md_lines.append(
            f"- `action` brake distribution dominant class: **brake = {top_brake}%** "
            f"({top_count}/{n_action} = {pct:.1f}%)"
        )
        md_lines.append(
            f"- A constant predictor that always outputs brake={top_brake}% scores "
            f"**{pct:.1f}%** on per-frame action accuracy by chance alone."
        )
        md_lines.append("")
    if steer_by_qtype["action"]:
        top_st, top_count = steer_by_qtype["action"].most_common(1)[0]
        pct = 100.0 * top_count / max(n_action, 1)
        md_lines.append(
            f"- `action` steering dominant class: **{top_st}** "
            f"({top_count}/{n_action} = {pct:.1f}%)"
        )
        md_lines.append("")

    md_lines.append(
        "**Interpretation**: heavily-skewed dimensions inflate aggregate accuracy. "
        "Slice metrics and class-imbalanced metrics (e.g., per-class recall) are the "
        "honest measures of capability."
    )

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[A1] Wrote human-readable markdown: {out_md}")

    # Console summary.
    print("\n[A1] Quick summary:")
    print(f"  Total samples: {total}")
    print(f"  By question_type: {dict(qtype_counter)}")
    if brake_by_qtype["action"]:
        top_brake, top_count = brake_by_qtype["action"].most_common(1)[0]
        print(
            f"  Most common 'action' brake: {top_brake}% "
            f"({top_count}/{n_action} = {_percent(top_count, n_action)})"
        )
    if brake_by_qtype["action_future"]:
        top_brake, top_count = brake_by_qtype["action_future"].most_common(1)[0]
        print(
            f"  Most common 'action_future' brake: {top_brake}% "
            f"({top_count}/{n_action_future} = {_percent(top_count, n_action_future)})"
        )


if __name__ == "__main__":
    main()

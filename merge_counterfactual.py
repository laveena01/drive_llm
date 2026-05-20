"""
merge_counterfactual.py — aggregate counterfactual_shard*.json into per-perturbation metrics.

For each perturbation, computes:
  - sensitivity_rate     (|delta_brake| > 20%)
  - direction_accuracy   (delta sign matches expected_direction)
  - mean_abs_delta
  - parse_ok_rate
  - applied_rate         (fraction of samples where perturbation was non-no-op)
For P4 additionally:
  - text_follower_rate   (action followed the swapped risk level)
  - vector_follower_rate (action stuck with original)
Plus dumps a flat list of delta_brake values per perturbation for CDF plotting.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

EXPECTED_DIRECTION = {
    "P1_remove_closest":     "down",
    "P2_inject_threat":      "up",
    "P3_halve_ttc":          "up",
    "P4_swap_risk_level":    "depends",
    "P5_mask_pedestrians":   "down",
    "P6_double_ego_speed":   "up",
}

SENSITIVITY_THRESHOLD = 20.0  # percent points on brake


def aggregate_perturbation(rows, kind):
    n_total = len(rows)
    n_applied = sum(1 for r in rows if r.get("perturbation_applied"))
    n_parse_ok = sum(
        1 for r in rows
        if r.get("base_brake") is not None and r.get("pert_brake") is not None
    )
    deltas = [r["delta_brake"] for r in rows if r.get("delta_brake") is not None]
    abs_deltas = [abs(d) for d in deltas]

    n_changed = sum(1 for d in abs_deltas if d > SENSITIVITY_THRESHOLD)
    sensitivity_rate = n_changed / n_parse_ok if n_parse_ok else 0.0
    mean_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0

    exp_dir = EXPECTED_DIRECTION.get(kind, "up")
    if exp_dir in ("up", "down") and n_changed:
        agreed = 0
        for d in deltas:
            if abs(d) <= SENSITIVITY_THRESHOLD:
                continue
            if exp_dir == "up" and d > 0:
                agreed += 1
            elif exp_dir == "down" and d < 0:
                agreed += 1
        direction_accuracy = agreed / n_changed
    else:
        direction_accuracy = None

    summary = {
        "n_total": n_total,
        "n_applied": n_applied,
        "applied_rate": n_applied / n_total if n_total else 0.0,
        "n_parse_ok": n_parse_ok,
        "parse_ok_rate": n_parse_ok / n_total if n_total else 0.0,
        "sensitivity_rate": sensitivity_rate,
        "direction_accuracy": direction_accuracy,
        "mean_abs_delta": mean_abs_delta,
        "expected_direction": exp_dir,
    }

    # P4-specific: text_follower vs vector_follower
    if kind == "P4_swap_risk_level":
        text_follow = 0
        vector_follow = 0
        considered = 0
        for r in rows:
            if not r.get("perturbation_applied"):
                continue
            db = r.get("delta_brake")
            if db is None:
                continue
            sd = r.get("p4_swap_direction")
            considered += 1
            if sd == "low_to_critical":
                # if model now brakes harder (db > 0), it followed the swapped text
                if db > SENSITIVITY_THRESHOLD:
                    text_follow += 1
                elif abs(db) <= SENSITIVITY_THRESHOLD:
                    vector_follow += 1
            elif sd == "critical_to_low":
                # if model now brakes less (db < 0), it followed the swapped text
                if db < -SENSITIVITY_THRESHOLD:
                    text_follow += 1
                elif abs(db) <= SENSITIVITY_THRESHOLD:
                    vector_follow += 1
        summary["p4_text_follower_rate"] = text_follow / considered if considered else 0.0
        summary["p4_vector_follower_rate"] = vector_follow / considered if considered else 0.0
        summary["p4_considered"] = considered

    summary["deltas"] = deltas  # raw distribution for plotting
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    cf_dir = os.path.join("runs", args.run, "counterfactual")
    paths = sorted(glob.glob(os.path.join(cf_dir, "counterfactual_shard*.json")))
    if not paths:
        raise RuntimeError(f"No shard files found under {cf_dir}")

    # Concatenate per-perturbation rows across shards
    merged = defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        for kind, rows in d.items():
            merged[kind].extend(rows)

    metrics = {}
    for kind, rows in merged.items():
        metrics[kind] = aggregate_perturbation(rows, kind)

    metrics["_meta"] = {
        "run": args.run,
        "num_shards": args.num_shards,
        "shard_paths": paths,
        "sensitivity_threshold_pct": SENSITIVITY_THRESHOLD,
    }

    out_path = os.path.join(cf_dir, "counterfactual_metrics.json")
    json.dump(metrics, open(out_path, "w"), indent=2)
    print(f"[CF-MERGE] wrote {out_path}")

    # Print compact summary table to stdout
    print("\nPerturbation               sensitivity  direction_acc  mean|Δbrake|  parse_ok")
    print("-" * 90)
    for kind in [
        "P1_remove_closest",
        "P2_inject_threat",
        "P3_halve_ttc",
        "P4_swap_risk_level",
        "P5_mask_pedestrians",
        "P6_double_ego_speed",
    ]:
        if kind not in metrics:
            continue
        m = metrics[kind]
        da = m["direction_accuracy"]
        da_str = f"{da:.2f}" if da is not None else "  n/a"
        print(f"{kind:25s}  {m['sensitivity_rate']:>10.3f}  {da_str:>13s}  {m['mean_abs_delta']:>12.2f}  {m['parse_ok_rate']:>7.2f}")
    if "P4_swap_risk_level" in metrics:
        m4 = metrics["P4_swap_risk_level"]
        print(f"\nP4 text_follower: {m4.get('p4_text_follower_rate', 0):.3f}")
        print(f"P4 vector_follower: {m4.get('p4_vector_follower_rate', 0):.3f}")


if __name__ == "__main__":
    main()

"""
A2 — Scene-level leakage diagnostic.

Loads a run's driving_qa_data.json, reconstructs the exact same train/val
split that `training.py` uses (`train_test_split(test_size=0.2, seed=42)`
from HuggingFace `datasets`), and reports how leaky that random split is
with respect to scene identity.

Three leakage levels are measured:

1. Scene-level overlap — how many val scenes also appear in train?
2. Frame-level leakage — how many val frames are from a scene that also
   has frames in train?
3. Adjacency leakage — how many val frames have an IMMEDIATELY adjacent
   frame (frame_in_scene ± 1) in train? This is the worst-case leakage.

Purpose: justify (or skip) the scene-level-split rerun based on measured
numbers. The leakage % itself is a reportable methodology finding.

Usage:
    python analyze_scene_leakage.py --run <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from datasets import Dataset


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
        "[A2] No --run argument, no RUN_ID env var, no .latest_build_run_id pointer."
    )


def _percent(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * n / total):.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="A2 — Scene-level leakage diagnostic")
    p.add_argument("--run", default=None, help="Run ID (folder under runs/)")
    p.add_argument(
        "--test_size", type=float, default=0.2,
        help="Match training.py's split test_size (default 0.2)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Match training.py's split seed (default 42)",
    )
    args = p.parse_args()

    run_id = _resolve_run_id(args.run)
    qa_path = os.path.join("runs", run_id, "data", "driving_qa_data.json")
    out_json = os.path.join("runs", run_id, "data", "scene_leakage_report.json")
    out_md = os.path.join("runs", run_id, "data", "scene_leakage_report.md")

    if not os.path.isfile(qa_path):
        sys.exit(f"[A2] QA data not found at {qa_path}")

    print(f"[A2] Loading QA data from {qa_path}")
    with open(qa_path, "r") as f:
        data: List[Dict] = json.load(f)
    print(f"[A2] Loaded {len(data)} samples")

    # Reproduce the EXACT split used by training.py.
    # See training.py:457, 1032, 1321.
    full_ds = Dataset.from_list(data)
    split_ds = full_ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    print(f"[A2] Split sizes: train={len(train_ds)}, val={len(val_ds)}")

    # --- Collect scene_idx sets and (scene_idx, frame_in_scene) sets ---
    train_scenes: Set = set()
    val_scenes: Set = set()
    train_frames: Set[Tuple] = set()
    val_frames: Set[Tuple] = set()

    # Group val by scene for adjacency analysis.
    val_by_scene: Dict[int, List[int]] = defaultdict(list)
    train_by_scene: Dict[int, Set[int]] = defaultdict(set)

    missing_field = 0
    for s in train_ds:
        sc = s.get("scene_idx")
        fr = s.get("frame_in_scene")
        if sc is None or fr is None:
            missing_field += 1
            continue
        train_scenes.add(sc)
        train_frames.add((sc, fr))
        train_by_scene[sc].add(fr)

    for s in val_ds:
        sc = s.get("scene_idx")
        fr = s.get("frame_in_scene")
        if sc is None or fr is None:
            missing_field += 1
            continue
        val_scenes.add(sc)
        val_frames.add((sc, fr))
        val_by_scene[sc].append(fr)

    if missing_field > 0:
        print(
            f"[A2] WARNING: {missing_field} samples missing scene_idx or "
            "frame_in_scene — excluded from analysis."
        )

    # --- Scene-level overlap ---
    scenes_in_both = train_scenes & val_scenes
    scenes_unique_to_val = val_scenes - train_scenes
    scenes_unique_to_train = train_scenes - val_scenes
    n_train_scenes = len(train_scenes)
    n_val_scenes = len(val_scenes)
    n_overlap = len(scenes_in_both)

    # --- Frame-level leakage ---
    # For each val sample (counted as a frame entry, not unique frames),
    # does its scene appear in train?
    n_val_total = len(val_ds)
    n_val_in_leaky_scene = 0
    for s in val_ds:
        sc = s.get("scene_idx")
        if sc in train_scenes:
            n_val_in_leaky_scene += 1
    n_val_in_pure_scene = n_val_total - n_val_in_leaky_scene

    # --- Adjacency leakage ---
    # For each unique val frame, is there a train frame adjacent in time?
    n_unique_val_frames = len(val_frames)
    n_adjacent_in_train = 0
    n_same_frame_in_train = 0  # shouldn't happen, but verify
    for (sc, fr) in val_frames:
        train_frs = train_by_scene.get(sc, set())
        if not train_frs:
            continue
        if fr in train_frs:
            n_same_frame_in_train += 1
        if (fr - 1) in train_frs or (fr + 1) in train_frs:
            n_adjacent_in_train += 1

    # --- Build JSON output ---
    report = {
        "run": run_id,
        "split": {
            "test_size": args.test_size,
            "seed": args.seed,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        "scene_level": {
            "unique_scenes_train": n_train_scenes,
            "unique_scenes_val": n_val_scenes,
            "scenes_in_both_sets": n_overlap,
            "scenes_unique_to_val": len(scenes_unique_to_val),
            "scenes_unique_to_train": len(scenes_unique_to_train),
            "scene_overlap_pct_of_val": (100.0 * n_overlap / max(n_val_scenes, 1)),
        },
        "frame_level": {
            "val_frames_in_leaky_scene": n_val_in_leaky_scene,
            "val_frames_in_pure_scene": n_val_in_pure_scene,
            "frame_level_leakage_pct": (100.0 * n_val_in_leaky_scene / max(n_val_total, 1)),
        },
        "adjacency_level": {
            "unique_val_frames": n_unique_val_frames,
            "val_frames_with_adjacent_train_frame": n_adjacent_in_train,
            "val_frames_with_same_frame_in_train": n_same_frame_in_train,
            "adjacency_leakage_pct": (100.0 * n_adjacent_in_train / max(n_unique_val_frames, 1)),
        },
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[A2] Wrote machine-readable JSON: {out_json}")

    # --- Decision rule application ---
    frame_leak_pct = report["frame_level"]["frame_level_leakage_pct"]
    adj_leak_pct = report["adjacency_level"]["adjacency_leakage_pct"]
    rerun_recommended = (frame_leak_pct >= 30.0) or (adj_leak_pct >= 10.0)

    # --- Build markdown output ---
    md: List[str] = []
    md.append(f"# Scene-level leakage report for run `{run_id}`\n")
    md.append(
        f"Split parameters: `train_test_split(test_size={args.test_size}, "
        f"seed={args.seed})` (replays the exact split used in training.py).\n"
    )

    md.append("## Split summary\n")
    md.append("| | Count |")
    md.append("|---|---:|")
    md.append(f"| Train samples | {len(train_ds)} |")
    md.append(f"| Val samples | {len(val_ds)} |")
    md.append(f"| Total | {len(train_ds) + len(val_ds)} |")
    md.append(f"| Unique scenes in train | {n_train_scenes} |")
    md.append(f"| Unique scenes in val | {n_val_scenes} |")
    md.append("")

    md.append("## Scene-level overlap\n")
    md.append(f"- Scenes appearing in **BOTH** train and val: **{n_overlap}**")
    md.append(
        f"  ({_percent(n_overlap, n_val_scenes)} of val scenes are also in train)"
    )
    md.append(f"- Scenes unique to val: {len(scenes_unique_to_val)}")
    md.append(f"- Scenes unique to train: {len(scenes_unique_to_train)}")
    md.append("")
    md.append(
        "*Interpretation*: if nearly all val scenes also have frames in train, "
        "the random sample-level split has essentially shuffled within each scene."
    )
    md.append("")

    md.append("## Frame-level leakage\n")
    md.append(
        f"- Val samples in a scene that **also** has train samples: "
        f"**{n_val_in_leaky_scene}** ({_percent(n_val_in_leaky_scene, n_val_total)} of val)"
    )
    md.append(
        f"- Val samples in a scene that is **only** in val: "
        f"{n_val_in_pure_scene} ({_percent(n_val_in_pure_scene, n_val_total)} of val)"
    )
    md.append("")
    md.append(
        "*Interpretation*: each val frame whose scene appears in train has been "
        "trained on nearly-identical neighbouring frames. This inflates accuracy."
    )
    md.append("")

    md.append("## Adjacency leakage (worst case)\n")
    md.append(f"- Unique val frames (scene_idx, frame_in_scene): {n_unique_val_frames}")
    md.append(
        f"- Val frames with the **immediately adjacent frame** (±1) in train: "
        f"**{n_adjacent_in_train}** "
        f"({_percent(n_adjacent_in_train, n_unique_val_frames)} of unique val frames)"
    )
    if n_same_frame_in_train > 0:
        md.append(
            f"- ⚠️ Val frames with the **exact same frame** also in train: "
            f"{n_same_frame_in_train} (should be 0 — duplicates in dataset)"
        )
    md.append("")
    md.append(
        "*Interpretation*: at 2 Hz keyframes, frame f and frame f±1 are 0.5s apart "
        "— visually almost identical. If a substantial fraction of val frames have "
        "an adjacent train frame, the model has effectively seen the val frame."
    )
    md.append("")

    md.append("## Decision (per A2 rule)\n")
    md.append(
        f"- Frame-level leakage: **{frame_leak_pct:.2f}%** "
        f"(threshold: ≥30% triggers scene-split rerun)"
    )
    md.append(
        f"- Adjacency leakage: **{adj_leak_pct:.2f}%** "
        f"(threshold: ≥10% triggers scene-split rerun)"
    )
    md.append("")
    if rerun_recommended:
        md.append(
            "**Recommendation**: ✅ **RUN** the scene-level-split retrain. "
            "Leakage exceeds threshold."
        )
    else:
        md.append(
            "**Recommendation**: ❌ **SKIP** the scene-level-split retrain. "
            "Leakage is below threshold; just report these numbers as the "
            "methodology note."
        )
    md.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[A2] Wrote human-readable markdown: {out_md}")

    # Console summary.
    print("\n[A2] Quick summary:")
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"  Unique scenes — train: {n_train_scenes}, val: {n_val_scenes}")
    print(f"  Scene overlap: {n_overlap} ({_percent(n_overlap, n_val_scenes)} of val)")
    print(f"  Frame-level leakage: {n_val_in_leaky_scene} val samples "
          f"({frame_leak_pct:.2f}%)")
    print(f"  Adjacency leakage: {n_adjacent_in_train} unique val frames "
          f"({adj_leak_pct:.2f}%)")
    if rerun_recommended:
        print("  → DECISION: run the scene-split retrain.")
    else:
        print("  → DECISION: skip scene-split retrain; report leakage numbers as-is.")


if __name__ == "__main__":
    main()

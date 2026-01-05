import os
import json
import glob
import argparse
from statistics import mean

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def summarize_dataset(json_path: str, name: str):
    if not os.path.exists(json_path):
        print(f"[DATA] {name}: NOT FOUND -> {json_path}")
        return None

    size = os.path.getsize(json_path)
    data = load_json(json_path)

    n = len(data)
    in_lens = [len((x.get("input") or "")) for x in data]
    tgt_lens = [len((x.get("target") or "")) for x in data]

    print(f"\n[DATA] {name}")
    print(f"  path        : {json_path}")
    print(f"  file size   : {human_bytes(size)}")
    print(f"  samples     : {n}")
    print(f"  input chars : avg={mean(in_lens):.1f}, max={max(in_lens) if in_lens else 0}")
    print(f"  tgt chars   : avg={mean(tgt_lens):.1f}, max={max(tgt_lens) if tgt_lens else 0}")

    # quick rough estimate: bytes/sample
    print(f"  bytes/sample: ~{(size / max(1,n)):.1f} bytes")

    # check optional keys (useful for your setup)
    keys = set()
    for x in data[:50]:
        keys |= set(x.keys())
    print(f"  sample keys (first 50 union): {sorted(keys)}")
    return {"n": n, "size": size}

def find_latest_trainer_state(stage_dir: str):
    if not os.path.isdir(stage_dir):
        return None

    # Prefer latest checkpoint trainer_state.json
    ckpts = sorted(glob.glob(os.path.join(stage_dir, "checkpoint-*")))
    ckpts = [c for c in ckpts if os.path.isdir(c)]

    # try from latest to earliest
    for c in reversed(ckpts):
        ts = os.path.join(c, "trainer_state.json")
        if os.path.exists(ts):
            return ts

    # fallback: sometimes saved at stage root
    ts_root = os.path.join(stage_dir, "trainer_state.json")
    if os.path.exists(ts_root):
        return ts_root

    return None

def summarize_trainer_state(stage_dir: str, stage_name: str):
    ts = find_latest_trainer_state(stage_dir)
    if ts is None:
        print(f"\n[TRAIN] {stage_name}: trainer_state.json not found in {stage_dir}")
        return

    st = load_json(ts)
    log_history = st.get("log_history", [])

    # find last entries with train metrics
    last_train = None
    last_eval = None
    for item in reversed(log_history):
        if last_train is None and ("train_runtime" in item or "train_steps_per_second" in item):
            last_train = item
        if last_eval is None and ("eval_loss" in item):
            last_eval = item
        if last_train and last_eval:
            break

    print(f"\n[TRAIN] {stage_name}")
    print(f"  trainer_state: {ts}")

    if last_train:
        tr = last_train.get("train_runtime", None)
        sps = last_train.get("train_steps_per_second", None)
        spsamp = last_train.get("train_samples_per_second", None)
        steps = last_train.get("train_steps", None)

        if tr is not None:
            print(f"  train_runtime            : {tr:.2f} sec")
        if sps is not None:
            print(f"  train_steps_per_second   : {sps:.4f}")
        if spsamp is not None:
            print(f"  train_samples_per_second : {spsamp:.4f}")
        if steps is not None:
            print(f"  train_steps              : {steps}")

    if last_eval:
        print(f"  last eval_loss: {last_eval.get('eval_loss')}")

def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g. runs/20260103_222637")
    args = ap.parse_args()

    run_dir = args.run_dir
    data_dir = os.path.join(run_dir, "data")
    s1_dir = os.path.join(run_dir, "stage1")
    s2_dir = os.path.join(run_dir, "stage2")

    print("=" * 90)
    print(f"[RUN] {run_dir}")
    print("=" * 90)

    # sizes
    for dname, dpath in [("data", data_dir), ("stage1", s1_dir), ("stage2", s2_dir)]:
        if os.path.isdir(dpath):
            print(f"[SIZE] {dname:6s}: {human_bytes(dir_size_bytes(dpath))}  ({dpath})")
        else:
            print(f"[SIZE] {dname:6s}: MISSING ({dpath})")

    # dataset stats
    cap_path = os.path.join(data_dir, "vector_captioning_data.json")
    qa_path  = os.path.join(data_dir, "driving_qa_data.json")
    summarize_dataset(cap_path, "Stage1 captioning JSON")
    summarize_dataset(qa_path,  "Stage2 QA JSON")

    # throughput stats (trainer_state)
    summarize_trainer_state(s1_dir, "Stage1")
    summarize_trainer_state(s2_dir, "Stage2")

    print("\nDone.")

if __name__ == "__main__":
    main()

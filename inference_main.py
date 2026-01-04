# inference_main.py
import argparse
import os
from llm_driving.inference import run_inference_two_stage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path like runs/20260103_214214")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")

    stage2_dir = os.path.join(run_dir, "stage2")
    qa_path = os.path.join(run_dir, "data", "driving_qa_data.json")
    out_path = os.path.join(stage2_dir, "inference_outputs.json")

    run_inference_two_stage(
        run_dir_stage2=stage2_dir,
        qa_path=qa_path,
        out_path=out_path,
        limit=args.limit,
    )

if __name__ == "__main__":
    main()

# inference_main.py
import argparse
import os
from llm_driving.inference import run_inference_two_stage
from llm_driving.logger import setup_logger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path like runs/20260103_214214")
    ap.add_argument("--qa_path", default=None, help="Optional explicit QA json path")
    args = ap.parse_args()

    run_dir = args.run_dir.rstrip("/")

    # Setup logger with single consolidated log file in run_dir
    log_file = os.path.join(run_dir, "inference.log")
    setup_logger(name="llm_driving", log_file=log_file)

    qa_path = args.qa_path
    if qa_path is None:
        qa_path = os.path.join(run_dir, "data", "driving_qa_data.json")

    run_inference_two_stage(
        run_dir_stage2=os.path.join(run_dir, "stage2"),
        qa_path=qa_path,
        out_path=os.path.join(run_dir, "stage2", "inference_outputs.json"),
        limit=None,
    )

if __name__ == "__main__":
    main()

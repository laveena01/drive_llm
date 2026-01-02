# inference_main.py

from llm_driving.inference import run_inference_two_stage
from llm_driving.config import STAGE2_OUTPUT_DIR

def main():
    print("=" * 80)
    print("[INFERENCE] Paper-style inference started.")
    print("=" * 80)

    run_inference_two_stage(
        out_path=f"{STAGE2_OUTPUT_DIR}/inference_outputs.json",
        limit=None,   # set e.g. 200 for quick test
    )

    print("\n[INFERENCE] Done.")
    print("=" * 80)

if __name__ == "__main__":
    main()

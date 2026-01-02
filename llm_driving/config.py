# llm_driving/config.py
import os
from datetime import datetime

# -----------------------------
# nuScenes
# -----------------------------
NUSC_ROOT = "/u/student/2024/cs24mtech11010/nuscenes"
NUSC_VERSION = "v1.0-mini"          # later: "v1.0-trainval"

# -----------------------------
# vectors
# -----------------------------
MAX_OBJECTS = 10
VECTOR_DIM = 7

# -----------------------------
# model
# -----------------------------
MODEL_NAME = "google/flan-t5-base"

# Run switches (useful for debugging)
RUN_STAGE1 = True
RUN_STAGE2 = True

SEED = 42

# -----------------------------
# Stage 1: vector -> caption
# -----------------------------
STAGE1_EPOCHS = 10
STAGE1_BATCH_SIZE = 2
STAGE1_LR = 2e-5
STAGE1_WEIGHT_DECAY = 0.0
STAGE1_MAX_INPUT_LEN = 256
STAGE1_MAX_TARGET_LEN = 256

# -----------------------------
# Stage 2: caption+question -> paper-style action output
# -----------------------------
STAGE2_EPOCHS = 8
STAGE2_BATCH_SIZE = 2
STAGE2_LR = 2e-5
STAGE2_WEIGHT_DECAY = 0.0
STAGE2_MAX_INPUT_LEN = 256
STAGE2_MAX_TARGET_LEN = 128

# -----------------------------
# Generation params (eval/pred dumps)
# -----------------------------
GEN_NUM_BEAMS = 4
GEN_EARLY_STOPPING = True
GEN_NO_REPEAT_NGRAM_SIZE = 4
GEN_REPETITION_PENALTY = 1.2
GEN_LENGTH_PENALTY = 1.0
GEN_MAX_NEW_TOKENS_STAGE1 = 160
GEN_MAX_NEW_TOKENS_STAGE2 = 80

# -----------------------------
# Trainer logging / saving / eval
# (you said ignore disk concerns, so defaults are not restrictive)
# -----------------------------
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"         # or "steps"
SAVE_STRATEGY = "epoch"         # or "steps"
SAVE_TOTAL_LIMIT = 3            # keep a few checkpoints

# -----------------------------
# Metrics toggles (we'll implement in training.py)
# -----------------------------
# Text similarity metrics
COMPUTE_ROUGE = True
COMPUTE_BLEU = True

# Structured parsing metrics for paper-format outputs
COMPUTE_PARSE_METRICS = True

# Numeric/control metrics (MAE on pedal %, steering accuracy, etc.)
COMPUTE_CONTROL_METRICS = True

# -----------------------------
# runs
# -----------------------------
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)

DATA_DIR = os.path.join(RUN_DIR, "data")
STAGE1_OUTPUT_DIR = os.path.join(RUN_DIR, "stage1")
STAGE2_OUTPUT_DIR = os.path.join(RUN_DIR, "stage2")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)
os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)

CAPTIONING_DATA_PATH = os.path.join(DATA_DIR, "vector_captioning_data.json")
QA_DATA_PATH = os.path.join(DATA_DIR, "driving_qa_data.json")

DEBUG = False

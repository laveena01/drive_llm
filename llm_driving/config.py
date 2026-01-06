# llm_driving/config.py
import os
from datetime import datetime

# -----------------------------
# Distributed (torchrun) helpers
# -----------------------------
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_RANK0 = (RANK == 0)

# -----------------------------
# nuScenes
# -----------------------------
NUSC_ROOT = os.environ.get("NUSC_ROOT", "/u/student/2024/cs24mtech11010/data/nuscenes")
NUSC_VERSION = os.environ.get("NUSC_VERSION", "v1.0-trainval")  # "v1.0-mini" or "v1.0-trainval"

# -----------------------------
# vectors
# -----------------------------
MAX_OBJECTS = 10
VECTOR_DIM = 7

# -----------------------------
# model
# -----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-base")

# Run switches (useful for debugging)
RUN_STAGE1 = True
RUN_STAGE2 = True

SEED = 42

# -----------------------------
# Stage 1: vector -> caption (match your training.py defaults)
# -----------------------------
STAGE1_EPOCHS = 10
STAGE1_PER_DEVICE_BATCH = 1
STAGE1_GRAD_ACCUM = 4
STAGE1_LR = 5e-4
STAGE1_WEIGHT_DECAY = 0.0
STAGE1_MAX_INPUT_LEN = 192
STAGE1_MAX_TARGET_LEN = 192

# -----------------------------
# Stage 2: caption+question -> paper-style action output
# -----------------------------
STAGE2_EPOCHS = 5
STAGE2_PER_DEVICE_BATCH = 1
STAGE2_GRAD_ACCUM = 4
STAGE2_LR = 5e-4
STAGE2_WEIGHT_DECAY = 0.0
STAGE2_MAX_INPUT_LEN = 192
STAGE2_MAX_TARGET_LEN = 192

# -----------------------------
# Generation params (eval/pred dumps)
# -----------------------------
GEN_NUM_BEAMS = 4
GEN_EARLY_STOPPING = True
GEN_NO_REPEAT_NGRAM_SIZE = 4
GEN_REPETITION_PENALTY = 1.2
GEN_LENGTH_PENALTY = 1.0
GEN_MAX_NEW_TOKENS_STAGE1 = 160
GEN_MAX_NEW_TOKENS_STAGE2 = 90

# -----------------------------
# Trainer logging / saving / eval
# -----------------------------
LOGGING_STEPS = 50
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2

# -----------------------------
# Metrics toggles
# -----------------------------
COMPUTE_ROUGE = True
COMPUTE_BLEU = True
COMPUTE_PARSE_METRICS = True
COMPUTE_CONTROL_METRICS = True

# -----------------------------
# runs
# -----------------------------
# Keep trainval runs separate by default (override via env RUNS_DIR if you want)
DEFAULT_RUNS_DIR = "runs_trainval" if NUSC_VERSION != "v1.0-mini" else "runs"
RUNS_DIR = os.environ.get("RUNS_DIR", DEFAULT_RUNS_DIR)

# If you want to resume/force a run id, you can export RUN_ID=<...>
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)

DATA_DIR = os.path.join(RUN_DIR, "data")
STAGE1_OUTPUT_DIR = os.path.join(RUN_DIR, "stage1")
STAGE2_OUTPUT_DIR = os.path.join(RUN_DIR, "stage2")

CAPTIONING_DATA_PATH = os.path.join(DATA_DIR, "vector_captioning_data.json")
QA_DATA_PATH = os.path.join(DATA_DIR, "driving_qa_data.json")

# IMPORTANT: only rank0 should create directories (DDP safety)
if IS_RANK0:
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STAGE1_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STAGE2_OUTPUT_DIR, exist_ok=True)

DEBUG = False

# llm_driving/config.py
import os
from datetime import datetime

# os.sep = '\\'  # Force forward slashes

# -----------------------------
# nuScenes
# -----------------------------
# NUSC_ROOT = "/u/student/2021/cs21resch15003/nuscenes"
# NUSC_VERSION = "v1.0-mini"          # later: "v1.0-trainval"

NUSC_ROOT = "/u/student/2021/cs21resch15003/data/nuscenes"
NUSC_VERSION = "v1.0-trainval"

# -----------------------------
# vectors
# -----------------------------
MAX_OBJECTS = 10
VECTOR_DIM = 8

# -----------------------------
# Risk Calculation (NEW)
# -----------------------------
DEFAULT_EGO_SPEED = 10.0  # m/s
RISK_FRONT_CONE_DEG = 45.0   # consider objects within +/- this angle in front
RISK_LATERAL_BAND_M = 3.0    # consider objects with |rel_y| <= this as "in-path"
RISK_REQUIRE_IN_FRONT = True # require rel_x > 0 for TTC/collision
USE_ADVANCED_RISK = True  # Use multi-dimensional risk instead of simple distance-based policy
RISK_WEIGHTS = {
    'collision': 0.40,
    'pedestrian': 0.30,
    'ttc': 0.20,
    'regulatory': 0.10,
}

# -----------------------------
# model
# -----------------------------
MODEL_NAME = "google/flan-t5-base"
T5_D_MODEL = 768                     # flan-t5-base hidden size

# Run switches
RUN_STAGE1 = True
RUN_STAGE2 = True
SEED = 42

# -----------------------------
# Vector-prefix (PART-2)
# -----------------------------
USE_VECTOR_PREFIX = True          # <-- key switch
PREFIX_LEN = 64                   # number of prefix tokens injected to encoder (paper: 64)
VEC_ENCODER_HIDDEN = 256
VEC_ENCODER_LAYERS = 2
VEC_ENCODER_HEADS = 4
VEC_ENCODER_DROPOUT = 0.1
NUM_OBJECT_TYPES = 4              # car=0, pedestrian=1, traffic_light=2, object=3
TYPE_EMBED_DIM = 16               # learned embedding dim for type_id
TOKENS_PER_OBJECT = 6             # prefix tokens per object (10 * 6 = 60, + 4 global = 64)

VECTOR_ENCODER_CONFIG = dict(
    max_objects=MAX_OBJECTS,        # 10
    vector_dim=VECTOR_DIM,         # 8
    hidden_dim=VEC_ENCODER_HIDDEN, # 256
    prefix_len=PREFIX_LEN,         # 64
    t5_d_model=T5_D_MODEL,        # 768
    n_layers=VEC_ENCODER_LAYERS,   # 2
    n_heads=VEC_ENCODER_HEADS,     # 4
    dropout=VEC_ENCODER_DROPOUT,   # 0.1
    num_types=NUM_OBJECT_TYPES,    # 4
    type_embed_dim=TYPE_EMBED_DIM, # 16
    tokens_per_object=TOKENS_PER_OBJECT,  # 6
)

# Whether to freeze base FLAN-T5 weights (False = full fine-tuning, like the paper)
FREEZE_BASE_MODEL = False

# -----------------------------
# LoRA (optional, PART-2)
# -----------------------------
USE_LORA = False
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# For T5/FLAN-T5 these names are commonly present in attention projections.
# If peft complains, we’ll adjust to your exact module names later.
LORA_TARGET_MODULES = ["q", "v"]

# -----------------------------
# Stage 1: vector -> caption
# -----------------------------
STAGE1_EPOCHS = 10
STAGE1_BATCH_SIZE = 4
STAGE1_LR = 2e-5
STAGE1_WEIGHT_DECAY = 0.0
STAGE1_MAX_INPUT_LEN = 128
STAGE1_MAX_TARGET_LEN = 256

# Stage1 text prompt — minimal, forces decoder to rely on prefix embeddings
STAGE1_TEXT_PROMPT = "Describe:"

# -----------------------------
# Stage 2: caption+question -> action
# -----------------------------
STAGE2_EPOCHS = 8
STAGE2_BATCH_SIZE = 4
STAGE2_LR = 2e-5
STAGE2_WEIGHT_DECAY = 0.0
STAGE2_MAX_INPUT_LEN = 192
STAGE2_MAX_TARGET_LEN = 128

STAGE2_QUESTION = "How should the car drive in this situation and why?"

# -----------------------------
# Generation params
# -----------------------------
GEN_NUM_BEAMS = 4
GEN_EARLY_STOPPING = True
GEN_NO_REPEAT_NGRAM_SIZE = 8
GEN_REPETITION_PENALTY = 1.5
GEN_LENGTH_PENALTY = 1.0
GEN_MAX_NEW_TOKENS_STAGE1 = 160
GEN_MAX_NEW_TOKENS_STAGE2 = 80

# -----------------------------
# Logging/saving
# -----------------------------
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 3
LOG_LEVEL = "INFO"          # "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FILE_NAME = "train.log" # stored under RUN_DIR
DISABLE_TQDM = False     

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
RUNS_DIR = "runs"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)

DATA_DIR = os.path.join(RUN_DIR, "data")
STAGE1_OUTPUT_DIR = os.path.join(RUN_DIR, "stage1")
STAGE2_OUTPUT_DIR = os.path.join(RUN_DIR, "stage2")

CAPTIONING_DATA_PATH = os.path.join(DATA_DIR, "vector_captioning_data.json")
QA_DATA_PATH = os.path.join(DATA_DIR, "driving_qa_data.json")

DEBUG = False

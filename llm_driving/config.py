# llm_driving/config.py
import os
from datetime import datetime

# os.sep = '\\'  # Force forward slashes

# -----------------------------
# nuScenes
# -----------------------------
NUSC_ROOT = "/u/student/2024/cs24mtech14014/nuscenes"
# NUSC_VERSION = "v1.0-mini"          # later: "v1.0-trainval"

NUSC_ROOT = "/u/student/2024/cs24mtech14014/data/nuscenes"
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

# -----------------------------
# Temporal context (Step 2 / Part B)
# -----------------------------
# When True, the encoder consumes a sliding window of K=TEMPORAL_WINDOW
# keyframes [t-K+1 .. t] (past frames + current). The shared
# VectorPrefixEncoder is applied per frame; a small TemporalTransformer
# aggregates across frames per prefix slot before the result is injected
# into T5. Output shape into T5 is unchanged: (B, PREFIX_LEN, T5_D_MODEL).
#
# When False, the data path produces single-frame samples (legacy behaviour
# bit-for-bit) — useful as the K=1 ablation row for the thesis table.
USE_TEMPORAL = False
TEMPORAL_WINDOW = 4
TEMPORAL_TRANSFORMER_LAYERS = 2
TEMPORAL_TRANSFORMER_HEADS = 4
TEMPORAL_TRANSFORMER_DROPOUT = 0.1

# Step 4: when True, the K-frame window aligns object identity across frames
# using nuScenes instance_token (so slot j refers to the same physical
# object across all K frames). When False, falls back to Step 2's
# sort-by-distance per-frame slotting (the buggy behaviour that produced
# null results — kept for ablation parity).
USE_TRACKED_TEMPORAL = False

# Step 4: when True, the dataset builder adds an extra "action_future"
# question per frame whose target is computed from the H-frame look-ahead
# (`compute_future_aware_action` in risk_calculator.py). This gives Stage 2
# a label channel that requires anticipation. The new question is tagged
# `question_type="action_future"` so existing eval metrics (which filter
# question_type=="action") remain comparable to Step 1 baseline.
USE_FUTURE_AWARE_SUPERVISION = True
FUTURE_AWARE_HORIZON = 4
# Per-sample loss weight applied to action_future questions during Stage 2
# training. > 1.0 boosts the gradient signal from this question type so the
# model actually uses temporal cues rather than ignoring them as a minority
# class (only 1 of 6 action questions). 3.0 brings its share to ~27% of
# action-question gradient.
ACTION_FUTURE_LOSS_WEIGHT = 3.0

# Step 4: when True, lanGen produces motion-aware captions that include
# compact temporal descriptors (e.g. "Obj1 closing 3.5m/s decel 4.9m/s")
# for the top-3 closest objects. Without this, Stage 1 captioning targets
# don't reward encoding temporal info into text, so the temporal signal in
# Stage 1's encoder gets discarded at the caption boundary.
USE_TEMPORAL_CAPTIONS = False
TEMPORAL_CAPTION_TOP_N = 3

# Ablation flag: when False, the `### RISK\n<risk_text>` block is stripped
# from every Stage 2 prompt (training, eval, inference). The risk_text
# field is still computed and stored on samples (for analysis), but the
# model never sees it as input. Tests the question "does the model need
# risk handed to it explicitly, or can the encoder learn it from vectors?".
# Default True = current behaviour (risk in prompt).
USE_RISK_IN_PROMPT = True

# Step 5-lite: oversample "hard brake-future" cases (LOW/MINIMAL current
# risk + brake_required_future) in Stage 2 training to break the
# LOW.future_brake_recall = 0% floor. Only `action_future` rows are
# duplicated (target=BRAKE); `action` rows on the same frames are NOT
# duplicated (target=CONTINUE would reinforce the wrong direction).
# Cross-question parameter sharing in Stage 2's FLAN-T5 means lifting
# the action_future channel generalizes back to the action question
# on the same scene — which is what the LOW.future_brake_recall metric
# rewards (it's computed on action questions, not action_future).
OVERSAMPLE_HARD_BRAKE_LOW = True
HARD_BRAKE_LOW_OVERSAMPLE_FACTOR = 5

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
# Step 4 / Part C: bumped from 256 -> 320. The single-frame dense caption
# (10 objects, ~20 tokens each via `describe_object`) already takes ~200
# tokens; adding 3 temporal lines for Part C pushed worst-case to 257.
# 320 leaves a comfortable ~60-token safety margin. Memory cost on
# FLAN-T5-base at batch 4 is ~+1 GB — negligible on A100 40 GB.
STAGE1_MAX_TARGET_LEN = 320

# Stage1 text prompt — minimal, forces decoder to rely on prefix embeddings
STAGE1_TEXT_PROMPT = "Describe:"

# -----------------------------
# Stage 2: caption+question -> action
# -----------------------------
STAGE2_EPOCHS = 8
STAGE2_BATCH_SIZE = 4
# Step 4: bumped from 192 → 384 to accommodate Part C's temporal captions.
# Stage 2 input layout: OBSERVATION + caption + RISK + risk_text + QUESTION
# + qa_question + OUTPUT FORMAT + paper_format. Headers + risk + question +
# format ≈ 140 tokens. With temporal-aware captions reaching ~200 tokens
# in dense scenes, the previous 192 limit truncated the OUTPUT FORMAT block
# and broke 5-line parsing. 384 gives a comfortable safety margin.
STAGE2_LR = 2e-5
STAGE2_WEIGHT_DECAY = 0.0
STAGE2_MAX_INPUT_LEN = 384
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

# RUN_ID resolution order (so `python build_data.py` followed by
# `accelerate launch main.py` shares runs/<RUN_ID>/data/ without the user
# having to manage an env var):
#
#   1. RUN_ID env var if set — explicit override always wins.
#   2. runs/.latest_build_run_id pointer file — written by build_data.py
#      after a successful build. Lets subsequent training runs auto-pick
#      up the most recent dataset build with no extra typing.
#   3. Fresh timestamp — used when neither of the above is available
#      (e.g. first invocation of build_data.py, or a totally fresh repo).
def _resolve_run_id() -> str:
    env = os.environ.get("RUN_ID")
    if env:
        return env
    pointer_path = os.path.join("runs", ".latest_build_run_id")
    try:
        if os.path.isfile(pointer_path):
            with open(pointer_path, "r") as f:
                pinned = f.read().strip()
            if pinned:
                return pinned
    except Exception:
        # Defensive: a corrupt pointer file should never block the run.
        pass
    return datetime.now().strftime("%Y%m%d_%H%M%S")


RUN_ID = _resolve_run_id()
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)

DATA_DIR = os.path.join(RUN_DIR, "data")
STAGE1_OUTPUT_DIR = os.path.join(RUN_DIR, "stage1")
STAGE2_OUTPUT_DIR = os.path.join(RUN_DIR, "stage2")

CAPTIONING_DATA_PATH = os.path.join(DATA_DIR, "vector_captioning_data.json")
QA_DATA_PATH = os.path.join(DATA_DIR, "driving_qa_data.json")

DEBUG = False

# Risk Implementation and Integration Summary

## Overview
This document explains where risk is implemented in the codebase and how it integrates into the main driver code.

## Risk Implementation Files

### 1. **Core Risk Calculation** (`llm_driving/nuscenes_risk_integration.py`)
**Location**: `c:\Users\niran\Desktop\sandbox\projects\drive_llm\llm_driving\nuscenes_risk_integration.py`

This is the **foundational risk module** with 676 lines of code that implements:

#### Key Components:
- **`RiskComponents` dataclass** (lines 24-35): Multi-dimensional risk breakdown
  - `collision_risk`: Distance and TTC-based collision risk
  - `pedestrian_risk`: Special weighting for vulnerable road users
  - `ttc_risk`: Time-to-collision based risk
  - `regulatory_risk`: Traffic light compliance
  - `total_risk`: Weighted combination
  - `risk_level`: Classification (CRITICAL/HIGH/MODERATE/LOW/MINIMAL)

- **`SimpleRiskCalculator` class** (lines 37-202): Main risk calculation engine
  - **Type weights** (lines 44-52): Different object types get different risk multipliers
    - Pedestrians: 2.5x
    - Bicycles: 2.0x
    - Motorcycles: 1.8x
    - Trucks/Buses: 1.3x
    - Cars: 1.0x
  
  - **`calculate_ttc()`** (lines 68-81): Time-to-collision calculation
    - Returns `float('inf')` if not approaching
    - Otherwise: `distance / closing_speed`
  
  - **`calculate_risk()`** (lines 83-202): Comprehensive risk assessment
    - Calculates TTC-based risk (lines 116-128)
    - Distance-based collision risk (lines 130-140)
    - Type-weighted collision risk (lines 143-145)
    - Pedestrian-specific risk (lines 147-156)
    - Regulatory risk from traffic lights (lines 158-163)
    - **Total risk formula** (lines 166-171):
      ```
      total_risk = collision_risk * 0.40 +
                   pedestrian_risk * 0.30 +
                   ttc_risk * 0.20 +
                   regulatory_risk * 0.10
      ```

#### Additional Features:
- **`extract_scene_with_advanced_risk()`** (lines 209-320): NuScenes integration
- **Prompt generation functions** (lines 327-459): LLM prompt creation with risk info
- **Training data generation** (lines 466-548): Risk-aware training samples

---

### 2. **Risk Calculator Bridge** (`llm_driving/risk_calculator.py`)
**Location**: `c:\Users\niran\Desktop\sandbox\projects\drive_llm\llm_driving\risk_calculator.py`

This is the **integration layer** (241 lines) that bridges the risk module with the LLM pipeline:

#### Key Components:
- **`FrameRiskData` dataclass** (lines 40-51): Aggregated frame-level risk
  - `risk_level`: Overall scene risk
  - `max_collision_risk`: Highest collision risk among all objects
  - `max_pedestrian_risk`: Highest pedestrian risk
  - `min_ttc`: Minimum time-to-collision
  - `avg_total_risk`: Average risk across objects
  - `num_risk_objects`: Count of risky objects
  - `object_risks`: Per-object risk details

- **`calculate_risk_from_vectors()`** (lines 54-162): **Main integration function**
  - **Input**: Vector array `[rel_x, rel_y, dist, rel_speed, heading, size, type_id]`
  - **Process**:
    1. Converts vectors to risk calculator format (lines 98-111)
    2. Calls `SimpleRiskCalculator.calculate_risk()` for each object (lines 114-120)
    3. Aggregates statistics (lines 123-137)
    4. Determines scene-level risk (lines 142-152)
  - **Output**: `FrameRiskData` with comprehensive risk assessment

- **`policy_from_risk()`** (lines 169-213): **Risk-based driving policy**
  - Replaces simple distance-based policy
  - Returns: `(accel, brake, steer, reason, policy_label)`
  - **Decision logic**:
    - **CRITICAL**: TTC < 2s or collision_risk ≥ 0.8 → Emergency brake (80-90%)
    - **HIGH**: TTC < 3s or collision_risk ≥ 0.6 → Hard brake (50-60%)
    - **MODERATE**: collision_risk ≥ 0.4 → Cautious (20-30% brake)
    - **LOW**: collision_risk ≥ 0.2 → Slight brake (5%)
    - **MINIMAL**: Safe to continue (20% accel, 0% brake)

- **`get_risk_summary_text()`** (lines 216-240): Human-readable risk description

---

## Integration Points

### ❌ **NOT Currently Integrated** (Implementation exists but not used)

The risk calculation modules exist but are **NOT actively used** in the main pipeline. Here's the evidence:

#### 1. **Dataset Builder** (`llm_driving/datasets_builder.py`)
**Current Implementation** (line 54-102):
```python
def _policy_from_min_dist(num_objects: int, min_dist: float, frame_idx: int = -1):
    """Simple distance-based policy - NO RISK CALCULATION"""
    if min_dist < 6.0:
        risk_level = "CRITICAL"
        accel, brake = 0, 70
    elif min_dist < 10.0:
        risk_level = "HIGH"
        accel, brake = 0, 40
    # ... etc
```

**What it should use**:
```python
from .risk_calculator import calculate_risk_from_vectors, policy_from_risk

# In _make_samples_from_frames():
risk_data = calculate_risk_from_vectors(
    vectors=frame["vectors"],
    num_objects=num_objects,
    ego_speed=10.0
)
accel, brake, steer, reason, policy_label = policy_from_risk(risk_data)
```

#### 2. **Language Generation** (`llm_driving/langen.py`)
**Current Implementation** (lines 71-143):
- Has placeholder for risk data (line 84): `risk_data = frame.get("risk_data", None)`
- Can append risk info to captions (lines 127-142)
- **BUT**: No frame currently contains `risk_data` field

**What it should receive**:
```python
frame = {
    "vectors": vectors,
    "num_objects": num_objects,
    "risk_data": {  # ← This is never populated!
        "risk_level": "HIGH",
        "min_ttc": 3.2,
        "max_collision_risk": 0.65,
        # ...
    }
}
```

#### 3. **Main Pipeline** (`main.py`)
- **No imports** of risk modules
- **No risk calculation** in the pipeline
- Uses simple distance-based policy throughout

---

## How to Integrate Risk (Step-by-Step)

### Step 1: Modify `nuscenes_data.py`
Add risk calculation when extracting frames:

```python
from .risk_calculator import calculate_risk_from_vectors

def get_scene_frames_vectors(nusc, scene_idx, max_frames=None):
    # ... existing code ...
    
    for sample in samples:
        # ... extract vectors ...
        
        # ADD RISK CALCULATION
        risk_data = calculate_risk_from_vectors(
            vectors=vectors,
            num_objects=num_objects,
            ego_speed=10.0  # or extract from nuScenes
        )
        
        frame = {
            "vectors": vectors,
            "num_objects": num_objects,
            "risk_data": risk_data.to_dict(),  # ← ADD THIS
        }
        frames.append(frame)
```

### Step 2: Modify `datasets_builder.py`
Replace `_policy_from_min_dist()` with risk-based policy:

```python
from .risk_calculator import calculate_risk_from_vectors, policy_from_risk

def _make_samples_from_frames(frames, captioning_samples, qa_samples, scene_idx=0):
    for idx, frame in enumerate(frames):
        # ... existing code ...
        
        # REPLACE THIS:
        # accel, brake, steer, reason, policy_label = _policy_from_min_dist(use_n, min_dist)
        
        # WITH THIS:
        risk_data = calculate_risk_from_vectors(
            vectors=frame["vectors"],
            num_objects=num_objects,
            ego_speed=10.0
        )
        accel, brake, steer, reason, policy_label = policy_from_risk(risk_data)
        
        # Update frame with risk data for lanGen
        frame["risk_data"] = risk_data.to_dict()
        caption = lanGen(frame)  # Now includes risk info
```

### Step 3: Update Logging in `training.py`
The training file already has risk logging functions:
- `_map_brake_to_risk()` (line 232)
- `_print_eval_risk_summary()` (line 245)

These will automatically work once risk data flows through the pipeline.

---

## Summary

### ✅ **What Exists**
1. **Complete risk calculation system** in `nuscenes_risk_integration.py`
2. **Integration bridge** in `risk_calculator.py`
3. **Risk-aware policy function** `policy_from_risk()`
4. **Risk-aware caption generation** in `langen.py` (ready but unused)
5. **Risk logging utilities** in `training.py`

### ❌ **What's Missing**
1. **No risk calculation in data extraction** (`nuscenes_data.py`)
2. **Still using simple distance-based policy** (`datasets_builder.py` line 136)
3. **Risk data never populated in frames** (no `risk_data` field)
4. **Main pipeline doesn't import or use risk modules**

### 🔧 **To Activate Risk System**
You need to modify **3 files**:
1. `nuscenes_data.py` - Add risk calculation to frame extraction
2. `datasets_builder.py` - Replace `_policy_from_min_dist()` with `policy_from_risk()`
3. Optionally update `config.py` to add risk-related configuration parameters

The risk implementation is **complete and ready to use**, but it's **not integrated into the main execution flow** yet.

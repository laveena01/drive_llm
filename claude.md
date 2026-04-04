**LLM Driving Project - Technical Handover Report**

*Scope: current uploaded codebase review, comparison with earlier underscore versions, and implementation handover for a new contributor.*

> Current codebase is NOT the same as the earlier version. It has shifted from a simple min-distance action pipeline to a risk-aware multi-question pipeline. The uploaded vector encoder is not integrated into dataset building, training, or inference. Stage-1 remains vector-string -> caption. Stage-2 now consumes caption + risk summary and predicts both action answers and risk answers.

# 1. What this project currently does

- Builds object-level vectors from nuScenes frames, capped to MAX_OBJECTS=10 and VECTOR_DIM=8.
- Generates a structured natural-language scene caption from vectors using lanGen.
- Computes frame-level risk using a dedicated risk module based on TTC, distance, front weighting, lateral conflict, pedestrian weighting, and uncertainty proxy.
- Builds a Stage-2 QA dataset with two task families: action questions and risk questions.
- Trains Stage-1 for vector -> caption and Stage-2 for caption + risk + question -> answer.

# 2. Confirmed change from the earlier version

| **Area** | **Earlier underscore version** | **Current uploaded version** |
| --- | --- | --- |
| **Stage-2 supervision** | Single action question per frame using min_dist policy. | Five action questions + three risk questions per frame using risk-based policy. |
| **Risk usage** | Not used in primary dataset path. | Integrated into Stage-2 prompt and target generation. |
| **Caption content** | Observation caption only. | Caption still excludes risk; risk is injected separately in Stage-2. |
| **Stage-2 model init** | Fresh model from MODEL_NAME base. | Deep copy of trained Stage-1 model. |
| **Evaluation** | Action-only metrics. | Action metrics + risk-level accuracy + safety-oriented metrics. |
| **Inference script** | Simple Stage-2 evaluation with overlap metrics. | Simpler loadable-checkpoint inference with format repair, but action-only. |
| **Vector encoder** | Config present but not active there too. | Still not integrated; file remains standalone/incomplete. |

# 3. End-to-end data flow

- nuScenes frame -> object annotations -> ego-frame object vectors.
- Vector format per object: \[rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id\].
- Stage-1 sample: input is vector string; target is lanGen caption.
- Risk module runs on the same vectors and produces frame risk summary plus per-object risk metadata.
- Stage-2 action sample: observation caption + risk summary + action question + fixed 5-line output format.
- Stage-2 risk sample: observation caption + risk summary + risk question + 1-2 line risk format.
- Training split is generated from JSON datasets using HuggingFace Dataset train_test_split(seed=42).

# 4. Module-by-module guide

## config.py

- Defines dataset paths, run directories, risk hyperparameters, model name, generation settings, and also vector-prefix/LoRA switches.
- Important note: vector-prefix and LoRA switches are declared here but are not consumed by the uploaded training path.

## nuscenes_data.py

- Initializes nuScenes and converts each annotated object into an 8D ego-frame vector.
- Vectors are sorted by distance and padded/truncated to MAX_OBJECTS.

## langen.py

- Converts vectors into a stable caption. Uses object type, coarse size bin, speed magnitude from rel_vx/rel_vy, and left/right/ahead direction.
- Adds fixed ego speed and route lines. If risk_data is present it can append risk lines, but dataset_builder explicitly strips/avoids risk in Stage-1 captions.

## risk_calculator.py

- Core active risk module. Public API: calculate_risk_from_vectors, get_risk_summary_text, policy_from_risk.
- Scene risk is aggregated from per-object risk with soft front gating, TTC logic, distance baseline, lateral conflict, uncertainty proxy, and log-sum-exp aggregation.

## datasets_builder.py

- Current dataset builder. Creates Stage-1 captioning data and Stage-2 QA data.
- For each frame it computes one caption sample and eight QA samples: 5 action + 3 risk.

## training.py

- Current training path. Trains Stage-1 first, then initializes Stage-2 from Stage-1 weights using copy.deepcopy(model_stage1).
- Stage-2 evaluation runs two modes: oracle_caption and stage1_caption.

## inference.py

- Loads a Stage-2 model directory or latest checkpoint and runs evaluation-style inference on a QA JSON file.
- Current script enforces/fixes 5-line action format and reports action_accuracy plus parse_ok_rate.

## vector_encoder.py

- Standalone encoder definition for vector-prefix idea.
- Not wired into training/inference. The class is incomplete for production use because forward() references members that are never defined in the uploaded file.

## logging_utils.py

- Redirects stdout/stderr into a named logger and writes a run log under the run directory.
- Useful for reproducible cluster runs.

# 5. Current active learning tasks

- Stage-1 task: vector string -> structured scene caption.
- Stage-2 action task: predict fixed 5-line control output.
- Stage-2 risk task: predict compact risk level + brief reason.
- Stage-2 action labels come from policy_from_risk(risk_data), not from min_dist thresholds.
- Stage-2 risk labels come from \_risk_target_from_risk_data(risk_data).

# 6. Output formats

## Action output

- Exactly 5 lines: header, accelerator pedal, brake pedal, steering, reason.
- Action parsing and post-processing are strict because metrics depend on extracting accelerator/brake/steering fields.

## Risk output

- 1-2 short lines only.
- Template: Risk level: \<...\>. Reason: \<...\>.

# 7. What is the real status of the vector encoder

- The project configuration still declares USE_VECTOR_PREFIX=True, PREFIX_LEN=16, FREEZE_BASE_MODEL=True, USE_LORA=True.
- No uploaded training, dataset, or inference file imports vector_encoder.py.
- No uploaded code constructs VectorPrefixEncoder or injects prefix embeddings into T5.
- Therefore the current runnable pipeline is not using vector-prefix conditioning.
- The uploaded vector_encoder.py is not complete enough to run as-is; it defines forward() but not the module construction needed by that forward path.

# 8. Input processing difference vs the base paper

Base paper pipeline: numeric object/state vectors are not flattened into prompt text. They are passed through a learned multimodal stack and fused into the LLM as embeddings.

Base paper vector inputs are structured by semantic groups such as cars, pedestrians, ego state, and route, with optional expert attention/action labels during data generation. A structured language generator (lanGen) is used to produce pseudo captions for supervision and QA labeling.

Base paper training objective: first align vector modality to LLM representation space through vector-caption pretraining, then finetune the multimodal model for Driving QA. The learned representation path is part of the model, not a preprocessing shortcut.

Current project pipeline: vectors are serialized directly into a plain text string via vector_to_string(...), then fed to a text-only seq2seq model for Stage-1 captioning. Stage-2 consumes generated caption text plus explicit risk text plus question text.

Practical distinction: the base paper lets the model learn how to encode numeric structure before language decoding, whereas the current project exposes numbers only through text tokenization. This is simpler to implement and easier to debug, but it loses explicit learned handling of object grouping, numeric relations, and modality-specific attention before the LLM.

Implication for contributors: improvements in this codebase currently happen by changing vector semantics, caption design, risk summaries, question templates, or supervision logic. In the base paper, a major improvement lever also exists in the learned vector-to-language fusion stage.

# 9. Important mismatches and technical risks

| **Issue** | **Impact** | **Recommendation** |
| --- | --- | --- |
| Config says vector-prefix/LoRA are enabled, but training ignores them. | A new contributor may assume prefix tuning is active when it is not. | Document these flags as inactive or remove them until integration is complete. |
| Stage-2 initialization changed from base model to Stage-1 copy. | This changes experiment meaning and makes results not directly comparable to the earlier paper-faithful setup. | State clearly in experiment logs whether Stage-2 starts from base or Stage-1. |
| Current inference.py is action-only while training.py supports both action and risk questions. | Offline evaluation may under-report current task scope. | Extend inference to branch on question_type like training evaluation does. |
| vector_encoder.py is incomplete. | Future integration attempts will fail or require reconstruction. | Implement \_\_init\_\_, object projection, transformer encoder, pooling, and T5 prefix injection end-to-end before enabling config flags. |

# 10. Minimal execution order for a new contributor

- Set NUSC_ROOT and NUSC_VERSION in config.py.
- Run dataset generation through build_datasets_full_mini() to produce captioning and QA JSON files.
- Train Stage-1 with train_stage1(captioning_path).
- Train Stage-2 with train_stage2(model_stage1, tokenizer, qa_path).
- Use eval_metrics.json plus validation prediction JSON files under stage1/ and stage2/ for inspection.
- Use inference.py only for action-format checking unless it is expanded for risk questions.

# 11. Files that matter most for modification

- Change vector semantics in nuscenes_data.py and keep langen.py + risk_calculator.py synchronized.
- Change task design in datasets_builder.py.
- Change risk heuristics and action supervision in risk_calculator.py.
- Change training/evaluation protocol in training.py.
- Change deployment-style evaluation in inference.py.

# 12. Bottom line

- Current codebase is a risk-aware two-stage text pipeline, not a vector-prefix pipeline.
- Risk is active in dataset creation, supervision, prompts, and evaluation.
- Vector encoder exists only as an unused experimental stub in the uploaded files.
- Anyone continuing this project should treat risk_calculator.py + datasets_builder.py + training.py as the true active path.
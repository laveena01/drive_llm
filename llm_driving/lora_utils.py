# llm_driving/lora_utils.py
"""
LoRA integration for VectorPrefixT5.

Uses the `peft` library to apply LoRA adapters to T5's q/v projections.
Only vector encoder + LoRA adapter parameters are trainable;
T5 base weights are frozen.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, TaskType

from llm_driving.vector_prefix_t5 import VectorPrefixT5
from llm_driving.vector_encoder import VectorEncoderConfig

logger = logging.getLogger("llm_driving")


def apply_lora(
    model: VectorPrefixT5,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> VectorPrefixT5:
    """
    Apply LoRA adapters to the T5 model inside VectorPrefixT5.
    
    1. Freeze all T5 base parameters
    2. Apply LoRA to q/v projections via peft
    3. Vector encoder params remain trainable (they're separate nn.Module)
    
    Args:
        model: VectorPrefixT5 instance
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        target_modules: T5 module names to apply LoRA to (default: ["q", "v"])
        
    Returns:
        Same VectorPrefixT5 instance with LoRA applied to model.t5
    """
    if target_modules is None:
        target_modules = ["q", "v"]

    # Step 1: Freeze T5 base
    model.freeze_t5_base()

    # Step 2: Apply LoRA to T5
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model.t5 = get_peft_model(model.t5, lora_config)

    # Step 3: Ensure vector encoder is still trainable
    for param in model.vector_encoder.parameters():
        param.requires_grad = True

    # Log parameter counts
    model.print_trainable_parameters()

    return model


def save_checkpoint(
    model: VectorPrefixT5,
    output_dir: str,
    epoch: Optional[int] = None,
):
    """
    Save vector encoder weights + LoRA adapter weights.
    
    Saves:
        - vector_encoder.pt — VectorPrefixEncoder state_dict
        - lora_adapter/ — peft LoRA adapter weights
        - encoder_config.pt — VectorEncoderConfig for reconstruction
        
    Does NOT save T5 base weights (they're frozen and unchanged).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save vector encoder
    encoder_path = os.path.join(output_dir, "vector_encoder.pt")
    torch.save(model.vector_encoder.state_dict(), encoder_path)
    logger.info(f"[save_checkpoint] Saved vector encoder to {encoder_path}")

    # Save encoder config
    config_path = os.path.join(output_dir, "encoder_config.pt")
    torch.save(vars(model.encoder_config), config_path)
    logger.info(f"[save_checkpoint] Saved encoder config to {config_path}")

    # Save LoRA adapter (peft's save method)
    lora_dir = os.path.join(output_dir, "lora_adapter")
    if hasattr(model.t5, "save_pretrained"):
        model.t5.save_pretrained(lora_dir)
        logger.info(f"[save_checkpoint] Saved LoRA adapter to {lora_dir}")
    else:
        logger.warning("[save_checkpoint] T5 model has no save_pretrained — LoRA not saved")

    # Save epoch info
    if epoch is not None:
        meta_path = os.path.join(output_dir, "training_meta.pt")
        torch.save({"epoch": epoch}, meta_path)

    logger.info(f"[save_checkpoint] Checkpoint saved to {output_dir}")


def load_checkpoint(
    model_name: str,
    checkpoint_dir: str,
    device: str = "cpu",
    apply_lora_config: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
) -> VectorPrefixT5:
    """
    Load a VectorPrefixT5 from a checkpoint directory.
    
    Reconstructs:
        1. VectorEncoderConfig from encoder_config.pt
        2. VectorPrefixT5 with fresh T5 base
        3. LoRA adapter weights from lora_adapter/ (if exists)
        4. Vector encoder weights from vector_encoder.pt
        
    Args:
        model_name: HuggingFace model ID (e.g., "google/flan-t5-base")
        checkpoint_dir: Directory containing saved checkpoint
        device: Target device
        apply_lora_config: Whether to apply LoRA (set False for non-LoRA checkpoints)
        
    Returns:
        Fully loaded VectorPrefixT5 ready for inference or continued training
    """
    from peft import PeftModel

    # Load encoder config
    config_path = os.path.join(checkpoint_dir, "encoder_config.pt")
    config_dict = torch.load(config_path, map_location=device, weights_only=True)
    encoder_config = VectorEncoderConfig(**config_dict)

    # Create model
    model = VectorPrefixT5(model_name, encoder_config)

    # Load LoRA adapter if exists
    lora_dir = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.isdir(lora_dir) and apply_lora_config:
        model.t5 = PeftModel.from_pretrained(model.t5, lora_dir)
        logger.info(f"[load_checkpoint] Loaded LoRA adapter from {lora_dir}")
    elif apply_lora_config:
        # No saved adapter — apply fresh LoRA config
        logger.info("[load_checkpoint] No saved LoRA adapter found, applying fresh LoRA config")
        apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                   target_modules=lora_target_modules)

    # Load vector encoder weights
    encoder_path = os.path.join(checkpoint_dir, "vector_encoder.pt")
    if os.path.isfile(encoder_path):
        state_dict = torch.load(encoder_path, map_location=device, weights_only=True)
        model.vector_encoder.load_state_dict(state_dict)
        logger.info(f"[load_checkpoint] Loaded vector encoder from {encoder_path}")

    model.to(device)
    logger.info(f"[load_checkpoint] Model loaded from {checkpoint_dir}")

    return model

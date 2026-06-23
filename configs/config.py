"""
configs/config.py
-----------------
Centralized configuration for the TinyVLM training pipeline.
Defines VLMConfig (model hyper-parameters) and the global CONFIG dict
(paths, training hyper-parameters, distillation settings, I/O).
"""

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Model hyper-parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class VLMConfig:
    """
    Unified configuration for a VisionLanguageModel instance built on top
    of the nanoVLM framework.

    Vision Encoder: SigLIP  (google/siglip-base-patch16-224)
    Language Model: SmolLM2 (HuggingFaceTB/SmolLM2-135M)
    """

    # ---- Vision Encoder (ViT / SigLIP) ----
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip-base-patch16-224"

    # ---- Language Model (SmolLM2) ----
    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100_000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_eos_token_id: int = 0
    lm_max_length: int = 128
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = "HuggingFaceTB/SmolLM2-135M"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-135M"
    lm_chat_template: str = "default"

    # ---- Modality Projector ----
    mp_pixel_shuffle_factor: int = 2
    mp_image_token_length: int = 64

    # ---- Checkpoint ----
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints/nanoVLM-default"
    vlm_extra_tokens: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pre-built configs
# ---------------------------------------------------------------------------

#: Full-size teacher model (220 M parameters)
TEACHER_CONFIG = VLMConfig(vit_img_size=224, lm_max_length=128)

#: Compressed student model (50 M parameters)
STUDENT_CONFIG = VLMConfig(
    vit_img_size=224,
    lm_max_length=128,
    # Vision: half the blocks
    vit_n_blocks=6,
    # Language: half the blocks, halved hidden dim
    lm_n_blocks=12,
    lm_hidden_dim=288,
    lm_inter_dim=768,
    lm_n_heads=6,
    lm_n_kv_heads=2,
)


# ---------------------------------------------------------------------------
# Global training / I-O configuration
# ---------------------------------------------------------------------------

_OUTPUT_DIR = "outputs"

CONFIG = {
    # ---- Model configs ----
    "teacher_vlm_config": TEACHER_CONFIG,
    "student_vlm_config": STUDENT_CONFIG,
    "caption_context_length": 128,

    # ---- Data paths (update to your local paths) ----
    "local_dataset_web_path": "/path/to/prepared_web_dataset",
    "hf_dataset_synth": "jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
    "test1_dir": "/path/to/Tests/Test1",
    "test2_dir": "/path/to/Tests/Test2",

    # ---- Training hyper-parameters ----
    "batch_size": 8,
    "gradient_accumulation_steps": 4,          # effective batch = 32
    "teacher_epochs": 3,
    "student_epochs": 3,
    "lr_mp": 1e-4,                             # modality projector LR
    "lr_backbones": 1e-5,                      # backbone LR
    "lr_student": 5e-5,                        # student distillation LR
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "use_amp": True,
    "validation_split": 0.05,
    "max_grad_norm": 1.0,

    # ---- Knowledge distillation ----
    "distillation_alpha": 0.5,                 # weight on soft (teacher) loss
    "distillation_temperature": 2.0,           # softening temperature T

    # ---- Quantization-Aware Training (QAT) ----
    "qat_epochs": 1,
    "qat_lr": 1e-5,
    "qat_weight_decay": 0.01,

    # ---- Output paths ----
    "output_dir": _OUTPUT_DIR,
    "teacher_checkpoint": "teacher_final.pt",
    "student_checkpoint": "student_distilled.pt",
    "quantized_student_model": "student_ptq_int8.pt",
    "qat_quantized_model": "student_qat_int8.pt",
    "submission_filename": "submission.csv",
    "results_table_csv": "table_results.csv",
    "hf_cache_dir": os.path.join(_OUTPUT_DIR, "hf_cache"),

    # ---- nanoVLM repo path ----
    "nanovlm_root": "/path/to/nanoVLM",       # set via --nanovlm_root CLI arg
}

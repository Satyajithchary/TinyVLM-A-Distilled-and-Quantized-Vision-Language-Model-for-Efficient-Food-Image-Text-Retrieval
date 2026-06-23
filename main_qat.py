"""
main_qat.py
-----------
TinyVLM — Quantization-Aware Training (QAT) Extension
======================================================
Runs after the standard pipeline (main.py) to fine-tune the distilled
student model with fake-quantize operators inserted, then converts to a
true INT8 model and evaluates PTQ vs QAT side-by-side (Table V in paper).

Prerequisites
-------------
  - A trained, distilled student checkpoint at
    ``outputs/student_distilled.pt`` (produced by main.py).

Usage
-----
    python main_qat.py \\
        --nanovlm_root /path/to/nanoVLM \\
        --web_data_path /path/to/prepared_web_dataset \\
        --student_ckpt outputs/student_distilled.pt \\
        --ptq_ckpt outputs/student_ptq_int8.pt \\
        --output_dir outputs

Paper
-----
"TinyVLM: A Distilled and Quantized Vision-Language Model for Efficient
Food Image-Text Retrieval in TinyML Settings"
COMSNETS 2026 AIoT Workshop
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, Subset
from datasets import load_from_disk, load_dataset

from configs.config import CONFIG
from data.dataset import UnifiedVLMDataset, collate_fn
from training.trainer import validate, train_qat_one_epoch
from quantization.quantize import (
    apply_dynamic_quantization,
    prepare_model_for_qat,
    convert_qat_model,
    save_quantized_model,
)
from utils.helpers import get_cosine_schedule_with_warmup, measure_model_size_mb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TinyVLM — QAT fine-tuning and PTQ vs QAT comparison"
    )
    p.add_argument("--nanovlm_root",    required=True)
    p.add_argument("--web_data_path",   default=CONFIG["local_dataset_web_path"])
    p.add_argument("--hf_dataset",      default=CONFIG["hf_dataset_synth"])
    p.add_argument("--student_ckpt",    required=True, help="Path to distilled student .pt")
    p.add_argument("--ptq_ckpt",        default=None,  help="Path to existing PTQ checkpoint (skip PTQ if provided)")
    p.add_argument("--output_dir",      default=CONFIG["output_dir"])
    p.add_argument("--qat_epochs",      type=int,   default=CONFIG["qat_epochs"])
    p.add_argument("--qat_lr",          type=float, default=CONFIG["qat_lr"])
    p.add_argument("--batch_size",      type=int,   default=CONFIG["batch_size"])
    p.add_argument("--no_amp",          action="store_true")
    p.add_argument("--backend",         default="fbgemm", choices=["fbgemm", "qnnpack"],
                   help="Quantization backend: fbgemm (x86) or qnnpack (ARM)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.nanovlm_root not in sys.path:
        sys.path.insert(0, args.nanovlm_root)

    from nanoVLM.data.processors import get_image_processor, get_tokenizer
    from models.vlm_wrapper import VLMWrapper

    cfg            = CONFIG.copy()
    cfg["use_amp"] = not args.no_amp
    cfg["output_dir"] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ====================================================================
    # Phase 1 — Data preparation (validation only for QAT)
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Data Preparation")
    print("=" * 60)

    web_hf   = load_from_disk(args.web_data_path)
    synth_hf = load_dataset(
        args.hf_dataset, split="train", cache_dir=os.path.join(args.output_dir, "hf_cache")
    )

    image_processor = get_image_processor(224, 224)
    tokenizer       = get_tokenizer(cfg["teacher_vlm_config"].lm_tokenizer)

    web_ds   = UnifiedVLMDataset(web_hf,   image_processor, tokenizer, cfg["caption_context_length"])
    synth_ds = UnifiedVLMDataset(synth_hf, image_processor, tokenizer, cfg["caption_context_length"])

    val_split = cfg["validation_split"]

    def split_train_val(ds):
        n_val = int(len(ds) * val_split)
        return (
            Subset(ds, range(0, len(ds) - n_val)),
            Subset(ds, range(len(ds) - n_val, len(ds))),
        )

    web_train,   web_val   = split_train_val(web_ds)
    synth_train, synth_val = split_train_val(synth_ds)

    train_dataset = ConcatDataset([web_train, synth_train])
    val_dataset   = ConcatDataset([web_val,   synth_val])

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # ====================================================================
    # Phase 2 — Load distilled student & establish PTQ baseline
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Load Distilled Student + PTQ Baseline")
    print("=" * 60)

    student = VLMWrapper(cfg["student_vlm_config"], image_processor=image_processor)
    student.load_state_dict(torch.load(args.student_ckpt, map_location="cpu"))
    student.to(device)

    # FP32 validation
    _, fp32_acc = validate(student, val_loader, device, cfg)
    fp32_size   = measure_model_size_mb(student)
    print(f"Student FP32 — Val Acc: {fp32_acc * 100:.2f}%  Size: {fp32_size:.1f} MB")

    # PTQ
    ptq_ckpt = args.ptq_ckpt or os.path.join(args.output_dir, cfg["quantized_student_model"])
    if args.ptq_ckpt and os.path.exists(args.ptq_ckpt):
        from quantization.quantize import load_dynamic_quantized_model
        base_for_ptq = VLMWrapper(cfg["student_vlm_config"], image_processor=image_processor)
        tiny_ptq     = load_dynamic_quantized_model(base_for_ptq, args.ptq_ckpt)
        print(f"Loaded existing PTQ model from {args.ptq_ckpt}")
    else:
        tiny_ptq = apply_dynamic_quantization(student)
        save_quantized_model(tiny_ptq, ptq_ckpt)

    _, ptq_acc  = validate(tiny_ptq, val_loader, "cpu", cfg)
    ptq_size    = measure_model_size_mb(tiny_ptq)
    print(f"TinyVLM PTQ  — Val Acc: {ptq_acc * 100:.2f}%  Size: {ptq_size:.1f} MB")

    del tiny_ptq
    torch.cuda.empty_cache()

    # ====================================================================
    # Phase 3 — QAT fine-tuning
    # ====================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 3: Quantization-Aware Training ({args.qat_epochs} epoch(s))")
    print("=" * 60)

    try:
        qat_model = prepare_model_for_qat(student, backend=args.backend)
    except RuntimeError as exc:
        print(f"QAT preparation failed: {exc}")
        sys.exit(1)

    qat_model.to(device)

    qat_cfg = cfg.copy()
    qat_cfg["gradient_accumulation_steps"] = cfg["gradient_accumulation_steps"]

    qat_optimizer   = optim.AdamW(qat_model.parameters(), lr=args.qat_lr, weight_decay=cfg["qat_weight_decay"])
    total_steps     = max(1, len(train_loader) * args.qat_epochs // cfg["gradient_accumulation_steps"])
    qat_scheduler   = get_cosine_schedule_with_warmup(qat_optimizer, cfg["warmup_steps"], total_steps)
    qat_scaler      = GradScaler(enabled=cfg["use_amp"])

    for epoch in range(1, args.qat_epochs + 1):
        tr_loss, tr_acc = train_qat_one_epoch(
            qat_model, train_loader, qat_optimizer, qat_scheduler, qat_scaler,
            device, epoch, qat_cfg,
        )
        vl_loss, vl_acc = validate(qat_model, val_loader, device, qat_cfg)
        print(
            f"QAT Epoch {epoch}/{args.qat_epochs} — "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}"
        )
        torch.cuda.empty_cache()

    # ====================================================================
    # Phase 4 — Convert QAT model to INT8 and save
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Convert QAT Model to INT8")
    print("=" * 60)

    try:
        qat_int8 = convert_qat_model(qat_model)
    except RuntimeError as exc:
        print(f"QAT conversion failed: {exc}")
        sys.exit(1)

    qat_ckpt_path = os.path.join(args.output_dir, cfg["qat_quantized_model"])
    save_quantized_model(qat_int8, qat_ckpt_path)

    _, qat_acc  = validate(qat_int8, val_loader, "cpu", qat_cfg)
    qat_size    = measure_model_size_mb(qat_int8)
    print(f"TinyVLM QAT  — Val Acc: {qat_acc * 100:.2f}%  Size: {qat_size:.1f} MB")

    # ====================================================================
    # Phase 5 — Table V: PTQ vs QAT comparison
    # ====================================================================
    print("\n" + "=" * 60)
    print("TABLE V: PTQ vs QAT Comparison")
    print("=" * 60)
    print(f"{'Method':<35} {'Val Acc (%)':>11} {'Size (MB)':>10}")
    print("-" * 60)
    print(f"{'PTQ (INT8)':<35} {ptq_acc * 100:>11.1f} {ptq_size:>10.1f}")
    print(f"{'QAT (INT8)':<35} {qat_acc * 100:>11.1f} {qat_size:>10.1f}")
    print("=" * 60)

    # Save as CSV
    csv_path = os.path.join(args.output_dir, "table_V_ptq_vs_qat.csv")
    with open(csv_path, "w") as f:
        f.write("Method,Val Acc (%),Model Size (MB)\n")
        f.write(f"PTQ (INT8),{ptq_acc * 100:.1f},{ptq_size:.1f}\n")
        f.write(f"QAT (INT8),{qat_acc * 100:.1f},{qat_size:.1f}\n")
    print(f"Table V saved → {csv_path}")

    print("\n" + "=" * 60)
    print("QAT pipeline complete.")
    print(f"  QAT INT8 model : {qat_ckpt_path}")
    print(f"  Table V CSV    : {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

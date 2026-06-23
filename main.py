"""
main.py
-------
TinyVLM — Full Training Pipeline
=================================
Stages
------
  1. Data preparation  — load web + synthetic datasets, build DataLoaders
  2. Teacher training  — contrastive fine-tuning of full-size VLM
  3. Knowledge distillation — train compact student with KL-divergence soft labels
  4. Post-Training Quantization (PTQ) — dynamic INT8 quantization
  5. Inference & submission — generate challenge submission CSV

Usage
-----
    python main.py \\
        --nanovlm_root /path/to/nanoVLM \\
        --web_data_path /path/to/prepared_web_dataset \\
        --test1_dir /path/to/Test1 \\
        --test2_dir /path/to/Test2 \\
        --output_dir outputs

All other hyper-parameters are read from configs/config.py and can be
overridden via CLI flags (see --help).

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
from training.trainer import train_one_epoch, validate, train_distill_one_epoch
from quantization.quantize import apply_dynamic_quantization, save_quantized_model
from inference.inference import load_test_data, run_inference, generate_submission
from utils.helpers import get_cosine_schedule_with_warmup


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="TinyVLM — Teacher training + Distillation + PTQ pipeline"
    )
    p.add_argument("--nanovlm_root",    required=True,  help="Path to nanoVLM repo root")
    p.add_argument("--web_data_path",   default=CONFIG["local_dataset_web_path"])
    p.add_argument("--hf_dataset",      default=CONFIG["hf_dataset_synth"])
    p.add_argument("--test1_dir",       default=CONFIG["test1_dir"])
    p.add_argument("--test2_dir",       default=CONFIG["test2_dir"])
    p.add_argument("--output_dir",      default=CONFIG["output_dir"])
    p.add_argument("--teacher_epochs",  type=int,   default=CONFIG["teacher_epochs"])
    p.add_argument("--student_epochs",  type=int,   default=CONFIG["student_epochs"])
    p.add_argument("--batch_size",      type=int,   default=CONFIG["batch_size"])
    p.add_argument("--no_amp",          action="store_true", help="Disable AMP (float16) training")
    p.add_argument("--skip_teacher",    action="store_true", help="Skip teacher training, load from checkpoint")
    p.add_argument("--skip_distil",     action="store_true", help="Skip distillation, load from checkpoint")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- nanoVLM path setup ----
    if args.nanovlm_root not in sys.path:
        sys.path.insert(0, args.nanovlm_root)

    from nanoVLM.data.processors import get_image_processor, get_tokenizer
    from models.vlm_wrapper import VLMWrapper

    cfg            = CONFIG.copy()
    cfg["use_amp"] = not args.no_amp
    cfg["output_dir"] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(cfg["hf_cache_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU: {prop.name}  ({prop.total_memory / 1e9:.1f} GB)")
        torch.cuda.empty_cache()

    # ====================================================================
    # Phase 1 — Data preparation
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Data Preparation")
    print("=" * 60)

    web_hf   = load_from_disk(args.web_data_path)
    synth_hf = load_dataset(
        args.hf_dataset, split="train", cache_dir=cfg["hf_cache_dir"]
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

    print(f"Train samples : {len(train_dataset):,}")
    print(f"Val   samples : {len(val_dataset):,}")
    print(f"Effective batch size: {args.batch_size * cfg['gradient_accumulation_steps']}")

    # ====================================================================
    # Phase 2 — Teacher training
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Teacher Training")
    print("=" * 60)

    teacher = VLMWrapper(cfg["teacher_vlm_config"], image_processor=image_processor).to(device)

    teacher_ckpt_path = os.path.join(args.output_dir, cfg["teacher_checkpoint"])

    if args.skip_teacher and os.path.exists(teacher_ckpt_path):
        print(f"Skipping training — loading teacher from {teacher_ckpt_path}")
        teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
    else:
        optimizer = optim.AdamW(
            [
                {"params": teacher.model.MP.parameters(), "lr": cfg["lr_mp"]},
                {"params": [p for n, p in teacher.named_parameters() if "MP." not in n],
                 "lr": cfg["lr_backbones"]},
            ],
            weight_decay=cfg["weight_decay"],
        )
        total_steps = max(1, len(train_loader) * args.teacher_epochs // cfg["gradient_accumulation_steps"])
        scheduler   = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], total_steps)
        scaler      = GradScaler(enabled=cfg["use_amp"])

        for epoch in range(1, args.teacher_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(
                teacher, train_loader, optimizer, scheduler, scaler, device, epoch, cfg
            )
            vl_loss, vl_acc = validate(teacher, val_loader, device, cfg)
            print(
                f"Epoch {epoch}/{args.teacher_epochs} — "
                f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
                f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}"
            )
            torch.cuda.empty_cache()

        torch.save(teacher.state_dict(), teacher_ckpt_path)
        print(f"Teacher checkpoint saved → {teacher_ckpt_path}")

    # ====================================================================
    # Phase 3 — Knowledge distillation
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Knowledge Distillation (Student)")
    print("=" * 60)

    student = VLMWrapper(cfg["student_vlm_config"], image_processor=image_processor).to(device)

    student_ckpt_path = os.path.join(args.output_dir, cfg["student_checkpoint"])

    if args.skip_distil and os.path.exists(student_ckpt_path):
        print(f"Skipping distillation — loading student from {student_ckpt_path}")
        student.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    else:
        optimizer   = optim.AdamW(student.parameters(), lr=cfg["lr_student"], weight_decay=cfg["weight_decay"])
        total_steps = max(1, len(train_loader) * args.student_epochs // cfg["gradient_accumulation_steps"])
        scheduler   = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], total_steps)
        scaler      = GradScaler(enabled=cfg["use_amp"])

        for epoch in range(1, args.student_epochs + 1):
            tr_loss, tr_acc = train_distill_one_epoch(
                teacher, student, train_loader, optimizer, scheduler, scaler, device, epoch, cfg
            )
            vl_loss, vl_acc = validate(student, val_loader, device, cfg)
            print(
                f"Epoch {epoch}/{args.student_epochs} — "
                f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
                f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}"
            )
            torch.cuda.empty_cache()

        torch.save(student.state_dict(), student_ckpt_path)
        print(f"Student checkpoint saved → {student_ckpt_path}")

    # Free teacher memory now
    del teacher
    torch.cuda.empty_cache()

    # ====================================================================
    # Phase 4 — Post-Training Quantization (PTQ)
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Post-Training Quantization (Dynamic INT8)")
    print("=" * 60)

    tiny_vlm = apply_dynamic_quantization(student)
    ptq_path = os.path.join(args.output_dir, cfg["quantized_student_model"])
    save_quantized_model(tiny_vlm, ptq_path)

    # Quick validation of quantized model
    _, ptq_acc = validate(tiny_vlm, val_loader, "cpu", cfg)
    print(f"TinyVLM PTQ validation accuracy: {ptq_acc * 100:.2f}%")

    # ====================================================================
    # Phase 5 — Inference & submission generation
    # ====================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Inference & Submission Generation")
    print("=" * 60)

    final_model = tiny_vlm
    sim_test1, sim_test2 = None, None
    n1, n2 = 0, 0

    try:
        files1, caps1, dir1 = load_test_data(args.test1_dir, captions_file="captions.txt")
        sim_test1 = run_inference(
            final_model, files1, caps1, dir1, "cpu", cfg["use_amp"],
            cfg["caption_context_length"],
        )
        n1 = len(files1)
        print(f"Test Set 1: {n1} images processed.")
    except FileNotFoundError as exc:
        print(f"Test Set 1 not found: {exc}")
    except Exception as exc:
        print(f"Test Set 1 error: {exc}")

    try:
        files2, caps2, dir2 = load_test_data(args.test2_dir, captions_file="captions.json")
        sim_test2 = run_inference(
            final_model, files2, caps2, dir2, "cpu", cfg["use_amp"],
            cfg["caption_context_length"],
        )
        n2 = len(files2)
        print(f"Test Set 2: {n2} images processed.")
    except FileNotFoundError as exc:
        print(f"Test Set 2 not found: {exc}")
    except Exception as exc:
        print(f"Test Set 2 error: {exc}")

    if sim_test1 is not None or sim_test2 is not None:
        sub_path = os.path.join(args.output_dir, cfg["submission_filename"])
        generate_submission(sim_test1, sim_test2, n1, n2, sub_path)
    else:
        print("No test sets processed — submission not generated.")

    # ====================================================================
    # Done
    # ====================================================================
    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Teacher checkpoint   : {os.path.join(args.output_dir, cfg['teacher_checkpoint'])}")
    print(f"  Student checkpoint   : {os.path.join(args.output_dir, cfg['student_checkpoint'])}")
    print(f"  TinyVLM (PTQ, INT8)  : {ptq_path}")
    if sim_test1 is not None or sim_test2 is not None:
        print(f"  Submission           : {os.path.join(args.output_dir, cfg['submission_filename'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()

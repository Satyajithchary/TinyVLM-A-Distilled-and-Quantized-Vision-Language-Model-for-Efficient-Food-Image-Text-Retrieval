"""
scripts/evaluate.py
-------------------
Evaluate all model variants (Teacher FP32, Student FP32, TinyVLM PTQ-INT8,
TinyVLM QAT-INT8) on the Dishcovery validation set and generate the results
tables from the paper (Table II and Table V).

Usage
-----
    python scripts/evaluate.py \\
        --nanovlm_root /path/to/nanoVLM \\
        --web_data_path /path/to/prepared_web_dataset \\
        --teacher_ckpt outputs/teacher_final.pt \\
        --student_ckpt outputs/student_distilled.pt \\
        --ptq_ckpt outputs/student_ptq_int8.pt \\
        --qat_ckpt outputs/student_qat_int8.pt \\
        --output_dir outputs
"""

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk, load_dataset

from training.trainer import validate
from utils.helpers import measure_model_size_mb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TinyVLM evaluation & table generation")
    p.add_argument("--nanovlm_root",    required=True)
    p.add_argument("--web_data_path",   required=True)
    p.add_argument("--hf_dataset",      default="jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth")
    p.add_argument("--teacher_ckpt",    default=None)
    p.add_argument("--student_ckpt",    default=None)
    p.add_argument("--ptq_ckpt",        default=None)
    p.add_argument("--qat_ckpt",        default=None)
    p.add_argument("--output_dir",      default="outputs")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--caption_length",  type=int, default=128)
    p.add_argument("--val_split",       type=float, default=0.05)
    p.add_argument("--no_amp",          action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluate a single model variant
# ---------------------------------------------------------------------------

def evaluate_variant(
    model: nn.Module,
    name: str,
    val_loader: DataLoader,
    use_amp: bool,
    n_cpu_batches: int = 3,
) -> dict:
    """
    Compute validation accuracy, on-disk model size, and simulated CPU latency.

    Returns
    -------
    dict with keys: Model, ValAcc (%), SizeMB, LatencyMs
    """
    config = {"use_amp": use_amp}

    # --- Validation accuracy (GPU/CPU) ---
    device = next(model.parameters()).device
    try:
        _, val_acc = validate(model, val_loader, str(device), config)
    except Exception as exc:
        print(f"[eval] Validation failed for {name}: {exc}")
        val_acc = 0.0

    # --- On-disk size ---
    try:
        size_mb = measure_model_size_mb(model)
    except Exception as exc:
        print(f"[eval] Size measurement failed for {name}: {exc}")
        size_mb = 0.0

    # --- Simulated CPU latency (throughput: images per second) ---
    latency_ms = 0.0
    try:
        cpu_model = copy.deepcopy(model).to("cpu").eval()
        imgs_done = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if batch is None or idx >= n_cpu_batches:
                    break
                images = batch["pixel_values"].to("cpu")
                ids    = batch["input_ids"].to("cpu")
                mask   = batch["attention_mask"].to("cpu")
                cpu_model(images, ids, mask)
                imgs_done += images.size(0)
        elapsed = time.perf_counter() - t0
        latency_ms = (elapsed / imgs_done * 1000.0) if imgs_done > 0 else 0.0
        del cpu_model
    except Exception as exc:
        print(f"[eval] Latency measurement failed for {name}: {exc}")

    return {
        "Model":      name,
        "ValAcc":     val_acc * 100.0,
        "SizeMB":     size_mb,
        "LatencyMs":  latency_ms,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table_ii(rows: list[dict], teacher_size_mb: float, output_path: str = None):
    """Print (and optionally save) the main results table (Table II)."""
    header = ["Model", "Val Acc (%)", "Model Size (MB)", "Relative Size", "CPU Latency (ms)"]
    print("\n" + "=" * 90)
    print("TABLE II: Main Experimental Results")
    print("=" * 90)
    print(f"{'Model':<48} {'Val Acc (%)':>11} {'Size (MB)':>10} {'Rel Size':>10} {'Latency (ms)':>13}")
    print("-" * 90)
    for r in rows:
        rel = f"{r['SizeMB'] / teacher_size_mb:.2f}x" if teacher_size_mb > 0 else "N/A"
        print(
            f"{r['Model']:<48} {r['ValAcc']:>11.1f} {r['SizeMB']:>10.1f}"
            f" {rel:>10} {r['LatencyMs']:>13.1f}"
        )
    print("=" * 90)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                rel = f"{r['SizeMB'] / teacher_size_mb:.2f}x" if teacher_size_mb > 0 else "N/A"
                f.write(
                    f"{r['Model']},{r['ValAcc']:.1f},{r['SizeMB']:.1f},{rel},{r['LatencyMs']:.1f}\n"
                )
        print(f"[evaluate] Table II saved → {output_path}")


def print_table_v(ptq_row: dict, qat_row: dict, output_path: str = None):
    """Print (and optionally save) the PTQ vs QAT comparison table (Table V)."""
    print("\n" + "=" * 55)
    print("TABLE V: PTQ vs QAT Comparison")
    print("=" * 55)
    print(f"{'Method':<35} {'Val Acc (%)':>11} {'Size (MB)':>10}")
    print("-" * 55)
    for r in (ptq_row, qat_row):
        if r:
            print(f"{r['Model']:<35} {r['ValAcc']:>11.1f} {r['SizeMB']:>10.1f}")
    print("=" * 55)

    if output_path:
        with open(output_path, "w") as f:
            f.write("Method,Val Acc (%),Model Size (MB)\n")
            for r in (ptq_row, qat_row):
                if r:
                    f.write(f"{r['Model']},{r['ValAcc']:.1f},{r['SizeMB']:.1f}\n")
        print(f"[evaluate] Table V saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.nanovlm_root not in sys.path:
        sys.path.insert(0, args.nanovlm_root)

    from nanoVLM.data.processors import get_image_processor, get_tokenizer
    from models.vlm_wrapper import VLMWrapper
    from configs.config import TEACHER_CONFIG, STUDENT_CONFIG
    from quantization.quantize import load_dynamic_quantized_model
    from data.dataset import UnifiedVLMDataset, collate_fn

    use_amp = not args.no_amp
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build validation dataloader ----
    print("Loading datasets…")
    web_hf   = load_from_disk(args.web_data_path)
    synth_hf = load_dataset(args.hf_dataset, split="train",
                             cache_dir=os.path.join(args.output_dir, "hf_cache"))

    image_processor = get_image_processor(224, 224)
    tokenizer       = get_tokenizer(TEACHER_CONFIG.lm_tokenizer)

    def make_val_subset(hf_ds):
        full   = UnifiedVLMDataset(hf_ds, image_processor, tokenizer, args.caption_length)
        n_val  = int(len(full) * args.val_split)
        return Subset(full, range(len(full) - n_val, len(full)))

    val_dataset = torch.utils.data.ConcatDataset([
        make_val_subset(web_hf),
        make_val_subset(synth_hf),
    ])
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=False,
    )

    rows:     list[dict] = []
    ptq_row:  dict       = {}
    qat_row:  dict       = {}

    # ---- Teacher FP32 ----
    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        print("\nEvaluating Teacher (FP32)…")
        m = VLMWrapper(TEACHER_CONFIG, image_processor=image_processor).to(device)
        m.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        row = evaluate_variant(m, "Teacher VLM (FP32)", val_loader, use_amp)
        rows.append(row)
        teacher_size_mb = row["SizeMB"]
        del m
        torch.cuda.empty_cache()
    else:
        teacher_size_mb = 847.17   # fallback from paper

    # ---- Student FP32 ----
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        print("\nEvaluating Student / Distilled (FP32)…")
        m = VLMWrapper(STUDENT_CONFIG, image_processor=image_processor).to(device)
        m.load_state_dict(torch.load(args.student_ckpt, map_location=device))
        rows.append(evaluate_variant(m, "Student VLM (with Distillation, FP32)", val_loader, use_amp))
        del m
        torch.cuda.empty_cache()

    # ---- TinyVLM PTQ-INT8 ----
    if args.ptq_ckpt and os.path.exists(args.ptq_ckpt):
        print("\nEvaluating TinyVLM PTQ (INT8)…")
        base  = VLMWrapper(STUDENT_CONFIG, image_processor=image_processor)
        tiny  = load_dynamic_quantized_model(base, args.ptq_ckpt)
        ptq_row = evaluate_variant(tiny, "TinyVLM (Distilled + PTQ, INT8)", val_loader, use_amp)
        rows.append(ptq_row)
        del base, tiny

    # ---- TinyVLM QAT-INT8 ----
    if args.qat_ckpt and os.path.exists(args.qat_ckpt):
        print("\nEvaluating TinyVLM QAT (INT8)…")
        from quantization.quantize import prepare_model_for_qat, convert_qat_model
        base        = VLMWrapper(STUDENT_CONFIG, image_processor=image_processor)
        qat_struct  = prepare_model_for_qat(base)
        state       = torch.load(args.qat_ckpt, map_location="cpu")
        qat_struct.load_state_dict(state)
        qat_int8    = convert_qat_model(qat_struct)
        qat_row = evaluate_variant(qat_int8, "TinyVLM (Distilled + QAT, INT8)", val_loader, use_amp)
        rows.append(qat_row)
        del base, qat_struct, qat_int8

    # ---- Print tables ----
    if rows:
        print_table_ii(
            rows,
            teacher_size_mb,
            output_path=os.path.join(args.output_dir, "table_II_results.csv"),
        )
    if ptq_row and qat_row:
        print_table_v(
            ptq_row, qat_row,
            output_path=os.path.join(args.output_dir, "table_V_ptq_vs_qat.csv"),
        )


if __name__ == "__main__":
    main()

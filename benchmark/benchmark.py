"""
benchmark/benchmark.py
----------------------
CPU inference-latency benchmark for Teacher, Student (FP32), and
TinyVLM (INT8 quantized) models.

Usage
-----
    python benchmark/benchmark.py \\
        --nanovlm_root /path/to/nanoVLM \\
        --teacher_ckpt outputs/teacher_final.pt \\
        --student_ckpt outputs/student_distilled.pt \\
        --quantized_ckpt outputs/student_ptq_int8.pt
"""

import argparse
import sys
import os
import time

import numpy as np
import torch
import torch.nn as nn

from configs.config import TEACHER_CONFIG, STUDENT_CONFIG

WARMUP_RUNS    = 20
BENCHMARK_RUNS = 100
BATCH_SIZE     = 1
CONTEXT_LENGTH = 128
IMAGE_SIZE     = 224
DEVICE         = "cpu"


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_model(name: str, model: nn.Module) -> float:
    """
    Measure average per-sample inference latency on CPU.

    Parameters
    ----------
    name  : str         display name
    model : nn.Module   model to benchmark

    Returns
    -------
    float  average latency in milliseconds
    """
    model.to(DEVICE).eval()

    dummy_images  = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    dummy_ids     = torch.randint(0, 30_000, (BATCH_SIZE, CONTEXT_LENGTH), device=DEVICE)
    dummy_mask    = torch.ones(BATCH_SIZE, CONTEXT_LENGTH, device=DEVICE)

    print(f"\n--- Benchmarking: {name} ---")
    timings = []

    with torch.no_grad():
        print(f"  Warm-up ({WARMUP_RUNS} runs)…")
        for _ in range(WARMUP_RUNS):
            _ = model(dummy_images, dummy_ids, dummy_mask)

        print(f"  Measurement ({BENCHMARK_RUNS} runs)…")
        for _ in range(BENCHMARK_RUNS):
            t0 = time.perf_counter()
            _  = model(dummy_images, dummy_ids, dummy_mask)
            timings.append((time.perf_counter() - t0) * 1_000)

    avg_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    print(f"  Avg latency : {avg_ms:.2f} ms  (± {std_ms:.2f} ms)")
    return avg_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TinyVLM latency benchmark")
    p.add_argument("--nanovlm_root",    required=True,  help="Path to the nanoVLM repo root")
    p.add_argument("--teacher_ckpt",    required=False, default=None)
    p.add_argument("--student_ckpt",    required=False, default=None)
    p.add_argument("--quantized_ckpt",  required=False, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # --- nanoVLM path setup ---
    if args.nanovlm_root not in sys.path:
        sys.path.insert(0, args.nanovlm_root)

    from models.vlm_wrapper import VLMWrapper
    from quantization.quantize import load_dynamic_quantized_model

    results: dict[str, float] = {}

    # Teacher
    if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
        print("Loading Teacher (FP32)…")
        teacher = VLMWrapper(TEACHER_CONFIG)
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=DEVICE))
        results["Teacher VLM (FP32)"] = benchmark_model("Teacher VLM (FP32)", teacher)
        del teacher

    # Student FP32
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        print("Loading Student (FP32)…")
        student = VLMWrapper(STUDENT_CONFIG)
        student.load_state_dict(torch.load(args.student_ckpt, map_location=DEVICE))
        results["Student VLM (with Distillation, FP32)"] = benchmark_model(
            "Student VLM (with Distillation, FP32)", student
        )
        del student

    # TinyVLM quantized (INT8)
    if args.quantized_ckpt and os.path.exists(args.quantized_ckpt):
        print("Loading TinyVLM (INT8)…")
        base  = VLMWrapper(STUDENT_CONFIG)
        tiny  = load_dynamic_quantized_model(base, args.quantized_ckpt, device=DEVICE)
        results["TinyVLM (Distilled + Quantized, INT8)"] = benchmark_model(
            "TinyVLM (Distilled + Quantized, INT8)", tiny
        )
        del base, tiny

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY (simulated CPU, single-sample)")
    print("=" * 60)
    print(f"{'Model':<45} {'Latency (ms)':>12}")
    print("-" * 60)
    for name, lat in results.items():
        print(f"{name:<45} {lat:>12.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

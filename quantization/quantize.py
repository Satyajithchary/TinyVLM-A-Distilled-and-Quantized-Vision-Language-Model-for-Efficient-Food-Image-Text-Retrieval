"""
quantization/quantize.py
------------------------
Utilities for Post-Training Quantization (PTQ) and
Quantization-Aware Training (QAT) of the VLMWrapper student model.

PTQ (dynamic)
~~~~~~~~~~~~~
Weights of all nn.Linear layers are converted to INT8 offline; activations
are quantized on-the-fly at inference time.  No re-training is required.

QAT (static, fbgemm)
~~~~~~~~~~~~~~~~~~~~
Fake-quantize nodes are inserted into the graph before fine-tuning so the
model can adapt to reduced-precision arithmetic.  The QAT model is then
converted to a true INT8 model via torch.quantization.convert().
"""

import copy
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Post-Training Dynamic Quantization (PTQ)
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to all nn.Linear layers.

    The model is first moved to CPU (dynamic quantization is CPU-only in
    PyTorch) and set to eval mode.

    Parameters
    ----------
    model : nn.Module  trained FP32 student model

    Returns
    -------
    nn.Module  dynamically quantized INT8 model (on CPU)
    """
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    return quantized


def save_quantized_model(model: nn.Module, path: str) -> None:
    """Save the state_dict of a (quantized) model to `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024.0 ** 2)
    print(f"[PTQ] Quantized model saved → {path}  ({size_mb:.1f} MB)")


def load_dynamic_quantized_model(base_model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """
    Reconstruct a dynamically quantized model from a saved state_dict.

    Steps
    -----
    1. Apply quantize_dynamic() to the base model structure (same as during saving).
    2. Load the INT8 state_dict.

    Parameters
    ----------
    base_model : nn.Module  freshly initialised (FP32) model with the student config
    path       : str        path to the saved INT8 state_dict
    device     : str        target device (must be "cpu" for dynamic quantization)

    Returns
    -------
    nn.Module  quantized model ready for inference
    """
    base_model.to("cpu")
    quantized = torch.quantization.quantize_dynamic(base_model, {nn.Linear}, dtype=torch.qint8)
    state_dict = torch.load(path, map_location=device)
    quantized.load_state_dict(state_dict)
    quantized.eval()
    return quantized


# ---------------------------------------------------------------------------
# Quantization-Aware Training (QAT)
# ---------------------------------------------------------------------------

def prepare_model_for_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """
    Insert fake-quantize nodes into a model for QAT fine-tuning.

    The returned model can be trained normally; after training call
    `convert_qat_model()` to obtain the actual INT8 model.

    Parameters
    ----------
    model   : nn.Module  FP32 student (typically a deep-copy of the distilled student)
    backend : str        'fbgemm' (x86) or 'qnnpack' (ARM)

    Returns
    -------
    nn.Module  QAT-prepared model (in-place modification + returned)

    Raises
    ------
    RuntimeError if torch.quantization.prepare_qat() fails.
    """
    qat_model = copy.deepcopy(model)
    try:
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(qat_model, inplace=True)
        print(f"[QAT] Model prepared with backend='{backend}'.")
    except Exception as exc:
        raise RuntimeError(
            f"[QAT] prepare_qat failed: {exc}\n"
            "Consider using backend='qnnpack' on ARM devices."
        ) from exc
    return qat_model


def convert_qat_model(qat_model: nn.Module) -> nn.Module:
    """
    Convert a QAT-trained model to a true INT8 model (CPU only).

    Parameters
    ----------
    qat_model : nn.Module  QAT-prepared and fine-tuned model

    Returns
    -------
    nn.Module  INT8 model on CPU
    """
    model_cpu = copy.deepcopy(qat_model).to("cpu")
    model_cpu.eval()
    try:
        torch.quantization.convert(model_cpu, inplace=True)
        print("[QAT] Conversion to INT8 complete.")
    except Exception as exc:
        raise RuntimeError(f"[QAT] convert() failed: {exc}") from exc
    return model_cpu

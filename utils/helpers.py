"""
utils/helpers.py
----------------
Shared utilities:
  - Cosine LR schedule with linear warmup
  - Attention-masked mean pooling
  - Model serialization size measurement
"""

import os
import math
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Cosine decay schedule with a linear warm-up phase.

    During warm-up the LR rises linearly from 0 to base_lr.
    After warm-up it follows a cosine curve from base_lr down to 0.

    Parameters
    ----------
    optimizer           : torch optimiser
    num_warmup_steps    : int  length of the warm-up phase
    num_training_steps  : int  total number of scheduler steps
    num_cycles          : float  number of cosine half-waves (default 0.5 = one descent)
    last_epoch          : int   passed through to LambdaLR

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# ---------------------------------------------------------------------------
# Mean pooling
# ---------------------------------------------------------------------------

def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the attention-weighted mean of token hidden states.

    Parameters
    ----------
    hidden_state   : Tensor [B, L, D]
    attention_mask : Tensor [B, L]  (1 = keep, 0 = pad)

    Returns
    -------
    Tensor [B, D]
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * mask_expanded, dim=1) / torch.clamp(
        mask_expanded.sum(dim=1), min=1e-9
    )


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def measure_model_size_mb(model: nn.Module) -> float:
    """
    Serialize the model's state_dict to a temporary file and return its size
    in megabytes.  This reflects the actual on-disk footprint, including any
    INT8 quantized parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    float  size in MB
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        os.remove(tmp_path)
    return size_bytes / (1024.0 ** 2)

"""
training/trainer.py
-------------------
Training and validation loops for TinyVLM.

  train_one_epoch        — standard contrastive training (teacher)
  validate               — validation loop (no gradient)
  train_distill_one_epoch — knowledge-distillation training (student)
  train_qat_one_epoch    — quantization-aware fine-tuning (QAT student)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from training.losses import contrastive_loss, combined_student_loss, compute_accuracy


# ---------------------------------------------------------------------------
# Standard contrastive training (teacher)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: str,
    epoch_num: int,
    config: dict,
) -> tuple[float, float]:
    """
    One epoch of contrastive training.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    model.train()
    grad_accum = config.get("gradient_accumulation_steps", 1)
    max_norm   = config.get("max_grad_norm", 1.0)
    use_amp    = config.get("use_amp", True)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} [train]")
    for step, batch in enumerate(pbar):
        if batch is None:
            continue

        images         = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast(dtype=torch.float16, enabled=use_amp):
            img_feats, txt_feats = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_feats, txt_feats, model.logit_scale.exp())
            loss = loss / grad_accum

        acc = compute_accuracy(img_feats, txt_feats, model.logit_scale.exp())
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * grad_accum
        total_acc  += acc
        n_batches  += 1

        pbar.set_postfix(
            loss=f"{loss.item() * grad_accum:.4f}",
            acc =f"{acc:.4f}",
            lr  =f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if step % 50 == 0:
            torch.cuda.empty_cache()

    return (total_loss / n_batches if n_batches else 0.0,
            total_acc  / n_batches if n_batches else 0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    device: str,
    config: dict,
) -> tuple[float, float]:
    """
    Validation loop (no gradients).

    Returns
    -------
    (avg_val_loss, avg_val_accuracy)
    """
    model.eval()
    use_amp = config.get("use_amp", True)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    pbar = tqdm(dataloader, desc="Validating")

    for batch in pbar:
        if batch is None:
            continue

        images         = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast(dtype=torch.float16, enabled=use_amp):
            img_feats, txt_feats = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_feats, txt_feats, model.logit_scale.exp())

        acc = compute_accuracy(img_feats, txt_feats, model.logit_scale.exp())

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1
        pbar.set_postfix(val_loss=f"{loss.item():.4f}", val_acc=f"{acc:.4f}")

    return (total_loss / n_batches if n_batches else 0.0,
            total_acc  / n_batches if n_batches else 0.0)


# ---------------------------------------------------------------------------
# Knowledge distillation training (student)
# ---------------------------------------------------------------------------

def train_distill_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: str,
    epoch_num: int,
    config: dict,
) -> tuple[float, float]:
    """
    One epoch of knowledge-distillation training for the student model.

    Teacher runs in eval/no-grad mode; only the student is updated.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    teacher.eval()
    student.train()

    alpha       = config.get("distillation_alpha", 0.5)
    temperature = config.get("distillation_temperature", 2.0)
    grad_accum  = config.get("gradient_accumulation_steps", 1)
    max_norm    = config.get("max_grad_norm", 1.0)
    use_amp     = config.get("use_amp", True)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} [distil]")
    for step, batch in enumerate(pbar):
        if batch is None:
            continue

        images         = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            t_img, t_txt = teacher(images, input_ids, attention_mask)

        with autocast(dtype=torch.float16, enabled=use_amp):
            s_img, s_txt = student(images, input_ids, attention_mask)
            loss = combined_student_loss(
                s_img, s_txt, t_img, t_txt,
                student.logit_scale.exp(), alpha, temperature,
            ) / grad_accum

        acc = compute_accuracy(s_img, s_txt, student.logit_scale.exp())
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * grad_accum
        total_acc  += acc
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", acc=f"{acc:.4f}")

        if step % 50 == 0:
            torch.cuda.empty_cache()

    return (total_loss / n_batches if n_batches else 0.0,
            total_acc  / n_batches if n_batches else 0.0)


# ---------------------------------------------------------------------------
# QAT fine-tuning loop
# ---------------------------------------------------------------------------

def train_qat_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: str,
    epoch_num: int,
    config: dict,
) -> tuple[float, float]:
    """
    One epoch of quantization-aware training (QAT).

    The model must have been prepared with torch.quantization.prepare_qat()
    before calling this function. The loop is identical to the standard
    contrastive loop — fake-quantize operators inserted by prepare_qat()
    handle the INT8 simulation transparently.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    model.train()
    grad_accum = config.get("gradient_accumulation_steps", 1)
    max_norm   = config.get("max_grad_norm", 1.0)
    use_amp    = config.get("use_amp", True)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} [QAT]")
    for step, batch in enumerate(pbar):
        if batch is None:
            continue

        images         = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast(dtype=torch.float16, enabled=use_amp):
            img_feats, txt_feats = model(images, input_ids, attention_mask)
            loss = contrastive_loss(img_feats, txt_feats, model.logit_scale.exp())
            loss = loss / grad_accum

        acc = compute_accuracy(img_feats, txt_feats, model.logit_scale.exp())
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * grad_accum
        total_acc  += acc
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}", acc=f"{acc:.4f}")

        if step % 50 == 0:
            torch.cuda.empty_cache()

    return (total_loss / n_batches if n_batches else 0.0,
            total_acc  / n_batches if n_batches else 0.0)

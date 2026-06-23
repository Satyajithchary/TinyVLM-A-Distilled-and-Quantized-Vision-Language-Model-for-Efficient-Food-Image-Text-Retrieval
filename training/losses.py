"""
training/losses.py
------------------
Loss functions and accuracy helpers for contrastive image-text training
and knowledge distillation.

  contrastive_loss  — symmetric InfoNCE (CLIP-style)
  distillation_loss — KL-divergence on teacher/student similarity matrices
  compute_accuracy  — batch-level i2t + t2i top-1 retrieval accuracy
"""

import torch
import torch.nn.functional as F


def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss.

    Given a batch of N (image, text) pairs, the diagonal of the similarity
    matrix contains the positive pairs; off-diagonal entries are negatives.

    Parameters
    ----------
    image_features : Tensor [N, d]  — L2-normalised image embeddings
    text_features  : Tensor [N, d]  — L2-normalised text  embeddings
    logit_scale    : scalar Tensor  — exp of learnable temperature parameter

    Returns
    -------
    scalar Tensor
    """
    logits_per_image = logit_scale * image_features @ text_features.t()   # [N, N]
    logits_per_text  = logits_per_image.t()                                # [N, N]
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)

    loss = (
        F.cross_entropy(logits_per_image, labels)
        + F.cross_entropy(logits_per_text, labels)
    ) / 2.0
    return loss


def distillation_loss(
    student_img_feats: torch.Tensor,
    student_txt_feats: torch.Tensor,
    teacher_img_feats: torch.Tensor,
    teacher_txt_feats: torch.Tensor,
    logit_scale: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    KL-divergence distillation loss on the similarity matrices.

    The student is trained to replicate the inter-sample similarity
    distribution produced by the teacher, soft-ened by `temperature`.

    Loss = KL( softmax(S_student / T) || softmax(S_teacher / T) ) * T²

    Parameters
    ----------
    student_img_feats : Tensor [N, d]
    student_txt_feats : Tensor [N, d]
    teacher_img_feats : Tensor [N, d]
    teacher_txt_feats : Tensor [N, d]
    logit_scale       : scalar Tensor
    temperature       : float   distillation temperature T

    Returns
    -------
    scalar Tensor
    """
    student_logits = logit_scale * student_img_feats @ student_txt_feats.t()
    teacher_logits = logit_scale * teacher_img_feats @ teacher_txt_feats.t()

    loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits  / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return loss


def combined_student_loss(
    student_img_feats: torch.Tensor,
    student_txt_feats: torch.Tensor,
    teacher_img_feats: torch.Tensor,
    teacher_txt_feats: torch.Tensor,
    logit_scale: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    """
    Weighted sum of hard (contrastive) and soft (distillation) losses.

    L = (1 - α) · L_contrastive + α · L_distill

    Parameters
    ----------
    alpha       : float  weight on the distillation term (0 → pure contrastive)
    temperature : float  distillation temperature
    """
    hard = contrastive_loss(student_img_feats, student_txt_feats, logit_scale)
    soft = distillation_loss(
        student_img_feats, student_txt_feats,
        teacher_img_feats, teacher_txt_feats,
        logit_scale, temperature,
    )
    return (1.0 - alpha) * hard + alpha * soft


@torch.no_grad()
def compute_accuracy(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> float:
    """
    Batch-level top-1 retrieval accuracy, averaged over i→t and t→i directions.

    Returns
    -------
    float in [0, 1]
    """
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text  = logits_per_image.t()
    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)

    i2t_acc = (logits_per_image.argmax(dim=1) == labels).float().mean().item()
    t2i_acc = (logits_per_text .argmax(dim=1) == labels).float().mean().item()
    return (i2t_acc + t2i_acc) / 2.0

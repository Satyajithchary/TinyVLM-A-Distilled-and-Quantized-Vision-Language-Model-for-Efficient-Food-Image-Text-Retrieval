"""
data/dataset.py
---------------
UnifiedVLMDataset: works with both the web-scraped dataset (local images
referenced by path) and the synthetic HuggingFace dataset (images stored
inline as PIL objects).

collate_fn filters out None items produced by failed __getitem__ calls,
so the DataLoader never crashes on a single corrupt sample.
"""

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UnifiedVLMDataset(Dataset):
    """
    Unified dataset for image-caption contrastive training.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        A HuggingFace dataset with at minimum 'caption' and either
        'local_path' (web split) or 'image' (synthetic split) columns.
    image_processor : callable
        Transforms a PIL Image to a float32 Tensor of shape [3, H, W].
    tokenizer : callable
        HuggingFace tokenizer for encoding captions.
    caption_length : int
        Maximum token length for caption padding/truncation.
    """

    def __init__(self, hf_dataset, image_processor, tokenizer, caption_length: int):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.caption_length = caption_length
        # Distinguish web dataset (file-system images) from HF dataset (inline PIL)
        self.is_web_dataset = "local_path" in self.dataset.column_names

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Optional[dict]:
        try:
            example = self.dataset[idx]
            caption = example.get("caption", "")

            if not isinstance(caption, str) or not caption.strip():
                return None

            # ---- Load image ------------------------------------------------
            if self.is_web_dataset:
                image_path = example.get("local_path")
                if not image_path or not os.path.exists(image_path):
                    return None
                image_obj = Image.open(image_path).convert("RGB")
            else:
                image_obj = example.get("image")
                if image_obj is None or not isinstance(image_obj, Image.Image):
                    return None
                image_obj = image_obj.convert("RGB")

            # ---- Pre-process image -----------------------------------------
            processed = self.image_processor(image_obj)

            if isinstance(processed, tuple):
                processed = processed[0]
            if isinstance(processed, dict):
                processed = processed.get(
                    "pixel_values",
                    processed.get("images", next(iter(processed.values()))),
                )
            if not isinstance(processed, torch.Tensor):
                if isinstance(processed, np.ndarray):
                    processed = torch.from_numpy(processed)
                else:
                    return None

            # Ensure shape [C, H, W]
            if processed.dim() == 4:
                processed = processed.squeeze(0)
            if processed.dim() != 3:
                return None

            processed = processed.float()

            # ---- Tokenise caption ------------------------------------------
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.caption_length,
                return_tensors="pt",
            )

            return {
                "image": processed,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

        except Exception as exc:
            print(f"[UnifiedVLMDataset] Warning: skipping item {idx} — {exc}")
            return None


def collate_fn(batch):
    """
    Custom collate: drops None samples silently so a single bad item
    does not break a whole batch.

    Returns None when the entire batch is empty (trainer skips it).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    return {
        "pixel_values":   torch.stack([item["image"]          for item in batch]),
        "input_ids":      torch.stack([item["input_ids"]      for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }

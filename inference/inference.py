"""
inference/inference.py
----------------------
Inference utilities for the Dishcovery: VLM MetaFood Challenge.

  load_test_data      — reads image filenames and captions from a test directory
  run_inference       — encodes all images and captions, returns similarity matrix
  generate_submission — writes a submission CSV from two similarity matrices
"""

import json
import os
from typing import Optional

import torch
from PIL import Image
from torch.cuda.amp import autocast
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(
    test_dir: str,
    images_file: str = "images.txt",
    captions_file: str = "captions.txt",
) -> tuple[list[str], list[str], str]:
    """
    Load image filenames and captions from a test split directory.

    Expected directory layout::

        test_dir/
          images.txt        — one filename per line
          captions.txt      — one caption per line  (or captions.json)
          imgs/             — directory containing the images

    Parameters
    ----------
    test_dir      : str  path to the test split directory
    images_file   : str  name of the file listing image filenames
    captions_file : str  name of the captions file (.txt or .json)

    Returns
    -------
    (image_filenames, captions, imgs_dir)
    """
    image_order_path = os.path.join(test_dir, images_file)
    captions_path    = os.path.join(test_dir, captions_file)
    imgs_dir         = os.path.join(test_dir, "imgs")

    for p in (image_order_path, captions_path, imgs_dir):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected path not found: {p}")

    with open(image_order_path, "r") as f:
        image_filenames = [line.strip() for line in f if line.strip()]

    if captions_file.endswith(".json"):
        with open(captions_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            captions = [data[k] for k in sorted(data.keys(), key=int)]
        else:
            captions = data
    else:
        with open(captions_path, "r") as f:
            captions = [line.strip() for line in f if line.strip()]

    return image_filenames, captions, imgs_dir


# ---------------------------------------------------------------------------
# Encoding and similarity matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    image_files: list[str],
    captions: list[str],
    imgs_dir: str,
    device: str,
    use_amp: bool,
    context_length: int,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Encode all images and captions and return a cosine-similarity matrix.

    Parameters
    ----------
    model          : VLMWrapper (eval mode expected)
    image_files    : list of image filenames in imgs_dir
    captions       : list of caption strings
    imgs_dir       : path to image directory
    device         : 'cuda' or 'cpu'
    use_amp        : whether to enable AMP (float16) inference
    context_length : max caption token length
    batch_size     : images/captions per forward pass

    Returns
    -------
    Tensor [n_images, n_captions]  float32 similarity matrix
    """
    model.eval()
    all_img_feats: list[torch.Tensor] = []
    all_txt_feats: list[torch.Tensor] = []

    # ---- Encode images ----
    for i in tqdm(range(0, len(image_files), batch_size), desc="Encoding images"):
        batch_imgs = []
        for fname in image_files[i : i + batch_size]:
            img_path = os.path.join(imgs_dir, fname)
            batch_imgs.append(Image.open(img_path).convert("RGB"))

        pixels_list = []
        for img in batch_imgs:
            processed = model.image_processor(img)
            if isinstance(processed, dict):
                processed = processed["pixel_values"]
            if isinstance(processed, torch.Tensor) and processed.dim() == 4:
                processed = processed.squeeze(0)
            pixels_list.append(processed)

        pixels = torch.stack(pixels_list).to(device)
        with autocast(dtype=torch.float16, enabled=use_amp):
            feats = model.encode_image(pixels)
        all_img_feats.append(feats.cpu())

    # ---- Encode captions ----
    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
        batch_caps = captions[i : i + batch_size]
        tokens = model.tokenizer(
            batch_caps,
            padding=True,
            truncation=True,
            max_length=context_length,
            return_tensors="pt",
        )
        with autocast(dtype=torch.float16, enabled=use_amp):
            feats = model.encode_text(
                tokens.input_ids.to(device), tokens.attention_mask.to(device)
            )
        all_txt_feats.append(feats.cpu())

    sim_matrix = (torch.cat(all_img_feats) @ torch.cat(all_txt_feats).t()).float()
    return sim_matrix


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------

def generate_submission(
    sim_test1: Optional[torch.Tensor],
    sim_test2: Optional[torch.Tensor],
    num_test1_imgs: int,
    num_test2_imgs: int,
    output_path: str,
    test1_threshold: float = 0.22,
) -> None:
    """
    Write a challenge submission CSV.

    Test Set 1 (multi-label): predict all captions whose similarity score
    exceeds `test1_threshold`.
    Test Set 2 (single-label): predict the argmax caption.

    Parameters
    ----------
    sim_test1        : Tensor [n1, n_caps]  or None
    sim_test2        : Tensor [n2, n_caps]  or None
    num_test1_imgs   : int
    num_test2_imgs   : int
    output_path      : str  path to output CSV
    test1_threshold  : float  similarity threshold for multi-label predictions
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("image_id,class_ids\n")

        if sim_test1 is not None:
            for i in range(num_test1_imgs):
                indices = torch.where(sim_test1[i] >= test1_threshold)[0].tolist()
                preds = "-".join(map(str, indices))
                f.write(f"{i},{preds}\n")

        if sim_test2 is not None:
            for i in range(num_test2_imgs):
                pred = torch.argmax(sim_test2[i]).item()
                f.write(f"{num_test1_imgs + i},{pred}\n")

    print(f"[Inference] Submission saved → {output_path}")

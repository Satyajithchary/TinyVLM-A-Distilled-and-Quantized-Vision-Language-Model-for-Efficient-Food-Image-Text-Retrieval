"""
models/vlm_wrapper.py
---------------------
VLMWrapper: a thin nn.Module that wraps a nanoVLM VisionLanguageModel and
exposes separate encode_image / encode_text encoders for contrastive learning.

Architecture
~~~~~~~~~~~~
  Image path:  images → vision_encoder (SigLIP ViT) → modality projector → mean-pool
  Text  path:  input_ids → token_embedding → LM decoder → mean-pool

Normalised embeddings are returned for computing cosine-similarity matrices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import VLMConfig
from utils.helpers import mean_pooling


class VLMWrapper(nn.Module):
    """
    Wraps a nanoVLM VisionLanguageModel for contrastive image-text training.

    Parameters
    ----------
    vlm_config : VLMConfig
        Model hyper-parameters.
    image_processor : callable, optional
        Pre-processing transform for raw PIL images (used during inference).
    """

    def __init__(self, vlm_config: VLMConfig, image_processor=None):
        super().__init__()

        # Import here so the module can be instantiated without nanoVLM on
        # the Python path — the caller (main.py) adds nanoVLM to sys.path
        # before constructing this object.
        from nanoVLM.models.vision_language_model import VisionLanguageModel
        from nanoVLM.data.processors import get_tokenizer

        self.model = VisionLanguageModel(vlm_config)
        self.tokenizer = get_tokenizer(vlm_config.lm_tokenizer)
        self.image_processor = image_processor

        # Learnable logit scale (initialised to log(1/0.07) ≈ 2.659, same as CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of pre-processed images to unit-length feature vectors.

        Parameters
        ----------
        images : Tensor  [B, 3, H, W]

        Returns
        -------
        Tensor  [B, proj_dim]
        """
        # Vision encoder is kept frozen (no gradients propagate back through it)
        with torch.no_grad():
            img_tokens = self.model.vision_encoder(images)

        # Unpack whichever format the encoder returns
        if hasattr(img_tokens, "last_hidden_state"):
            vision_features = img_tokens.last_hidden_state
        elif isinstance(img_tokens, torch.Tensor) and img_tokens.dim() == 3:
            vision_features = img_tokens
        else:
            vision_features = img_tokens

        # Modality projector (receives gradients during training)
        projected = self.model.MP(vision_features)         # [B, T, proj_dim]
        return projected.mean(dim=1)                       # [B, proj_dim]

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of tokenised captions to feature vectors.

        Parameters
        ----------
        input_ids      : Tensor  [B, L]
        attention_mask : Tensor  [B, L]

        Returns
        -------
        Tensor  [B, hidden_dim]
        """
        with torch.no_grad():
            token_embeddings = self.model.decoder.token_embedding(input_ids)
            text_outputs = self.model.decoder(
                token_embeddings, attention_mask=attention_mask, start_pos=0
            )

        # Unpack whichever format the decoder returns
        if isinstance(text_outputs, torch.Tensor):
            hidden_states = text_outputs
        elif hasattr(text_outputs, "last_hidden_state"):
            hidden_states = text_outputs.last_hidden_state
        elif isinstance(text_outputs, tuple):
            hidden_states = text_outputs[0]
        else:
            hidden_states = text_outputs

        return mean_pooling(hidden_states, attention_mask)  # [B, hidden_dim]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images, input_ids, attention_mask):
        """
        Returns L2-normalised image and text embeddings.

        Returns
        -------
        image_features : Tensor  [B, d]
        text_features  : Tensor  [B, d]
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        return (
            F.normalize(image_features, dim=-1),
            F.normalize(text_features,  dim=-1),
        )

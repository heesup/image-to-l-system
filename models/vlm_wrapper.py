import json
import re
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Any, Optional, List
from dataset.lsystem import LSystem, LSystemTokenizer

def get_device() -> torch.device:
    """Return Apple Silicon MPS device if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generate lower-triangular causal mask for autoregressive Transformer decoder."""
    mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    return mask

class PureLSystemVLM(nn.Module):
    """100% Pure Autoregressive Vision-Language Model (VLM).
    Predicts axiom, rules, angle, iterations, step_size, line_width end-to-end as text tokens.
    No classification heads, no regression heads.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 256
    ):
        super().__init__()
        self.tokenizer = LSystemTokenizer()
        vocab_size = self.tokenizer.vocab_size

        # Vision Encoder (ResNet-18 feature extractor)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # Output: (B, 512, 8, 8)
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        
        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 + max_seq_len, embed_dim))
        
        # Token Embedding for Decoder
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Pure Language Modeling Head
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image tensor into sequence of visual tokens (B, 64, embed_dim)."""
        feats = self.encoder(images)             # (B, 512, 8, 8)
        feats = self.proj(feats)                 # (B, embed_dim, 8, 8)
        feats = feats.flatten(2).transpose(1, 2) # (B, 64, embed_dim)
        return feats

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass computing Language Modeling logits over target text tokens."""
        img_feats = self.encode_image(images)
        tgt_embed = self.token_embed(input_ids)
        
        batch_sz, tgt_len = input_ids.size()

        if tgt_mask is None and tgt_len > 1:
            tgt_mask = generate_causal_mask(tgt_len, images.device)

        pos_img = self.pos_embed[:, :64, :]
        pos_tgt = self.pos_embed[:, 64:64+tgt_len, :]

        memory = img_feats + pos_img
        tgt = tgt_embed + pos_tgt

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        logits = self.lm_head(output)
        return logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int = 160, repetition_penalty: float = 1.5) -> List[str]:
        """Autoregressively generate JSON text tokens with repetition penalty."""
        self.eval()
        batch_sz = images.size(0)
        device = images.device

        input_ids = torch.full((batch_sz, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        finished = [False] * batch_sz
        decoded_strings = [""] * batch_sz

        for step in range(max_len):
            logits = self.forward(images, input_ids)
            next_logits = logits[:, -1, :].clone()  # (B, vocab_size)

            # Apply Repetition Penalty to avoid looping tokens (e.g. "FFFFFFFF")
            for i in range(batch_sz):
                for token_id in set(input_ids[i].tolist()):
                    if token_id in (self.tokenizer.bos_id, self.tokenizer.pad_id):
                        continue
                    if next_logits[i, token_id] > 0:
                        next_logits[i, token_id] /= repetition_penalty
                    else:
                        next_logits[i, token_id] *= repetition_penalty

            # Greedy sampling
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

            input_ids = torch.cat([input_ids, next_tokens], dim=1)

            for i in range(batch_sz):
                if not finished[i]:
                    token_id = next_tokens[i].item()
                    if token_id in (self.tokenizer.eos_id, self.tokenizer.pad_id):
                        finished[i] = True
                    else:
                        char = self.tokenizer.id_to_char.get(token_id, "")
                        decoded_strings[i] += char

            if all(finished):
                break

        return decoded_strings


    def predict_lsystem(self, image: torch.Tensor) -> LSystem:
        """Predict LSystem specification object by parsing generated JSON text tokens."""
        json_texts = self.generate(image, max_len=160)
        pred_json = json_texts[0]

        # Try parsing JSON directly
        try:
            data = json.loads(pred_json)
            return LSystem.from_dict(data)
        except Exception:
            # Robust Regex Parsing fallback for numbers and string fields
            angle_match = re.search(r'"angle":\s*([0-9.]+)', pred_json)
            iter_match = re.search(r'"iterations":\s*([0-9]+)', pred_json)

            angle = float(angle_match.group(1)) if angle_match else 25.0
            iterations = int(iter_match.group(1)) if iter_match else 3

            return LSystem(
                axiom="X",
                rules={"X": "F[+X][-X]FX", "F": "FF"},
                angle=max(10.0, min(90.0, angle)),
                iterations=max(2, min(5, iterations)),
                step_size=1.0,
                line_width=2.0
            )

class LSystemVLM:
    """Wrapper class for PureLSystemVLM on Apple Silicon MPS."""

    def __init__(self, model_name: str = "standalone", device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = PureLSystemVLM().to(self.device)

    def to(self, device: torch.device):
        self.device = device
        self.model = self.model.to(device)
        return self

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Any, Optional

def get_device() -> torch.device:
    """Return Apple Silicon MPS device if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class StandaloneLSystemModel(nn.Module):
    """Lightweight PyTorch Vision-Transformer/ResNet Encoder + Transformer Decoder model.
    Serves as an efficient standalone VLM architecture for Apple Silicon MPS.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 256
    ):
        super().__init__()
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
        
        # Output Heads
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Angle, Iterations, Step Size, Line Width
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image tensor into sequence of visual tokens (B, 64, embed_dim)."""
        feats = self.encoder(images)           # (B, 512, 8, 8)
        feats = self.proj(feats)               # (B, embed_dim, 8, 8)
        feats = feats.flatten(2).transpose(1, 2) # (B, 64, embed_dim)
        return feats

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        img_feats = self.encode_image(images)
        tgt_embed = self.token_embed(input_ids)
        
        # Combine image feats and token embeddings
        seq = torch.cat([img_feats, tgt_embed], dim=1) + self.pos_embed[:, :64 + input_ids.size(1), :]
        
        memory = seq[:, :64, :]
        tgt = seq[:, 64:, :]

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        logits = self.lm_head(output)
        params = self.param_head(output[:, -1, :])

        return {"logits": logits, "pred_params": params}

class LSystemVLM:
    """Wrapper class supporting Hugging Face VLM (SmolVLM / PaliGemma + LoRA) and standalone PyTorch model."""

    def __init__(self, model_name: str = "standalone", device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.is_hf = False

        if model_name == "standalone":
            self.model = StandaloneLSystemModel().to(self.device)
        else:
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                from peft import get_peft_model, LoraConfig, TaskType

                self.processor = AutoProcessor.from_pretrained(model_name)
                base_model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float32)

                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(base_model, peft_config).to(self.device)
                self.is_hf = True
            except Exception as e:
                print(f"[Warning] Failed to load HF model '{model_name}': {e}. Falling back to StandaloneLSystemModel.")
                self.model = StandaloneLSystemModel().to(self.device)
                self.model_name = "standalone"

    def to(self, device: torch.device):
        self.device = device
        self.model = self.model.to(device)
        return self

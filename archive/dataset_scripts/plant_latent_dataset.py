"""
Paired Dataset for Latent Flow Matching: (RGB Image, 512D Target Plant Latents).

Loads paired rendered images and Helios XMLs, encodable via the pretrained PlantOrganVAE
into target continuous latent matrices Z_1 in R^{N x 512}.
"""

import os
import glob
import re
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.plant_vae import PlantOrganVAE


class PlantLatentDataset(Dataset):
    """
    Loads paired (Rendered RGB Image, Plant Organ Latent Matrix Z in R^{N x 512}).
    """

    def __init__(
        self,
        data_root: str = "dataset/helios_data",
        image_size: int = 128,
        max_organs: int = 1600,
        vae_model: Optional[PlantOrganVAE] = None,
        vae_ckpt: str = "diffusion_based/checkpoints/plant_organ_vae_best.pt",
        device: str = "cpu",
        max_samples: Optional[int] = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.image_size = image_size
        self.max_organs = max_organs
        self.max_samples = max_samples
        self.device = torch.device(device)

        # Image transform (RGB normalized to [-1, 1])
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Load / prepare VAE encoder
        if vae_model is not None:
            self.vae = vae_model
        else:
            self.vae = PlantOrganVAE(latent_dim=512, hidden_dim=512).to(self.device)
            if os.path.exists(vae_ckpt):
                ckpt = torch.load(vae_ckpt, map_location=self.device)
                self.vae.load_state_dict(ckpt["model_state_dict"])
                self.vae.eval()

        # Discover pairs (image + XML)
        self.samples = self._discover_pairs()
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _discover_pairs(self) -> List[Dict[str, Any]]:
        pairs = []
        xml_files = sorted(glob.glob(os.path.join(self.data_root, "**", "*.xml"), recursive=True))
        if self.max_samples is not None and len(xml_files) > self.max_samples:
            xml_files = xml_files[:self.max_samples]

        for xml_path in xml_files:
            xml_dir = os.path.dirname(xml_path)
            xml_base = os.path.splitext(os.path.basename(xml_path))[0]

            img_path = None
            candidate_names = [
                xml_path.replace(".xml", ".jpeg"),
                xml_path.replace(".xml", ".png"),
                xml_path.replace(".xml", ".jpg"),
            ]
            
            m = re.match(r"(rad_dap\d+_\d+)", xml_base)
            if m:
                prefix = m.group(1)
                candidate_names.extend([
                    os.path.join(xml_dir, f"{prefix}_rad.jpeg"),
                    os.path.join(xml_dir, f"{prefix}_rad.png"),
                    os.path.join(xml_dir, f"{prefix}_RGB.png"),
                    os.path.join(xml_dir, f"{prefix}.png"),
                ])

            for c in candidate_names:
                if os.path.exists(c):
                    img_path = c
                    break

            dap = 50.0
            dap_m = re.search(r"dap0*(\d+)", xml_path)
            if dap_m:
                dap = float(dap_m.group(1))

            pairs.append({
                "xml_path": xml_path,
                "img_path": img_path,
                "dap": dap,
            })

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # 1. Load Image
        if item["img_path"] is not None and os.path.exists(item["img_path"]):
            try:
                img = Image.open(item["img_path"]).convert("RGB")
                img_tensor = self.img_transform(img)
            except Exception:
                img_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        else:
            img_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)

        # 2. Parse XML into 40D Organ Array
        arr = PlantOrganArray.from_xml_file(item["xml_path"])
        X = arr.tensor  # (N_organs, 40)
        N = X.shape[0]

        # 3. Encode into 512D Latent Space via Pretrained VAE
        with torch.no_grad():
            mu, _ = self.vae.encode(X.to(self.device))
            z = mu.cpu()  # (N_organs, 512)

        # 4. Pad / Clip to max_organs
        actual_n = min(N, self.max_organs)
        padded_z = torch.zeros((self.max_organs, 512), dtype=torch.float32)
        padded_z[:actual_n] = z[:actual_n]

        mask = torch.zeros(self.max_organs, dtype=torch.bool)
        mask[:actual_n] = True

        return {
            "image": img_tensor,
            "latents": padded_z,          # (max_organs, 512)
            "existence_mask": mask,       # (max_organs,)
            "num_organs": actual_n,
            "dap": item["dap"],
            "xml_path": item["xml_path"],
        }


def collate_latent_flow_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Trims batch sequence length to the maximum active organ count in the mini-batch for optimal throughput.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    latents = torch.stack([b["latents"] for b in batch], dim=0)
    masks = torch.stack([b["existence_mask"] for b in batch], dim=0)
    num_organs = torch.tensor([b["num_organs"] for b in batch], dtype=torch.long)
    daps = torch.tensor([b["dap"] for b in batch], dtype=torch.float32)

    max_n = max(int(num_organs.max().item()), 1)
    trimmed_latents = latents[:, :max_n]
    trimmed_masks = masks[:, :max_n]

    return {
        "images": images,
        "latents": trimmed_latents,
        "masks": trimmed_masks,
        "num_organs": num_organs,
        "daps": daps,
    }

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Callable, Dict, Any, List

class LSystemDataset(Dataset):
    """PyTorch Dataset for Image-to-L-System VLM training and evaluation."""

    PROMPT = "Estimate the L-system specification (axiom, rules, angle, iterations, step_size, line_width) for this plant image."

    def __init__(self, data_dir: str, transform: Optional[Callable] = None, processor: Optional[Any] = None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.processor = processor

        index_path = os.path.join(data_dir, "index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.items = json.load(f)
        else:
            self.items = []
            annotations_dir = os.path.join(data_dir, "annotations")
            if os.path.exists(annotations_dir):
                for fname in sorted(os.listdir(annotations_dir)):
                    if fname.endswith(".json"):
                        with open(os.path.join(annotations_dir, fname), "r") as f:
                            self.items.append(json.load(f))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        target_text = json.dumps(item["lsystem"], separators=(',', ':'))

        if self.processor is not None:
            # Hugging Face VLM processor formatting
            inputs = self.processor(
                text=f"<image>\n{self.PROMPT}",
                images=image,
                return_tensors="pt"
            )
            inputs["labels"] = target_text
            inputs["item"] = item
            return inputs
        else:
            # Standard tensor fallback
            image_tensor = self.transform(image)
            return {
                "image": image_tensor,
                "prompt": self.PROMPT,
                "target_text": target_text,
                "lsystem_dict": item["lsystem"],
                "image_path": image_path
            }

def create_dataloaders(data_dir: str, batch_size: int = 8, shuffle: bool = True, processor: Optional[Any] = None) -> DataLoader:
    dataset = LSystemDataset(data_dir, processor=processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

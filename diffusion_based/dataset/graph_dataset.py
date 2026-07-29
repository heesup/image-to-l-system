import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, List, Optional
from dataset.generator import LSystemDatasetGenerator
from dataset.graph_extractor import PlantGraphExtractor

class PlantGraphDataset(Dataset):
    """PyTorch Dataset returning paired target plant images and ground-truth 2D plant graphs (V, A, e)."""

    def __init__(self, data_dir: Optional[str] = None, num_synthetic_samples: int = 100, max_nodes: int = 64, image_size: int = 256):
        self.image_size = image_size
        self.max_nodes = max_nodes
        self.extractor = PlantGraphExtractor(image_size=(image_size, image_size), max_nodes=max_nodes)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples: List[Dict[str, Any]] = []
        generator = LSystemDatasetGenerator(seed=42)
        dataset_items = generator.generate_dataset(num_samples=num_synthetic_samples)

        for item in dataset_items:
            img = item["image"]
            lsys = item["lsystem"]
            graph_data = self.extractor.extract_graph(lsys)
            self.samples.append({
                "image": img,
                "nodes": graph_data["nodes"],              # (max_nodes, 5)
                "adj_matrix": graph_data["adj_matrix"],    # (max_nodes, max_nodes)
                "parent_indices": graph_data["parent_indices"], # (max_nodes,)
                "existence_mask": graph_data["existence_mask"], # (max_nodes,)
                "num_nodes": graph_data["num_nodes"]
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_tensor = self.transform(sample["image"])
        return {
            "image": img_tensor,
            "nodes": sample["nodes"],
            "adj_matrix": sample["adj_matrix"],
            "parent_indices": sample["parent_indices"],
            "existence_mask": sample["existence_mask"],
            "num_nodes": torch.tensor(sample["num_nodes"], dtype=torch.long)
        }

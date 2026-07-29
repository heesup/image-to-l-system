import os
import json
import random
from typing import List, Dict, Any, Tuple
from .lsystem import LSystem
from .renderer import TurtleRenderer

PRESET_GRAMMARS = [
    # 1. Classic Binary Plant
    {"axiom": "X", "rules": {"X": "F[+X][-X]FX", "F": "FF"}},
    # 2. Monopodial Bush
    {"axiom": "X", "rules": {"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"}},
    # 3. Ternary Branching Tree
    {"axiom": "X", "rules": {"X": "F[+X][X][-X]", "F": "FF"}},
    # 4. Asymmetric Fern / Weed
    {"axiom": "X", "rules": {"X": "F[+X]F[-X]+X", "F": "FF"}},
    # 5. Curly Branching Plant
    {"axiom": "X", "rules": {"X": "F-[+X][+X]-F-X", "F": "FF"}},
    # 6. Dragon / Vine Curve
    {"axiom": "FX", "rules": {"X": "X+YF+", "Y": "-FX-Y"}},
    # 7. Simple Branch
    {"axiom": "F", "rules": {"F": "F[+F]F[-F]F"}},
]

COLOR_PALETTES = [
    "forestgreen",
    "darkgreen",
    "seagreen",
    "olivedrab",
    "darkslategray",
    "black"
]

class LSystemDatasetGenerator:
    """Generates synthetic dataset of L-System plant images and target annotations."""

    def __init__(self, image_size: Tuple[int, int] = (256, 256), seed: int = 42):
        self.image_size = image_size
        self.seed = seed
        random.seed(seed)

    def sample_lsystem(self) -> LSystem:
        """Sample a deterministic L-System specification with random parameters."""
        preset = random.choice(PRESET_GRAMMARS)
        angle = round(random.uniform(15.0, 36.0), 1)
        iterations = random.randint(2, 4)
        step_size = round(random.uniform(0.8, 1.5), 2)
        line_width = random.choice([1.0, 2.0, 3.0])

        return LSystem(
            axiom=preset["axiom"],
            rules=preset["rules"],
            angle=angle,
            iterations=iterations,
            step_size=step_size,
            line_width=line_width
        )

    def generate_dataset(self, num_samples: int, output_dir: str) -> List[Dict[str, Any]]:
        """Generate num_samples plant images and annotations into output_dir."""
        images_dir = os.path.join(output_dir, "images")
        annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        metadata = []

        for i in range(num_samples):
            sample_id = f"plant_{i:05d}"
            fg_color = random.choice(COLOR_PALETTES)
            renderer = TurtleRenderer(image_size=self.image_size, fg_color=fg_color)

            lsystem = self.sample_lsystem()
            img = renderer.render(lsystem)

            img_path = os.path.join(images_dir, f"{sample_id}.png")
            json_path = os.path.join(annotations_dir, f"{sample_id}.json")

            img.save(img_path)
            
            annotation_data = {
                "id": sample_id,
                "image_path": img_path,
                "lsystem": lsystem.to_dict(),
                "json_str": lsystem.to_json()
            }

            with open(json_path, "w") as f:
                json.dump(annotation_data, f, indent=2)

            metadata.append(annotation_data)

        index_path = os.path.join(output_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

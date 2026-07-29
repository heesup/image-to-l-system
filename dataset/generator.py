import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
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
    # 8. Pentaplex Star (72 deg angle)
    {"axiom": "F++F++F++F++F", "rules": {"F": "F++F++F+++++F-F++F"}},
    # 9. Hilbert Curve (90 deg angle)
    {"axiom": "A", "rules": {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"}},
    # 10. Sierpinski Triangle (60 deg angle)
    {"axiom": "F-G-G", "rules": {"F": "F-G+F+G-F", "G": "GG"}},
    # 11. Koch Curve (90 deg angle)
    {"axiom": "F", "rules": {"F": "F+F-F-F+F"}}
]


COLOR_PALETTES = [
    "forestgreen",
    "darkgreen",
    "seagreen",
    "olivedrab",
    "darkslategray",
    "black"
]

def generate_random_rule(max_len: int = 12) -> str:
    """Dynamically generate a random syntactically balanced L-System production rule string."""
    tokens = ["F", "F", "F", "+", "-", "X"]
    seq = ["F"]
    open_brackets = 0

    for _ in range(random.randint(4, max_len)):
        t = random.choice(tokens + (["["] if open_brackets < 2 else []) + (["]"] if open_brackets > 0 else []))
        if t == '[':
            open_brackets += 1
            seq.append("[")
            seq.append(random.choice(["+X", "-X", "+F", "-F"]))
        elif t == ']':
            open_brackets -= 1
            seq.append("]")
        else:
            seq.append(t)

    # Close any unclosed brackets
    seq.append("]" * open_brackets)
    rule_str = "".join(seq)

    # Ensure brackets are balanced
    if not LSystem.validate_brackets(rule_str):
        return "F[+X][-X]FX"
    return rule_str

class LSystemDatasetGenerator:
    """Generates synthetic dataset of L-System plant images and target annotations."""

    def __init__(self, image_size: Tuple[int, int] = (256, 256), seed: int = 42):
        self.image_size = image_size
        self.seed = seed
        random.seed(seed)

    def sample_lsystem(self) -> LSystem:
        """Sample an L-System specification from presets (50%) or random grammar synthesis (50%)."""
        if random.random() < 0.5:
            preset = random.choice(PRESET_GRAMMARS)
            axiom = preset["axiom"]
            rules = preset["rules"]
        else:
            axiom = "X"
            rule_x = generate_random_rule()
            rules = {"X": rule_x, "F": "FF"}

        angle = round(random.uniform(12.0, 60.0), 1)
        iterations = random.randint(2, 4)
        step_size = round(random.uniform(0.6, 1.8), 2)
        line_width = random.choice([1.0, 2.0, 3.0])

        return LSystem(
            axiom=axiom,
            rules=rules,
            angle=angle,
            iterations=iterations,
            step_size=step_size,
            line_width=line_width
        )


    def generate_dataset(self, num_samples: int, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate num_samples plant images and annotations into output_dir (or in-memory if output_dir is None)."""
        if output_dir:
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

            annotation_data = {
                "id": sample_id,
                "image": img,
                "lsystem": lsystem,
                "lsystem_dict": lsystem.to_dict(),
                "json_str": lsystem.to_json()
            }

            if output_dir:
                img_path = os.path.join(images_dir, f"{sample_id}.png")
                json_path = os.path.join(annotations_dir, f"{sample_id}.json")
                img.save(img_path)
                annotation_data["image_path"] = img_path
                with open(json_path, "w") as f:
                    json.dump(annotation_data["lsystem_dict"], f, indent=2)

            metadata.append(annotation_data)

        if output_dir:
            index_path = os.path.join(output_dir, "index.json")
            with open(index_path, "w") as f:
                json.dump([m["lsystem_dict"] for m in metadata], f, indent=2)
            print(f"Successfully generated {num_samples} L-System plant samples in '{output_dir}'.")

        return metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic L-system plant dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of plant samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory path")
    args = parser.parse_args()

    gen = LSystemDatasetGenerator(seed=42)
    gen.generate_dataset(num_samples=args.num_samples, output_dir=args.output_dir)


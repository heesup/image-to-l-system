import os
import random
import json
import matplotlib.pyplot as plt
from PIL import Image
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
from infer import predict_image

# Unseen, Out-of-Distribution (OOD) Grammars
OOD_GRAMMARS = [
    # 1. Hilbert Curve / Square Vine (Unseen rule)
    {"axiom": "A", "rules": {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"}, "angle": 90.0, "iterations": 3},
    # 2. Pentaplex / Star Plant (5-fold symmetry angle 72 deg)
    {"axiom": "F++F++F++F++F", "rules": {"F": "F++F++F+++++F-F++F"}, "angle": 72.0, "iterations": 2},
    # 3. Dense Willow Branch (High angle 45 deg)
    {"axiom": "X", "rules": {"X": "F[+X][-X][++X][--X]F", "F": "FF"}, "angle": 45.0, "iterations": 3},
    # 4. Asymmetric Tall Reed (Steep angle 15 deg)
    {"axiom": "X", "rules": {"X": "FF+[+X]-[-X]+X", "F": "F"}, "angle": 15.0, "iterations": 4},
    # 5. Hexagonal Crystal Tree (60 deg angle)
    {"axiom": "X", "rules": {"X": "F[+X]F[-X]+X", "F": "F+F--F+F"}, "angle": 60.0, "iterations": 3}
]

def test_ood_plants(output_dir: str = "data/ood_test", plots_dir: str = "plots/ood"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"=== Generating {len(OOD_GRAMMARS)} Out-of-Distribution (OOD) Test Plants ===")

    for i, spec in enumerate(OOD_GRAMMARS):
        lsystem = LSystem(
            axiom=spec["axiom"],
            rules=spec["rules"],
            angle=spec["angle"],
            iterations=spec["iterations"],
            step_size=1.0,
            line_width=2.0
        )
        renderer = TurtleRenderer(image_size=(256, 256), fg_color="darkgreen")
        img = renderer.render(lsystem)

        img_path = os.path.join(images_dir, f"ood_plant_{i:02d}.png")
        img.save(img_path)

        plot_path = os.path.join(plots_dir, f"ood_comparison_{i:02d}.png")

        print(f"\n--- Testing OOD Sample {i+1}/{len(OOD_GRAMMARS)}: '{img_path}' ---")
        predict_image(
            image_path=img_path,
            checkpoint_path="checkpoints/rl_model.pt",
            output_plot=plot_path
        )

    print(f"\n=== OOD Testing Complete! Comparison plots saved to '{plots_dir}/' ===")

if __name__ == "__main__":
    test_ood_plants()

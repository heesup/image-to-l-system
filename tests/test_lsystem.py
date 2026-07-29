import pytest
import torch
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
from dataset.generator import LSystemDatasetGenerator
from models.vlm_wrapper import StandaloneLSystemModel, get_device
from training.rewards import calculate_mask_iou

def test_lsystem_expansion():
    # Test Koch snowflake rule expansion
    lsys = LSystem(axiom="F", rules={"F": "F+F-F-F+F"}, iterations=1)
    expanded = lsys.expand()
    assert expanded == "F+F-F-F+F"

    lsys2 = LSystem(axiom="F", rules={"F": "F+F-F-F+F"}, iterations=2)
    expanded2 = lsys2.expand()
    assert len(expanded2) == 25

def test_bracket_validation():
    assert LSystem.validate_brackets("F[+F][-F]") is True
    assert LSystem.validate_brackets("F[+F][-F") is False
    assert LSystem.validate_brackets("F+F]") is False
    assert LSystem.validate_brackets("F[[+F]--F]") is True

def test_renderer_output():
    lsys = LSystem(axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0, iterations=2)
    renderer = TurtleRenderer(image_size=(128, 128))
    img = renderer.render(lsys)
    assert img.size == (128, 128)
    assert img.mode == "RGB"

def test_dataset_generator(tmp_path):
    gen = LSystemDatasetGenerator(seed=123)
    out_dir = str(tmp_path / "test_data")
    items = gen.generate_dataset(num_samples=3, output_dir=out_dir)
    assert len(items) == 3
    assert items[0]["lsystem"]["axiom"] in ["X", "FX", "F"]

def test_standalone_model_forward():
    device = get_device()
    model = StandaloneLSystemModel().to(device)
    dummy_img = torch.randn(2, 3, 256, 256, device=device)
    dummy_input_ids = torch.randint(1, 30, (2, 16), device=device)
    
    out = model(dummy_img, dummy_input_ids)
    assert "logits" in out
    assert "pred_params" in out
    assert out["logits"].shape == (2, 16, 128)
    assert out["pred_params"].shape == (2, 4)

def test_mask_iou_identity():
    lsys = LSystem(axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0, iterations=2)
    renderer = TurtleRenderer(image_size=(128, 128))
    img1 = renderer.render(lsys)
    img2 = renderer.render(lsys)
    iou = calculate_mask_iou(img1, img2)
    assert iou == pytest.approx(1.0, rel=1e-3)

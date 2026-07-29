import unittest
import torch
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
from dataset.generator import LSystemDatasetGenerator
from models.vlm_wrapper import StandaloneLSystemModel, get_device
from training.rewards import calculate_mask_iou

class TestLSystemPipeline(unittest.TestCase):

    def test_lsystem_expansion(self):
        # Test Koch snowflake rule expansion
        lsys = LSystem(axiom="F", rules={"F": "F+F-F-F+F"}, iterations=1)
        expanded = lsys.expand()
        self.assertEqual(expanded, "F+F-F-F+F")

        lsys2 = LSystem(axiom="F", rules={"F": "F+F-F-F+F"}, iterations=2)
        expanded2 = lsys2.expand()
        self.assertEqual(len(expanded2), 49)


    def test_bracket_validation(self):
        self.assertTrue(LSystem.validate_brackets("F[+F][-F]"))
        self.assertFalse(LSystem.validate_brackets("F[+F][-F"))
        self.assertFalse(LSystem.validate_brackets("F+F]"))
        self.assertTrue(LSystem.validate_brackets("F[[+F]--F]"))

    def test_renderer_output(self):
        lsys = LSystem(axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0, iterations=2)
        renderer = TurtleRenderer(image_size=(128, 128))
        img = renderer.render(lsys)
        self.assertEqual(img.size, (128, 128))
        self.assertEqual(img.mode, "RGB")

    def test_dataset_generator(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            gen = LSystemDatasetGenerator(seed=123)
            items = gen.generate_dataset(num_samples=3, output_dir=tmp_dir)
            self.assertEqual(len(items), 3)
            self.assertIn(items[0]["lsystem"]["axiom"], ["X", "FX", "F"])

    def test_standalone_model_forward(self):
        device = get_device()
        model = StandaloneLSystemModel().to(device)
        dummy_img = torch.randn(2, 3, 256, 256, device=device)
        dummy_input_ids = torch.randint(1, 30, (2, 16), device=device)
        
        out = model(dummy_img, dummy_input_ids)
        self.assertIn("logits", out)
        self.assertIn("pred_params", out)
        self.assertEqual(out["logits"].shape, (2, 16, 128))
        self.assertEqual(out["pred_params"].shape, (2, 4))

    def test_mask_iou_identity(self):
        lsys = LSystem(axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0, iterations=2)
        renderer = TurtleRenderer(image_size=(128, 128))
        img1 = renderer.render(lsys)
        img2 = renderer.render(lsys)
        iou = calculate_mask_iou(img1, img2)
        self.assertAlmostEqual(iou, 1.0, places=3)

if __name__ == "__main__":
    unittest.main()

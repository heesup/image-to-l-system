import unittest
import torch
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
from dataset.generator import LSystemDatasetGenerator
from dataset.graph_extractor import PlantGraphExtractor
from lm_based.models.vlm_wrapper import PureLSystemVLM, get_device
from lm_based.training.rewards import calculate_mask_iou
from diffusion_based.models.graph_diffuser import PlantGraphDiffuser

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
            self.assertIn(items[0]["lsystem"].axiom, ["X", "FX", "F"])

    def test_vlm_model_forward(self):
        device = get_device()
        model = PureLSystemVLM().to(device)
        dummy_img = torch.randn(2, 3, 256, 256, device=device)
        dummy_input_ids = torch.randint(1, 10, (2, 16), device=device)
        
        logits = model(dummy_img, dummy_input_ids)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 16)

    def test_graph_extractor(self):
        lsys = LSystem(axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0, iterations=2)
        extractor = PlantGraphExtractor(image_size=(128, 128), max_nodes=32)
        graph = extractor.extract_graph(lsys)
        self.assertIn("nodes", graph)
        self.assertIn("adj_matrix", graph)
        self.assertEqual(graph["nodes"].shape, (32, 5))
        self.assertEqual(graph["adj_matrix"].shape, (32, 32))

    def test_graph_diffuser_forward(self):
        device = get_device()
        model = PlantGraphDiffuser(max_nodes=32).to(device)
        dummy_nodes = torch.randn(2, 32, 5, device=device)
        dummy_existence = torch.ones(2, 32, 1, device=device)
        timesteps = torch.tensor([100, 200], device=device).long()
        dummy_images = torch.randn(2, 3, 256, 256, device=device)

        out = model(dummy_nodes, dummy_existence, timesteps, dummy_images)
        self.assertEqual(out["pred_node_noise"].shape, (2, 32, 5))
        self.assertEqual(out["pred_existence_logits"].shape, (2, 32))
        self.assertEqual(out["pred_adj_logits"].shape, (2, 32, 32))

    def test_mask_iou_identity(self):
        lsys = LSystem(axiom="X", rules={"X": "F[+X][-X]FX", "F": "FF"}, angle=25.0, iterations=2)
        renderer = TurtleRenderer(image_size=(128, 128))
        img1 = renderer.render(lsys)
        img2 = renderer.render(lsys)
        iou = calculate_mask_iou(img1, img2)
        self.assertAlmostEqual(iou, 1.0, places=3)

if __name__ == "__main__":
    unittest.main()

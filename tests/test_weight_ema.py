"""WeightEMA: EMA of parameters after each optimizer step (registered as a post-step hook), buffers copied,
state dict has the model's layout so the EMA file loads into the same model class."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.training.train_hierarchical_flow_matching import WeightEMA


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)
        self.register_buffer("mu", torch.zeros(4))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        return self.lin(x - self.mu)


class TestWeightEMA(unittest.TestCase):
    def test_update_matches_formula_and_buffers_follow(self):
        torch.manual_seed(0)
        net = _Net(); ema = WeightEMA(net, 0.9)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        opt.register_step_post_hook(lambda o, a, k: ema.update(net))
        w0 = net.lin.weight.detach().clone()
        loss = net(torch.randn(3, 4)).pow(2).mean(); loss.backward(); opt.step()
        w1 = net.lin.weight.detach().clone()
        self.assertTrue(torch.allclose(ema.module.lin.weight, 0.9 * w0 + 0.1 * w1, atol=1e-6))
        net.mu.fill_(2.0); net.count.fill_(7); ema.update(net)
        self.assertTrue(torch.allclose(ema.module.mu, torch.full((4,), 2.0))); self.assertEqual(int(ema.module.count), 7)
        self.assertFalse(any(p.requires_grad for p in ema.module.parameters()))

    def test_state_dict_round_trip(self):
        net = _Net(); ema = WeightEMA(net, 0.99)
        fresh = _Net(); fresh.load_state_dict(ema.state_dict())
        self.assertEqual(set(fresh.state_dict().keys()), set(net.state_dict().keys()))


if __name__ == "__main__":
    unittest.main()

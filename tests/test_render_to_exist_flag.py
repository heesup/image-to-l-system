"""render_to_exist / render_to_latent CLI+launcher wiring: both flags accepted, default off, no crash importing."""
import os
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestRenderToExistFlag(unittest.TestCase):
    def test_help_lists_both_flags(self):
        out = subprocess.run([sys.executable, os.path.join(REPO, "plant_recon/training/train_hierarchical_flow_matching.py"), "--help"],
                             capture_output=True, text=True, timeout=60)
        self.assertIn("--render_to_exist", out.stdout)
        self.assertIn("--render_to_latent", out.stdout)

    def test_launcher_accepts_render_to_exist_env(self):
        script = os.path.join(REPO, "slurm_scripts/train_hierarchical_flow_matching.sh")
        r = subprocess.run(["bash", "-n", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("RENDER_TO_EXIST", open(script).read())


if __name__ == "__main__":
    unittest.main()

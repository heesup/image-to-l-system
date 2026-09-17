"""Helios-procedural cold start: build a "typical plant of this age" from a DAP estimate ALONE
(no image conditioning at all) and hand it to the same refinement loop Approach 2 already runs.

Why this exists. Approach 1 (real_world/eval/run_approach1_cold.py) starts refinement from the
trained network's own image-conditioned sample. On real crops that sample is often very sparse
(measured 2026-09-16 on 20 AgML crops: 14 of 20 plants came back with a single live phytomer),
which leaves the differentiable renderer optimizing from a starting point that barely resembles a
plant. The alternative tested here: skip the network for the initial guess and ask the SAME Helios
PlantArchitecture generator that produced the synthetic training set for a procedurally grown
cowpea at the estimated DAP -- no image information, but guaranteed on-manifold plant topology,
organ counts and proportions for that age -- then let refinement fit it to the real photo. The two
cold starts are directly comparable because both end up as the identical
(pos, rot, scale, latent, exist, parent) state that refine_one() optimizes.

Cost: the generator runs a real day-stepped growth simulation, ~20 s per plant at DAP 60 even with
--renderer none (rendering skipped, XML only), so generated plants are cached on disk by
(species, DAP, seed) and reused.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_HELIOS_ROOT = REPO_ROOT / "submodules/Digital-Crops" / "projects" / "syntheticdata_generation"
BUILD_DIR = _HELIOS_ROOT / "build"
MAIN_BIN = BUILD_DIR / "main"
DEFAULT_CACHE = REPO_ROOT / "use_cases" / "real_world" / "data" / "helios_procedural"

# The packet slot layout must match the VAE whose latents the pipeline consumes: the v9 packet
# cache this project's checkpoints train against was built with terminal_leaflet_last=True, and
# build_phytomer_packets reads that as a process-level env var rather than an argument.
os.environ.setdefault("PHYTOMER_TERMINAL_LAST", "1")


def generate_helios_xml(dap: int, seed: int = 0, species: str = "cowpea",
                         cache_dir: Optional[Path] = None, genotype: str = "") -> Path:
    """Procedurally grows one plant of age `dap` days and returns its structure XML path.

    Delegates to `plant_recon/dataset/generate_helios_dataset.render_one`, the same function that produced this
    project's synthetic training set, with `--renderer none` so only the structure XML is written.
    That function already handles the binary's working-directory requirement, the isolated temp dir,
    the output-file moves and the naming convention, and it skips work when the sample is already on
    disk — so a repeated (species, dap, seed) costs nothing instead of re-running the ~20 s growth
    simulation.
    """
    if str(REPO_ROOT) not in sys.path:      # plant_recon imports as a package at the repo root
        sys.path.insert(0, str(REPO_ROOT))
    from plant_recon.dataset.generate_helios_dataset import render_one, _sample_name

    cache_dir = Path(cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = _HELIOS_ROOT / "configs" / f"params_{species}.json"
    name = _sample_name(species, int(dap), int(seed), genotype)
    xml_path = cache_dir / species / f"{name}_0000_plant_0000.xml"
    r = render_one((species, genotype, int(dap), int(seed), str(cache_dir), str(cfg), 0, False, "none"))
    if not xml_path.exists():
        raise RuntimeError(f"Helios generation produced no XML for dap={dap} seed={seed}: {r}")
    return xml_path


def helios_state_from_xml(xml_path: Path, pvae, device: torch.device, M: int = 10):
    """XML -> the same (pos, rot, scale, latent, exist, parent_pos, parent_idx) state that
    run_approach1_cold.sample_cold() returns, so either can seed refine_one() unchanged.

    The conversion chain is the training set's own (plant_recon/dataset/generate_cache.py,
    `--mode pkt`): XML -> 14D part tensor -> 26D FM nodes -> 10-slot phytomer packets; then the
    phytomer scale s_a and the PhytomerVAE latent are read off those packets exactly as
    eval_gt_substitution_ablation.py reads them for ground-truth substitution.
    """
    from plant_recon.models.plant_organ_array import PlantOrganArray, P_COL_ORGAN_TYPE, ORGAN_NONE
    from plant_recon.dataset.part_array_dataset import encode_fm, attach_parent_links
    from plant_recon.dataset.phytomer_packets import phytomer_scale
    from plant_recon.dataset.generate_cache import build_pkt_targets, extract_phytomer_ids

    arr = PlantOrganArray.from_xml_file(str(xml_path))
    part = arr.to_part_tensor()
    nodes = encode_fm(part)
    existence = (part[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
    ids = extract_phytomer_ids(arr)
    # build_pkt_targets is the training cache's own packet builder (generate_cache.py --mode pkt):
    # it applies the canonical packet conventions and encodes the frozen-VAE latent the same way the
    # cached targets were built, so an XML converted here is indistinguishable from a cached sample.
    pkt = build_pkt_targets(nodes, existence, ids, vae=pvae, device=device)
    if pkt is None:
        raise RuntimeError(f"no phytomer packets built from {xml_path}")
    attach_parent_links(pkt)

    packets_d = pkt["packets"].to(device).float()
    pos = pkt["centers"].to(device).float()
    rot = pkt["refs"].to(device).float()
    scale = phytomer_scale(packets_d)
    lat = pkt["latent"].to(device).float()
    exist = torch.ones(pos.shape[0], device=device)
    parent_pos = torch.nan_to_num(pkt["parent_pos"].to(device).float())
    parent_idx = pkt["parent_idx"].to(device).long()
    return pos, rot, scale, lat, exist, parent_pos, parent_idx


def helios_cold_start(dap: int, pvae, device: torch.device, M: int = 10, seed: int = 0,
                       species: str = "cowpea", cache_dir: Optional[Path] = None
                       ) -> Tuple[torch.Tensor, float, tuple]:
    """Drop-in counterpart of run_approach1_cold.sample_cold(): returns (parts, dap, state)."""
    from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes
    xml_path = generate_helios_xml(dap, seed=seed, species=species, cache_dir=cache_dir)
    pos, rot, scale, lat, exist, par, parent_idx = helios_state_from_xml(xml_path, pvae, device, M)
    with torch.no_grad():
        parts = plant_from_nodes(pvae, pos, rot, scale, lat, exist, par, M)
    return parts, float(dap), (pos, rot, scale, lat, exist, par, parent_idx)

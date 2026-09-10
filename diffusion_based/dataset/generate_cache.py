"""
Dataset Cache Generator — XML -> rendered image + 26D nodes + phytomer packets.

Modes (--mode):
  cache - one .pt PER SAMPLE (PartArrayDataset fast path), containing:
            nodes          (max_slots, 26) FM organ rows (EMPTY-padded)
            dap, num_organs, existence_mask
            image          (4*zooms, H, W) rendered pyramid
            num_zooms, zooms
            phytomer_ids   (N, 2) XML (shoot_id, phytomer_idx) membership
            pkt            {packets, presence, centers, refs, latent}
                           (latent only when --vae-checkpoint is given)
          Rendering the pyramid image is the expensive step; `pkt` is derived
          from the same XML rows for free, so the training loop never has to
          rebuild/encode packets per step.

  pkt   - XML-direct packet targets only (NO rendering), one small .pt per
          sample: {packets, presence, centers, refs, latent}. Use this to build
          or refresh packet targets for an existing image cache without
          re-rendering (reads ~250KB XML instead of the ~630KB cache file).
          Requires --vae-checkpoint to store `latent`.

Pyramid option (--pyramid):
  none     - (4,  H, W)        RGB(3, [-1,1]) + CHM depth(1, meters)
  concat   - (16, H, W)        the 4-channel image rendered at zoom 1x, 2x, 4x, 8x,
                               concatenated along the channel dimension.

Data selection (--species):
  cowpea  - only dataset/helios_data/cowpea (default)
  bean    - only bean
  all     - both

Crop-named outputs (default):
  cache mode -> dataset/cache/<crop>_curv26/
  pkt   mode -> dataset/cache/<crop>_curv26_pkt/
"""

import os
import sys
import glob
import re
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray, ORGAN_NONE, P_COL_ORGAN_TYPE,
    T_COL_ORGAN_TYPE, T_COL_SHOOT_ID, T_COL_PHYTOMER_IDX,
    ORGAN_ROOT_META, ORGAN_SHOOT_META,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    encode_fm, FM_NODE_DIM, EMPTY_IDX,
)
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets
from diffusion_based.models.phytomer_vae import PhytomerVAE

PYRAMID_ZOOMS = [1.0, 2.0, 4.0, 8.0]
DEFAULT_VAE_CHECKPOINT = os.path.join(
    repo_root, "diffusion_based", "checkpoints", "phytomer_vae_xml", "phytomer_vae_64d_best.pt")

PKT_VERSION = 3  # 1: 8-slot, 2: 10-slot absolute latent, 3: 10-slot + normalized-space latent


def parse_args():
    parser = argparse.ArgumentParser(description="Plant Dataset Cache Generator (cache / pkt modes)")
    parser.add_argument("--mode", type=str, default="cache", choices=["cache", "pkt"],
                        help="cache: render image + nodes + pkt per sample; "
                             "pkt: XML-direct packet targets only (no rendering)")
    parser.add_argument("--species", type=str, default="cowpea", choices=["cowpea", "bean", "all"],
                        help="Which crop's XML models to generate from (default: cowpea only)")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the output directory (default depends on mode + crop)")
    parser.add_argument("--no-crop-suffix", action="store_false", dest="crop_suffix", default=True,
                        help="Do not append the crop name to the output dir")
    parser.add_argument("--pyramid", type=str, default="concat", choices=["none", "concat"],
                        help="none: single 4-ch image; concat: 4 channels x 4 zoom levels = 16-ch")
    parser.add_argument("--num-workers", type=int, default=20,
                        help="Total SLURM workers (with --worker-id) or DataLoader workers in pkt mode")
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers for pkt mode (0 = use --num-workers)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="XML files per producer batch in pkt mode")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-slots", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-checkpoint", type=str, default=DEFAULT_VAE_CHECKPOINT,
                        help="Frozen PhytomerVAE checkpoint for `latent` (empty string = skip latent)")
    parser.add_argument("--file-list", type=str, default="",
                        help="pkt mode: text file of XML paths to process (one per line). "
                             "When set, --data-root discovery is skipped.")
    parser.add_argument("--force", action="store_true",
                        help="cache mode: overwrite existing .pt files")
    parser.add_argument("--dap-min", type=float, default=None, help="Filter XMLs by minimum DAP")
    parser.add_argument("--dap-max", type=float, default=None, help="Filter XMLs by maximum DAP")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="Print progress every N samples (0 = auto: cache 10, pkt 100)")
    parsed = parser.parse_args()
    crop = "all" if parsed.species == "all" else parsed.species
    if parsed.output_dir is None:
        base = os.path.join(repo_root, "dataset", "cache")
        suffix = "_pkt" if parsed.mode == "pkt" else ""
        parsed.output_dir = os.path.join(base, f"{crop}_curv26{suffix}") if parsed.crop_suffix else base
    return parsed


def load_species_xml_samples(data_root: str, species: str, dap_min: Optional[float] = None, dap_max: Optional[float] = None) -> List[Dict[str, Any]]:
    """Discovers plant XML files for the given crop (or both with 'all')."""
    if species == "all":
        search_path = os.path.join(data_root, "**", "*_plant_*.xml")
    else:
        search_path = os.path.join(data_root, species, "**", "*_plant_*.xml")

    xml_files = sorted(glob.glob(search_path, recursive=True))
    if not xml_files:
        flat = os.path.join(data_root, species, "*_plant_*.xml") if species != "all" else os.path.join(data_root, "*_plant_*.xml")
        xml_files = sorted(glob.glob(flat))

    samples = []
    for x in xml_files:
        bn = os.path.basename(x)
        dap = 30.0
        m = re.search(r"dap(\d+)", bn)
        if m:
            dap = float(m.group(1))
        if dap_min is not None and dap < dap_min:
            continue
        if dap_max is not None and dap > dap_max:
            continue
        samples.append({"xml": x, "dap": dap, "filename": bn})
    return samples


def extract_phytomer_ids(arr: PlantOrganArray, num_organs: Optional[int] = None) -> torch.Tensor:
    """Per-organ (shoot_id, phytomer_idx) from the XML 40D tensor, (-1,-1) for meta rows.

    Row order matches the cache `nodes` rows (cache nodes[i] == 40D row i). This is
    the EXACT XML phytomer membership that build_phytomer_packets groups by —
    nearest-center re-clustering mis-assigns ~77% of mature-plant leaflets.
    """
    t = arr.tensor
    n = t.shape[0] if num_organs is None else min(int(num_organs), t.shape[0])
    ids = torch.full((n, 2), -1, dtype=torch.int64)
    ot_col = t[:, T_COL_ORGAN_TYPE]
    sid_col = t[:, T_COL_SHOOT_ID]
    pidx_col = t[:, T_COL_PHYTOMER_IDX]
    for i in range(n):
        ot = int(ot_col[i].item())
        if ot not in (ORGAN_ROOT_META, ORGAN_SHOOT_META):
            ids[i, 0] = int(sid_col[i].item())
            ids[i, 1] = int(pidx_col[i].item())
    return ids


def load_vae(checkpoint: str, device: torch.device) -> Optional[PhytomerVAE]:
    if not checkpoint:
        return None
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")
    vae = PhytomerVAE(latent_dim=64, hidden_dim=256).to(device)
    vae.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def build_pkt_targets(
    nodes_26d: torch.Tensor,
    existence_mask: torch.Tensor,
    phytomer_ids: torch.Tensor,
    vae: Optional[PhytomerVAE] = None,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Canonical packet targets + optional frozen-VAE latent for one sample."""
    packets, presence, centers, refs = build_phytomer_packets(
        nodes_26d, existence_mask=existence_mask, phytomer_ids=phytomer_ids)
    if packets.shape[0] == 0:
        return None
    pkt = {
        "packets": packets.half().cpu(),  # ABSOLUTE (assembly + s_a need absolute)
        "presence": presence.cpu(),
        "centers": centers.cpu(),
        "refs": refs.cpu(),
        # v3: 10 slots, latent from scale-NORMALIZED VAE input, s_a carried by
        # the 76D flow state (petiole scale row, see anchor_scale).
        "pkt_version": 3,
    }
    if vae is not None:
        dev = device if device is not None else next(vae.parameters()).device
        with torch.no_grad():
            lat = vae.encode(vae.pack_input(packets.to(dev).float(), presence.to(dev)))[0]
        pkt["latent"] = lat.cpu().half()
    return pkt


def encode_sample(
    arr: PlantOrganArray,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    max_slots: int,
    use_pyramid: bool,
    dap: float,
    vae: Optional[PhytomerVAE] = None,
) -> Optional[Dict[str, Any]]:
    """
    Renders one XML into a training sample dict (nodes + image + pkt).
    Image channels: 4 (RGB[-1,1] + CHM meters) per zoom level; 16 channels total
    when --pyramid concat (zooms 1,2,4,8 concatenated along channels).
    """
    part_13d = arr.to_part_tensor()
    num_nodes = min(part_13d.shape[0], max_slots)

    nodes_fm = torch.zeros((max_slots, FM_NODE_DIM))
    nodes_fm[:num_nodes] = encode_fm(part_13d[:num_nodes])
    nodes_fm[num_nodes:, EMPTY_IDX] = 1.0  # EMPTY padding rows

    existence = (part_13d[:num_nodes, P_COL_ORGAN_TYPE] > ORGAN_NONE)
    phytomer_ids = extract_phytomer_ids(arr, num_nodes)
    pkt = build_pkt_targets(nodes_fm[:num_nodes], existence.float(), phytomer_ids,
                            vae=vae, device=device)

    mesh = renderer.geo_builder.build_mesh_from_part_tensor(
        arr.to_part_tensor(device=device), device=device)

    zooms = PYRAMID_ZOOMS if use_pyramid else [1.0]
    channel_imgs = []
    for zoom in zooms:
        with torch.no_grad():
            rgbd = renderer.render_mesh(
                mesh,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=5.0,
                background="ground",
                focus_plant=True,
                include_depth=True,
                zoom_factor=float(zoom),
                reference_window_size=1.2,
            )  # (4, H, W)
        rgb_norm = (rgbd[:3].clamp(0.0, 1.0) - 0.5) / 0.5
        depth_ch = rgbd[3:4].clamp(min=0.0)
        channel_imgs.append(torch.cat([rgb_norm, depth_ch], dim=0))

    image = torch.cat(channel_imgs, dim=0)  # (4*len(zooms), H, W)
    return {
        "nodes": nodes_fm.cpu(),
        "dap": torch.tensor(dap, dtype=torch.float32),
        "num_organs": torch.tensor(num_nodes, dtype=torch.long),
        "existence_mask": existence.cpu(),
        "image": image.half().cpu(),
        "num_zooms": len(zooms),
        "zooms": zooms,
        "phytomer_ids": phytomer_ids,
        "pkt": pkt,
    }


def generate_cache(
    species: str,
    data_root: str,
    output_dir: str,
    num_workers: int,
    worker_id: int,
    image_size: int,
    max_slots: int,
    device_str: str,
    use_pyramid: bool,
    vae_checkpoint: str = "",
    force: bool = False,
    progress_every: int = 10,
    dap_min: Optional[float] = None,
    dap_max: Optional[float] = None,
):
    """One .pt per sample, written to <output_dir>/<prefix>.pt (PartArrayDataset fast path)."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    all_xml = load_species_xml_samples(data_root, species, dap_min=dap_min, dap_max=dap_max)
    if not all_xml:
        raise FileNotFoundError(f"No XML plant models for species '{species}' in {data_root} (dap_range: {dap_min}-{dap_max})")

    if num_workers <= 1:
        lo = 0
        hi = len(all_xml)
        shard_slice = all_xml
    else:
        per_worker = (len(all_xml) + num_workers - 1) // num_workers
        lo = worker_id * per_worker
        hi = min(lo + per_worker, len(all_xml))
        shard_slice = all_xml[lo:hi]
    print(f"[Cache worker {worker_id}/{num_workers}] {lo} -> {hi} ({hi - lo} samples) -> {output_dir}")

    vae = load_vae(vae_checkpoint, device)
    print(f"VAE latent: {'ON (' + os.path.basename(vae_checkpoint) + ')' if vae else 'OFF (packets only)'}")

    renderer = HeliosPyTorchRenderer(image_size=image_size, device=device)
    t0 = time.time()
    ok = skipped = empty = err = 0
    for k, s_info in enumerate(shard_slice):
        prefix = os.path.basename(s_info["xml"]).split("_plant_")[0]
        out_pt = os.path.join(output_dir, f"{prefix}.pt")
        if os.path.exists(out_pt) and not force:
            skipped += 1
            ok += 1
            continue
        try:
            arr = PlantOrganArray.from_xml_file(s_info["xml"])
            sample = encode_sample(arr, renderer, device, max_slots, use_pyramid,
                                   dap=s_info["dap"], vae=vae)
            if sample is None or sample.get("pkt") is None:
                empty += 1
                continue
            sample["prefix"] = prefix
            sample["xml"] = s_info["xml"]
            torch.save(sample, out_pt)
            ok += 1
        except Exception:
            err += 1
            continue
        if progress_every and (k + 1) % progress_every == 0:
            el = time.time() - t0
            rate = (k + 1) / max(el, 1e-3)
            print(f"  [{k + 1}/{hi - lo}] ok={ok} skip={skipped} empty={empty} err={err} "
                  f"{el:.0f}s ({rate:.1f}/s)", flush=True)

    el = time.time() - t0
    print(f"[Cache worker {worker_id}] DONE {ok}/{hi - lo} samples in {el:.1f}s "
          f"(skip={skipped}, empty={empty}, err={err}, {ok / max(el, 1e-3):.1f}/s) -> {output_dir}")


class _XmlPktDataset(Dataset):
    """pkt mode: XML -> packet targets (no rendering). Producer side of the pipeline."""

    def __init__(self, xml_files: List[str], out_dir: str, need_latent: bool):
        self.xml_files = xml_files
        self.out_dir = out_dir
        self.need_latent = need_latent

    def __len__(self) -> int:
        return len(self.xml_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.xml_files[idx]
        prefix = os.path.basename(path).split("_plant_")[0]
        out_path = os.path.join(self.out_dir, f"{prefix}.pt")
        if os.path.exists(out_path):
            try:
                d = torch.load(out_path, map_location="cpu", weights_only=True)
                if "packets" in d and (not self.need_latent or "latent" in d):
                    return {"status": "skip", "prefix": prefix, "out_path": out_path}
            except Exception:
                pass
        try:
            arr = PlantOrganArray.from_xml_file(path)
            part = arr.to_part_tensor()
            nodes = encode_fm(part)
            existence = (part[:, P_COL_ORGAN_TYPE] > ORGAN_NONE).float()
            ids = extract_phytomer_ids(arr)
            packets, presence, centers, refs = build_phytomer_packets(
                nodes, existence_mask=existence, phytomer_ids=ids)
            if packets.shape[0] == 0:
                return {"status": "empty", "prefix": prefix, "out_path": out_path}
            return {
                "status": "ok",
                "prefix": prefix,
                "out_path": out_path,
                "packets": packets.half(),
                "presence": presence,
                "centers": centers,
                "refs": refs,
            }
        except Exception as e:
            return {"status": f"err:{e}", "prefix": prefix, "out_path": out_path}


def generate_pkt(
    species: str,
    data_root: str,
    output_dir: str,
    num_workers: int,
    worker_id: int,
    device_str: str,
    vae_checkpoint: str = "",
    file_list: str = "",
    progress_every: int = 100,
    dap_min: Optional[float] = None,
    dap_max: Optional[float] = None,
):
    """XML-direct packet targets only: {packets, presence, centers, refs, latent}."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    if file_list and os.path.exists(file_list):
        with open(file_list) as f:
            xml_files = [ln.strip() for ln in f if ln.strip()]
    else:
        xml_files = [s["xml"] for s in load_species_xml_samples(
            data_root, species, dap_min=dap_min, dap_max=dap_max)]
    if not xml_files:
        raise FileNotFoundError(f"No XML plant models for species '{species}' in {data_root}")

    vae = load_vae(vae_checkpoint, device)
    if vae is None:
        print("WARNING: no --vae-checkpoint given; writing packets WITHOUT latent")

    todo = []
    done_count = 0
    for p in xml_files:
        prefix = os.path.basename(p).split("_plant_")[0]
        out_path = os.path.join(output_dir, f"{prefix}.pt")
        if os.path.exists(out_path):
            try:
                d = torch.load(out_path, map_location="cpu", weights_only=True)
                if ("packets" in d and d.get("pkt_version", 0) >= PKT_VERSION
                        and (vae is None or "latent" in d)):
                    done_count += 1
                    continue
            except Exception:
                pass
        todo.append(p)
    print(f"[Pkt worker {worker_id}/{num_workers}] {len(todo)} todo ({done_count} already done) -> {output_dir}")
    print(f"VAE latent: {'ON (' + os.path.basename(vae_checkpoint) + ')' if vae else 'OFF'}")

    n_loader_workers = num_workers if num_workers > 1 else 0
    dataset = _XmlPktDataset(todo, output_dir, need_latent=vae is not None)
    loader = DataLoader(dataset, batch_size=max(1, min(128, len(todo) if todo else 1)),
                        num_workers=n_loader_workers, collate_fn=lambda b: b, pin_memory=False)

    t0 = time.time()
    done = ok = skip = empty = err = 0
    for batch in loader:
        items = [b for b in batch if b["status"] == "ok"]
        if items and vae is not None:
            all_in = [vae.pack_input(it["packets"].float(), it["presence"]) for it in items]
            mega = torch.cat(all_in, dim=0).to(device)
            chunk = 262144
            lats = []
            with torch.no_grad():
                for i in range(0, mega.shape[0], chunk):
                    lats.append(vae.encode(mega[i:i + chunk])[0])
            lats = torch.cat(lats, dim=0).cpu()
            off = 0
            for it in items:
                n = it["packets"].shape[0]
                d = {
                    "packets": it["packets"],
                    "presence": it["presence"],
                    "centers": it["centers"],
                    "refs": it["refs"],
                    "latent": lats[off:off + n].half(),
                }
                off += n
                torch.save(d, it["out_path"])
        elif items:
            for it in items:
                torch.save({
                    "packets": it["packets"],
                    "presence": it["presence"],
                    "centers": it["centers"],
                    "refs": it["refs"],
                }, it["out_path"])
        for b in batch:
            done += 1
            if b["status"] == "ok":
                ok += 1
            elif b["status"] == "skip":
                skip += 1
            elif b["status"] == "empty":
                empty += 1
            else:
                err += 1
        if progress_every and done % progress_every < len(batch):
            el = time.time() - t0
            rate = done / max(el, 1e-3)
            eta = (len(todo) - done) / max(rate, 1e-3) / 60
            print(f"[{done}/{len(todo)}] ok={ok} skip={skip} empty={empty} err={err} "
                  f"{el:.0f}s ({rate:.0f}/s, ETA {eta:.0f}min)", flush=True)

    print(f"[Pkt worker {worker_id}] DONE ok={ok} skip={skip} empty={empty} err={err} "
          f"in {time.time() - t0:.0f}s -> {output_dir}", flush=True)


def main():
    args = parse_args()
    use_pyramid = (args.pyramid == "concat")
    if args.mode == "cache":
        generate_cache(
            species=args.species, data_root=args.data_root, output_dir=args.output_dir,
            num_workers=args.num_workers, worker_id=args.worker_id,
            image_size=args.image_size, max_slots=args.max_slots,
            device_str=args.device, use_pyramid=use_pyramid,
            vae_checkpoint=args.vae_checkpoint, force=args.force,
            progress_every=args.progress_every or 10,
            dap_min=args.dap_min, dap_max=args.dap_max,
        )
    else:
        generate_pkt(
            species=args.species, data_root=args.data_root, output_dir=args.output_dir,
            num_workers=args.workers or args.num_workers, worker_id=args.worker_id,
            device_str=args.device, vae_checkpoint=args.vae_checkpoint,
            file_list=args.file_list,
            progress_every=args.progress_every or 100,
            dap_min=args.dap_min, dap_max=args.dap_max,
        )


if __name__ == "__main__":
    main()

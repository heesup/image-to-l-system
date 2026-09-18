"""Download a real cowpea/bean field dataset from AgML (https://github.com/Project-AgML/AgML) and
convert it to the same on-disk layout use_cases/real_world/download_roboflow_dataset.py's Roboflow export used
(images/ + labels/ YOLO .txt + data.yaml, train/valid/test split), so every downstream script
(use_cases/real_world/detector/train_yolo_detector.py, real_plant_crop_utils.py, real_field_dataset.py) reads
either source unchanged.

Default dataset: `gemini_plant_detection_2022` -- object-detection (COCO bounding boxes, no
segmentation masks), classes ['plant', 'weed'], 402 images at 2592x2048, from the GEMINI cowpea
breeding project (http://gemini-breeding.github.io/) — the SAME nadir tunnel-cart rig (metal rails,
integrated LED bars, identical 2592x2048 framing, confirmed by inspection 2026-09-16) as the Roboflow
`t4_plant_weed_seg` export already used, just a different capture batch/date (filenames carry
Davis-COWPEAMAGIC plot IDs and a full timestamp, like the Roboflow one — dap_from_timestamp.py's
parser should extend to these once its regex is checked against the new filename format).

No segmentation masks means the fine-tuned detector will be a plain YOLO11n (not -seg). First test
(2026-09-16, 6 plants) with no fallback silhouette at all reproduced the project's known "canvas
inflation" failure on several plants worse than before -- without a real per-plant mask, Approach 2's
Dice term had only the blurry Depth-Anything pseudo-CHM to threshold against, a much looser target
than a tight instance mask. Fixed by giving `detect_plants` a bounding-box-rectangle mask fallback
when the detector has no segmentation output (`bbox_to_mask` in real_plant_crop_utils.py) -- coarser
than a real mask but still a real silhouette bound, unlike thresholding the pseudo-depth blob.

Other bean/cowpea options in AgML's catalog as of 2026-09-16 (agml.data.public_data_sources()):
  - gemini_pod_detection_2022 (98 img), gemini_leaf_detection_2022 (25 img),
    gemini_flower_detection_2022 (134 img): same project, organ-level detection, single class
    'object' each -- useful later for per-organ evaluation, not plant/weed localization.
  - bean_synthetic_earlygrowth_aerial: SYNTHETIC (location='digital'), not a real-image test.
  - bean_disease_uganda: real but whole-image disease classification, no localization at all.
  - iNatAg(-mini)/vigna_unguiculata (cowpea) and .../phaseolus_vulgaris (common bean): real but
    image_classification (species ID from iNaturalist photos), no bounding boxes.
  None of these five have both (a) real field imagery and (b) object-detection annotations, which
  is why gemini_plant_detection_2022 is the only real match for "bean or cowpea bounding boxes".
"""
import argparse
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def convert_coco_to_yolo(agml_dir: Path, out_dir: Path, val_frac: float, test_frac: float, seed: int) -> None:
    ann = json.load(open(agml_dir / "annotations.json"))
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    by_image = {im["id"]: im for im in ann["images"]}
    boxes_by_image = {}
    for a in ann["annotations"]:
        boxes_by_image.setdefault(a["image_id"], []).append(a)

    ids = list(by_image.keys())
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val_frac); n_test = round(len(ids) * test_frac)
    split_ids = {"valid": set(ids[:n_val]), "test": set(ids[n_val:n_val + n_test]), "train": set(ids[n_val + n_test:])}

    for split in ("train", "valid", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    id_to_split = {i: s for s, s_ids in split_ids.items() for i in s_ids}
    n_boxes = 0
    for img_id, im in by_image.items():
        split = id_to_split[img_id]
        src = agml_dir / "images" / im["file_name"]
        dst_img = out_dir / split / "images" / im["file_name"]
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        W, H = im["width"], im["height"]
        lines = []
        for a in boxes_by_image.get(img_id, []):
            x, y, w, h = a["bbox"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            lines.append(f"{cat_id_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes += 1
        (out_dir / split / "labels" / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )
    counts = {s: len(v) for s, v in split_ids.items()}
    print(f"Wrote YOLO-format dataset to {out_dir} ({len(ids)} images, {n_boxes} boxes, classes={names}, split={counts})")
    print(f"data.yaml: {data_yaml}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gemini_plant_detection_2022",
                    help="AgML dataset name (see agml.data.public_data_sources() and this file's docstring)")
    ap.add_argument("--out", default=str(REPO_ROOT / "real_world" / "data" / "agml_gemini_plant_detection_2022"))
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import agml
    loader = agml.data.AgMLDataLoader(a.dataset)  # downloads to ~/.agml/datasets/<dataset>/ if not already present
    agml_dir = Path.home() / ".agml" / "datasets" / a.dataset
    print(f"AgML dataset ready at {agml_dir} ({len(loader)} images)")
    convert_coco_to_yolo(agml_dir, Path(a.out), a.val_frac, a.test_frac, a.seed)


if __name__ == "__main__":
    main()

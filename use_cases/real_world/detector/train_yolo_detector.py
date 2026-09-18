"""Fine-tune a lightweight YOLO detector on the Roboflow t4_plant_weed_seg export to localize
individual cowpea plants (class 'plant') in real nadir tunnel-cart images, separating them from
weeds (class 'weed'). Output feeds use_cases/real_world/dataset/real_plant_crop_utils.py::detect_plants.

Per-run artifacts (weights, curves, args.yaml) are written next to that run's own directory
under outputs/logs/, not scattered into outputs/eval.
"""
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO_ROOT / "use_cases/real_world/data/roboflow_t4_plant_weed_seg/1/data.yaml"))
    ap.add_argument("--weights", default=str(REPO_ROOT / "outputs/weights/yolo11n-seg.pt"), help="pretrained base (nano = lightweight)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280, help="native images are 2592x2048; keep detail on small plant/weed leaves")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", default=str(REPO_ROOT / "outputs/logs"))
    ap.add_argument("--name", default="real_plant_detector")
    a = ap.parse_args()

    # Ultralytics prepends runs/<task>/ to a relative --project path (confirmed empirically,
    # 2026-09-15) — pass an absolute path so weights/curves land exactly under --project/--name,
    # matching the per-run-artifact convention (next to that run's own SLURM log).
    project_abs = str(Path(a.project).resolve())

    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                project=project_abs, name=a.name, patience=30, exist_ok=True)


if __name__ == "__main__":
    main()

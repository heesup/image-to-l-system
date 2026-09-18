"""Download the real cowpea field dataset from Roboflow (project t4_plant_weed_seg).

Resolves the workspace from the API key, finds the project, and downloads the requested
export format to use_cases/real_world/data/roboflow_<project>/<version>/. Never prints or logs the
API key. Run once, then inspect the result (data.yaml, class names, a few images) before
writing anything downstream that assumes a particular format.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / "real_world" / ".env")


def resolve_project(rf, project_slug: str):
    """Find project_slug across the key's accessible workspaces."""
    try:
        ws = rf.workspace()
    except Exception as e:
        print(f"Default workspace lookup failed ({e}); listing all workspaces.")
        ws = None
    if ws is not None:
        try:
            return ws.project(project_slug), ws.url
        except Exception as e:
            print(f"Project '{project_slug}' not in default workspace ({getattr(ws, 'url', '?')}): {e}")
    # Fall back: Roboflow's public API doesn't expose "list all my workspaces" without
    # knowing their slugs, so surface what we do know and let the user redirect.
    raise SystemExit(
        f"Could not find project '{project_slug}' under the API key's default workspace. "
        "Pass --workspace <slug> explicitly (find it in the Roboflow web UI project URL: "
        "app.roboflow.com/<workspace>/<project>/<version>)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="t4_plant_weed_seg")
    ap.add_argument("--workspace", default="", help="workspace slug; if omitted, use the API key's default workspace")
    ap.add_argument("--version", type=int, default=0, help="0 = latest version")
    ap.add_argument("--format", default="yolov8", choices=["yolov8", "yolov8-obb", "coco", "voc"],
                     help="Roboflow export format. yolov8 includes seg masks if the project is instance-segmentation.")
    ap.add_argument("--out", default=str(REPO_ROOT / "real_world" / "data"))
    a = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY not set. Put it in use_cases/real_world/.env (see .env.example).")

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    if a.workspace:
        project = rf.workspace(a.workspace).project(a.project)
        ws_slug = a.workspace
    else:
        project, ws_slug = resolve_project(rf, a.project)

    version_num = a.version or int(project.versions()[-1].version.split("/")[-1])
    version = project.version(version_num)

    out_dir = Path(a.out) / f"roboflow_{a.project}" / str(version_num)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ws_slug}/{a.project}/v{version_num} ({a.format}) -> {out_dir}")
    version.download(a.format, location=str(out_dir))

    data_yaml = out_dir / "data.yaml"
    if data_yaml.exists():
        print(f"\nOK: {data_yaml}")
        print(data_yaml.read_text())
    else:
        found = list(out_dir.rglob("*.yaml")) + list(out_dir.rglob("*.json"))
        print(f"\nNo data.yaml at the expected path; found: {found[:5]}")


if __name__ == "__main__":
    main()

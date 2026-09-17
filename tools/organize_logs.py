#!/usr/bin/env python
"""Organize outputs/logs/ into one folder per start date (YYYYMMDD).

Moves job logs (hierarchical_fm_<jobid>.log), local-run logs (local_*.log), per-run figure folders
(run_<jobid>/, run_local_<timestamp>/) and evaluation folders (run_sub10_* etc.) from the top level into
outputs/logs/<YYYYMMDD>/ by their start date; re-links each run folder's run.log symlink relatively.
Items modified within --live-minutes (default 20) are left alone so running jobs are never touched.
Dry-run by default; --apply performs the moves. Nothing is ever deleted.
"""
import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "logs")
DATE_RE = re.compile(r"^\d{8}$")
DATE_LINE = re.compile(r"^Date: (\w{3}) (\w{3}) +(\d+) .* (\d{4})$")


def header_date(path: str):
    """Start date from the launcher's 'Date: Mon Sep 15 ...' header line, if present."""
    try:
        with open(path, errors="replace") as f:
            for _ in range(60):
                line = f.readline()
                if not line:
                    break
                m = DATE_LINE.match(line.strip())
                if m:
                    return datetime.strptime(f"{m.group(2)} {m.group(3)} {m.group(4)}", "%b %d %Y").strftime("%Y%m%d")
    except OSError:
        pass
    return None


def item_date(path: str):
    name = os.path.basename(path)
    m = re.match(r"run_local_(\d{8})_", name)
    if m:
        return m.group(1)
    if os.path.isfile(path):
        d = header_date(path)
        if d:
            return d
    if os.path.isdir(path):
        # a run folder: date of its earliest panel / file
        try:
            t = min(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)) if os.listdir(path) else os.path.getmtime(path)
        except (OSError, ValueError):
            t = os.path.getmtime(path)
        return time.strftime("%Y%m%d", time.localtime(t))
    return time.strftime("%Y%m%d", time.localtime(os.path.getctime(path)))


def is_live(path: str, minutes: float) -> bool:
    newest = os.path.getmtime(path)
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
    return (time.time() - newest) < minutes * 60


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="perform the moves (default: dry run)")
    ap.add_argument("--live-minutes", type=float, default=20.0, help="leave items modified this recently")
    ap.add_argument("--include-live", action="store_true", help="move live items too (only when their runs are known to be finished)")
    a = ap.parse_args()
    logs = os.path.abspath(LOGS)
    moves, skipped = [], []
    for name in sorted(os.listdir(logs)):
        path = os.path.join(logs, name)
        if DATE_RE.match(name) or name.startswith("archive_") or name in ("README.md",):
            continue
        if not a.include_live and is_live(path, a.live_minutes):
            skipped.append(name)
            continue
        date = item_date(path)
        # keep a run folder with its job log: run_<jobid> takes the log's date when both exist
        m = re.match(r"run_(\d+)$", name)
        if m and os.path.exists(os.path.join(logs, f"hierarchical_fm_{m.group(1)}.log")):
            date = item_date(os.path.join(logs, f"hierarchical_fm_{m.group(1)}.log"))
        moves.append((name, date))
    for name, date in moves:
        dst_dir = os.path.join(logs, date)
        print(f"{'MOVE' if a.apply else 'would move'}  {name:<48} -> {date}/")
        if a.apply:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(os.path.join(logs, name), os.path.join(dst_dir, name))
            link = os.path.join(dst_dir, name, "run.log")
            if os.path.islink(link):
                target = os.readlink(link)
                base = os.path.basename(target)
                os.unlink(link)
                os.symlink(os.path.join("..", base), link)     # relative: ../hierarchical_fm_<jobid>.log
    if skipped:
        print(f"left in place (modified in the last {a.live_minutes:.0f} min): " + ", ".join(skipped))
    print(f"{len(moves)} item(s) {'moved' if a.apply else 'to move'}; run with --apply to perform.")


if __name__ == "__main__":
    sys.exit(main())

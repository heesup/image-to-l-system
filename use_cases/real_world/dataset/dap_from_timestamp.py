"""Derive a real plant's DAP (Days After Planting) from the capture timestamp embedded in its
Roboflow filename, given a known/estimated planting date — a much more reliable age signal than
Stage 1's self-predicted DAP, which was trained only on Helios-simulated appearances (see
docs/experiments/20260915-real-image-first-test/20260915-real-image-first-test.md §4.1/§6 for the calibration-error evidence this
was built to quantify).

Filenames look like `2023-06-20_Plot109-MAGIC002_camA_camA-1687286498964934887_jpg.rf....jpg` —
the trailing digits before `_jpg` are a nanosecond Unix epoch timestamp (confirmed 2026-09-15:
1687286498964934887 -> 2023-06-20 18:41:38 UTC, matching the filename's own date prefix), present
on every file including the few without a leading date prefix, so this is the robust source of
truth rather than a filename regex on the prefix.
"""
import datetime
import re

_TS_RE = re.compile(r"(\d{16,19})_jpg")


def capture_datetime(path_or_name: str) -> "datetime.datetime | None":
    m = _TS_RE.search(path_or_name)
    if not m:
        return None
    ns = int(m.group(1))
    return datetime.datetime.fromtimestamp(ns / 1e9, tz=datetime.timezone.utc)


def dap_from_filename(path_or_name: str, planted_date: datetime.date) -> "float | None":
    dt = capture_datetime(path_or_name)
    if dt is None:
        return None
    return (dt.date() - planted_date).days

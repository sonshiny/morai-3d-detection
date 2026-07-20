#!/usr/bin/env python3
"""Read-only OCCLUSION_MIN_PTS impact aggregation.

Uses existing occlusion sidecars when present. If the dataset is absent, it
falls back to the aggregate rows in _tmp_occlusion_gen.log. Log-derived counts
are estimates because that log stores removal rates rounded to one decimal.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


THRESHOLDS = (0, 1, 3, 5)
BUCKETS = ((0.0, 20.0), (20.0, 40.0), (40.0, 55.0), (55.0, math.inf))


def aggregate_sidecars(paths: list[Path]):
    import numpy as np

    points = []
    distances = []
    for path in paths:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"unexpected occlusion shape {arr.shape}: {path}")
        if arr.shape[0]:
            points.append(arr[:, 1].astype(np.float64, copy=False))
            distances.append(arr[:, 2].astype(np.float64, copy=False))

    if not points:
        return []
    points = np.concatenate(points)
    distances = np.concatenate(distances)
    rows = []
    for lo, hi in BUCKETS:
        mask = (distances >= lo) & (distances < hi)
        n = int(mask.sum())
        for threshold in THRESHOLDS:
            removed = int(((points[mask] >= 0) & (points[mask] < threshold)).sum())
            rows.append((lo, hi, threshold, n, n - removed, removed,
                         100.0 * removed / n if n else 0.0, False))
    return rows


def aggregate_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = text.rfind("occlusion GT: ALL(")
    if marker < 0:
        raise ValueError(f"ALL aggregate not found in {path}")
    block = text[marker:]
    pattern = re.compile(
        r"\[\s*(\d+)\s*-\s*(\d+)\)\s+(\d+)"
        r"\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"
    )
    parsed = []
    for match in pattern.finditer(block):
        lo, hi, n = map(int, match.group(1, 2, 3))
        rates = dict(zip((1, 3, 5, 10), map(float, match.group(4, 5, 6, 7))))
        parsed.append((float(lo), math.inf if hi == 999 else float(hi), n, rates))
    if not parsed:
        raise ValueError(f"distance/threshold aggregate not found in {path}")

    rows = []
    for lo, hi, n, rates in parsed:
        for threshold in THRESHOLDS:
            rate = 0.0 if threshold == 0 else rates[threshold]
            removed = round(n * rate / 100.0)
            rows.append((lo, hi, threshold, n, n - removed, removed, rate,
                         threshold != 0))
    return rows


def bucket_label(lo: float, hi: float) -> str:
    return f"[{lo:.0f},{'inf' if math.isinf(hi) else f'{hi:.0f}'})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo = args.repo.resolve()
    dataset = repo / "dataset"
    sidecars = sorted(dataset.glob("scen*/occlusion/*.npy")) if dataset.is_dir() else []

    if sidecars:
        rows = aggregate_sidecars(sidecars)
        scenarios = len({p.parents[1].name for p in sidecars})
        print(f"source=sidecar files={len(sidecars)} scenarios={scenarios}")
    else:
        log_path = repo / "_tmp_occlusion_gen.log"
        rows = aggregate_log(log_path)
        print(f"source=log-estimate dataset_exists={dataset.is_dir()} sidecar_files=0 log={log_path.name}")

    print("bucket threshold total surviving removed removal_rate_pct estimated_counts")
    for lo, hi, threshold, total, surviving, removed, rate, estimated in rows:
        print(
            f"{bucket_label(lo, hi)} {threshold} {total} {surviving} "
            f"{removed} {rate:.1f} {'yes' if estimated else 'no'}"
        )


if __name__ == "__main__":
    main()

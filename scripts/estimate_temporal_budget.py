#!/usr/bin/env python3
"""Estimate 150-scene wall time from an actual temporal preflight run.

This deliberately does not infer throughput from a GPU product name.  It uses
the production loader sizes in run_config.json and synchronized timings emitted
by train.py into throughput.jsonl.
"""

import argparse
import json
import math
import os
from pathlib import Path


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _weighted_rate(rows, count_key):
    seconds = sum(float(r["seconds"]) for r in rows)
    count = sum(int(r[count_key]) for r in rows)
    if seconds <= 0 or count <= 0:
        raise ValueError(f"invalid timing totals: seconds={seconds}, {count_key}={count}")
    return count / seconds


def _validation_count(epochs, cadence):
    scheduled = set(range(cadence, epochs + 1, cadence))
    scheduled.add(epochs)  # train.py always validates the final epoch
    return len(scheduled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--cadences", default="1,5,10")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_json(run_dir / "run_config.json")
    rows = _load_jsonl(run_dir / "throughput.jsonl")
    train_rows = [r for r in rows if r.get("kind") == "train" and r.get("dataloader_steps", 0)]
    val_rows = [r for r in rows if r.get("kind") == "validation" and r.get("frames", 0)]
    if not train_rows or not val_rows:
        raise SystemExit("training/validation timing record가 모두 필요합니다.")

    contract = {
        "use_temporal_memory": True,
        "use_streaming_sampler": True,
        "temp_gnn_mode": "gated",
        "train_gt_version": "v3",
        "val_gt_version": "v3",
    }
    mismatches = [
        f"{k}: got={cfg.get(k)!r}, expected={v!r}"
        for k, v in contract.items() if cfg.get(k) != v
    ]
    if mismatches:
        raise SystemExit("temporal production contract mismatch:\n  " + "\n  ".join(mismatches))

    step_rate = _weighted_rate(train_rows, "dataloader_steps")
    train_fps = _weighted_rate(train_rows, "frames")
    val_fps = _weighted_rate(val_rows, "frames")
    train_batches = int(cfg["train_batches_per_epoch"])
    val_frames = int(cfg["val_dataset_frames"])
    train_seconds_per_epoch = train_batches / step_rate
    full_val_seconds = val_frames / val_fps

    scenarios = []
    for cadence in [int(x) for x in args.cadences.split(",") if x.strip()]:
        if cadence < 1:
            raise SystemExit("validation cadence must be >=1")
        n_val = _validation_count(args.epochs, cadence)
        total_seconds = args.epochs * train_seconds_per_epoch + n_val * full_val_seconds
        scenarios.append({
            "validate_every_epochs": cadence,
            "validation_runs": n_val,
            "train_hours_total": args.epochs * train_seconds_per_epoch / 3600.0,
            "validation_hours_total": n_val * full_val_seconds / 3600.0,
            "wall_hours_estimate": total_seconds / 3600.0,
            "wall_days_estimate": total_seconds / 86400.0,
        })

    result = {
        "status": "MEASURED_EXTRAPOLATION_NOT_GUARANTEE",
        "run_dir": str(run_dir),
        "gpu_name": cfg.get("gpu_name"),
        "torch": cfg.get("torch"),
        "cuda": cfg.get("cuda"),
        "contract": contract,
        "configuration": {
            k: cfg.get(k) for k in (
                "batch_size", "grad_accum_steps", "num_workers", "use_amp",
                "allow_tf32", "use_dense_depth", "sequence_length",
            )
        },
        "dataset": {
            "train_dataset_frames": int(cfg["train_dataset_frames"]),
            "train_frames_per_epoch": int(cfg["train_frames_per_epoch"]),
            "streaming_train_dropped_frames": int(cfg["streaming_train_dropped_frames"]),
            "train_batches_per_epoch": train_batches,
            "val_dataset_frames": val_frames,
        },
        "measured": {
            "train_timing_records": len(train_rows),
            "validation_timing_records": len(val_rows),
            "dataloader_steps_per_second": step_rate,
            "train_frames_per_second": train_fps,
            "validation_frames_per_second": val_fps,
            "estimated_train_hours_per_epoch": train_seconds_per_epoch / 3600.0,
            "estimated_full_validation_hours": full_val_seconds / 3600.0,
        },
        "epochs": args.epochs,
        "scenarios": scenarios,
        "warnings": [
            "GPU 이름 환산이 아니라 해당 PC/데이터/temporal 코드경로의 짧은 실측 외삽이다.",
            "thermal throttling, 장시간 I/O, checkpoint 저장은 짧은 preflight보다 느릴 수 있다.",
            "GPU 성능은 필요한 optimizer update 수를 줄이는 근거가 아니다.",
        ],
    }
    out = Path(args.output).resolve() if args.output else run_dir / "temporal_budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"wrote {out}")
    print(
        f"measured train={train_fps:.3f} frame/s, val={val_fps:.3f} frame/s | "
        f"train/epoch={train_seconds_per_epoch/3600:.2f}h, full-val={full_val_seconds/3600:.2f}h"
    )
    for s in scenarios:
        print(
            f"  val every {s['validate_every_epochs']:>2} epoch: "
            f"{s['wall_hours_estimate']:.1f}h = {s['wall_days_estimate']:.2f}d "
            f"({s['validation_runs']} validations)"
        )


if __name__ == "__main__":
    main()

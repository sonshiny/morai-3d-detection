#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

from scripts.check_gpu_environment import version_tuple


REPO = Path(__file__).resolve().parent


def test_gpu_version_parser():
    assert version_tuple("2.7.1+cu128") == (2, 7, 1)
    assert version_tuple("12.8") == (12, 8)


def test_production_wrappers_force_temporal_contract():
    for name in ("preflight_150.sh", "train_150_v3.sh", "resume_150.sh"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "USE_TEMPORAL_MEMORY=1" in text
        assert "STREAMING_SAMPLER=1" in text
        assert "TEMP_GNN_MODE=gated" in text
    prod = (REPO / "scripts" / "train_150_v3.sh").read_text(encoding="utf-8")
    assert "MAX_STEPS_PER_EPOCH=0" in prod
    assert "FAST_VAL_MAX_FRAMES=0" in prod


def test_budget_estimator_uses_measured_temporal_run(tmp_path):
    cfg = {
        "gpu_name": "test-gpu",
        "torch": "test",
        "cuda": "test",
        "use_temporal_memory": True,
        "use_streaming_sampler": True,
        "temp_gnn_mode": "gated",
        "train_gt_version": "v3",
        "val_gt_version": "v3",
        "batch_size": 4,
        "grad_accum_steps": 2,
        "num_workers": 0,
        "use_amp": False,
        "allow_tf32": False,
        "use_dense_depth": False,
        "sequence_length": 150,
        "train_dataset_frames": 4010,
        "train_frames_per_epoch": 4000,
        "streaming_train_dropped_frames": 10,
        "train_batches_per_epoch": 1000,
        "val_dataset_frames": 200,
    }
    (tmp_path / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    rows = [
        {
            "kind": "train", "seconds": 50.0, "frames": 400,
            "dataloader_steps": 100,
        },
        {"kind": "validation", "seconds": 10.0, "frames": 20},
    ]
    (tmp_path / "throughput.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8"
    )
    out = tmp_path / "budget.json"
    subprocess.run(
        [
            sys.executable, str(REPO / "scripts" / "estimate_temporal_budget.py"),
            "--run-dir", str(tmp_path), "--epochs", "10", "--cadences", "5",
            "--output", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["measured"]["train_frames_per_second"] == 8.0
    assert result["measured"]["estimated_train_hours_per_epoch"] == 500.0 / 3600.0
    scenario = result["scenarios"][0]
    assert scenario["validation_runs"] == 2  # epochs 5 and 10
    assert scenario["wall_hours_estimate"] == 5200.0 / 3600.0


def test_resume_contract_records_temporal_and_numeric_modes():
    text = (REPO / "train.py").read_text(encoding="utf-8")
    for key in (
        "'use_temporal_memory': USE_TEMPORAL_MEMORY",
        "'use_streaming_sampler': USE_STREAMING_SAMPLER",
        "'temp_gnn_mode': TEMP_GNN_MODE",
        "'use_amp': USE_AMP",
        "'allow_tf32': ALLOW_TF32",
    ):
        assert text.count(key) >= 2  # run provenance + resumable checkpoint
    assert "optimizer.load_state_dict(ckpt['optimizer_state'])" in text

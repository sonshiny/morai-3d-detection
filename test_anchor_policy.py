#!/usr/bin/env python3
"""
test_anchor_policy.py  (task D)
===============================
anchor 정책 검증:
  1) versioned anchor 의 meta 가 provenance(label_dir, gt_version, k, seed, input_label_sha256,
     anchor_full_sha256)를 기록하고, 기록된 SHA-256 이 실제 .npy 와 일치한다.
  2) 동일 (label_dir, split, k, seed) 로 재생성하면 anchor tensor 가 bitwise 동일하다(재현성).
  3) baseline(train v2)·candidate(train v3) 두 run 이 같은 versioned anchor 파일을 공유하면
     모델이 로드하는 anchor tensor 가 bitwise 동일하다(anchor 는 A/B 변수로 섞이면 안 된다).
  4) SHA-256 가 어긋나면(파일 1바이트만 바뀌어도) 감지된다(학습 시작 fail-fast 트리거).
  5) gt_version -> label_dir 매핑.
이 테스트는 실제 dataset(./dataset, scen05/77/144)과 사전 생성 versioned anchor 를 사용한다.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

import make_kmeans
from make_kmeans import (
    anchor_meta_matches_run, ensure_kmeans_files, hash_train_label_files,
    label_dir_for_gt, metadata_is_valid, resolve_label_dir_name, sha256_file,
)
import anchor_generator

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(REPO, "dataset")
ANCHOR_DIR = os.path.join(REPO, "anchors", "v3_train_scen05_scen77_k900")
FULL = os.path.join(ANCHOR_DIR, "anchor_kmeans_full.npy")
XY = os.path.join(ANCHOR_DIR, "anchor_kmeans_xy.npy")
META = os.path.join(ANCHOR_DIR, "anchor_kmeans_meta.json")

_have_dataset = os.path.isdir(os.path.join(DATASET, "scen05", "labels_3d_v3"))
_have_anchor = os.path.isfile(FULL) and os.path.isfile(META)
needs_data = pytest.mark.skipif(not _have_dataset, reason="./dataset scen05 v3 없음")
needs_anchor = pytest.mark.skipif(not _have_anchor, reason="versioned anchor 미생성")


def test_gt_version_mapping():
    assert label_dir_for_gt("v2") == "labels_3d_v2"
    assert label_dir_for_gt("v3") == "labels_3d_v3"
    assert resolve_label_dir_name(label_dir_name="labels_3d_v3") == "labels_3d_v3"
    assert resolve_label_dir_name(gt_version="v2") == "labels_3d_v2"
    with pytest.raises(ValueError):
        label_dir_for_gt("v9")


@needs_anchor
def test_versioned_anchor_meta_and_sha():
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)
    for key in ("k", "seed", "label_dir", "gt_version", "input_label_sha256",
                "anchor_full_sha256", "anchor_xy_sha256", "train_scenarios",
                "val_scenarios"):
        assert key in meta, f"meta에 {key} 없음"
    assert meta["k"] == 900
    assert meta["label_dir"] == "labels_3d_v3"
    assert meta["gt_version"] == "v3"
    assert meta["train_scenarios"] == ["scen05", "scen77"]
    assert meta["val_scenarios"] == ["scen144"]
    # 기록 SHA == 실제 파일 SHA (학습 시작 fail-fast 검증의 근거)
    assert meta["anchor_full_sha256"] == sha256_file(FULL)
    assert meta["anchor_xy_sha256"] == sha256_file(XY)
    arr = np.load(FULL)
    assert arr.shape == (900, 11)
    print(f"  meta ok: gt={meta['gt_version']} k={meta['k']} "
          f"full_sha={meta['anchor_full_sha256'][:16]} input_sha={meta['input_label_sha256'][:16]}")


@needs_data
@needs_anchor
def test_anchor_reproducible_bitwise():
    """같은 seed/label_dir/split 로 재생성하면 anchor 가 bitwise 동일해야 한다."""
    with tempfile.TemporaryDirectory() as td:
        xy = os.path.join(td, "xy.npy")
        full = os.path.join(td, "full.npy")
        meta = os.path.join(td, "meta.json")
        split = os.path.join(td, "split.json")
        ensure_kmeans_files(
            dataset_root=DATASET, val_scenarios=["scen144"], k=900,
            xy_out=xy, full_out=full, meta_out=meta, split_out=split,
            seed=42, force=True, gt_version="v3",
        )
        regen = np.load(full)
        committed = np.load(FULL)
        assert regen.shape == committed.shape == (900, 11)
        assert np.array_equal(regen, committed), "재생성 anchor가 bitwise 다름 (비결정 kmeans?)"
        assert sha256_file(full) == sha256_file(FULL)
    print("  regenerated anchor is bitwise identical (deterministic seed=42)")


@needs_anchor
def test_shared_anchor_identical_across_v2_v3_runs():
    """baseline(v2)·candidate(v3) 두 run 이 같은 ANCHOR_DIR 를 쓰면 로드 tensor 가 동일."""
    prev_full = os.environ.get("ANCHOR_FULL_FILE")
    prev_xy = os.environ.get("ANCHOR_XY_FILE")
    try:
        os.environ["ANCHOR_FULL_FILE"] = os.path.abspath(FULL)
        os.environ["ANCHOR_XY_FILE"] = os.path.abspath(XY)
        a_baseline = anchor_generator.generate_anchors_full()   # train v2 run 이 로드할 anchor
        a_candidate = anchor_generator.generate_anchors_full()  # train v3 run 이 로드할 anchor
        assert a_baseline.shape == (anchor_generator.NUM_ANCHORS, 11)
        assert torch.equal(a_baseline, a_candidate), "두 run의 anchor tensor가 다름"
        # 파일에서 직접 읽은 값과도 bitwise 동일
        assert torch.equal(a_baseline, torch.from_numpy(np.load(FULL).astype(np.float32)))
    finally:
        for k, v in (("ANCHOR_FULL_FILE", prev_full), ("ANCHOR_XY_FILE", prev_xy)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("  baseline(v2)/candidate(v3) load bitwise-identical anchor tensor")


@needs_anchor
def test_sha_mismatch_detected():
    """1바이트만 바뀌어도 SHA 가 달라진다 → fail-fast 근거."""
    with open(FULL, "rb") as f:
        data = bytearray(f.read())
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)
    data[-1] ^= 0x01  # flip one bit
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
        tf.write(data)
        tampered = tf.name
    try:
        assert sha256_file(tampered) != meta["anchor_full_sha256"]
    finally:
        os.remove(tampered)
    print("  tampered anchor SHA differs from recorded meta → fail-fast would trigger")


@needs_anchor
def test_anchor_meta_matches_run_wrong_split_stale_seed():
    """3-scene anchor 를 다른 split/seed/label 에 지정하면 fail-fast(불일치 감지)."""
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)
    good_kwargs = dict(
        k=meta["k"], gt_version=meta["gt_version"],
        train_scenarios=meta["train_scenarios"], val_scenarios=meta["val_scenarios"],
        seed=meta["seed"], input_label_sha256=meta["input_label_sha256"])
    ok, mm = anchor_meta_matches_run(meta, **good_kwargs)
    assert ok and not mm, f"올바른 설정인데 불일치: {mm}"

    # wrong split: 150-scene split 을 지정 → train_scenarios 불일치
    kw = dict(good_kwargs); kw["train_scenarios"] = [f"scen{i:02d}" for i in range(1, 151)]
    ok, mm = anchor_meta_matches_run(meta, **kw)
    assert not ok and any("train_scenarios" in x for x in mm), mm

    # stale label: 입력 hash 변경 → 불일치
    kw = dict(good_kwargs); kw["input_label_sha256"] = "deadbeef" * 8
    ok, mm = anchor_meta_matches_run(meta, **kw)
    assert not ok and any("input_label_sha256" in x for x in mm), mm

    # wrong seed
    kw = dict(good_kwargs); kw["seed"] = meta["seed"] + 1
    ok, mm = anchor_meta_matches_run(meta, **kw)
    assert not ok and any("seed" in x for x in mm), mm

    # wrong gt_version
    kw = dict(good_kwargs); kw["gt_version"] = "v2"
    ok, mm = anchor_meta_matches_run(meta, **kw)
    assert not ok and any("gt_version" in x for x in mm), mm
    print("  wrong-split / stale-label / wrong-seed / wrong-gt 전부 fail-fast로 감지 -> PASS")


@needs_data
@needs_anchor
def test_metadata_is_valid_rejects_stale_seed():
    """make_kmeans reuse: seed 가 바뀌면 stale anchor 를 재사용하지 않는다(재생성 트리거)."""
    # v3 meta 를 임시 위치에 만들고(seed=42), seed=43 로 물으면 False 여야 한다.
    with tempfile.TemporaryDirectory() as td:
        xy = os.path.join(td, "xy.npy"); full = os.path.join(td, "full.npy")
        meta = os.path.join(td, "meta.json"); split = os.path.join(td, "split.json")
        ensure_kmeans_files(dataset_root=DATASET, val_scenarios=["scen144"], k=900,
                            xy_out=xy, full_out=full, meta_out=meta, split_out=split,
                            seed=42, force=True, gt_version="v3")
        assert metadata_is_valid(meta, DATASET, ["scen144"], 900,
                                 label_dir_name="labels_3d_v3", seed=42) is True
        assert metadata_is_valid(meta, DATASET, ["scen144"], 900,
                                 label_dir_name="labels_3d_v3", seed=43) is False
    print("  seed 43 → metadata_is_valid False (stale 재사용 방지) -> PASS")


if __name__ == "__main__":
    test_gt_version_mapping()
    if _have_anchor:
        test_versioned_anchor_meta_and_sha()
        test_shared_anchor_identical_across_v2_v3_runs()
        test_sha_mismatch_detected()
        test_anchor_meta_matches_run_wrong_split_stale_seed()
    if _have_dataset and _have_anchor:
        test_anchor_reproducible_bitwise()
        test_metadata_is_valid_rejects_stale_seed()
    print("ANCHOR POLICY TESTS PASSED")

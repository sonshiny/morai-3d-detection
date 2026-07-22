#!/usr/bin/env python3
"""
test_gt_version_split.py  (task B/C)
====================================
train GT 와 validation GT 를 분리해도 val 로더의 frame/stem 순서가 GT 버전과 무관하게
동일함을 검증한다(→ 동일 프레임을 v2/v3 어느 GT 로 채점하든 같은 순서로 매칭 가능).
또한 대표 split(train=scen05+scen77, val=scen144)의 프레임 수를 고정 검증한다.
production 데이터셋 클래스(MoraiTemporalDataset)를 그대로 사용한다.
"""
import os
import pytest

from morai_dataset import MoraiTemporalDataset

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(REPO, "dataset")
_have = os.path.isdir(os.path.join(DATASET, "scen144", "labels_3d_v3"))
needs_data = pytest.mark.skipif(not _have, reason="./dataset scen144 v3 없음")


def _order(ds):
    return [(it["scen_name"], it["stem"]) for it in ds.items]


@needs_data
def test_val_loader_order_identical_across_gt_versions():
    v2 = MoraiTemporalDataset(dataset_root=DATASET, split="val",
                              val_scenarios=["scen144"], load_depth=False, gt_version="v2")
    v3 = MoraiTemporalDataset(dataset_root=DATASET, split="val",
                              val_scenarios=["scen144"], load_depth=False, gt_version="v3")
    o2, o3 = _order(v2), _order(v3)
    assert o2 == o3, "v2/v3 val 로더의 (scen,stem) 순서가 다름"
    # 모든 stem 이 scen144, 1421 프레임
    assert len(o3) == 1421, f"scen144 val frame 수 기대 1421, got {len(o3)}"
    assert all(s == "scen144" for s, _ in o3)
    print(f"  val order identical across v2/v3: {len(o3)} frames (scen144)")


@needs_data
def test_train_loader_order_identical_across_gt_versions():
    t2 = MoraiTemporalDataset(dataset_root=DATASET, split="train",
                              val_scenarios=["scen144"], load_depth=False, gt_version="v2")
    t3 = MoraiTemporalDataset(dataset_root=DATASET, split="train",
                              val_scenarios=["scen144"], load_depth=False, gt_version="v3")
    o2, o3 = _order(t2), _order(t3)
    assert o2 == o3, "v2/v3 train 로더의 (scen,stem) 순서가 다름"
    # train = scen05 + scen77 = 557 + 1359 = 1916
    assert len(o3) == 1916, f"train frame 수 기대 1916, got {len(o3)}"
    scen = sorted(set(s for s, _ in o3))
    assert scen == ["scen05", "scen77"], f"train 시나리오 기대 [scen05,scen77], got {scen}"
    print(f"  train order identical across v2/v3: {len(o3)} frames {scen}")


if __name__ == "__main__":
    test_val_loader_order_identical_across_gt_versions()
    test_train_loader_order_identical_across_gt_versions()
    print("GT-VERSION SPLIT TESTS PASSED")

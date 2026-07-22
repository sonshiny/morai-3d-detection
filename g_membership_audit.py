#!/usr/bin/env python3
"""
task G: v2/v3 default-filter membership 감사.
production 필터 경로(MoraiTemporalDataset._load_labels_v2)를 그대로 실행해
각 GT 버전이 학습에 실제로 쓰는 박스 집합을 (scen, frame_id, track_id) 로 구성한다.
  - natural membership : 각 버전이 자기 필터 결과대로(=production A/B가 쓰는 것)
  - common intersection: 두 버전 모두 통과한 박스만(=correction-only 진단 모드)
50m radial 경계에서 v2→v3 보정으로 in/out 이 바뀌는 박스를 전수 보고한다.
"""
import csv
import json
import math
import os

from morai_dataset import MoraiTemporalDataset

ROOT = os.environ.get("DATASET_ROOT", "./dataset")
SCENS = ["scen05", "scen77", "scen144"]


def kept_keys(gt_version):
    """production _load_labels_v2 로 필터한 뒤 kept (scen,frame_id,track_id) → radial dist."""
    keys = {}
    per_scen = {}
    for scen in SCENS:
        ds = MoraiTemporalDataset(dataset_root=ROOT, split="val", val_scenarios=[scen],
                                  load_depth=False, gt_version=gt_version)
        scen_dir = os.path.join(ROOT, scen)
        n_kept = 0
        for it in ds.items:
            stem = it["stem"]
            boxes, labels, track_ids, vv, vsrc, visw = ds._load_labels_v2(scen_dir, stem)
            fr = ds.scene_infos[scen][stem]["frame_id"]
            for bi in range(boxes.shape[0]):
                x, y = float(boxes[bi, 0]), float(boxes[bi, 1])
                tid = int(track_ids[bi])
                keys[(scen, fr, tid)] = math.hypot(x, y)
                n_kept += 1
        per_scen[scen] = n_kept
    return keys, per_scen


def raw_radial(gt_version, scen, frame_id, track_id):
    """필터 이전 raw CSV 에서 해당 박스의 radial dist(있으면)."""
    label_dir = "labels_3d_v2" if gt_version == "v2" else "labels_3d_v3"
    # frame_id -> stem 은 scene_info 로. 간단히 모든 stem CSV 를 스캔하지 않고 scene_info 사용.
    info_name = "scene_info.json" if gt_version == "v2" else "scene_info_v3.json"
    with open(os.path.join(ROOT, scen, info_name), encoding="utf-8") as f:
        info = json.load(f)
    stem = None
    for fr in info["frames"]:
        if fr["frame_id"] == frame_id:
            stem = fr["stem"]
            break
    if stem is None:
        return None
    path = os.path.join(ROOT, scen, label_dir, stem + ".csv")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["track_id"])) == track_id:
                return math.hypot(float(row["x"]), float(row["y"]))
    return None


def main():
    v2, v2_scen = kept_keys("v2")
    v3, v3_scen = kept_keys("v3")
    s2, s3 = set(v2), set(v3)
    common = s2 & s3
    v2_only = sorted(s2 - s3)
    v3_only = sorted(s3 - s2)

    print("=== task G: v2/v3 default-filter membership (production _load_labels_v2) ===")
    print(f"  v2 natural kept : {len(s2)}   per-scen={v2_scen}")
    print(f"  v3 natural kept : {len(s3)}   per-scen={v3_scen}")
    print(f"  common (∩)      : {len(common)}")
    print(f"  v2-only         : {len(v2_only)}")
    print(f"  v3-only         : {len(v3_only)}")
    print(f"  expected         : v2=9124 v3=9113 common=9112 v2-only=12 v3-only=1")

    diffs = []
    print("\n--- v2-only boxes (v2 kept, v3 dropped) — radial in v2 vs v3 raw ---")
    for k in v2_only:
        scen, fr, tid = k
        r2 = v2[k]
        r3raw = raw_radial("v3", scen, fr, tid)
        cross = (r2 <= 50.0) and (r3raw is not None and r3raw > 50.0)
        diffs.append({"kind": "v2_only", "scen": scen, "frame_id": fr, "track_id": tid,
                      "v2_radial": r2, "v3_radial": r3raw, "boundary_cross": bool(cross)})
        print(f"  {scen} frame={fr} track={tid}: v2_r={r2:.3f}m v3_r={r3raw!r} "
              f"{'50m-cross' if cross else 'OTHER'}")
    print("\n--- v3-only boxes (v3 kept, v2 dropped) — radial in v3 vs v2 raw ---")
    for k in v3_only:
        scen, fr, tid = k
        r3 = v3[k]
        r2raw = raw_radial("v2", scen, fr, tid)
        cross = (r3 <= 50.0) and (r2raw is not None and r2raw > 50.0)
        diffs.append({"kind": "v3_only", "scen": scen, "frame_id": fr, "track_id": tid,
                      "v3_radial": r3, "v2_radial": r2raw, "boundary_cross": bool(cross)})
        print(f"  {scen} frame={fr} track={tid}: v3_r={r3:.3f}m v2_r={r2raw!r} "
              f"{'50m-cross' if cross else 'OTHER'}")

    n_cross = sum(1 for d in diffs if d["boundary_cross"])
    print(f"\n  50m boundary-cross among {len(diffs)} diffs: {n_cross}/{len(diffs)}")

    out = {
        "v2_natural": len(s2), "v3_natural": len(s3), "common": len(common),
        "v2_only": len(v2_only), "v3_only": len(v3_only),
        "v2_per_scen": v2_scen, "v3_per_scen": v3_scen,
        "common_intersection_mode_boxes": len(common),
        "expected": {"v2": 9124, "v3": 9113, "common": 9112, "v2_only": 12, "v3_only": 1},
        "diffs": diffs,
        "n_boundary_cross": n_cross,
    }
    os.makedirs("pretrain_verify", exist_ok=True)
    with open("pretrain_verify/g_membership_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote pretrain_verify/g_membership_audit.json")

    ok = (len(s2) == 9124 and len(s3) == 9113 and len(common) == 9112
          and len(v2_only) == 12 and len(v3_only) == 1)
    print("MATCH expected" if ok else "‼️ DOES NOT MATCH expected — 조사 필요")


if __name__ == "__main__":
    main()

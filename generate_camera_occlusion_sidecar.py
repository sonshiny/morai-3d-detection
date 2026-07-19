#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_camera_occlusion_sidecar.py  (C안 규칙용 YOLO 트랙 다수결 sidecar 생성)
================================================================================
scratch_full_occlusion_rule_verify.py 가 산출한 검증된 트랙 다수결 판정
(<scen>/_rule_verify/rule_eval.csv 의 track_occluded 컬럼)을 프레임별 sidecar 로
저장한다. 학습 때마다 YOLO 를 돌리지 않기 위한 사전계산이다.

산출:  dataset/<scen>/camera_occlusion/<stem>.npy
  shape (M, 2) float32 = [track_id, track_undetected_flag(0/1)]
  (occlusion/*.npy 와 동일 패턴; loader 는 track_id 로 매칭)
트랙 단위 판정이므로 같은 track_id 는 시나리오 내 모든 프레임에서 동일 flag.

rule_eval.csv 가 없으면 해당 scen 은 건너뛰고 경고한다
(먼저 scratch_full_occlusion_rule_verify.py --scens <scen> 실행 필요).

사용:  python3 generate_camera_occlusion_sidecar.py [--root ROOT] [--scens s1,s2]
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(_HERE, "dataset")


def process_scen(scen_dir):
    csv_path = os.path.join(scen_dir, "_rule_verify", "rule_eval.csv")
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] rule_eval.csv 없음: {csv_path}")
        return 0, 0
    # stem -> {tid: flag}
    per_stem = defaultdict(dict)
    track_flag = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["tid"])
            flag = row["track_occluded"] == "True"
            per_stem[row["stem"]][tid] = flag
            # 트랙 단위 일관성 검사
            if tid in track_flag and track_flag[tid] != flag:
                raise ValueError(f"track {tid} flag 불일치: {csv_path}")
            track_flag[tid] = flag

    out_dir = os.path.join(scen_dir, "camera_occlusion")
    os.makedirs(out_dir, exist_ok=True)
    for stem, tids in per_stem.items():
        arr = np.array([[tid, 1.0 if fl else 0.0] for tid, fl in sorted(tids.items())],
                       dtype=np.float32).reshape(-1, 2)
        np.save(os.path.join(out_dir, stem + ".npy"), arr)
    n_occ = sum(1 for v in track_flag.values() if v)
    print(f"  {os.path.basename(scen_dir)}: {len(per_stem)}프레임, 트랙 {len(track_flag)}개 "
          f"(미검출 flag=1: {sorted(t for t, v in track_flag.items() if v)})")
    return len(per_stem), n_occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--scens", default=None, help="쉼표구분; 미지정시 root 의 모든 scenNN")
    args = ap.parse_args()

    if args.scens:
        names = args.scens.split(",")
    else:
        names = sorted(n for n in os.listdir(args.root)
                       if n.startswith("scen") and n[4:].isdigit())
    total_f = 0
    for name in names:
        total_f += process_scen(os.path.join(args.root, name))[0]
    print(f"[camera_occlusion] 완료: {len(names)} scen, {total_f} 프레임")


if __name__ == "__main__":
    main()

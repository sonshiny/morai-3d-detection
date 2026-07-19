#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""수정 후 replay 산출물 검사: 게이트 dt, timestamp=ref_src 여부, 라벨 존재."""

import os
import sys
import csv
import statistics as st

SCEN = sys.argv[1] if len(sys.argv) > 1 else "/home/autonav/replay_regress_fix/scen01"


def main():
    with open(os.path.join(SCEN, "sync_log.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["action"] == "save"]

    def vals(k):
        return [float(r[k]) for r in rows if r[k] not in ("", None)]

    print(f"save 프레임: {len(rows)}")
    for k in ("ego_src_dt_ms", "obj_src_dt_ms", "left_src_dt_ms",
              "right_src_dt_ms", "lidar_src_dt_ms"):
        v = vals(k)
        print(f"  {k:16s}: median={st.median(v):+7.1f}  max|dt|={max(abs(x) for x in v):7.1f} ms")

    # ego_pose timestamp == ref_src?
    diffs = []
    for r in rows[:50]:
        stem = r["stem"]
        p = os.path.join(SCEN, "ego_pose", stem + ".csv")
        with open(p) as f:
            ts = float(list(csv.DictReader(f))[0]["timestamp"])
        diffs.append(ts - float(r["ref_src_ts"]))
    print(f"ego_pose.timestamp - ref_src_ts: max|diff|={max(abs(d) for d in diffs):.6f}s "
          f"(0이면 촬영시각 저장 확인)")

    n_lbl = sum(1 for f in os.listdir(os.path.join(SCEN, "labels_3d")) if f.endswith(".csv"))
    n_obj = 0
    for f in os.listdir(os.path.join(SCEN, "labels_3d")):
        with open(os.path.join(SCEN, "labels_3d", f)) as fh:
            n_obj += max(0, len(fh.readlines()) - 1)
    print(f"labels_3d: {n_lbl}개 파일, 총 객체 행 {n_obj}")


if __name__ == "__main__":
    main()

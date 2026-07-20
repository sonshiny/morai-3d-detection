#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전 시나리오 sync_log 오염 서명 스캔: ego/obj src_dt, 카메라 배달지연."""

import os
import csv
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "dataset")


def med(rows, k):
    v = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
    return st.median(v) if v else float("nan")


def main():
    for d in sorted(os.listdir(ROOT)):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        s = os.path.join(p, "sync_log.csv")
        if not os.path.isfile(s):
            print(f"{d:16s}: sync_log 없음 (구파이프라인/오프라인 생성)")
            continue
        with open(s) as f:
            rows = [r for r in csv.DictReader(f) if r["action"] == "save"]
        if not rows:
            print(f"{d:16s}: save 0행")
            continue
        ego_dt = med(rows, "ego_src_dt_ms")
        obj_dt = med(rows, "obj_src_dt_ms")
        cam_lag = med(rows, "ref_recv_ts") - med(rows, "ref_src_ts")
        ego_mn = min(float(r["ego_src_dt_ms"]) for r in rows if r["ego_src_dt_ms"])
        obj_mn = min(float(r["obj_src_dt_ms"]) for r in rows if r["obj_src_dt_ms"])
        bad = "  <-- 오염" if (abs(ego_dt) > 200 or abs(obj_dt) > 200) else ""
        print(f"{d:16s}: n={len(rows):4d} ego_dt(med/min)={ego_dt:9.0f}/{ego_mn:9.0f}ms "
              f"obj_dt={obj_dt:11.0f}/{obj_mn:11.0f}ms cam_lag={cam_lag:7.1f}s{bad}")


if __name__ == "__main__":
    main()

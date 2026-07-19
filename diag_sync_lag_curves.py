#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scen17 sync_log: 토픽별 심시각 어긋남(src-ref_src)·배달지연(recv-src) 곡선."""

import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(HERE, "dataset", "scen17", "sync_log.csv")

TOPICS = ["ego", "obj", "left", "right", "lidar"]


def main():
    with open(SL) as f:
        rows = [r for r in csv.DictReader(f) if r["action"] == "save"]

    def fl(r, k):
        return float(r[k]) if r[k] not in ("", None) else np.nan

    t0 = fl(rows[0], "ref_src_ts")
    print(f"save 프레임: {len(rows)},  ref_src t0 = {t0:.3f} (epoch)")
    print("\n원시 스탬프 크기 확인 (프레임 0): 같은 epoch 계열인지")
    r0 = rows[0]
    print(f"  ref_src ={fl(r0,'ref_src_ts'):.3f}  ref_recv={fl(r0,'ref_recv_ts'):.3f}")
    for t in TOPICS:
        print(f"  {t:5s}_src={fl(r0, t+'_src_ts'):.3f}  {t}_recv={fl(r0, t+'_recv_ts'):.3f}")

    print("\n프레임별 곡선 (10프레임 간격):")
    hdr = (f"{'f':>4s} {'ref_src-t0':>10s} {'cam recv-src':>12s} "
           + "".join(f"{t+' src-ref':>12s}" for t in TOPICS)
           + "".join(f"{t+' recv-src':>13s}" for t in ("ego", "obj")))
    print(hdr)
    for i, r in enumerate(rows):
        if i % 10 and i != len(rows) - 1:
            continue
        rs, rr = fl(r, "ref_src_ts"), fl(r, "ref_recv_ts")
        line = f"{int(r['frame_id']):4d} {rs - t0:10.2f} {rr - rs:12.2f}"
        for t in TOPICS:
            line += f"{fl(r, t + '_src_ts') - rs:12.2f}"
        for t in ("ego", "obj"):
            line += f"{fl(r, t + '_recv_ts') - fl(r, t + '_src_ts'):13.2f}"
        print(line)

    print("\n요약 (src - ref_src, 초):")
    for t in TOPICS:
        d = np.array([fl(r, t + "_src_ts") - fl(r, "ref_src_ts") for r in rows])
        print(f"  {t:5s}: median={np.nanmedian(d):+9.3f}  min={np.nanmin(d):+9.3f}  "
              f"max={np.nanmax(d):+9.3f}")
    print("요약 (recv - src = 배달지연, 초):")
    for t in ["ref"] + TOPICS:
        if t == "ref":
            d = np.array([fl(r, "ref_recv_ts") - fl(r, "ref_src_ts") for r in rows])
        else:
            d = np.array([fl(r, t + "_recv_ts") - fl(r, t + "_src_ts") for r in rows])
        print(f"  {t:5s}: median={np.nanmedian(d):+9.3f}  min={np.nanmin(d):+9.3f}  "
              f"max={np.nanmax(d):+9.3f}")


if __name__ == "__main__":
    main()

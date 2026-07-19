#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scen17 sync_log에서 lidar recv gap(ref_recv 대비) 분포 실측.

주의: scen17은 receive 모드 = lidar를 _latest_recv(결정시점 최신)로 골랐다.
따라서 이 분포는 'latest 선택의 나이'이며, nearest 선택의 gap 상한 근거로 쓴다
(nearest는 정의상 latest-나이보다 작거나 같은 |gap|을 고른다).
"""

import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

for scen in ("scen17",):
    p = os.path.join(HERE, "dataset", scen, "sync_log.csv")
    rows = [r for r in csv.DictReader(open(p)) if r["action"] == "save"]
    g = []
    for r in rows:
        if r["lidar_recv_ts"] and r["ref_recv_ts"]:
            g.append(float(r["lidar_recv_ts"]) - float(r["ref_recv_ts"]))
    g = np.array(g) * 1000.0
    print(f"{scen}: n={len(g)}  lidar_recv - ref_recv (ms, latest 선택):")
    print(f"  mean={g.mean():+.1f} p50={np.percentile(g,50):+.1f} "
          f"p1={np.percentile(g,1):+.1f} p99={np.percentile(g,99):+.1f} "
          f"min={g.min():+.1f} max={g.max():+.1f}  |g| max={np.abs(g).max():.1f}")

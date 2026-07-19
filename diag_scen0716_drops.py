#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scen07~16 빈 수집의 drop 사유 집계."""

import os
import csv
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

for d in [f"scen{i:02d}" for i in range(7, 17)]:
    p = os.path.join(ROOT, d, "sync_log.csv")
    if not os.path.isfile(p):
        continue
    rows = list(csv.DictReader(open(p)))
    c = Counter((r["drop_reason"] or "").split(":")[0]
                for r in rows if r["action"] == "drop")
    lag = ""
    for r in rows:
        if r["ref_src_ts"] and r["ref_recv_ts"]:
            lag = f"cam_lag(first)={float(r['ref_recv_ts']) - float(r['ref_src_ts']):+.1f}s"
            break
    print(f"{d}: rows={len(rows):5d} drops={dict(c)} {lag}")

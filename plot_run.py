#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_run.py — training_history.csv 로 loss + P/R/F1 곡선을 그린다.

train.py 를 import 하지 않으므로(순수 csv+matplotlib) **도는 학습을 전혀 건드리지 않고**
언제든 실행할 수 있다. 첫 validation 이후 training_history.csv 가 생기면 사용 가능.

사용:
  python3 plot_run.py                      # runs/ 하위 최신 run 자동 선택
  python3 plot_run.py runs/prod_v3_depth_20260726
  python3 plot_run.py <RUN_DIR> -o out.png
"""
import os
import csv
import sys
import glob
import math
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_run_dir(arg):
    if arg:
        return arg.rstrip("/")
    cands = sorted(glob.glob("runs/*/training_history.csv"), key=os.path.getmtime)
    if not cands:
        raise SystemExit("[plot_run] training_history.csv 를 못 찾음 — 첫 validation(epoch1) 전이거나 "
                         "RUN_DIR 을 인자로 지정하세요.")
    return os.path.dirname(cands[-1])


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(rows, name):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, "")))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _has_data(ys):
    return any(math.isfinite(y) for y in ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None, help="RUN_DIR (생략시 runs/ 최신)")
    ap.add_argument("-o", "--out", default=None, help="출력 png (기본 RUN_DIR/metrics_overview.png)")
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)
    csv_path = os.path.join(run_dir, "training_history.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"[plot_run] 없음: {csv_path} (첫 validation 후 생성됨)")
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"[plot_run] 빈 CSV: {csv_path}")

    ep = col(rows, "epoch")
    # 성능(P/R/F1) 전용 2패널.
    fig, (ax_prf, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))

    # (좌) val P/R/F1 — primary (softcalibrated @0.15, early-stop 기준)
    for c, lbl in (("softcalibrated_precision_015", "precision"),
                   ("softcalibrated_recall_015", "recall"),
                   ("softcalibrated_f1_015", "f1")):
        ys = col(rows, c)
        if _has_data(ys):
            ax_prf.plot(ep, ys, marker="o", lw=1.8, label=lbl)
    ax_prf.set_title("Val P/R/F1 - softcalibrated @0.15 (primary / early-stop metric)")
    ax_prf.set_xlabel("epoch"); ax_prf.set_ylabel("score"); ax_prf.set_ylim(bottom=0)
    ax_prf.grid(alpha=.3); ax_prf.legend()

    # (우) val F1 — score threshold 별 (softcalibrated)
    for thr in ("005", "010", "015", "025"):
        ys = col(rows, f"softcalibrated_f1_{thr}")
        if _has_data(ys):
            # suffix "005" 는 int 5 → score_thr 0.05 (0.005 아님). 라벨을 실제 값으로.
            ax_f1.plot(ep, ys, marker=".", lw=1.5, label=f"f1@{int(thr) / 100:.2f}")
    ax_f1.set_title("Val F1 by score threshold (softcalibrated)")
    ax_f1.set_xlabel("epoch"); ax_f1.set_ylabel("f1"); ax_f1.set_ylim(bottom=0)
    ax_f1.grid(alpha=.3); ax_f1.legend()

    fig.suptitle(f"{os.path.basename(run_dir)} — {len(rows)} validations", fontsize=13)
    fig.tight_layout()
    out = args.out or os.path.join(run_dir, "metrics_overview.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    best_f1 = max((v for v in col(rows, "softcalibrated_f1_015") if math.isfinite(v)), default=float("nan"))
    print(f"[plot_run] 저장: {out}  ({len(rows)} epochs, best softcal f1@0.15={best_f1:.4f})")


if __name__ == "__main__":
    main()

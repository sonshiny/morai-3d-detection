#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decision_dashboard.py — training_history.csv 로 "계속/중단" 판단 지표를 뽑는다.
학습 프로세스와 독립(csv 읽기만). epoch 8~10 관찰용.

epoch 별로:
  primary   : softcalibrated f1@0.15 (고정 실사용 지표, early-stop 기준)
  raw@0.25  : raw f1@0.25
  best_f1   : 전 mode(cal/soft/raw) × 전 threshold 중 최고 F1 + 그 위치 (threshold-독립 검출력)
  R-S gap   : 같은 최적 threshold 부근 raw vs soft 최고 F1 차 (calibration 의존도)
그리고 안정성(최적 threshold 위치가 epoch마다 얼마나 흔들리는지)을 표시.

사용: python3 decision_dashboard.py [RUN_DIR]   (생략시 runs/ 최신)
"""
import os
import csv
import sys
import glob

MODES = ("calibrated", "softcalibrated", "raw")
THRS = ("001", "003", "005", "010", "015", "025")


def find_run(arg):
    if arg:
        return arg.rstrip("/")
    c = sorted(glob.glob("runs/*/training_history.csv"), key=os.path.getmtime)
    if not c:
        raise SystemExit("training_history.csv 없음 — RUN_DIR 지정")
    return os.path.dirname(c[-1])


def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main():
    run = find_run(sys.argv[1] if len(sys.argv) > 1 else None)
    rows = list(csv.DictReader(open(os.path.join(run, "training_history.csv"), newline="")))
    if not rows:
        raise SystemExit("빈 CSV")

    print(f"\n=== 의사결정 대시보드: {os.path.basename(run)} ({len(rows)} epochs) ===")
    print(f"{'ep':>3} {'primary@15':>10} {'raw@0.25':>9} {'best_f1':>8} {'best_at':>16} {'R-S gap':>8}")
    prev_at = None
    best_at_hist = []
    overall_best = (-1.0, None, None)
    for r in rows:
        ep = int(f(r, "epoch"))
        primary = f(r, "softcalibrated_f1_015")
        raw025 = f(r, "raw_f1_025")
        # 전 mode×threshold 최고 F1 + 위치
        best = (-1.0, None, None)
        soft_best = -1.0
        raw_best = -1.0
        for m in MODES:
            for t in THRS:
                v = f(r, f"{m}_f1_{t}")
                if v == v and v > best[0]:   # not nan
                    best = (v, m, t)
                if m == "softcalibrated" and v == v:
                    soft_best = max(soft_best, v)
                if m == "raw" and v == v:
                    raw_best = max(raw_best, v)
        best_at = f"{best[1]}@{int(best[2]) / 100:.2f}" if best[1] else "n/a"
        rs_gap = (raw_best - soft_best)
        stab = "" if prev_at is None else ("  ← 위치 이동" if best_at != prev_at else "  (위치 유지)")
        print(f"{ep:>3} {primary:>10.4f} {raw025:>9.4f} {best[0]:>8.4f} {best_at:>16} {rs_gap:>+8.4f}{stab}")
        prev_at = best_at
        best_at_hist.append(best_at)
        if best[0] > overall_best[0]:
            overall_best = (best[0], ep, best_at)

    # 판단 힌트
    print("\n--- 판단 힌트 ---")
    print(f"전체 최고 F1(threshold-독립): {overall_best[0]:.4f} @ epoch {overall_best[1]} ({overall_best[2]})")
    last3 = best_at_hist[-3:]
    print(f"최근 최적 threshold 위치: {last3}  → {'안정' if len(set(last3)) == 1 else '진동(calibration 불안정)'}")
    prim = [f(r, 'softcalibrated_f1_015') for r in rows]
    print(f"primary@15 최근 3: {[round(x,4) for x in prim[-3:]]}  (최고 {max(prim):.4f} @ epoch {prim.index(max(prim))+1})")
    print("\n[계속 근거] 전체최고F1 신기록 갱신 / 최적 threshold 2~3ep 연속 유지 / R-S gap 축소")
    print("[중단 근거] 5 validation 동안 primary·전체최고F1 신기록 없음 + threshold 계속 진동 + quality 계속 상승")


if __name__ == "__main__":
    main()

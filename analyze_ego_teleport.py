#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_ego_teleport.py
=======================
ego 텔레포트(리포지션) 경계를 검출해 temporal 학습용 연속 시퀀스를 비파괴적으로
segment 로 분할한다. scene_info.json 의 글로벌 ego pose + 실측 timestamp 사용.

정책 (코덱스 검토 반영 — dt>0.3 hard-split 폐기):
  HARD SPLIT (seq 무결성 필수):
    - implied_speed > SPEED_MAX (기본 30 m/s)  ← 정상 p99.9=17.3, 텔레포트 400~6000 → 유일 필수
    - dt <= 0 (timestamp 비단조; 방어적)
  SOFT/RESET (선택 — instance bank가 max_time_interval=2.0s 로 이미 런타임 자동 리셋):
    - dt > RESET_DT (기본 2.0s) → segment는 이어가되 reset 지점으로 표기
  REPORT ONLY (분할 아님):
    - dt > REPORT_DT (기본 0.3s) 지터 갭 카운트

핵심: bank의 _get_temporal_memory 가 dt<=0 / dt>2.0 은 이미 리셋하므로,
bank가 못 막는 건 "정상 dt인데 speed>30(리포지션)"뿐 → speed>30 이 필수 hard-split.

산출물(--write): dataset/<scen>/segments.json (offset[=정렬 stem idx] 기준 [start,end] 연속 구간).
기본은 dry-run(리포트만).
"""
import os
import csv
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, "dataset")

SPEED_MAX = 30.0     # m/s (hard split)
RESET_DT = 2.0       # s (soft reset, = model max_time_interval)
REPORT_DT = 0.3      # s (report only)


def scene_frames(scen_dir):
    """scene_info.json → (stems, ego_xy[N,2], ts[N]) frame_id 정렬."""
    p = os.path.join(scen_dir, "scene_info.json")
    if not os.path.isfile(p):
        return None
    fr = sorted(json.load(open(p))["frames"], key=lambda f: f["frame_id"])
    stems = [f["stem"] for f in fr]
    xy = np.array([[f["ego"]["x"], f["ego"]["y"]] for f in fr], dtype=np.float64)
    ts = np.array([f["timestamp"] for f in fr], dtype=np.float64)
    return stems, xy, ts


def find_boundaries(xy, ts, speed_max=SPEED_MAX):
    """hard-split 경계 offset 집합 반환. boundary b = frame b-1 -> b 사이가 불법."""
    if len(ts) < 2:
        return [], {}, np.zeros(0), np.zeros(0)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(ts)
    speed = d / np.clip(dt, 1e-6, None)
    # boundary at index i (프레임 i가 새 segment 시작). json 직렬화 위해 python int로.
    hard = [int(x) for x in (np.where((speed > speed_max) | (dt <= 0))[0] + 1)]
    stats = {
        "speed_gt_max": int((speed > speed_max).sum()),
        "dt_le_0": int((dt <= 0).sum()),
        "dt_gt_reset": int((dt > RESET_DT).sum()),
        "dt_gt_report": int((dt > REPORT_DT).sum()),
        "speed_p999": float(np.percentile(speed, 99.9)) if speed.size else 0.0,
        "speed_max": float(speed.max()) if speed.size else 0.0,
    }
    return list(hard), stats, speed, dt


def segments_from_boundaries(n, boundaries):
    """경계 offset 리스트 → [start,end] 연속 구간(끝 포함)."""
    cuts = sorted(set(boundaries))
    segs = []
    start = 0
    for c in cuts:
        segs.append([start, c - 1])
        start = c
    segs.append([start, n - 1])
    return segs


def max_obj_speed_at(scen_dir, stems, i):
    """경계 i(프레임 i-1 -> i)에서 객체 글로벌 위치로 계산한 최대 이동속도(velocity 오염 점검)."""
    def load(stem):
        p = os.path.join(scen_dir, "labels_3d_v2", stem + ".csv")
        d = {}
        if os.path.isfile(p):
            for r in csv.DictReader(open(p)):
                d[int(float(r["track_id"]))] = (float(r["gx"]), float(r["gy"]), float(r["timestamp"]),
                                                float(r.get("vx_ego", 0.0)), float(r.get("vy_ego", 0.0)))
        return d
    a, b = load(stems[i - 1]), load(stems[i])
    best = 0.0; best_rawv = 0.0
    for tid, (gx, gy, tb, vx, vy) in b.items():
        if tid in a:
            gax, gay, ta, _, _ = a[tid]
            dt = max(tb - ta, 1e-6)
            best = max(best, np.hypot(gx - gax, gy - gay) / dt)
            best_rawv = max(best_rawv, np.hypot(vx, vy))
    return best, best_rawv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-max", type=float, default=SPEED_MAX)
    ap.add_argument("--seq-len", type=int, default=150, help="현재 chunk 길이(경계 crossing 집계용)")
    ap.add_argument("--write", action="store_true", help="segments.json 저장(기본 dry-run)")
    ap.add_argument("--scens", default=None)
    args = ap.parse_args()

    only = set(args.scens.split(",")) if args.scens else None
    scens = sorted(
        (n for n in os.listdir(DATASET_ROOT)
         if n.startswith("scen") and n[4:].isdigit()),
        key=lambda s: int(s[4:]),
    )
    if only:
        scens = [s for s in scens if s in only]

    tot = dict(frames=0, hard=0, speed=0, dt0=0, reset=0, report=0,
               scen_with_tp=0, seg=0, crossing=0)
    seg_lens = []
    tp_examples = []
    for scen in scens:
        sd = os.path.join(DATASET_ROOT, scen)
        fr = scene_frames(sd)
        if fr is None:
            continue
        stems, xy, ts = fr
        n = len(stems)
        bnds, st, speed, dt = find_boundaries(xy, ts, args.speed_max)
        segs = segments_from_boundaries(n, bnds)
        seg_lens += [e - s + 1 for s, e in segs]

        # 현재 150-chunk 내부를 가로지르는 경계 수
        crossing = sum(1 for b in bnds if (b // args.seq_len) == ((b - 1) // args.seq_len))
        tot["frames"] += n
        tot["hard"] += len(bnds)
        tot["speed"] += st["speed_gt_max"]
        tot["dt0"] += st["dt_le_0"]
        tot["reset"] += st["dt_gt_reset"]
        tot["report"] += st["dt_gt_report"]
        tot["seg"] += len(segs)
        tot["crossing"] += crossing
        if bnds:
            tot["scen_with_tp"] += 1
            for b in bnds:
                osp, orv = max_obj_speed_at(sd, stems, b)
                tp_examples.append((scen, stems[b - 1], stems[b], float(dt[b - 1]),
                                    float(speed[b - 1]), osp, orv))

        if args.write:
            reset_pts = [int(i + 1) for i in range(len(dt)) if dt[i] > RESET_DT]
            out = {
                "scenario": scen, "n_frames": n,
                "policy": {"speed_max": args.speed_max, "hard": "speed>max or dt<=0",
                           "reset_dt": RESET_DT, "report_dt": REPORT_DT},
                "segments": segs,                    # offset(정렬 stem idx) 기준 [start,end]
                "hard_boundaries": [int(b) for b in bnds],
                "reset_points": reset_pts,           # bank 자동리셋 지점(참고)
            }
            json.dump(out, open(os.path.join(sd, "segments.json"), "w"), indent=2)

    segl = np.array(seg_lens)
    print("=" * 78)
    print(f"  ego 텔레포트 분석 | scen {len(scens)} | frames {tot['frames']}")
    print(f"  정책: HARD split = speed>{args.speed_max}m/s OR dt<=0")
    print("=" * 78)
    print(f"  hard boundaries : {tot['hard']}  (speed>{args.speed_max}: {tot['speed']}, dt<=0: {tot['dt0']})")
    print(f"  텔레포트 포함 scen : {tot['scen_with_tp']} / {len(scens)}")
    print(f"  현재 150-chunk 내부 crossing : {tot['crossing']}  ← 지금 켜면 오염되는 경계 수")
    print(f"  segments 생성 : {tot['seg']}  | 길이 median {int(np.median(segl))} "
          f"min {int(segl.min())} p10 {int(np.percentile(segl,10))} max {int(segl.max())}")
    print(f"  [참고] soft-reset(dt>{RESET_DT}s, bank 자동처리): {tot['reset']} | "
          f"report-only(dt>{REPORT_DT}s): {tot['report']}")
    print("-" * 78)
    print("  텔레포트 경계 (scen, prev, cur, dt, ego_speed, obj_globalspeed, obj_rawv):")
    for e in sorted(tp_examples, key=lambda x: -x[4])[:12]:
        print(f"    {e[0]} {e[1]}->{e[2]} dt={e[3]:.3f}s ego={e[4]:.0f}m/s "
              f"objGlobal={e[5]:.0f}m/s objRawV={e[6]:.1f}m/s")
    print(f"\n  write={args.write} (segments.json {'저장됨' if args.write else 'dry-run'})")


if __name__ == "__main__":
    main()

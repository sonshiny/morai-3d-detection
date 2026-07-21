#!/usr/bin/env python3
"""
scratch_full_occlusion_rule_verify.py  (완전가림 제거 규칙 검증 — 파이프라인 미수정)
====================================================================================
규칙:  제거 <=> (radial_dist <= 50m) AND (라이다 npts == 0) AND (YOLO 트랙 다수결 미검출>70%)
이 규칙의 오제거/놓침을 scen02~06 전 박스 3신호 라벨링으로 검증한다.

3신호:
  - 라이다 npts       : occlusion/<stem>.npy  (프레임 단위)
  - YOLO 트랙 다수결   : yolo11m.pt COCO, conf 0.5, IoU 0.3, ped<->bicycle 개방,
                        커버된 in-range 프레임 중 미검출 비율 > 70% -> 트랙 가림
  - z-buffer ratio    : visibility/<stem>.npz (amodal 과대평가 감안, 참고용)

산출: <scen>/_rule_verify/rule_eval.csv, removed_*.jpg(제거 박스 전수 오버레이), 콘솔 리포트.
기존 함수 재사용: scratch_scen02_camera_occlusion_filter(F), generate_occlusion_gt,
verify_lidar_camera_overlay.project_body. 파이프라인 파일 수정 없음.
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import cv2
from ultralytics import YOLO

from camera_configs import CAM_ORDER, INTRINSICS
from verify_lidar_camera_overlay import project_body
from generate_occlusion_gt import read_boxes_v2, load_lidar_body
import scratch_scen02_camera_occlusion_filter as F

ROOT = os.environ.get("DATASET_ROOT", "/home/jang/dataset")
RANGE_M = 50.0
MISS_THR = 0.70
NEIGHBOR_K = 3          # 근처 프레임 라이다 확인 범위(±)
ZBUF_VIS = 0.5          # 제거 박스인데 zbuf가 이 이상이면 위험 flag
RISK2_NPTS_MAX = 10     # 유지 박스 경계 flag: 1<=npts<=10
RISK2_ZBUF = 0.10


def load_visibility_map(scen_dir, stem):
    p = os.path.join(scen_dir, "visibility", stem + ".npz")
    if not os.path.isfile(p):
        return {}
    d = np.load(p, allow_pickle=True)
    return {int(d["track_id"][i]): float(d["best_visible_ratio"][i])
            for i in range(len(d["track_id"]))}


def bearing_deg(b):
    import math
    return math.degrees(math.atan2(b["cy"], b["cx"]))


def process_scen(scen, model):
    scen_dir = os.path.join(ROOT, scen)
    lbl_dir = os.path.join(scen_dir, "labels_3d_v2")
    occ_dir = os.path.join(scen_dir, "occlusion")
    out_dir = os.path.join(scen_dir, "_rule_verify")
    os.makedirs(out_dir, exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv"))

    rows = []                      # per-box records
    yolo_cache = {}                # stem -> {cam: dets}
    track_frames = defaultdict(list)   # tid -> [(stem, covered, frame_match, in_range)]
    npts_seq = defaultdict(dict)       # tid -> {stem_idx: npts}

    for si, stem in enumerate(stems):
        boxes = read_boxes_v2(os.path.join(lbl_dir, stem + ".csv"))
        occ = np.load(os.path.join(occ_dir, stem + ".npy"))
        npts_map = {int(r[0]): int(r[1]) for r in occ} if occ.ndim == 2 and occ.shape[0] else {}
        vis_map = load_visibility_map(scen_dir, stem)

        dets_by_cam = {}
        for cam in CAM_ORDER:
            img = cv2.imread(os.path.join(scen_dir, "images", cam, stem + ".jpg"))
            dets_by_cam[cam] = F.run_yolo(model, img) if img is not None else []
        yolo_cache[stem] = dets_by_cam

        for b in boxes:
            tid = int(b["track_id"]); cls = int(b["class_id"])
            rd = float(np.hypot(b["cx"], b["cy"]))
            in_range = rd <= RANGE_M
            npts = npts_map.get(tid, -1)
            zbuf = vis_map.get(tid, float("nan"))
            covered = False; frame_match = False; best_cam = None; best_area = -1
            allowed = F.gt_allowed_groups(cls)
            for cam in CAM_ORDER:
                cov, bb, gc = F.project_gt_to_camera(b, cam)
                if not cov:
                    continue
                covered = True
                area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                if area > best_area:
                    best_area = area; best_cam = cam
                im, ig, biou, cm, cg = F.match_box(bb, gc, dets_by_cam[cam], allowed)
                if im:
                    frame_match = True
            rows.append(dict(scen=scen, stem=stem, si=si, tid=tid, cls=cls, rd=rd,
                             in_range=in_range, npts=npts, zbuf=zbuf,
                             covered=covered, frame_match=frame_match,
                             best_cam=best_cam, box=b))
            npts_seq[tid][si] = npts
            if in_range:
                track_frames[tid].append((stem, covered, frame_match))
        if (si + 1) % 50 == 0:
            print(f"    {scen}: {si + 1}/{len(stems)}")

    # YOLO 트랙 다수결
    track_miss = {}
    for tid, fr in track_frames.items():
        cov = [f for f in fr if f[1]]
        if not cov:
            track_miss[tid] = (None, True)      # 커버 0 -> 판단불가, 보수적으로 '가림' 취급하되 표기
            continue
        miss = sum(1 for f in cov if not f[2]) / len(cov)
        track_miss[tid] = (miss, miss > MISS_THR)

    # 규칙 적용 + 위험 flag
    for r in rows:
        miss, occluded = track_miss.get(r["tid"], (None, False))
        r["track_miss"] = miss
        r["track_occluded"] = occluded
        r["removed"] = bool(r["in_range"] and r["npts"] == 0 and occluded)
        # 위험1 후보 신호
        near = [npts_seq[r["tid"]].get(r["si"] + d, None)
                for d in range(-NEIGHBOR_K, NEIGHBOR_K + 1) if d != 0]
        r["neighbor_lidar_max"] = max([n for n in near if n is not None] + [0])
        same_frame = [x for x in rows if x["stem"] == r["stem"] and x["in_range"]
                      and abs((bearing_deg(x["box"]) - bearing_deg(r["box"]) + 180) % 360 - 180) < 15.0]
        r["nearest_front"] = bool(same_frame and min(same_frame, key=lambda x: x["rd"]) is r)
    return rows, track_miss, stems, yolo_cache, out_dir


def render_removed(rows, yolo_cache, scen_dir, out_dir):
    """제거 판정 박스 전수 오버레이: 프레임당 1장(박스별 최대투영 캠 기준으로 그룹)."""
    lidar_dir = os.path.join(scen_dir, "lidar")
    by_frame_cam = defaultdict(list)
    for r in rows:
        if r["removed"] and r["best_cam"]:
            by_frame_cam[(r["stem"], r["best_cam"])].append(r)
    n_img = 0
    for (stem, cam), rs in sorted(by_frame_cam.items()):
        img = cv2.imread(os.path.join(scen_dir, "images", cam, stem + ".jpg"))
        if img is None:
            continue
        vis = img.copy()
        # YOLO
        for d in yolo_cache[stem][cam]:
            x1, y1, x2, y2 = [int(t) for t in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 1)
            cv2.putText(vis, f"{d['coco']} {d['conf']:.2f}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1, cv2.LINE_AA)
        # 라이다 전체 투영(참고: occluder 확인용) — 박스 2D 영역 내 점만
        lp = os.path.join(lidar_dir, stem + ".npy")
        p_body = load_lidar_body(lp) if os.path.isfile(lp) else np.zeros((0, 3))
        if p_body.shape[0]:
            ph = np.concatenate([p_body, np.ones((p_body.shape[0], 1))], axis=1)
            K = INTRINSICS[cam].astype(np.float64)
            u, v, dep, val = project_body(ph, cam, K, 0.1)
        for r in rs:
            cov, bb, gc = F.project_gt_to_camera(r["box"], cam)
            if not cov:
                continue
            x1, y1, x2, y2 = [int(t) for t in bb]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 235), 2)
            if p_body.shape[0]:
                inb = val & (u >= bb[0]) & (u <= bb[2]) & (v >= bb[1]) & (v <= bb[3])
                occl = inb & (dep < r["rd"] - 1.5)          # 박스보다 앞 = 가림체 표면
                atbox = inb & (np.abs(dep - r["rd"]) <= 1.5)
                for uu, vv in zip(u[occl][::3], v[occl][::3]):
                    cv2.circle(vis, (int(uu), int(vv)), 2, (255, 0, 255), -1)
                for uu, vv in zip(u[atbox], v[atbox]):
                    cv2.circle(vis, (int(uu), int(vv)), 3, (0, 255, 0), -1)
            zb = "nan" if np.isnan(r["zbuf"]) else f"{r['zbuf']:.2f}"
            tag = (f"tid{r['tid']} d{r['rd']:.0f}m np{r['npts']} zb{zb} "
                   f"trkmiss{r['track_miss']:.0%} REMOVED")
            cv2.putText(vis, tag, (x1, min(895, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 235), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{stem} {cam} | magenta=lidar(occluder surface) green=lidar(at box depth)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"removed_{stem}_{cam}.jpg"), vis)
        n_img += 1
    return n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scens", default="scen02,scen03,scen04,scen05,scen06")
    args = ap.parse_args()
    scens = args.scens.split(",")

    model = YOLO(F.WEIGHTS)
    all_rows = []
    all_miss = {}

    for scen in scens:
        print(f"[{scen}] 처리 중...")
        rows, track_miss, stems, yolo_cache, out_dir = process_scen(scen, model)
        scen_dir = os.path.join(ROOT, scen)
        n_img = render_removed(rows, yolo_cache, scen_dir, out_dir)
        # CSV
        with open(os.path.join(out_dir, "rule_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stem", "tid", "cls", "rd", "in_range", "npts", "zbuf",
                        "covered", "frame_match", "track_miss", "track_occluded",
                        "removed", "neighbor_lidar_max", "nearest_front"])
            for r in rows:
                w.writerow([r["stem"], r["tid"], r["cls"], f"{r['rd']:.2f}", r["in_range"],
                            r["npts"], "" if np.isnan(r["zbuf"]) else f"{r['zbuf']:.3f}",
                            r["covered"], r["frame_match"],
                            "" if r["track_miss"] is None else f"{r['track_miss']:.3f}",
                            r["track_occluded"], r["removed"],
                            r["neighbor_lidar_max"], r["nearest_front"]])
        print(f"  오버레이 {n_img}장, CSV 저장 -> {out_dir}")
        all_rows.extend(rows)
        all_miss[scen] = track_miss

    # ================= 리포트 =================
    print("\n" + "=" * 100)
    print("  트랙 YOLO 다수결 (in-range 커버 프레임 기준)")
    print("=" * 100)
    for scen in scens:
        parts = []
        for tid, (miss, occ) in sorted(all_miss[scen].items()):
            m = "n/a" if miss is None else f"{miss:.0%}"
            parts.append(f"tid{tid}:{m}{'[가림]' if occ else ''}")
        print(f"  {scen}: " + "  ".join(parts))

    removed = [r for r in all_rows if r["removed"]]
    print("\n" + "=" * 100)
    print(f"  제거 판정 박스: {len(removed)}개")
    print("=" * 100)
    by_st = defaultdict(int)
    for r in removed:
        by_st[(r["scen"], r["tid"])] += 1
    for (scen, tid), n in sorted(by_st.items()):
        sample = [r for r in removed if r["scen"] == scen and r["tid"] == tid]
        zb = [x["zbuf"] for x in sample if not np.isnan(x["zbuf"])]
        print(f"  {scen} tid{tid}: {n}박스  zbuf범위 {min(zb):.2f}-{max(zb):.2f}" if zb
              else f"  {scen} tid{tid}: {n}박스")

    # 위험1: 제거인데 실제 보일 가능성
    print("\n" + "=" * 100)
    print(f"  [위험1] 제거 박스 중 가시 가능성 flag (zbuf>={ZBUF_VIS} or 최전면 or ±{NEIGHBOR_K}프레임 lidar>0)")
    print("=" * 100)
    r1 = [r for r in removed if
          (not np.isnan(r["zbuf"]) and r["zbuf"] >= ZBUF_VIS) or r["nearest_front"]
          or r["neighbor_lidar_max"] > 0]
    if not r1:
        print("  없음")
    for r in r1:
        print(f"  {r['scen']} {r['stem']} tid{r['tid']} d={r['rd']:.1f} zbuf={r['zbuf']:.2f} "
              f"neighbor_np={r['neighbor_lidar_max']} front={r['nearest_front']} "
              f"frame_yolo={'O' if r['frame_match'] else 'X'}")

    # 위험2: 유지인데 사실상 가림 경계
    print("\n" + "=" * 100)
    print(f"  [위험2] 유지 박스 중 경계 flag (1<=npts<={RISK2_NPTS_MAX} & zbuf<{RISK2_ZBUF})"
          f" + (npts==0 & 트랙은 가시 -> AND 불발로 유지된 프레임)")
    print("=" * 100)
    r2a = [r for r in all_rows if r["in_range"] and not r["removed"]
           and 1 <= r["npts"] <= RISK2_NPTS_MAX
           and not np.isnan(r["zbuf"]) and r["zbuf"] < RISK2_ZBUF]
    r2b = [r for r in all_rows if r["in_range"] and not r["removed"] and r["npts"] == 0
           and not r["track_occluded"]]
    print(f"  (a) 라이다 소수점+zbuf<0.1 유지: {len(r2a)}건")
    for r in r2a[:12]:
        print(f"      {r['scen']} {r['stem']} tid{r['tid']} np={r['npts']} zbuf={r['zbuf']:.2f} "
              f"frame_yolo={'O' if r['frame_match'] else 'X'} trkmiss={r['track_miss']:.0%}")
    print(f"  (b) npts==0 & 트랙가시 -> 유지: {len(r2b)}건")
    for r in r2b[:12]:
        zb = "nan" if np.isnan(r["zbuf"]) else f"{r['zbuf']:.2f}"
        print(f"      {r['scen']} {r['stem']} tid{r['tid']} cls{r['cls']} d={r['rd']:.1f} zbuf={zb} "
              f"frame_yolo={'O' if r['frame_match'] else 'X'} trkmiss={r['track_miss']:.0%}")

    print("\n산출물: <scen>/_rule_verify/{rule_eval.csv, removed_*.jpg}")


if __name__ == "__main__":
    main()

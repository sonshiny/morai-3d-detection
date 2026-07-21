#!/usr/bin/env python3
"""
scratch_scen02_camera_occlusion_filter.py  (C안 2단계 검증 — 학습 파이프라인 미수정)
====================================================================================
카메라 기반 가림 필터를 scen02 에서 검증한다. 완전가림 track 0·1 을 제거하면서
가시 객체(track 2 <=50m, 5/6/7/8)는 보존하는지로 로직 정확성을 확인한다.

절대 학습 파이프라인 파일(train.py / morai_dataset.py 등)을 수정하지 않는다.
기존 기하/투영 함수만 재사용:
  camera_configs.CAM_ORDER / INTRINSICS
  verify_lidar_camera_overlay.project_body / _BOX_EDGES
  generate_visibility_gt.box_corners_from_v2
  generate_occlusion_gt.read_boxes_v2

파라미터(확정):
  인지범위 50m (radial_dist>50m 는 필터 이전에 제외)
  YOLO conf 0.5, 가중치 = DAECHANGMO/Msg/YOLOWeight/yolo11m.pt (COCO 사전학습)
  클래스매핑: COCO car/bus/truck -> vehicle(0), person -> pedestrian(1)
  bicycle 는 검증 단계에서 pedestrian 과 매칭 가능하도록 열어둠(집계만)
  IoU 매칭 임계 0.3 (+ 중심-포함 매칭 병행 계산)
  카메라축: 투영되는 카메라 중 한 대라도 매칭되면 그 프레임은 '가시'
  시간축: track 단위, 커버된 프레임 중 미검출 비율>70% 이면 그 track '가림'
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import cv2
from ultralytics import YOLO

from camera_configs import CAM_ORDER, INTRINSICS
from verify_lidar_camera_overlay import project_body, _BOX_EDGES
from generate_visibility_gt import box_corners_from_v2
from generate_occlusion_gt import read_boxes_v2

# ---------------- 확정 파라미터 ----------------
RANGE_M = 50.0
CONF = 0.5
IOU_THR = 0.3
MISS_RATE_OCCLUDED = 0.70
MIN_DEPTH = 0.1
MIN_FRONT_CORNERS = 2
IMG_W, IMG_H = 1600, 900

WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11m.pt")  # COCO 사전학습; 없으면 ultralytics 자동 다운로드

# COCO id -> 매칭 그룹
COCO_VEHICLE = {2, 5, 7}          # car, bus, truck
COCO_PERSON = {0}                  # person
COCO_BICYCLE = {1}                 # bicycle
YOLO_CLASSES = [0, 1, 2, 5, 7]     # 요청 클래스만 추론
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 5: "bus", 7: "truck"}

# GT class -> 매칭 허용 그룹
def gt_allowed_groups(class_id):
    if class_id == 0:
        return {"vehicle"}
    return {"person", "bicycle"}   # pedestrian: 둘 다 열어둠

def coco_group(cid):
    if cid in COCO_VEHICLE:
        return "vehicle"
    if cid in COCO_PERSON:
        return "person"
    if cid in COCO_BICYCLE:
        return "bicycle"
    return None

def dist_bucket(d):
    if d < 20.0:
        return "00-20"
    if d < 40.0:
        return "20-40"
    if d <= 50.0:
        return "40-50"
    return ">50"


# ---------------- 기하 ----------------
def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)

def point_in_box(px, py, b):
    return (b[0] <= px <= b[2]) and (b[1] <= py <= b[3])


def project_gt_to_camera(box, camera):
    """box(read_boxes_v2 dict) -> (covered, gt_bbox_xyxy_clipped, gt_center_uv).
    covered: 카메라 앞쪽 코너>=MIN_FRONT_CORNERS 이고 이미지와 교차하면 True."""
    K = INTRINSICS[camera].astype(np.float64)
    corners = box_corners_from_v2(box)                        # (8,3)
    ch = np.concatenate([corners, np.ones((8, 1))], axis=1)
    u, v, depth, valid = project_body(ch, camera, K, MIN_DEPTH)
    front = np.where(valid)[0]
    if front.size < MIN_FRONT_CORNERS:
        return False, None, None
    uu = u[front]; vv = v[front]
    x1 = float(np.min(uu)); y1 = float(np.min(vv))
    x2 = float(np.max(uu)); y2 = float(np.max(vv))
    # 이미지로 클립
    cx1 = max(0.0, x1); cy1 = max(0.0, y1)
    cx2 = min(float(IMG_W), x2); cy2 = min(float(IMG_H), y2)
    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        return False, None, None
    # 중심점: 3D 박스중심 투영(앞쪽이면), 아니면 bbox 중심
    center_h = np.array([[box["cx"], box["cy"], box["cz"], 1.0]])
    cu, cv, cd, cvalid = project_body(center_h, camera, K, MIN_DEPTH)
    if bool(cvalid[0]):
        gc = (float(cu[0]), float(cv[0]))
    else:
        gc = (0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2))
    return True, (cx1, cy1, cx2, cy2), gc


# ---------------- YOLO ----------------
def run_yolo(model, image):
    res = model(image, verbose=False, conf=CONF, classes=YOLO_CLASSES)[0]
    dets = []
    for b in res.boxes:
        cid = int(b.cls[0])
        grp = coco_group(cid)
        if grp is None:
            continue
        x1, y1, x2, y2 = [float(t) for t in b.xyxy[0]]
        dets.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(b.conf[0]),
            "coco": COCO_NAMES.get(cid, str(cid)),
            "group": grp,
        })
    return dets


def match_box(gt_bbox, gt_center, dets, allowed_groups):
    """반환: (iou_matched, iou_group, best_iou, center_matched, center_group)."""
    best_iou = 0.0; iou_group = None
    center_group = None
    for d in dets:
        if d["group"] not in allowed_groups:
            continue
        i = iou_xyxy(gt_bbox, d["bbox"])
        if i > best_iou:
            best_iou = i; iou_group = d["group"]
        if center_group is None and point_in_box(gt_center[0], gt_center[1], d["bbox"]):
            center_group = d["group"]
    iou_matched = best_iou >= IOU_THR
    return iou_matched, (iou_group if iou_matched else None), best_iou, (center_group is not None), center_group


# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen-dir", default="/home/autonav/visibility_test_dataset/scen02")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--viz-stems", default="live_000000,live_000033,live_000060,live_000075,live_000085,live_000090")
    args = ap.parse_args()

    scen_dir = os.path.abspath(args.scen_dir)
    out_dir = args.out_dir or os.path.join(scen_dir, "_cam_occ_filter")
    os.makedirs(out_dir, exist_ok=True)
    viz_stems = set(args.viz_stems.split(","))

    lbl_dir = os.path.join(scen_dir, "labels_3d_v2")
    occ_dir = os.path.join(scen_dir, "occlusion")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv"))

    print(f"YOLO 로드: {WEIGHTS}")
    model = YOLO(WEIGHTS)

    # 누적 구조
    # per (track): list of frame records (in_range only)
    track_class = {}
    track_records = defaultdict(list)   # tid -> list of dict(stem, covered, iou_match, center_match, iou_grp, center_grp, rd)
    # 전 인스턴스(필터영향 집계용)
    instances = []   # dict(stem, tid, class, rd, in_range, npts, covered, iou_match, center_match)
    # bicycle 측정용: pedestrian track 의 그룹 매칭 카운트
    ped_group_counts = defaultdict(lambda: {"person": 0, "bicycle": 0, "either": 0, "covered": 0})

    range_excluded = 0

    for si, stem in enumerate(stems):
        boxes = read_boxes_v2(os.path.join(lbl_dir, stem + ".csv"))
        # occlusion npy -> {tid: npts}
        occ_path = os.path.join(occ_dir, stem + ".npy")
        npts_map = {}
        if os.path.isfile(occ_path):
            arr = np.load(occ_path)
            if arr.ndim == 2 and arr.shape[0]:
                npts_map = {int(r[0]): int(r[1]) for r in arr}

        # YOLO 3카메라 (프레임당 1회)
        yolo = {}
        images = {}
        for cam in CAM_ORDER:
            ip = os.path.join(scen_dir, "images", cam, stem + ".jpg")
            img = cv2.imread(ip)
            images[cam] = img
            yolo[cam] = run_yolo(model, img) if img is not None else []

        for b in boxes:
            tid = int(b["track_id"]); cls = int(b["class_id"])
            rd = float(np.hypot(b["cx"], b["cy"]))
            track_class[tid] = cls
            npts = npts_map.get(tid, -1)
            in_range = rd <= RANGE_M
            if not in_range:
                range_excluded += 1
                instances.append(dict(stem=stem, tid=tid, cls=cls, rd=rd, in_range=False,
                                      npts=npts, covered=False, iou_match=False, center_match=False))
                continue

            allowed = gt_allowed_groups(cls)
            covered_any = False
            iou_match_any = False; center_match_any = False
            iou_grp_hit = None; center_grp_hit = None
            per_cam = {}
            for cam in CAM_ORDER:
                cov, gt_bbox, gt_center = project_gt_to_camera(b, cam)
                if not cov:
                    per_cam[cam] = None
                    continue
                covered_any = True
                im, ig, biou, cm, cg = match_box(gt_bbox, gt_center, yolo[cam], allowed)
                per_cam[cam] = dict(gt_bbox=gt_bbox, gt_center=gt_center,
                                    iou_match=im, iou_grp=ig, best_iou=biou,
                                    center_match=cm, center_grp=cg)
                if im:
                    iou_match_any = True
                    if ig and iou_grp_hit is None:
                        iou_grp_hit = ig
                if cm:
                    center_match_any = True
                    if cg and center_grp_hit is None:
                        center_grp_hit = cg

            track_records[tid].append(dict(stem=stem, covered=covered_any, rd=rd,
                                           iou_match=iou_match_any, center_match=center_match_any,
                                           iou_grp=iou_grp_hit, center_grp=center_grp_hit))
            instances.append(dict(stem=stem, tid=tid, cls=cls, rd=rd, in_range=True,
                                  npts=npts, covered=covered_any,
                                  iou_match=iou_match_any, center_match=center_match_any))

            if cls == 1 and covered_any:
                pc = ped_group_counts[tid]
                pc["covered"] += 1
                if iou_grp_hit == "person" or center_grp_hit == "person":
                    pc["person"] += 1
                if iou_grp_hit == "bicycle" or center_grp_hit == "bicycle":
                    pc["bicycle"] += 1
                if iou_match_any or center_match_any:
                    pc["either"] += 1

            # 시각화
            if stem in viz_stems:
                pass  # 아래 별도 렌더 패스에서 처리

        if (si + 1) % 40 == 0:
            print(f"  진행 {si + 1}/{len(stems)} 프레임")

    # ---------------- track 단위 판정 ----------------
    def verdict_for(records, key):
        cov = [r for r in records if r["covered"]]
        n_cov = len(cov)
        if n_cov == 0:
            return dict(n_frames=len(records), n_cov=0, n_miss=0, miss_rate=None, verdict="NO_CAM_COVERAGE")
        n_miss = sum(1 for r in cov if not r[key])
        mr = n_miss / n_cov
        vd = "OCCLUDED" if mr > MISS_RATE_OCCLUDED else "VISIBLE"
        return dict(n_frames=len(records), n_cov=n_cov, n_miss=n_miss, miss_rate=mr, verdict=vd)

    track_verdict = {}
    for tid, recs in track_records.items():
        track_verdict[tid] = {
            "iou": verdict_for(recs, "iou_match"),
            "center": verdict_for(recs, "center_match"),
            "rd_min": min(r["rd"] for r in recs),
            "rd_max": max(r["rd"] for r in recs),
            "class": track_class[tid],
        }

    # ---------------- CSV: track별 판정 ----------------
    csv_path = os.path.join(out_dir, "track_verdicts.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "class_id", "rd_min_m", "rd_max_m",
                    "n_inrange_frames", "n_covered",
                    "iou_n_miss", "iou_miss_rate", "iou_verdict",
                    "center_n_miss", "center_miss_rate", "center_verdict"])
        for tid in sorted(track_verdict):
            tv = track_verdict[tid]
            io = tv["iou"]; ce = tv["center"]
            w.writerow([tid, tv["class"], f"{tv['rd_min']:.2f}", f"{tv['rd_max']:.2f}",
                        io["n_frames"], io["n_cov"],
                        io["n_miss"], "" if io["miss_rate"] is None else f"{io['miss_rate']:.3f}", io["verdict"],
                        ce["n_miss"], "" if ce["miss_rate"] is None else f"{ce['miss_rate']:.3f}", ce["verdict"]])
    print(f"\n[CSV] track별 판정 -> {csv_path}")

    # ---------------- 리포트: item4 정답 대조 ----------------
    print("\n" + "=" * 92)
    print("  [검증 4] 정답 대조표 (핵심 track)  — 범위 50m, IoU>=0.3, miss>70%->가림")
    print("=" * 92)
    hdr = f"  {'tid':>3} {'cls':>3} {'rd(m)':>12} {'cov':>4} | {'IoU miss%':>9} {'IoU판정':>10} | {'Ctr miss%':>9} {'Ctr판정':>10}  기대"
    print(hdr)
    expect = {0: "가림", 1: "가림", 2: "가시(<=50)", 5: "가시", 6: "가시", 7: "가시", 8: "가시"}
    for tid in [0, 1, 2, 5, 6, 7, 8]:
        if tid not in track_verdict:
            print(f"  {tid:>3}  (없음)")
            continue
        tv = track_verdict[tid]; io = tv["iou"]; ce = tv["center"]
        iou_mr = "-" if io["miss_rate"] is None else f"{100*io['miss_rate']:.0f}%"
        ctr_mr = "-" if ce["miss_rate"] is None else f"{100*ce['miss_rate']:.0f}%"
        print(f"  {tid:>3} {tv['class']:>3} {tv['rd_min']:>5.1f}-{tv['rd_max']:>5.1f} {io['n_cov']:>4} | "
              f"{iou_mr:>9} {io['verdict']:>10} | {ctr_mr:>9} {ce['verdict']:>10}  {expect.get(tid,'')}")

    # ---------------- 리포트: item2 bicycle 측정 ----------------
    print("\n" + "=" * 92)
    print("  [검증 2] bicycle 측정 — pedestrian(class1) track 이 person/bicycle 중 무엇으로 매칭됐나")
    print("=" * 92)
    print(f"  {'tid':>3} {'covered':>7} {'person매칭':>10} {'bicycle매칭':>11} {'either':>7}   (프레임 수)")
    for tid in sorted(ped_group_counts):
        pc = ped_group_counts[tid]
        print(f"  {tid:>3} {pc['covered']:>7} {pc['person']:>10} {pc['bicycle']:>11} {pc['either']:>7}")

    # ---------------- 리포트: item5 필터영향 & 라이다 비교 ----------------
    print("\n" + "=" * 92)
    print("  [검증 5] 카메라 필터가 제거하는 GT (거리구간 x 클래스) — track판정=가림인 in-range 인스턴스")
    print("=" * 92)
    occluded_tracks_iou = {tid for tid, tv in track_verdict.items() if tv["iou"]["verdict"] == "OCCLUDED"}
    occluded_tracks_ctr = {tid for tid, tv in track_verdict.items() if tv["center"]["verdict"] == "OCCLUDED"}
    print(f"  가림판정 track (IoU): {sorted(occluded_tracks_iou)}")
    print(f"  가림판정 track (Center): {sorted(occluded_tracks_ctr)}")

    rem = defaultdict(int); tot = defaultdict(int)
    for ins in instances:
        if not ins["in_range"]:
            continue
        key = (dist_bucket(ins["rd"]), ins["cls"])
        tot[key] += 1
        if ins["tid"] in occluded_tracks_iou:
            rem[key] += 1
    print(f"\n  {'dist':>6} {'class':>6} {'in-range':>9} {'cam제거(IoU)':>12}")
    for bucket in ["00-20", "20-40", "40-50"]:
        for cls in [0, 1]:
            k = (bucket, cls)
            if tot[k]:
                print(f"  {bucket:>6} {cls:>6} {tot[k]:>9} {rem[k]:>12}")

    # 라이다 필터(OCCLUSION_MIN_PTS, npts==0) 와 교차표 (in-range 인스턴스)
    both = lidar_only = cam_only = neither = 0
    lidar_removed_total = cam_removed_total = 0
    inrange_total = 0
    for ins in instances:
        if not ins["in_range"]:
            continue
        inrange_total += 1
        lidar_rm = (ins["npts"] == 0)
        cam_rm = ins["tid"] in occluded_tracks_iou
        lidar_removed_total += int(lidar_rm)
        cam_removed_total += int(cam_rm)
        if lidar_rm and cam_rm:
            both += 1
        elif lidar_rm:
            lidar_only += 1
        elif cam_rm:
            cam_only += 1
        else:
            neither += 1

    print("\n" + "=" * 92)
    print("  [검증 5] 라이다(OCCLUSION_MIN_PTS: npts==0) vs 카메라(track 가림) 교차표 — in-range 인스턴스")
    print("=" * 92)
    print(f"  in-range 인스턴스 총 {inrange_total} | 라이다 제거 {lidar_removed_total} | 카메라 제거 {cam_removed_total}")
    print(f"    둘다 제거    : {both}")
    print(f"    라이다만 제거: {lidar_only}")
    print(f"    카메라만 제거: {cam_only}")
    print(f"    둘다 유지    : {neither}")
    print(f"\n  범위(>50m) 제외 인스턴스: {range_excluded}")

    # ---------------- 시각화 ----------------
    render_viz(scen_dir, out_dir, viz_stems, model, track_verdict)

    print(f"\n[완료] 산출물 디렉토리: {out_dir}")


def render_viz(scen_dir, out_dir, viz_stems, model, track_verdict):
    lbl_dir = os.path.join(scen_dir, "labels_3d_v2")
    for stem in sorted(viz_stems):
        boxes = read_boxes_v2(os.path.join(lbl_dir, stem + ".csv"))
        for cam in CAM_ORDER:
            ip = os.path.join(scen_dir, "images", cam, stem + ".jpg")
            img = cv2.imread(ip)
            if img is None:
                continue
            vis = img.copy()
            dets = run_yolo(model, img)
            # YOLO (얇은 파랑)
            for d in dets:
                x1, y1, x2, y2 = [int(t) for t in d["bbox"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 1)
                cv2.putText(vis, f"{d['coco']} {d['conf']:.2f}", (x1, max(0, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1, cv2.LINE_AA)
            # GT 투영
            for b in boxes:
                tid = int(b["track_id"]); cls = int(b["class_id"])
                rd = float(np.hypot(b["cx"], b["cy"]))
                cov, gt_bbox, gt_center = project_gt_to_camera(b, cam)
                if not cov:
                    continue
                if rd > RANGE_M:
                    color = (140, 140, 140)       # 회색: 범위 제외
                    tag = f"tid{tid} >50m EXCL"
                else:
                    allowed = gt_allowed_groups(cls)
                    im, ig, biou, cmm, cg = match_box(gt_bbox, gt_center, dets, allowed)
                    vd = track_verdict.get(tid, {}).get("iou", {}).get("verdict", "?")
                    if im:
                        color = (0, 220, 0)        # 초록: 이 카메라에서 IoU 매칭
                    else:
                        color = (0, 0, 235)        # 빨강: 미매칭
                    tag = f"tid{tid} c{cls} iou{biou:.2f} [{vd}]"
                x1, y1, x2, y2 = [int(t) for t in gt_bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, tag, (x1, min(IMG_H - 4, y2 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            cv2.putText(vis, f"{stem} {cam} | GT green=match red=miss gray=>50m | yolo=orange",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            op = os.path.join(out_dir, f"{stem}_{cam}.jpg")
            cv2.imwrite(op, vis)


if __name__ == "__main__":
    main()

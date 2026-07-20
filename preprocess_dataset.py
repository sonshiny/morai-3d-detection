#!/usr/bin/env python3
"""
preprocess_dataset.py  (Phase 0 — SparseDrive temporal 이행 전처리)
====================================================================
신규 dataset(scen01~60)의 labels_3d 원본에는 학습에 그대로 쓰면 안 되는 3개의
인코딩 이슈가 있다(생성기 morai_3d_label_generator.py:94-121 대조 + 수치검증 완료).
이 스크립트는 그것들을 일괄 복구하고, temporal 인지 학습에 필요한 트랙 ID와
프레임별 ego->global 변환을 함께 산출한다. 원본 파일은 수정하지 않는다.

[복구 항목]
  1) yaw   : 저장값 rel = (npc_heading - ego_heading) - pi/2  (생성기 line 112).
             복구  yaw_ego   = atan2(sin_yaw, cos_yaw) + pi/2      (ego/body frame)
                   yaw_global = yaw_ego + ego_yaw_rad
  2) vel   : MORAI는 객체-로컬(전방,횡) 속도를 km/h로 주는데 생성기(line 115-117)가
             이를 world로 착각해 R(-ego_yaw)로 회전시켜 저장(버그).
             복구  v_local_kmh = R(+ego_yaw) @ [vx,vy]            (버그 역변환)
                   v_ego_ms    = R(yaw_ego)  @ v_local_kmh / 3.6  (ego/body frame, m/s)
                   v_world_ms  = R(ego_yaw)  @ v_ego_ms           (검증용)
  3) z     : 저장 z는 바닥(ego ground-relative) 기준 -> z_center = z + h/2

[트랙 ID 재생성]
  (object_source, object_index)는 슬롯이라 소멸/생성 시 재사용된다.
  각 슬롯 시계열을 (a)연속 프레임 위치 점프 > gate, 또는 (b)치수 시그니처 변화 시
  잘라 세그먼트로 나누고, 세그먼트 경계가 시공간적으로 이어지면(끝<->시작 gate 내,
  같은 클래스/치수) 병합해 물리 객체 단위 track_id를 부여한다.

[좌표 프레임 결정]
  박스는 ego/body frame 유지(기존 v10 가중치가 이 프레임을 기대 -> 전이학습 보존).
  temporal 전파용 T_global 은 body->global (lidar 오프셋은 상수라 상쇄, 일관성만 유지).

[산출물]  dataset/<scen>/
  labels_3d_v2/live_NNNNNN.csv : frame_id,timestamp,track_id,class_id,
       x,y,z_center,w,l,h,yaw_ego,vx_ego,vy_ego,vz,
       yaw_global,gx,gy,               # 검증/트래킹평가 편의
       sin_yaw_ego,cos_yaw_ego         # 학습 로더 편의(재계산 불필요)
  scene_info.json : 프레임별 {stem, frame_id, timestamp, T_ego2global(4x4), ego(x,y,z,yaw)}
                    + 시나리오 요약(트랙 수, 검증 지표)

사용:  python3 preprocess_dataset.py            # 전체 60 시나리오
       python3 preprocess_dataset.py --scen scen01 scen30   # 일부만
       python3 preprocess_dataset.py --report-only          # 재생성 없이 검증만
"""

import os
import csv
import json
import math
import argparse
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, "dataset")


def _scenario_sort_key(name):
    if name.startswith("scen") and name[4:].isdigit():
        return int(name[4:])
    return 10**9

# 트랙 재연결 게이트 (클래스별) — 프레임수가 아니라 실제 dt 기반 물리속도로 판정
MAX_SPEED = {0: 35.0, 1: 3.5}      # m/s 상한 (vehicle ~126km/h, pedestrian ~12.6km/h)
BASE_GATE = {0: 2.0, 1: 1.0}       # sub-frame 지터 흡수용 최소 허용 이동 (m)
DIM_TOL = 0.30                      # 치수 시그니처 동일 판정 허용오차 (m)
RESET_DT = 2.0                     # 이 시간보다 큰 갭은 무조건 identity 분리(SparseDrive와 동일)
MERGE_MAX_DT = 0.5                  # 세그먼트 병합은 짧은 프레임 드랍(<0.5s)만 보수적으로 연결


# ------------------------------------------------------------------
# 파일 IO
# ------------------------------------------------------------------
def _read_ego_pose(path):
    """ego_pose CSV -> dict. 1행."""
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {
        "frame_id": int(float(row["frame_id"])),
        "timestamp": float(row["timestamp"]),
        "x": float(row["ego_x"]), "y": float(row["ego_y"]), "z": float(row["ego_z"]),
        "yaw": float(row["ego_yaw_rad"]),          # == radians(ego_heading_deg), 검증됨
    }


def _read_labels(path):
    """labels_3d 원본 CSV -> list[dict] (파싱만, 미복구)."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "object_source": row["object_source"],
                "object_index": int(float(row["object_index"])),
                "class_id": int(float(row["class_id"])),
                "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]),
                "w": float(row["w"]), "l": float(row["l"]), "h": float(row["h"]),
                "sin_yaw": float(row["sin_yaw"]), "cos_yaw": float(row["cos_yaw"]),
                "vx": float(row["vx"]), "vy": float(row["vy"]), "vz": float(row["vz"]),
            })
    return out


def _rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


# ------------------------------------------------------------------
# 복구
# ------------------------------------------------------------------
def _decode_object(o, ego):
    """원본 object dict + ego -> 복구된 필드 dict."""
    ego_yaw = ego["yaw"]

    # yaw
    yaw_ego = math.atan2(o["sin_yaw"], o["cos_yaw"])
    yaw_ego = math.atan2(math.sin(yaw_ego), math.cos(yaw_ego))   # wrap [-pi,pi)
    yaw_global = math.atan2(math.sin(yaw_ego + ego_yaw), math.cos(yaw_ego + ego_yaw))

    # velocity: 버그 역변환 -> ego frame m/s
    v_stored = np.array([o["vx"], o["vy"]], dtype=np.float64)
    v_local_kmh = _rot2(+ego_yaw) @ v_stored                     # undo R(-ego_yaw)
    v_ego = _rot2(yaw_ego) @ v_local_kmh / 3.6                    # ego frame, m/s
    v_world = _rot2(ego_yaw) @ v_ego                              # 검증용

    # z: bottom -> center
    z_center = o["z"] + o["h"] / 2.0

    # global position: dp = R(ego_yaw) @ [x_e, y_e]; global = ego + dp
    dp = _rot2(ego_yaw) @ np.array([o["x"], o["y"]], dtype=np.float64)
    gx, gy = ego["x"] + dp[0], ego["y"] + dp[1]

    return {
        "class_id": o["class_id"],
        "x": o["x"], "y": o["y"], "z_center": z_center,
        "w": o["w"], "l": o["l"], "h": o["h"],
        "yaw_ego": yaw_ego, "yaw_global": yaw_global, "ego_yaw": ego_yaw,
        "vx_ego": float(v_ego[0]), "vy_ego": float(v_ego[1]), "vz": o["vz"],
        "v_world": (float(v_world[0]), float(v_world[1])),
        "raw_speed": float(math.hypot(o["vx"], o["vy"])),   # km/h; 0이면 MORAI 미제공
        "vel_source": "raw",                                 # fill 패스에서 갱신될 수 있음
        "gx": float(gx), "gy": float(gy),
        "slot": (o["object_source"], o["object_index"]),
        "dim_sig": (round(o["w"], 2), round(o["l"], 2), round(o["h"], 2)),
    }


def _fill_velocity_from_motion(frames):
    """
    raw 속도가 트랙 전체에서 사실상 0인데 실제로는 이동하는 객체(주로 보행자 —
    MORAI가 속도 미제공)에 대해, 재연결된 global 트랙 위치의 중심차분으로 속도를
    유도해 v_ego/v_world 를 덮어쓴다. 차량(raw 유효)은 그대로 둔다.
    반환: 모션기반으로 채운 트랙 수.
    """
    from collections import defaultdict as _dd
    track_pts = _dd(list)   # tid -> [obj], with (fi,t) available via wrapper
    idx = _dd(list)
    for fi, t, objs in frames:
        for o in objs:
            track_pts[o["track_id"]].append((fi, t, o))

    n_filled = 0
    W = 2   # 중심차분 윈도우(±프레임)
    for tid, pts in track_pts.items():
        pts.sort(key=lambda e: e[0])
        raw_speeds = [o["raw_speed"] for _, _, o in pts]
        disp = math.hypot(pts[-1][2]["gx"] - pts[0][2]["gx"],
                          pts[-1][2]["gy"] - pts[0][2]["gy"]) if len(pts) > 1 else 0.0
        median_raw = float(np.median(raw_speeds)) if raw_speeds else 0.0
        # raw 미제공(중앙값<0.3km/h) 이고 실제로 움직임(>2m) -> 모션기반 유도
        if not (median_raw < 0.3 and disp > 2.0):
            continue
        n = len(pts)
        for i in range(n):
            fi_i, t_i, o_i = pts[i]
            lo = max(0, i - W); hi = min(n - 1, i + W)
            # dt 갭이 큰 경계는 안쪽으로 좁힘
            while lo < i and (t_i - pts[lo][1]) > 1.0:
                lo += 1
            while hi > i and (pts[hi][1] - t_i) > 1.0:
                hi -= 1
            if hi == lo:
                vwx = vwy = 0.0
            else:
                dt = pts[hi][1] - pts[lo][1]
                vwx = (pts[hi][2]["gx"] - pts[lo][2]["gx"]) / dt
                vwy = (pts[hi][2]["gy"] - pts[lo][2]["gy"]) / dt
            v_ego = _rot2(-o_i["ego_yaw"]) @ np.array([vwx, vwy], dtype=np.float64)
            o_i["vx_ego"], o_i["vy_ego"] = float(v_ego[0]), float(v_ego[1])
            o_i["v_world"] = (vwx, vwy)
            o_i["vel_source"] = "motion"
        n_filled += 1
    return n_filled


# ------------------------------------------------------------------
# 트랙 재생성
# ------------------------------------------------------------------
def _dims_match(a, b):
    return (abs(a[0] - b[0]) <= DIM_TOL and
            abs(a[1] - b[1]) <= DIM_TOL and
            abs(a[2] - b[2]) <= DIM_TOL)


def _reassign_tracks(frames):
    """
    frames: list[ (frame_idx, timestamp, [decoded objects]) ]  시간순.
    각 decoded object 에 'track_id' 를 in-place 부여. 반환: 트랙 수, 통계.
    """
    # 1) 슬롯별 시계열 수집
    slot_seq = defaultdict(list)   # slot -> list[(fi, t, obj)]
    for fi, t, objs in frames:
        for o in objs:
            slot_seq[o["slot"]].append((fi, t, o))

    # 2) 슬롯 시계열을 점프/치수변화 지점에서 세그먼트로 분할
    segments = []  # 각 seg: dict(objs=[(fi,t,o)...], class_id, dim_sig, start,end,...)
    for slot, seq in slot_seq.items():
        seq.sort(key=lambda e: e[0])
        cur = []
        for i, (fi, t, o) in enumerate(seq):
            if not cur:
                cur = [(fi, t, o)]
                continue
            pfi, pt, po = cur[-1]
            cls = o["class_id"]
            dist = math.hypot(o["gx"] - po["gx"], o["gy"] - po["gy"])
            dt = max(t - pt, 1e-3)
            # 허용 이동 = max(지터 흡수 최소치, 물리속도상한 * 실제dt)
            allowed = max(BASE_GATE.get(cls, 2.0), MAX_SPEED.get(cls, 35.0) * dt)
            broke = (dt > RESET_DT) or (dist > allowed) \
                    or (not _dims_match(o["dim_sig"], po["dim_sig"])) \
                    or (cls != po["class_id"])
            if broke:
                segments.append(cur)
                cur = [(fi, t, o)]
            else:
                cur.append((fi, t, o))
        if cur:
            segments.append(cur)

    # 3) 세그먼트 병합: 끝점->다음 시작점이 시공간적으로 이어지면 같은 트랙
    #    (같은 클래스, 치수 일치, dt<=MERGE_MAX_DT, 예측 위치 gate 내)
    def seg_info(seg):
        fis = [e[0] for e in seg]
        return {
            "seg": seg, "class_id": seg[0][2]["class_id"],
            "dim_sig": seg[0][2]["dim_sig"],
            "start_fi": fis[0], "end_fi": fis[-1],
            "start": seg[0], "end": seg[-1],
        }
    infos = [seg_info(s) for s in segments]
    infos.sort(key=lambda a: a["start_fi"])
    n = len(infos)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        ei_fi, ei_t, ei_o = infos[i]["end"]
        best = None
        for j in range(n):
            if i == j:
                continue
            sj_fi, sj_t, sj_o = infos[j]["start"]
            if sj_fi <= ei_fi:                       # j 는 i 끝 이후 시작해야
                continue
            if infos[j]["class_id"] != infos[i]["class_id"]:
                continue
            if not _dims_match(infos[j]["dim_sig"], infos[i]["dim_sig"]):
                continue
            dt = sj_t - ei_t
            if dt <= 0 or dt > MERGE_MAX_DT:         # 짧은 프레임드랍만 연결
                continue
            # ei 의 속도로 예측한 위치 vs sj 실제 위치, 물리속도 상한으로 게이트
            px = ei_o["gx"] + ei_o["v_world"][0] * dt
            py = ei_o["gy"] + ei_o["v_world"][1] * dt
            dist = math.hypot(sj_o["gx"] - px, sj_o["gy"] - py)
            allowed = max(BASE_GATE.get(infos[i]["class_id"], 2.0),
                          MAX_SPEED.get(infos[i]["class_id"], 35.0) * dt)
            if dist <= allowed:
                if best is None or dist < best[1]:
                    best = (j, dist)
        if best is not None:
            union(i, best[0])

    # 4) track_id 부여
    root_to_tid = {}
    next_tid = 0
    for i in range(n):
        r = find(i)
        if r not in root_to_tid:
            root_to_tid[r] = next_tid
            next_tid += 1
        tid = root_to_tid[r]
        for (fi, t, o) in infos[i]["seg"]:
            o["track_id"] = tid

    return next_tid, {"n_segments": n, "n_slots": len(slot_seq)}


# ------------------------------------------------------------------
# 검증 지표
# ------------------------------------------------------------------
def _track_quality(frames):
    """트랙별 치수 편차, 연속프레임 위치 점프, 속도 vs 유한차분 잔차."""
    track_pts = defaultdict(list)   # tid -> [(fi, t, gx, gy, dims, v_world)]
    for fi, t, objs in frames:
        for o in objs:
            track_pts[o["track_id"]].append(
                (fi, t, o["gx"], o["gy"], (o["w"], o["l"], o["h"]), o["v_world"]))

    dim_spreads, pos_jumps, vel_resid = [], [], []
    for tid, pts in track_pts.items():
        pts.sort(key=lambda e: e[0])
        ws = [p[4][0] for p in pts]; ls = [p[4][1] for p in pts]; hs = [p[4][2] for p in pts]
        if len(pts) > 1:
            dim_spreads.append(max(max(ws) - min(ws), max(ls) - min(ls), max(hs) - min(hs)))
        for k in range(1, len(pts)):
            fi0, t0, x0, y0, _, v0 = pts[k - 1]
            fi1, t1, x1, y1, _, _ = pts[k]
            if fi1 - fi0 != 1:
                continue
            d = math.hypot(x1 - x0, y1 - y0)
            pos_jumps.append(d)
            dt = t1 - t0
            if dt > 1e-3:
                fd = np.array([(x1 - x0) / dt, (y1 - y0) / dt])   # world 유한차분 속도
                vel_resid.append(float(np.linalg.norm(fd - np.array(v0))))
    def stats(a):
        if not a:
            return {"n": 0}
        a = np.array(a)
        return {"n": int(a.size), "median": float(np.median(a)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    return {
        "n_tracks": len(track_pts),
        "dim_spread_m": stats(dim_spreads),
        "pos_jump_m": stats(pos_jumps),
        "vel_resid_ms": stats(vel_resid),
    }


# ------------------------------------------------------------------
# 시나리오 처리
# ------------------------------------------------------------------
LIDAR_DZ = 1.35   # camera_configs.LIDAR_TRANSLATION z (body frame)


def _T_ego2global(ego):
    T = np.eye(4, dtype=np.float64)
    c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3], T[2, 3] = ego["x"], ego["y"], ego["z"]
    return T


def process_scenario(scen_dir, write=True):
    lbl_dir = os.path.join(scen_dir, "labels_3d")
    ego_dir = os.path.join(scen_dir, "ego_pose")
    if not (os.path.isdir(lbl_dir) and os.path.isdir(ego_dir)):
        return None
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv"))

    frames = []          # (frame_idx, timestamp, [decoded])
    scene_frames = []    # scene_info 프레임 엔트리
    for stem in stems:
        ego_path = os.path.join(ego_dir, stem + ".csv")
        if not os.path.isfile(ego_path):
            continue
        ego = _read_ego_pose(ego_path)
        raw = _read_labels(os.path.join(lbl_dir, stem + ".csv"))
        decoded = [_decode_object(o, ego) for o in raw]
        frames.append((ego["frame_id"], ego["timestamp"], decoded))
        scene_frames.append({
            "stem": stem, "frame_id": ego["frame_id"], "timestamp": ego["timestamp"],
            "T_ego2global": _T_ego2global(ego).tolist(),
            "ego": {"x": ego["x"], "y": ego["y"], "z": ego["z"], "yaw": ego["yaw"]},
        })

    frames.sort(key=lambda e: e[0])
    n_tracks, seg_stats = _reassign_tracks(frames)
    n_vel_filled = _fill_velocity_from_motion(frames)
    seg_stats["n_vel_motion_tracks"] = n_vel_filled
    quality = _track_quality(frames)

    if write:
        out_dir = os.path.join(scen_dir, "labels_3d_v2")
        os.makedirs(out_dir, exist_ok=True)
        cols = ["frame_id", "timestamp", "track_id", "class_id",
                "x", "y", "z_center", "w", "l", "h", "yaw_ego",
                "vx_ego", "vy_ego", "vz", "yaw_global", "gx", "gy",
                "sin_yaw_ego", "cos_yaw_ego", "vel_source"]
        # frame_idx -> timestamp
        for fi, t, objs in frames:
            stem = f"live_{fi:06d}"
            with open(os.path.join(out_dir, stem + ".csv"), "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(cols)
                for o in objs:
                    wr.writerow([
                        fi, f"{t:.6f}", o["track_id"], o["class_id"],
                        f"{o['x']:.4f}", f"{o['y']:.4f}", f"{o['z_center']:.4f}",
                        f"{o['w']:.4f}", f"{o['l']:.4f}", f"{o['h']:.4f}",
                        f"{o['yaw_ego']:.6f}",
                        f"{o['vx_ego']:.4f}", f"{o['vy_ego']:.4f}", f"{o['vz']:.4f}",
                        f"{o['yaw_global']:.6f}", f"{o['gx']:.4f}", f"{o['gy']:.4f}",
                        f"{math.sin(o['yaw_ego']):.6f}", f"{math.cos(o['yaw_ego']):.6f}",
                        o["vel_source"],
                    ])
        scene_info = {
            "scenario": os.path.basename(scen_dir),
            "n_frames": len(frames),
            "n_tracks": n_tracks,
            "lidar_dz_body": LIDAR_DZ,
            "quality": quality,
            "seg_stats": seg_stats,
            "frames": scene_frames,
        }
        with open(os.path.join(scen_dir, "scene_info.json"), "w", encoding="utf-8") as f:
            json.dump(scene_info, f)

    return {"scenario": os.path.basename(scen_dir), "n_frames": len(frames),
            "n_tracks": n_tracks, **seg_stats, "quality": quality}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", nargs="*", default=None, help="처리할 시나리오(미지정=전체)")
    ap.add_argument("--report-only", action="store_true", help="파일 생성 없이 검증만")
    ap.add_argument("--root", default=DATASET_ROOT)
    args = ap.parse_args()

    if args.scen:
        scens = args.scen
    else:
        scens = sorted(
            (
                d for d in os.listdir(args.root)
                if os.path.isdir(os.path.join(args.root, d))
                and d.startswith("scen")
                and d[4:].isdigit()
            ),
            key=_scenario_sort_key,
        )

    print(f"{'scen':8s} {'frames':>7s} {'tracks':>7s} {'segs':>6s} "
          f"{'dimSpr(med/p95/max)':>22s} {'posJmp(med/p95/max)':>22s} {'velResid(med/p95/max)':>24s}")
    agg = {"frames": 0, "tracks": 0, "dim_max": [], "jump_p95": [], "vel_p95": []}
    for scen in scens:
        scen_dir = os.path.join(args.root, scen)
        r = process_scenario(scen_dir, write=not args.report_only)
        if r is None:
            print(f"{scen:8s}  (skip: no labels/ego)")
            continue
        q = r["quality"]
        ds, pj, vr = q["dim_spread_m"], q["pos_jump_m"], q["vel_resid_ms"]
        def fmt(s):
            return "n=0" if s["n"] == 0 else f"{s['median']:.2f}/{s['p95']:.2f}/{s['max']:.2f}"
        print(f"{scen:8s} {r['n_frames']:7d} {r['n_tracks']:7d} {r['n_segments']:6d} "
              f"{fmt(ds):>22s} {fmt(pj):>22s} {fmt(vr):>24s}")
        agg["frames"] += r["n_frames"]; agg["tracks"] += r["n_tracks"]
        if ds["n"]: agg["dim_max"].append(ds["max"])
        if pj["n"]: agg["jump_p95"].append(pj["p95"])
        if vr["n"]: agg["vel_p95"].append(vr["p95"])

    print("-" * 110)
    print(f"TOTAL frames={agg['frames']}  tracks={agg['tracks']}")
    if agg["dim_max"]:
        print(f"  치수편차 max(시나리오별)  최악={max(agg['dim_max']):.3f}m  (트랙 내 물리 일관성; <0.3 목표)")
    if agg["jump_p95"]:
        print(f"  위치점프 p95(시나리오별)  최악={max(agg['jump_p95']):.3f}m  (연속프레임; <3 목표)")
    if agg["vel_p95"]:
        print(f"  속도잔차 p95(시나리오별)  최악={max(agg['vel_p95']):.3f}m/s (복구속도 vs 유한차분; <0.5 목표)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
correct_source_time.py  (P4 — front-camera source-time GT correction)
======================================================================
labels_3d(원본) / ego_pose 파일의 timestamp 는 front-camera ref_src_ts 이지만,
좌표 계산에는 sync_log 가 그 프레임에 매칭한 **별도의** Ego/Object 메시지가 쓰였다.
즉 저장 박스는

    r_stored = R(-yaw_e(t_e)) @ (p_o(t_o) - p_e(t_e))            (t_e, t_o = ego/obj src time)

인 반면, 라벨 시각(front-camera) 기준의 정답은

    r_ref    = R(-yaw_e(t_ref)) @ (p_o(t_ref) - p_e(t_ref))      (t_ref = ref_src_ts)

이다. ego/obj 는 t_ref 대비 평균 +30~46ms 미래로 치우쳐 있고(상관 0.86~0.89),
그 시차는 zero-mean white noise 가 아니라 미래 방향 편향 + 시간 상관을 갖는다.
따라서 (v_obj-v_ego)*dt 나 along-track-only 근사는 t_e≈t_o 이고 ego 직진일 때만 맞는다.

이 스크립트는 위 물리식을 그대로 구현해 GT 와 scene_info 를 t_ref 기준으로
**일관되게** 재구성한다. 원본(labels_3d, ego_pose, sync_log.csv)과 기존
labels_3d_v2 / scene_info.json 은 절대 건드리지 않고, versioned 산출만 만든다.

알고리즘 (프레임마다)
  1) sync_log 의 action=save 행에서  t_ref=ref_src_ts, t_e=ego_src_ts, t_o=obj_src_ts.
  2) ego_pose 값 = t_e 상태 → (t_e, ego_state) 표를 시나리오 전체에서 수집.
  3) 같은 t_e 반복 시 상태 일치 검증 후 dedup.
  4) ego x/y/z 선형보간, yaw 는 unwrap 후 선형보간 → t_ref pose. 경계는 명시적
     최대 extrap/gap + correction_valid 플래그로 처리(조용한 raw fallback 금지).
  5) 저장 상대 x,y + t_e ego pose 로 object world center(t_o)를 복원(= 기존 decode gx,gy).
  6) _decode_object 의 velocity 규약으로 object world velocity 획득.
  7) raw velocity 객체:  p_o(t_ref) = p_o(t_o) + v_world*(t_ref - t_o)  (z 도 vz 로 동일).
     비 raw(motion/unknown) 객체: object self-motion 은 전파하지 않고 ego 프레임 변화만 반영.
  8) corrected x/y/z 는 t_ref ego pose 기준으로 변환, global yaw 는 t_e 로 복원 후
     t_ref yaw 기준으로 재표현, vx/vy 는 같은 world velocity 를 t_ref ego frame 으로 회전.
  9) global gx/gy, track 재할당, 품질 지표는 시간보정 이후에 계산.
 10) scene_info_v3 의 T_ego2global / ego pose 는 반드시 t_ref 보간 pose.

class/dimensions/frame·stem 수는 보정 때문에 바꾸지 않는다.

산출물 dataset/<scen>/
  labels_3d_v3/live_NNNNNN.csv : labels_3d_v2 와 동일 스키마
       + corr_dx,corr_dy,corr_dz,corr_dist,correction_valid  (박스별 감사)
  scene_info_v3.json           : 프레임별 t_ref pose + timing 감사 블록 + 시나리오 요약
  timing_correction_report.json: 시나리오 correction 통계(감사)

사용:  python3 correct_source_time.py --scen scen05 scen77 scen144
       python3 correct_source_time.py --report-only          # 파일 생성 없이 통계만
"""

import os
import csv
import json
import math
import argparse
import shutil
import tempfile
from collections import defaultdict

import numpy as np

# 검증된 decode/track 규약을 그대로 재사용한다(재구현 금지 — v2 와 동일 규약 보존).
from preprocess_dataset import (
    _read_ego_pose,
    _read_labels,
    _decode_object,
    _reassign_tracks,
    _fill_velocity_from_motion,
    _track_quality,
    _rot2,
    _T_ego2global,
    LIDAR_DZ,
    _scenario_sort_key,
    DATASET_ROOT,
)

# ego 보간 경계 정책 (조용한 raw fallback 금지 — 초과분은 correction_valid=0 으로 명시)
MAX_INTERP_GAP = 0.5    # s : 브래킷 두 샘플 간격이 이보다 크면 correction_valid=0
MAX_EXTRAP = 0.2        # s : 표 범위 밖 외삽이 이보다 크면 correction_valid=0
DUP_XY_TOL = 1e-3       # m : 같은 t_e 중복 시 위치 일치 허용오차
DUP_YAW_TOL = 1e-4      # rad
VZ_KMH_TO_MS = 1.0 / 3.6  # vz 는 수평 속도와 같은 km/h 규약(문서화된 가정; z 보정 <3.4cm 로 무시가능)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ang_close(a, b, tol):
    return abs(_wrap(a - b)) <= tol


# ------------------------------------------------------------------
# sync_log
# ------------------------------------------------------------------
def _read_sync_saves(scen_dir):
    """sync_log.csv 의 action=save 행 -> {stem: {t_ref,t_e,t_o,epoch_id}}.
    드랍 행은 라벨 파일이 없으므로 무시한다."""
    path = os.path.join(scen_dir, "sync_log.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("action") != "save":
                continue
            try:
                out[r["stem"]] = {
                    "t_ref": float(r["ref_src_ts"]),
                    "t_e": float(r["ego_src_ts"]),
                    "t_o": float(r["obj_src_ts"]),
                    "epoch_id": r.get("epoch_id", ""),
                }
            except (KeyError, ValueError):
                continue
    return out


# ------------------------------------------------------------------
# ego source-time 보간 표
# ------------------------------------------------------------------
class EgoSourceInterp:
    """(t_e, ego_state) 표. x/y/z 선형, yaw unwrap+선형 보간. 중복 t_e dedup."""

    def __init__(self, samples):
        # samples: list[(t_e, x, y, z, yaw)]
        by_te = {}
        self.n_conflict = 0
        for t_e, x, y, z, yaw in samples:
            key = round(float(t_e), 6)
            if key in by_te:
                px, py, pz, pyaw = by_te[key]
                if (abs(px - x) > DUP_XY_TOL or abs(py - y) > DUP_XY_TOL
                        or abs(pz - z) > DUP_XY_TOL or not _ang_close(pyaw, yaw, DUP_YAW_TOL)):
                    self.n_conflict += 1
                    raise ValueError(
                        "동일 ego_src_ts에 서로 다른 pose가 있습니다: "
                        f"t={key:.6f}, first=({px:.6f},{py:.6f},{pz:.6f},{pyaw:.9f}), "
                        f"next=({x:.6f},{y:.6f},{z:.6f},{yaw:.9f})")
                continue  # 첫 값 유지(대표 데이터는 conflict=0 검증됨)
            by_te[key] = (x, y, z, yaw)
        ts = sorted(by_te.keys())
        self.ts = np.array(ts, dtype=np.float64)
        self.xs = np.array([by_te[t][0] for t in ts], dtype=np.float64)
        self.ys = np.array([by_te[t][1] for t in ts], dtype=np.float64)
        self.zs = np.array([by_te[t][2] for t in ts], dtype=np.float64)
        raw_yaw = np.array([by_te[t][3] for t in ts], dtype=np.float64)
        self.yaws = np.unwrap(raw_yaw) if raw_yaw.size > 1 else raw_yaw
        self.n_unique = len(ts)

    def at(self, t):
        """t_ref 에서의 (x,y,z,yaw) 와 감사 메타 반환."""
        ts = self.ts
        n = ts.size
        if n == 0:
            return None, {"method": "empty", "gap": float("inf"), "extrap": float("inf")}
        if n == 1:
            return (float(self.xs[0]), float(self.ys[0]), float(self.zs[0]), _wrap(float(self.yaws[0]))), \
                   {"method": "single", "gap": 0.0, "extrap": abs(float(t - ts[0]))}
        i = int(np.searchsorted(ts, t))
        if i <= 0:
            lo, hi = 0, 1
            method = "extrap_low" if t < ts[0] else "interp"
        elif i >= n:
            lo, hi = n - 2, n - 1
            method = "extrap_high"
        else:
            lo, hi = i - 1, i
            method = "interp"
        t0, t1 = ts[lo], ts[hi]
        dt = t1 - t0
        frac = 0.0 if dt <= 0 else (t - t0) / dt

        def _ip(a):
            return float(a[lo] + (a[hi] - a[lo]) * frac)

        x, y, z = _ip(self.xs), _ip(self.ys), _ip(self.zs)
        yaw = _wrap(_ip(self.yaws))
        extrap = 0.0
        if t < ts[0]:
            extrap = float(ts[0] - t)
        elif t > ts[-1]:
            extrap = float(t - ts[-1])
        return (x, y, z, yaw), {"method": method, "gap": float(dt), "extrap": extrap}


class EpochedEgoInterp:
    """epoch_id(또는 명시적 segment) 별로 **독립** EgoSourceInterp 를 유지한다.

    task F(리포지션/텔레포트 안전장치): 서로 다른 epoch/segment 의 ego 표본을 절대
    한 보간 브래킷에 섞지 않는다. 표본이 있는 epoch 안에서만 x/y/z 선형 + yaw
    unwrap 보간이 일어나고, epoch 경계를 가로지르는 interpolation/extrapolation 은
    구조적으로 불가능하다. 조회한 epoch 에 표본이 없으면(=경계) extrap=inf 인 invalid
    meta 를 돌려주어 correction_valid=0 로 명시된다(조용한 cross-epoch fallback 금지).

    단일 epoch(대표 데이터: 모두 epoch_id='0')이면 표 하나만 만들어지므로 기존
    EgoSourceInterp 단일표 동작과 완전히 동일하다(bitwise no-op).
    """

    def __init__(self, samples):
        # samples: list[(t_e, x, y, z, yaw, epoch_id)]
        by_epoch = defaultdict(list)
        for s in samples:
            t_e, x, y, z, yaw, ep = s
            by_epoch[ep].append((t_e, x, y, z, yaw))
        self.interps = {ep: EgoSourceInterp(v) for ep, v in by_epoch.items()}
        self.epochs = sorted(by_epoch.keys(), key=str)
        self.n_epochs = len(self.interps)
        self.n_unique = sum(i.n_unique for i in self.interps.values())
        self.n_conflict = sum(i.n_conflict for i in self.interps.values())

    def at(self, t, epoch_id):
        interp = self.interps.get(epoch_id)
        if interp is None:
            # 해당 epoch 에 ego 표본이 전혀 없음 → 경계. 다른 epoch 로 넘어가 보간/외삽하지 않는다.
            return None, {"method": "no_epoch_samples", "gap": float("inf"),
                          "extrap": float("inf")}
        return interp.at(t)


def _interp_valid(meta):
    if meta["method"] in ("empty", "no_epoch_samples", "no_sync"):
        return False
    if meta["extrap"] > MAX_EXTRAP:
        return False
    if meta["gap"] > MAX_INTERP_GAP:
        return False
    return True


# ------------------------------------------------------------------
# object 보정
# ------------------------------------------------------------------
def _correct_object(dec, ego_te, ego_ref, t_ref, t_o):
    """decoded object(dec) 를 t_ref front-camera 시각 기준으로 재구성(in-place).
    ego_te: t_e 상태 dict(x,y,z,yaw) — decode 에 쓰인 원본 ego.
    ego_ref: (x,y,z,yaw) — t_ref 보간 ego pose.
    corr_dx/dy/dz/dist(감사)를 dec 에 추가한다."""
    ex_r, ey_r, ez_r, yaw_r = ego_ref

    # 저장 상대 x,y + t_e ego pose 로 복원한 object world center(t_o) = decode 의 gx,gy
    p_o_o = np.array([dec["gx"], dec["gy"]], dtype=np.float64)
    gz_o = ego_te["z"] + dec["z_center"]
    v_world = np.array(dec["v_world"], dtype=np.float64)   # m/s, frame-independent
    vz_ms = dec["vz"] * VZ_KMH_TO_MS
    dt_o = t_ref - t_o

    if dec.get("vel_source", "raw") == "raw":
        p_o_ref = p_o_o + v_world * dt_o
        gz_ref = gz_o + vz_ms * dt_o
    else:
        # motion-derived/unknown velocity 는 self-motion 전파에 신뢰하지 않는다.
        # ego 프레임 변화(t_e->t_ref)만 반영(object world 위치는 t_o 로 고정).
        p_o_ref = p_o_o.copy()
        gz_ref = gz_o

    x0, y0, z0 = dec["x"], dec["y"], dec["z_center"]     # 보정 전(원본 decode) 값

    rel = _rot2(-yaw_r) @ (p_o_ref - np.array([ex_r, ey_r], dtype=np.float64))
    x2, y2 = float(rel[0]), float(rel[1])
    z2 = float(gz_ref - ez_r)

    yaw_global = dec["yaw_global"]                        # world heading(t_o), 각속도 미제공→상수
    yaw_ego2 = _wrap(yaw_global - yaw_r)
    v_ego2 = _rot2(-yaw_r) @ v_world

    dec["x"], dec["y"], dec["z_center"] = x2, y2, z2
    dec["yaw_ego"] = yaw_ego2
    dec["ego_yaw"] = yaw_r                                # 이제 t_ref ego frame 기준
    dec["vx_ego"], dec["vy_ego"] = float(v_ego2[0]), float(v_ego2[1])
    dec["gx"], dec["gy"] = float(p_o_ref[0]), float(p_o_ref[1])
    dec["corr_dx"] = x2 - x0
    dec["corr_dy"] = y2 - y0
    dec["corr_dz"] = z2 - z0
    dec["corr_dist"] = math.hypot(x2 - x0, y2 - y0)
    return dec


# ------------------------------------------------------------------
# 시나리오 처리
# ------------------------------------------------------------------
def process_scenario(scen_dir, write=True):
    lbl_dir = os.path.join(scen_dir, "labels_3d")
    ego_dir = os.path.join(scen_dir, "ego_pose")
    if not (os.path.isdir(lbl_dir) and os.path.isdir(ego_dir)):
        return None

    sync = _read_sync_saves(scen_dir)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".csv"))

    # 1) 프레임 로드 + t_e 상태 표 수집
    frame_records = []       # dict per stem
    ego_samples = []         # (t_e, x, y, z, yaw)
    for stem in stems:
        ego_path = os.path.join(ego_dir, stem + ".csv")
        if not os.path.isfile(ego_path):
            raise FileNotFoundError(f"라벨과 대응하는 ego_pose가 없습니다: {ego_path}")
        ego = _read_ego_pose(ego_path)                   # x,y,z,yaw = t_e 상태
        raw = _read_labels(os.path.join(lbl_dir, stem + ".csv"))
        decoded = [_decode_object(o, ego) for o in raw]  # t_e/t_o 기준(미보정)
        tm = sync.get(stem)
        if tm is not None:
            ego_samples.append((tm["t_e"], ego["x"], ego["y"], ego["z"], ego["yaw"],
                                tm.get("epoch_id", "")))
        frame_records.append({
            "stem": stem, "frame_id": ego["frame_id"], "ego_te": ego,
            "timing": tm, "decoded": decoded,
        })

    # epoch/segment 경계를 넘는 보간 금지(task F). 단일 epoch 데이터에선 no-op.
    interp = EpochedEgoInterp(ego_samples)

    # 2) 프레임별 t_ref 보간 + object 보정
    frames = []              # (frame_id, t_ref, [decoded])  — 트래킹/품질용
    scene_frames = []
    n_no_sync = 0
    method_counts = defaultdict(int)
    frame_valid_counts = {"valid": 0, "invalid": 0}
    for rec in frame_records:
        tm = rec["timing"]
        ego_te = rec["ego_te"]
        decoded = rec["decoded"]
        if tm is None:
            # sync timing 부재: 보정 불가 → 원본 decode 유지하되 명시적으로 invalid 표시
            # (조용히 raw 를 정답인 척 섞지 않음). t_ref pose 는 t_e pose 로 대체.
            n_no_sync += 1
            t_ref = ego_te["timestamp"]
            ego_ref = (ego_te["x"], ego_te["y"], ego_te["z"], ego_te["yaw"])
            meta = {"method": "no_sync", "gap": float("inf"), "extrap": float("inf")}
            valid = False
            for d in decoded:
                d["corr_dx"] = d["corr_dy"] = d["corr_dz"] = d["corr_dist"] = 0.0
                d["correction_valid"] = 0
        else:
            t_ref = tm["t_ref"]
            ego_ref, meta = interp.at(t_ref, tm.get("epoch_id", ""))
            valid = _interp_valid(meta)
            if ego_ref is None:
                ego_ref = (ego_te["x"], ego_te["y"], ego_te["z"], ego_te["yaw"])
            for d in decoded:
                _correct_object(d, ego_te, ego_ref, t_ref, tm["t_o"])
                d["correction_valid"] = 1 if valid else 0

        method_counts[meta["method"]] += 1
        frame_valid_counts["valid" if valid else "invalid"] += 1

        ego_ref_dict = {"x": ego_ref[0], "y": ego_ref[1], "z": ego_ref[2], "yaw": ego_ref[3]}
        frames.append((rec["frame_id"], t_ref, decoded))
        scene_frames.append({
            "stem": rec["stem"], "frame_id": rec["frame_id"], "timestamp": t_ref,
            "T_ego2global": _T_ego2global(ego_ref_dict).tolist(),
            "ego": {"x": ego_ref[0], "y": ego_ref[1], "z": ego_ref[2], "yaw": ego_ref[3]},
            "timing": {
                "ref_ts": (tm["t_ref"] if tm else None),
                "ego_src_ts": (tm["t_e"] if tm else None),
                "obj_src_ts": (tm["t_o"] if tm else None),
                "ego_dt_ms": ((tm["t_e"] - tm["t_ref"]) * 1000.0 if tm else None),
                "obj_dt_ms": ((tm["t_o"] - tm["t_ref"]) * 1000.0 if tm else None),
                "interp_method": meta["method"],
                "interp_gap_s": (None if not math.isfinite(meta["gap"]) else meta["gap"]),
                "extrap_s": (None if not math.isfinite(meta["extrap"]) else meta["extrap"]),
                "correction_valid": 1 if valid else 0,
                "n_boxes": len(decoded),
            },
        })

    # 3) 트랙 재할당 / velocity motion-fill / 품질 — 전부 시간보정 이후
    frames.sort(key=lambda e: e[0])
    stem_by_frame_id = {}
    for rec in frame_records:
        fi = rec["frame_id"]
        if fi in stem_by_frame_id and stem_by_frame_id[fi] != rec["stem"]:
            raise ValueError(
                f"frame_id가 서로 다른 stem에 중복됩니다: {fi} -> "
                f"{stem_by_frame_id[fi]}, {rec['stem']}")
        stem_by_frame_id[fi] = rec["stem"]
    n_tracks, seg_stats = _reassign_tracks(frames)
    n_vel_filled = _fill_velocity_from_motion(frames)    # 대표 데이터 기대값 0
    seg_stats["n_vel_motion_tracks"] = n_vel_filled
    quality = _track_quality(frames)

    # 4) correction 통계(감사)
    corr = np.array([o["corr_dist"] for _, _, objs in frames for o in objs], dtype=np.float64)
    vsrc = defaultdict(int)
    for _, _, objs in frames:
        for o in objs:
            vsrc[o.get("vel_source", "unknown")] += 1
    n_boxes = int(corr.size)
    corr_valid = np.array(
        [o["correction_valid"] for _, _, objs in frames for o in objs], dtype=np.int64)

    def _pct(a, q):
        return float(np.percentile(a, q)) if a.size else 0.0

    corr_stats = {
        "n_boxes": n_boxes,
        "p50_m": _pct(corr, 50), "p95_m": _pct(corr, 95),
        "max_m": float(corr.max()) if corr.size else 0.0,
        "mean_m": float(corr.mean()) if corr.size else 0.0,
        "frac_gt_0p2m": float((corr > 0.2).mean()) if corr.size else 0.0,
        "frac_gt_0p5m": float((corr > 0.5).mean()) if corr.size else 0.0,
        "n_correction_valid": int(corr_valid.sum()),
        "n_correction_invalid": int((corr_valid == 0).sum()),
        "vel_source_counts": dict(vsrc),
        "n_vel_motion_tracks": n_vel_filled,
        "interp_method_counts": dict(method_counts),
        "ego_interp_unique": interp.n_unique,
        "ego_interp_conflicts": interp.n_conflict,
        "ego_interp_n_epochs": interp.n_epochs,
        "ego_interp_epochs": [str(e) for e in interp.epochs],
        "n_frames_no_sync": n_no_sync,
        "frame_valid_counts": frame_valid_counts,
        "max_interp_gap_s": (float(max((f["timing"]["interp_gap_s"] or 0.0) for f in scene_frames))
                             if scene_frames else 0.0),
    }

    if write:
        out_dir = os.path.join(scen_dir, "labels_3d_v3")
        # 기존 디렉터리에 현재 입력에 없는 CSV가 남는 stale-output 문제를 막기 위해
        # 같은 파일시스템의 임시 디렉터리에 완성한 뒤 교체한다.
        tmp_out_dir = tempfile.mkdtemp(prefix=".labels_3d_v3.tmp.", dir=scen_dir)
        cols = ["frame_id", "timestamp", "track_id", "class_id",
                "x", "y", "z_center", "w", "l", "h", "yaw_ego",
                "vx_ego", "vy_ego", "vz", "yaw_global", "gx", "gy",
                "sin_yaw_ego", "cos_yaw_ego", "vel_source",
                "corr_dx", "corr_dy", "corr_dz", "corr_dist", "correction_valid"]
        try:
            for fi, t, objs in frames:
                stem = stem_by_frame_id[fi]
                with open(os.path.join(tmp_out_dir, stem + ".csv"), "w", newline="", encoding="utf-8") as f:
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
                            o.get("vel_source", "raw"),
                            f"{o['corr_dx']:.4f}", f"{o['corr_dy']:.4f}", f"{o['corr_dz']:.4f}",
                            f"{o['corr_dist']:.4f}", o["correction_valid"],
                        ])
        except Exception:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)
            raise
        scene_info = {
            "scenario": os.path.basename(scen_dir),
            "gt_version": "v3",
            "n_frames": len(frames),
            "n_tracks": n_tracks,
            "lidar_dz_body": LIDAR_DZ,
            "source_time_correction": corr_stats,
            "quality": quality,
            "seg_stats": seg_stats,
            "frames": scene_frames,
        }
        scene_tmp = os.path.join(scen_dir, ".scene_info_v3.json.tmp")
        report_tmp = os.path.join(scen_dir, ".timing_correction_report.json.tmp")
        try:
            with open(scene_tmp, "w", encoding="utf-8") as f:
                json.dump(scene_info, f)
            with open(report_tmp, "w", encoding="utf-8") as f:
                json.dump({"scenario": os.path.basename(scen_dir), **corr_stats}, f, indent=2)

            backup_dir = os.path.join(scen_dir, ".labels_3d_v3.previous")
            if os.path.isdir(backup_dir):
                shutil.rmtree(backup_dir)
            if os.path.isdir(out_dir):
                os.replace(out_dir, backup_dir)
            try:
                os.replace(tmp_out_dir, out_dir)
            except Exception:
                if os.path.isdir(backup_dir) and not os.path.exists(out_dir):
                    os.replace(backup_dir, out_dir)
                raise
            if os.path.isdir(backup_dir):
                shutil.rmtree(backup_dir)
            os.replace(scene_tmp, os.path.join(scen_dir, "scene_info_v3.json"))
            os.replace(report_tmp, os.path.join(scen_dir, "timing_correction_report.json"))
        except Exception:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)
            for p in (scene_tmp, report_tmp):
                if os.path.isfile(p):
                    os.remove(p)
            raise

    return {"scenario": os.path.basename(scen_dir), "n_frames": len(frames),
            "n_tracks": n_tracks, "corr": corr_stats, "quality": quality}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", nargs="*", default=None, help="처리할 시나리오(미지정=전체)")
    ap.add_argument("--report-only", action="store_true", help="파일 생성 없이 통계만")
    ap.add_argument("--root", default=DATASET_ROOT)
    args = ap.parse_args()

    if args.scen:
        scens = args.scen
    else:
        scens = sorted(
            (d for d in os.listdir(args.root)
             if os.path.isdir(os.path.join(args.root, d))
             and d.startswith("scen") and d[4:].isdigit()),
            key=_scenario_sort_key,
        )

    print(f"{'scen':8s} {'frames':>7s} {'boxes':>6s} {'tracks':>6s} "
          f"{'p50':>7s} {'p95':>7s} {'max':>7s} {'>0.2m':>7s} {'valid':>7s} {'noSync':>6s} {'motion':>6s}")
    tot_frames = tot_boxes = 0
    for scen in scens:
        scen_dir = os.path.join(args.root, scen)
        r = process_scenario(scen_dir, write=not args.report_only)
        if r is None:
            print(f"{scen:8s}  (skip: no labels/ego)")
            continue
        c = r["corr"]
        tot_frames += r["n_frames"]; tot_boxes += c["n_boxes"]
        print(f"{scen:8s} {r['n_frames']:7d} {c['n_boxes']:6d} {r['n_tracks']:6d} "
              f"{c['p50_m']:7.3f} {c['p95_m']:7.3f} {c['max_m']:7.3f} "
              f"{c['frac_gt_0p2m']*100:6.1f}% {c['n_correction_valid']:7d} "
              f"{c['n_frames_no_sync']:6d} {c['n_vel_motion_tracks']:6d}")
    print("-" * 100)
    print(f"TOTAL frames={tot_frames} boxes={tot_boxes}")


if __name__ == "__main__":
    main()

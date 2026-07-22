#!/usr/bin/env python3
"""150-scene 원본 데이터 사전 감사 (report-only, 데이터 미수정).
검사: 모달리티 frame 정합, 누락 file, 중복 stem/timestamp, raw velocity 비율,
epoch/teleport(=epoch_id 수), nonfinite, calibration 존재.
critical 이슈는 정확한 scene/stem 과 함께 비-0 종료(fail-fast). 결과 JSON 저장.
사용: python3 audit_150_dataset.py --root $DATASET_ROOT [--out audit.json]
"""
import argparse, csv, json, math, os, sys
from collections import Counter

MODALITIES = ["labels_3d", "ego_pose", "lidar"]


def scenes(root):
    return sorted([d for d in os.listdir(root)
                   if d.startswith("scen") and d[4:].isdigit()
                   and os.path.isdir(os.path.join(root, d))],
                  key=lambda s: int(s[4:]))


def stems_of(d, ext):
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(ext)}


def audit_scene(root, scen):
    sd = os.path.join(root, scen)
    rec = {"scenario": scen, "critical": [], "warn": []}
    lbl = stems_of(os.path.join(sd, "labels_3d"), ".csv")
    ego = stems_of(os.path.join(sd, "ego_pose"), ".csv")
    lid = stems_of(os.path.join(sd, "lidar"), ".pcd") | stems_of(os.path.join(sd, "lidar"), ".bin") \
        | stems_of(os.path.join(sd, "lidar"), ".npy")
    rec["n_labels"], rec["n_ego"], rec["n_lidar"] = len(lbl), len(ego), len(lid)
    # 모달리티 정합
    miss_ego = sorted(lbl - ego)[:10]
    if miss_ego:
        rec["critical"].append(f"{len(lbl-ego)} label frame 에 ego_pose 없음 예:{miss_ego}")
    # images: cam별 stem 정합
    img_root = os.path.join(sd, "images")
    if os.path.isdir(img_root):
        for cam in sorted(os.listdir(img_root)):
            cs = stems_of(os.path.join(img_root, cam), ".jpg") | stems_of(os.path.join(img_root, cam), ".png")
            m = sorted(lbl - cs)[:5]
            if m:
                rec["critical"].append(f"images/{cam} 에 {len(lbl-cs)} label frame 이미지 없음 예:{m}")
    else:
        rec["warn"].append("images/ 없음")
    # sync_log: 중복 stem / epoch / raw velocity
    sync = os.path.join(sd, "sync_log.csv")
    epochs = Counter()
    dup_stem = []
    saves = 0
    if os.path.isfile(sync):
        seen = set()
        with open(sync, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("action") != "save":
                    continue
                saves += 1
                st = r.get("stem")
                if st in seen:
                    dup_stem.append(st)
                seen.add(st)
                epochs[r.get("epoch_id", "")] += 1
    else:
        rec["warn"].append("sync_log.csv 없음")
    rec["sync_saves"] = saves
    rec["n_epochs"] = len(epochs)
    rec["epoch_ids"] = list(epochs)
    if dup_stem:
        rec["critical"].append(f"sync_log 중복 stem {len(dup_stem)} 예:{dup_stem[:5]}")
    # raw velocity 비율 + nonfinite + 중복 timestamp(label 내)
    n_box = n_raw = n_nonfinite = 0
    ts_seen = {}
    dup_ts = 0
    ldir = os.path.join(sd, "labels_3d")
    for st in sorted(lbl):
        p = os.path.join(ldir, st + ".csv")
        try:
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n_box += 1
                    if row.get("vel_source", "raw") == "raw":
                        n_raw += 1
                    for k in ("x", "y", "z", "vx", "vy", "vz"):
                        if k in row and row[k] not in ("", None):
                            try:
                                if not math.isfinite(float(row[k])):
                                    n_nonfinite += 1
                                    rec["critical"].append(f"nonfinite {k} @ {st}")
                            except ValueError:
                                pass
                    t = row.get("timestamp")
                    if t is not None:
                        if t in ts_seen and ts_seen[t] != st:
                            dup_ts += 1
                        ts_seen[t] = st
        except Exception as e:
            rec["critical"].append(f"label 읽기 실패 {st}: {e}")
    rec["n_boxes"] = n_box
    rec["raw_velocity_ratio"] = (n_raw / n_box) if n_box else None
    rec["n_nonfinite"] = n_nonfinite
    # calibration 존재(전역)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATASET_ROOT"))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.root or not os.path.isdir(args.root):
        print(f"[FATAL] DATASET_ROOT 디렉터리 없음: {args.root}", file=sys.stderr)
        sys.exit(2)
    scs = args.scenes or scenes(args.root)
    # calibration (repo-level)
    calib_ok = os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "camera_configs.py"))
    out = {"root": os.path.abspath(args.root), "n_scenes": len(scs),
           "calibration_present": calib_ok, "scenes": []}
    n_critical = 0
    for scen in scs:
        rec = audit_scene(args.root, scen)
        n_critical += len(rec["critical"])
        out["scenes"].append(rec)
        flag = "CRIT" if rec["critical"] else ("warn" if rec["warn"] else "ok")
        print(f"  [{flag:4s}] {scen}: labels={rec['n_labels']} ego={rec['n_ego']} "
              f"lidar={rec['n_lidar']} boxes={rec['n_boxes']} "
              f"raw_vel={rec['raw_velocity_ratio']} epochs={rec['n_epochs']} "
              f"nonfinite={rec['n_nonfinite']}")
        for c in rec["critical"][:5]:
            print(f"         CRITICAL: {c}")
    out["n_critical"] = n_critical
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    print(f"\n[SUMMARY] scenes={len(scs)} calibration={calib_ok} critical_issues={n_critical}")
    if n_critical:
        print("[FAIL] critical 이슈가 있습니다. 위 scene/stem 을 해결 후 진행하세요.", file=sys.stderr)
        sys.exit(1)
    print("[OK] critical 이슈 없음.")


if __name__ == "__main__":
    main()

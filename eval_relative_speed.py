#!/usr/bin/env python3
"""
eval_relative_speed.py  (P3 — relative-speed conditioned evaluation)
====================================================================
GT 를 |v_obj_world - v_ego_world| (상대속도, m/s) 로 bin 하고, bin 별로
  n_gt, n_tracks, n_frames, (예측이 주어지면) n_matched / recall / matched center distance
를 리포트한다. 목적: P4(source-time 보정) 이후 남는 오차가 상대속도에 따라 실제로 커지는지
확인해 P1(anisotropic loss) GO/NO-GO 판단 근거를 만든다.

설계 원칙(요구사항 그대로):
  - ego 속도는 source-time ego pose(scene_info_v3 의 t_ref ego x,y,timestamp)에서 계산한다.
      * 같은 epoch(sync_log epoch_id) 안에서만.
      * 인접 한 프레임 단순 차분 금지 → ±HALFWIDTH(기본 3) local linear regression(최소제곱 기울기).
      * 경계(윈도가 잘리는 곳)에서는 실제 사용 표본 수와 fallback 을 기록한다.
  - object 속도: v_obj_world = R(ego_yaw) @ [vx_ego, vy_ego]  (labels_3d_v3, m/s).
  - GT 상대속도 = |v_obj_world - v_ego_world|.
  - 매칭 반경 2.0m, greedy·class-aware·score 내림차순 = train.py 의 by-distance 지표와 동일 규약.
  - GT bin 은 GT-conditioned(각 GT 가 매칭됐는지) → recall 만 정의된다. unmatched prediction 은
    어느 GT bin 소속인지 정의 불가하므로 이 표를 precision/F1 이라 부르지 않는다.
  - corr_dist 는 'amount_removed'(=v2→v3 로 제거된 시간오염 프록시)로만 표기한다.
    v3 잔존 noise floor 가 아니다.

사용:
  # GT 분포만(예측 없이) — 상대속도 bin 카운트/track/frame
  python3 eval_relative_speed.py --gt-only --scen scen05 scen77 scen144 --gt-version v3 \
      --out-json pretrain_verify/p3_gt_distribution.json

  # 예측(npz: per-frame arrays)까지 주면 bin 별 recall/center-distance
  python3 eval_relative_speed.py --preds preds.npz --scen scen144 --out-json p3.json
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from preprocess_dataset import _rot2

# audit 기본 bin (m/s). configurable(--bins) 하지만 현 감사 기준은 이 경계.
DEFAULT_BIN_EDGES = [0.0, 1.0, 3.0, 6.0, 8.0, float("inf")]
DEFAULT_HALFWIDTH = 3          # ±3 sample local linear regression
MATCH_RADIUS_M = 2.0           # train.py by-distance 지표와 동일
HIGH_SPEED_MIN_N = 30          # 이보다 표본 적은 고속 bin 은 모델 우열 판단 금지 경고


def _read_epoch_ids(scen_dir):
    """sync_log.csv 의 action=save 행 -> {stem: epoch_id}. 없으면 빈 dict(단일 epoch 취급)."""
    path = os.path.join(scen_dir, "sync_log.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("action") == "save":
                out[r["stem"]] = r.get("epoch_id", "")
    return out


def bin_index(speed, edges):
    """speed 가 속한 bin 인덱스. bin i = [edges[i], edges[i+1]). 마지막은 [edges[-2], inf)."""
    for i in range(len(edges) - 1):
        if edges[i] <= speed < edges[i + 1]:
            return i
    if speed >= edges[-2]:            # inf 경계 안전망
        return len(edges) - 2
    return None


def bin_label(edges, i):
    lo, hi = edges[i], edges[i + 1]
    hs = "inf" if math.isinf(hi) else f"{hi:g}"
    return f"[{lo:g},{hs})"


def local_linear_velocity(ts, xs, ys, epoch_ids, i, halfwidth=DEFAULT_HALFWIDTH):
    """프레임 i 의 ego world 속도 (vx,vy) m/s 를 같은 epoch 안 ±halfwidth 표본
    least-squares 기울기로 추정. 반환: (vx, vy, audit).
    audit: {n_used, is_boundary, method, span_s}."""
    n = len(ts)
    ep = epoch_ids[i]
    lo = max(0, i - halfwidth)
    hi = min(n - 1, i + halfwidth)
    js = [j for j in range(lo, hi + 1) if epoch_ids[j] == ep]
    tw = np.array([ts[j] for j in js], dtype=np.float64)
    is_boundary = (lo != i - halfwidth) or (hi != i + halfwidth) or (len(js) < (2 * halfwidth + 1))
    if len(js) < 2 or (tw.max() - tw.min()) <= 0:
        # fallback: 회귀 불가(고립 프레임/동일 시각) → 속도 0, 명시적 기록(조용한 차분 금지)
        return 0.0, 0.0, {"n_used": len(js), "is_boundary": True,
                          "method": "insufficient(v=0)", "span_s": 0.0}
    xw = np.array([xs[j] for j in js], dtype=np.float64)
    yw = np.array([ys[j] for j in js], dtype=np.float64)
    tc = tw - tw.mean()
    denom = float((tc * tc).sum())
    vx = float((tc * (xw - xw.mean())).sum() / denom)
    vy = float((tc * (yw - yw.mean())).sum() / denom)
    method = "linreg" if len(js) >= 3 else "2pt-slope"
    return vx, vy, {"n_used": len(js), "is_boundary": bool(is_boundary),
                    "method": method, "span_s": float(tw.max() - tw.min())}


def load_gt_with_relspeed(dataset_root, scen, gt_version="v3", halfwidth=DEFAULT_HALFWIDTH):
    """한 시나리오의 GT 박스에 상대속도를 붙여 반환.
    returns: (frames, ego_audit)
      frames: list per frame dict {scen, stem, timestamp, boxes:[...]}
        box: {class_id, track_id, x, y, gx, gy, rel_speed, obj_speed, ego_speed, corr_dist}
      ego_audit: {n_frames, boundary_frames, method_counts, ego_speed_p50/p95/max}
    """
    scen_dir = os.path.join(dataset_root, scen)
    info_name = "scene_info_v3.json" if gt_version == "v3" else "scene_info.json"
    label_dir = os.path.join(scen_dir, "labels_3d_v3" if gt_version == "v3" else "labels_3d_v2")
    with open(os.path.join(scen_dir, info_name), encoding="utf-8") as f:
        info = json.load(f)
    epoch_map = _read_epoch_ids(scen_dir)

    frames_info = sorted(info["frames"], key=lambda fr: fr["stem"])
    stems = [fr["stem"] for fr in frames_info]
    ts = [float(fr["timestamp"]) for fr in frames_info]
    ex = [float(fr["ego"]["x"]) for fr in frames_info]
    ey = [float(fr["ego"]["y"]) for fr in frames_info]
    eyaw = [float(fr["ego"]["yaw"]) for fr in frames_info]
    epoch_ids = [epoch_map.get(s, "") for s in stems]

    ego_v = []
    audit = {"n_frames": len(stems), "boundary_frames": 0,
             "method_counts": defaultdict(int), "n_used_hist": defaultdict(int)}
    for i in range(len(stems)):
        vx, vy, a = local_linear_velocity(ts, ex, ey, epoch_ids, i, halfwidth)
        ego_v.append((vx, vy))
        audit["method_counts"][a["method"]] += 1
        audit["n_used_hist"][a["n_used"]] += 1
        if a["is_boundary"]:
            audit["boundary_frames"] += 1

    out_frames = []
    ego_speeds = []
    for i, fr in enumerate(frames_info):
        stem = fr["stem"]
        vex, vey = ego_v[i]
        ego_speeds.append(math.hypot(vex, vey))
        yaw_r = eyaw[i]
        R = _rot2(yaw_r)
        boxes = []
        csv_path = os.path.join(label_dir, stem + ".csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if gt_version == "v3" and int(float(row.get("correction_valid", "1"))) != 1:
                    continue
                vxe, vye = float(row["vx_ego"]), float(row["vy_ego"])
                vwo = R @ np.array([vxe, vye], dtype=np.float64)   # object world velocity m/s
                rel = math.hypot(vwo[0] - vex, vwo[1] - vey)
                boxes.append({
                    "class_id": int(float(row["class_id"])),
                    "track_id": int(float(row["track_id"])),
                    "x": float(row["x"]), "y": float(row["y"]),
                    "gx": float(row.get("gx", 0.0)), "gy": float(row.get("gy", 0.0)),
                    "obj_speed": float(math.hypot(vwo[0], vwo[1])),
                    "ego_speed": float(math.hypot(vex, vey)),
                    "rel_speed": float(rel),
                    "corr_dist": float(row.get("corr_dist", 0.0)),
                })
        out_frames.append({"scen": scen, "stem": stem, "timestamp": ts[i], "boxes": boxes})

    esp = np.array(ego_speeds, dtype=np.float64)
    audit["method_counts"] = dict(audit["method_counts"])
    audit["n_used_hist"] = dict(sorted(audit["n_used_hist"].items()))
    audit["ego_speed_p50"] = float(np.percentile(esp, 50)) if esp.size else 0.0
    audit["ego_speed_p95"] = float(np.percentile(esp, 95)) if esp.size else 0.0
    audit["ego_speed_max"] = float(esp.max()) if esp.size else 0.0
    audit["halfwidth"] = halfwidth
    return out_frames, audit


def gt_distribution(frames, edges=DEFAULT_BIN_EDGES):
    """GT-only: bin 별 n_gt, n_tracks(고유 (scen,track)), n_frames, corr_dist(amount_removed) 통계."""
    nb = len(edges) - 1
    n_gt = [0] * nb
    tracks = [set() for _ in range(nb)]
    frames_seen = [set() for _ in range(nb)]
    corr = [[] for _ in range(nb)]
    rel_all = []
    for fr in frames:
        for b in fr["boxes"]:
            bi = bin_index(b["rel_speed"], edges)
            if bi is None:
                continue
            n_gt[bi] += 1
            tracks[bi].add((fr["scen"], b["track_id"]))
            frames_seen[bi].add((fr["scen"], fr["stem"]))
            corr[bi].append(b["corr_dist"])
            rel_all.append(b["rel_speed"])
    rows = []
    for i in range(nb):
        c = np.array(corr[i], dtype=np.float64)
        rows.append({
            "bin": bin_label(edges, i),
            "lo": edges[i], "hi": (None if math.isinf(edges[i + 1]) else edges[i + 1]),
            "n_gt": n_gt[i],
            "n_tracks": len(tracks[i]),
            "n_frames": len(frames_seen[i]),
            "amount_removed_p50": float(np.percentile(c, 50)) if c.size else 0.0,
            "amount_removed_p95": float(np.percentile(c, 95)) if c.size else 0.0,
            "amount_removed_mean": float(c.mean()) if c.size else 0.0,
            "amount_removed_max": float(c.max()) if c.size else 0.0,
        })
    return rows, rel_all


def _greedy_match_frame(gt_boxes, pred, dist_thr=MATCH_RADIUS_M):
    """train.py compute_detection_counts_by_distance 와 동일 규약:
    greedy, class-aware, score 내림차순, 가장 가까운 미매칭 GT(<=thr)에 매칭.
    반환: matched(bool list, gt 순서), matched_dist(list; 매칭 안 되면 None)."""
    n = len(gt_boxes)
    matched = [False] * n
    mdist = [None] * n
    gx = np.array([[b["x"], b["y"]] for b in gt_boxes], dtype=np.float64) if n else np.zeros((0, 2))
    gc = np.array([b["class_id"] for b in gt_boxes], dtype=np.int64) if n else np.zeros((0,), np.int64)
    order = np.argsort(-pred["score"]) if len(pred["score"]) else np.array([], dtype=int)
    for pi in order:
        if n == 0:
            break
        pcls = int(pred["label"][pi])
        cand = [j for j in range(n) if (not matched[j]) and gc[j] == pcls]
        if not cand:
            continue
        pxy = np.array([pred["x"][pi], pred["y"][pi]], dtype=np.float64)
        d = np.linalg.norm(gx[cand] - pxy, axis=1)
        k = int(np.argmin(d))
        if d[k] <= dist_thr:
            matched[cand[k]] = True
            mdist[cand[k]] = float(d[k])
    return matched, mdist


def load_filtered_gt_with_relspeed(dataset_root, scen, gt_version="v3",
                                   halfwidth=DEFAULT_HALFWIDTH):
    """production validation filter(MoraiTemporalDataset._load_labels_v2: 50m+visibility+occlusion)
    이후 GT 에 rel_speed 를 붙여 반환한다. rel_speed 는 raw all-box 계산값을
    (scen, stem, track_id) 로 매칭해 attach → recall 채점 대상이 모델이 실제 본 GT 와 동일 membership.
    returns: (frames, audit, n_missing_relspeed)."""
    from morai_dataset import MoraiTemporalDataset
    all_frames, audit = load_gt_with_relspeed(dataset_root, scen, gt_version, halfwidth)
    relmap = {}
    for fr in all_frames:
        for b in fr["boxes"]:
            relmap[(scen, fr["stem"], b["track_id"])] = b
    ds = MoraiTemporalDataset(dataset_root, "val", [scen], load_depth=False, gt_version=gt_version)
    frames, n_missing = [], 0
    for it in ds.items:
        stem = it["stem"]
        boxes_t, labels_t, tids_t, _vv, _vsrc, _visw = ds._load_labels_v2(it["scen_dir"], stem)
        boxes = []
        for i in range(int(boxes_t.shape[0])):
            tid = int(tids_t[i])
            src = relmap.get((scen, stem, tid))
            if src is None:
                n_missing += 1
            boxes.append({
                "x": float(boxes_t[i, 0]), "y": float(boxes_t[i, 1]),
                "class_id": int(labels_t[i]), "track_id": tid,
                "rel_speed": float(src["rel_speed"]) if src else 0.0,
                "corr_dist": float(src["corr_dist"]) if src else 0.0,
            })
        frames.append({"scen": scen, "stem": stem, "boxes": boxes})
    return frames, audit, n_missing


def evaluate_relative_speed(frames, preds_by_key, edges=DEFAULT_BIN_EDGES,
                            dist_thr=MATCH_RADIUS_M, require_nonempty=False):
    """예측이 있는 경우 bin 별 recall/center-distance.
    preds_by_key: {(scen,stem): {"x":arr,"y":arr,"label":arr,"score":arr}}.
    GT-conditioned(각 GT 매칭 여부) → recall 만. unmatched prediction 은 별도로 센다(precision 아님).
    require_nonempty=True 면 평가 frame/GT/pred 중 하나라도 0 이면 fail-fast(ValueError).
    (join 실패로 조용히 0/0 이 나오는 것을 막는다.)"""
    nb = len(edges) - 1
    n_gt = [0] * nb
    n_matched = [0] * nb
    mdists = [[] for _ in range(nb)]
    tracks = [set() for _ in range(nb)]
    frames_seen = [set() for _ in range(nb)]
    n_pred_total = 0
    n_pred_matched = 0
    for fr in frames:
        key = (fr["scen"], fr["stem"])
        pred = preds_by_key.get(key, {"x": np.zeros(0), "y": np.zeros(0),
                                      "label": np.zeros(0, int), "score": np.zeros(0)})
        n_pred_total += len(pred["score"])
        matched, mdist = _greedy_match_frame(fr["boxes"], pred, dist_thr)
        n_pred_matched += sum(1 for m in matched if m)
        for j, b in enumerate(fr["boxes"]):
            bi = bin_index(b["rel_speed"], edges)
            if bi is None:
                continue
            n_gt[bi] += 1
            tracks[bi].add((fr["scen"], b["track_id"]))
            frames_seen[bi].add(key)
            if matched[j]:
                n_matched[bi] += 1
                mdists[bi].append(mdist[j])
    rows = []
    for i in range(nb):
        md = np.array(mdists[i], dtype=np.float64)
        rows.append({
            "bin": bin_label(edges, i),
            "n_gt": n_gt[i], "n_tracks": len(tracks[i]), "n_frames": len(frames_seen[i]),
            "n_matched": n_matched[i],
            "recall": (n_matched[i] / n_gt[i]) if n_gt[i] else None,
            "match_dist_mean": float(md.mean()) if md.size else None,
            "match_dist_median": float(np.median(md)) if md.size else None,
            "match_dist_p95": float(np.percentile(md, 95)) if md.size else None,
        })
    unmatched_pred = n_pred_total - n_pred_matched
    total_gt = sum(n_gt)
    n_frames_eval = len(frames)
    if require_nonempty and (n_frames_eval == 0 or total_gt == 0 or n_pred_total == 0):
        raise ValueError(
            f"[P3] empty evaluation (frame/GT/pred join 실패 의심): frames={n_frames_eval} "
            f"total_gt={total_gt} n_pred_total={n_pred_total}. stem/scenario 키 정합 확인 필요.")
    return {"bins": rows,
            "note": ("GT-conditioned recall only. unmatched prediction("
                     f"{unmatched_pred}/{n_pred_total})은 GT bin 소속 정의 불가 → precision/F1 아님."),
            "n_frames_evaluated": n_frames_eval, "total_gt": total_gt,
            "n_pred_total": n_pred_total, "n_pred_unmatched": unmatched_pred}


def format_gt_table(rows, edges=DEFAULT_BIN_EDGES):
    lines = [f"{'bin(m/s)':>12s} {'n_gt':>7s} {'n_trk':>6s} {'n_frm':>6s} "
             f"{'amtRem_p50':>10s} {'amtRem_p95':>10s} {'amtRem_max':>10s}"]
    lines.append("-" * 72)
    tot = 0
    for r in rows:
        tot += r["n_gt"]
        lines.append(f"{r['bin']:>12s} {r['n_gt']:>7d} {r['n_tracks']:>6d} {r['n_frames']:>6d} "
                     f"{r['amount_removed_p50']:>10.4f} {r['amount_removed_p95']:>10.4f} "
                     f"{r['amount_removed_max']:>10.4f}")
    lines.append("-" * 72)
    lines.append(f"{'TOTAL':>12s} {tot:>7d}")
    lines.append("amtRem = amount_removed(=v2→v3 제거된 시간오염 프록시). v3 잔존 noise floor 아님.")
    return "\n".join(lines)


def format_eval_table(res):
    lines = [f"{'bin(m/s)':>12s} {'n_gt':>7s} {'n_trk':>6s} {'n_frm':>6s} "
             f"{'n_match':>7s} {'recall':>7s} {'md_mean':>8s} {'md_p95':>8s}"]
    lines.append("-" * 76)
    for r in res["bins"]:
        rc = "  n/a " if r["recall"] is None else f"{r['recall']:.4f}"
        mm = "   n/a" if r["match_dist_mean"] is None else f"{r['match_dist_mean']:.4f}"
        mp = "   n/a" if r["match_dist_p95"] is None else f"{r['match_dist_p95']:.4f}"
        lines.append(f"{r['bin']:>12s} {r['n_gt']:>7d} {r['n_tracks']:>6d} {r['n_frames']:>6d} "
                     f"{r['n_matched']:>7d} {rc:>7s} {mm:>8s} {mp:>8s}")
    lines.append("-" * 76)
    lines.append("GT-conditioned recall만 정의. " + res["note"])
    return "\n".join(lines)


def high_speed_warnings(rows, edges=DEFAULT_BIN_EDGES, min_n=HIGH_SPEED_MIN_N):
    warns = []
    for r in rows:
        if r["lo"] >= 6.0 and r["n_gt"] < min_n:
            warns.append(f"bin {r['bin']}: n_gt={r['n_gt']} (<{min_n}, n_tracks={r['n_tracks']}) "
                         "→ 표본 부족, 이 구간으로 모델 우열/노이즈 결론 금지.")
    return warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", "./dataset"))
    ap.add_argument("--scen", nargs="+", required=True)
    ap.add_argument("--gt-version", default="v3", choices=["v2", "v3"])
    ap.add_argument("--halfwidth", type=int, default=DEFAULT_HALFWIDTH)
    ap.add_argument("--bins", type=float, nargs="+", default=None,
                    help="bin 경계(예: 0 1 3 6 8 inf). 미지정 시 audit 기본.")
    ap.add_argument("--gt-only", action="store_true", help="예측 없이 GT 분포만")
    ap.add_argument("--preds", default=None, help="예측 npz(키: scen,stem,x,y,label,score)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    edges = args.bins if args.bins else list(DEFAULT_BIN_EDGES)
    if math.isinf(edges[-1]) is False and edges[-1] != float("inf"):
        edges = edges + [float("inf")] if edges[-1] < 1e9 else edges
    # 'inf' 문자열 처리
    edges = [float("inf") if (isinstance(e, str) and e.lower() == "inf") else float(e) for e in edges]

    all_frames = []
    ego_audits = {}
    for scen in args.scen:
        frames, audit = load_gt_with_relspeed(args.dataset_root, scen, args.gt_version, args.halfwidth)
        all_frames.extend(frames)
        ego_audits[scen] = audit

    rows, rel_all = gt_distribution(all_frames, edges)
    result = {
        "gt_version": args.gt_version, "scenarios": args.scen,
        "bin_edges": [None if math.isinf(e) else e for e in edges],
        "match_radius_m": MATCH_RADIUS_M, "halfwidth": args.halfwidth,
        "n_boxes_total": sum(r["n_gt"] for r in rows),
        "gt_distribution": rows,
        "ego_speed_audit": ego_audits,
        "high_speed_warnings": high_speed_warnings(rows, edges),
    }

    print(f"[P3] GT 상대속도 분포 (gt={args.gt_version}, scen={args.scen}, "
          f"halfwidth=±{args.halfwidth} local-linreg ego vel)")
    print(format_gt_table(rows, edges))
    for scen, a in ego_audits.items():
        print(f"  ego[{scen}]: frames={a['n_frames']} boundary={a['boundary_frames']} "
              f"method={a['method_counts']} ego_speed p50/p95/max="
              f"{a['ego_speed_p50']:.2f}/{a['ego_speed_p95']:.2f}/{a['ego_speed_max']:.2f} m/s")
    for w in result["high_speed_warnings"]:
        print("  ⚠️ " + w)

    if args.preds:
        data = np.load(args.preds, allow_pickle=True)
        preds_by_key = defaultdict(lambda: {"x": [], "y": [], "label": [], "score": []})
        for scen, stem, x, y, lab, sc in zip(data["scen"], data["stem"], data["x"],
                                             data["y"], data["label"], data["score"]):
            k = (str(scen), str(stem))
            preds_by_key[k] = {"x": np.atleast_1d(x).astype(float),
                               "y": np.atleast_1d(y).astype(float),
                               "label": np.atleast_1d(lab).astype(int),
                               "score": np.atleast_1d(sc).astype(float)}
        ev = evaluate_relative_speed(all_frames, preds_by_key, edges)
        result["recall_by_bin"] = ev
        print("\n[P3] 예측 매칭(2.0m) 상대속도 bin 별 recall")
        print(format_eval_table(ev))

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""한 A/B run 의 최종 checkpoint 를 동일 v3 validation GT **전체**로 평가(subset 아님).
production 매칭 함수(train.compute_detection_counts_by_class / _by_distance / decode_detections)를
그대로 재사용하되, 보고 대상 threshold 만 계산해 full-val 을 빠르게 처리한다(validate 의
0.01/0.03 저-threshold NMS sweep 을 생략 → 수치는 동일 함수·동일 threshold 에서 일치).
- overall/by-class: mode {raw, softcalibrated} × thr {0.10,0.15,0.25}
- by-distance + center-dist mean/median/p95: softcalibrated@0.15 (greedy 2.0m, class-aware)
- filtered production P3: validation filter 이후 GT membership + rel_speed bin recall/cdist
결과 → <run_dir>/eval_metrics.json.
"""
import argparse, hashlib, json, os, sys
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = "/home/autonav/projects/morai-3d-detection"
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, torch

THRS = (0.10, 0.15, 0.25)
MODES = {"raw": (False, 1.0), "softcalibrated": (True, 0.5)}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--val-scen", required=True)
    ap.add_argument("--val-gt", default="v3")
    ap.add_argument("--anchor-full", required=True)
    ap.add_argument("--ckpt", default="last_checkpoint.pth")
    ap.add_argument("--dataset-root", default="./dataset")
    args = ap.parse_args()

    os.environ["ANCHOR_FULL_FILE"] = os.path.abspath(args.anchor_full)
    os.environ["ANCHOR_XY_FILE"] = os.path.abspath(
        args.anchor_full.replace("anchor_kmeans_full.npy", "anchor_kmeans_xy.npy"))

    import train
    from morai_dataset import MoraiTemporalDataset, morai_temporal_collate_fn
    from loss_calculator import CustomLoss
    from torch.utils.data import DataLoader
    import eval_relative_speed as p3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.run_dir, args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = train.AutoNavModel(num_decoder_layers=6, pretrained_backbone=True,
                               use_temporal_memory=False, num_temp_instances=600,
                               use_grid_mask=True, use_dense_depth=False).to(device)
    msd = model.state_dict()
    msd.update({k: v for k, v in state.items() if k in msd and v.shape == msd[k].shape})
    model.load_state_dict(msd); model.eval()
    if hasattr(model, "reset_temporal_memory"):
        model.reset_temporal_memory()

    va = MoraiTemporalDataset(args.dataset_root, "val", [args.val_scen],
                              load_depth=False, gt_version=args.val_gt)
    vl = DataLoader(va, batch_size=4, shuffle=False, collate_fn=morai_temporal_collate_fn,
                    num_workers=0)
    crit = CustomLoss(num_classes=2, quality_weight=0.2).to(device)
    class_ids = tuple(sorted(train.CLASS_ID_NAMES))
    buckets = train.DISTANCE_BUCKETS

    # 누적기
    cnt = {m: {t: {c: [0, 0, 0] for c in class_ids} for t in THRS} for m in MODES}
    dbk = {i: {"tp": 0, "fp": 0, "fn": 0, "dist": []} for i in range(len(buckets))}
    cdist_all = []
    preds = {"scen": [], "stem": [], "x": [], "y": [], "label": [], "score": []}
    vloss_sum = 0.0; nb = 0

    with torch.no_grad():
        for batch in vl:
            images = batch["images"].to(device); intr = batch["intrinsics"].to(device)
            extr = batch["extrinsics"].to(device); ego = batch["ego_pose"].to(device)
            out = model(images, intr, extr, stems=batch["stem"], ego_poses=ego,
                        timestamps=batch.get("timestamp"), return_intermediate=True)
            dc = out["det_cls"].detach().cpu(); db = out["det_box"].detach().cpu()
            dq = out["det_quality"].detach().cpu()
            bl, _, _, _ = train.compute_auxiliary_detection_loss(out, batch, crit, device)
            vloss_sum += float(bl.item()); nb += 1
            for i in range(images.shape[0]):
                gtb = batch["dynamic_gt_boxes"][i].cpu()
                gtc = batch["dynamic_gt_labels"][i].cpu()
                for m, (aq, qp) in MODES.items():
                    for t in THRS:
                        per = train.compute_detection_counts_by_class(
                            dc[i], db[i], gtc, gtb, det_quality=dq[i], distance_thr=2.0,
                            score_thresh=t, apply_quality=aq, quality_power=qp, class_ids=class_ids)
                        for c in class_ids:
                            tp, fp, fn = per[c]
                            cnt[m][t][c][0] += tp; cnt[m][t][c][1] += fp; cnt[m][t][c][2] += fn
                # by-distance + center distances (softcalibrated@0.15)
                bd = train.compute_detection_counts_by_distance(
                    dc[i], db[i], gtc, gtb, det_quality=dq[i], distance_thr=2.0,
                    score_thresh=0.15, apply_quality=True, quality_power=0.5, buckets=buckets)
                for bi, d in bd.items():
                    dbk[bi]["tp"] += d["tp"]; dbk[bi]["fp"] += d["fp"]; dbk[bi]["fn"] += d["fn"]
                # per-match center distance (soft@0.15 greedy, class-aware, 2.0m)
                pb, pl, ps = train.decode_detections(dc[i], db[i], det_quality=dq[i],
                                                     score_thresh=0.15, apply_quality=True, quality_power=0.5)
                gm = _greedy(gtb.numpy(), gtc.numpy(), pb.numpy(), pl.numpy(), ps.numpy(), 2.0)
                for dist, gd in gm:
                    cdist_all.append(dist)
                    bidx = _bucket(gd, buckets)
                    if bidx is not None:
                        dbk[bidx]["dist"].append(dist)
                # P3 preds (soft@0.15)
                preds["scen"].append(batch["scenario"][i])
                preds["stem"].append(batch["stem"][i].split("/")[-1])
                preds["x"].append(pb[:, 0].numpy()); preds["y"].append(pb[:, 1].numpy())
                preds["label"].append(pl.numpy()); preds["score"].append(ps.numpy())

    def stats(a):
        if not a:
            return {"mean": None, "median": None, "p95": None, "n": 0}
        arr = np.array(a, dtype=np.float64)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)), "n": int(arr.size)}

    overall = {}
    by_class = {}
    for m in MODES:
        overall[m] = {}; by_class[m] = {}
        for t in THRS:
            tp = sum(cnt[m][t][c][0] for c in class_ids)
            fp = sum(cnt[m][t][c][1] for c in class_ids)
            fn = sum(cnt[m][t][c][2] for c in class_ids)
            overall[m][str(t)] = prf(tp, fp, fn)
            by_class[m][str(t)] = {str(c): prf(*cnt[m][t][c]) for c in class_ids}

    by_dist = {}
    for bi in range(len(buckets)):
        lo, hi = buckets[bi]
        d = dbk[bi]
        by_dist[f"{lo:.0f}_{hi:.0f}m"] = {**prf(d["tp"], d["fp"], d["fn"]),
                                          "center_dist": stats(d["dist"])}

    # filtered production P3
    frames, _a, n_missing = p3.load_filtered_gt_with_relspeed(args.dataset_root, args.val_scen, args.val_gt)
    pbk = {}
    for i in range(len(preds["stem"])):
        pbk[(preds["scen"][i], preds["stem"][i])] = {
            "x": np.atleast_1d(preds["x"][i]).astype(float), "y": np.atleast_1d(preds["y"][i]).astype(float),
            "label": np.atleast_1d(preds["label"][i]).astype(int), "score": np.atleast_1d(preds["score"][i]).astype(float)}
    ev = p3.evaluate_relative_speed(frames, pbk, p3.DEFAULT_BIN_EDGES, dist_thr=2.0, require_nonempty=True)
    ev["n_relspeed_missing"] = n_missing; ev["membership"] = "production_validation_filter(_load_labels_v2)"
    p3_boxes = []
    for fr in frames:
        pr = pbk.get((fr["scen"], fr["stem"]), {"x": np.zeros(0), "y": np.zeros(0), "label": np.zeros(0, int), "score": np.zeros(0)})
        matched, mdist = p3._greedy_match_frame(fr["boxes"], pr, 2.0)
        for j, b in enumerate(fr["boxes"]):
            p3_boxes.append({"scen": fr["scen"], "stem": fr["stem"], "track_id": b["track_id"],
                             "class_id": b["class_id"], "rel_speed": round(b["rel_speed"], 4),
                             "matched": bool(matched[j]),
                             "center_dist": (round(mdist[j], 4) if matched[j] else None)})

    result = {
        "run_dir": os.path.abspath(args.run_dir), "val_scen": args.val_scen, "val_gt": args.val_gt,
        "ckpt": args.ckpt, "ckpt_sha256": sha_file(ckpt_path),
        "ckpt_epoch": (ckpt.get("epoch") if isinstance(ckpt, dict) else None),
        "ckpt_update_step": (ckpt.get("global_update_step") if isinstance(ckpt, dict) else None),
        "anchor_full": os.path.abspath(args.anchor_full), "anchor_full_sha256": sha_file(args.anchor_full),
        "val_frames": len(va.items), "val_loss": (vloss_sum / nb if nb else None),
        "thresholds": list(THRS), "modes": list(MODES),
        "overall": overall, "by_class": by_class,
        "by_distance_softcalibrated@0.15": by_dist,
        "center_dist_overall_softcalibrated@0.15": stats(cdist_all),
        "p3_filtered": ev, "p3_boxes": p3_boxes,
    }
    outp = os.path.join(args.run_dir, "eval_metrics.json")
    json.dump(result, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    s = overall["softcalibrated"]["0.15"]
    print(f"[eval] {args.run_dir} valframes={len(va.items)} valloss={result['val_loss']:.3f} "
          f"soft@0.15 P/R/F1={s['precision']:.3f}/{s['recall']:.3f}/{s['f1']:.3f} "
          f"cdist={result['center_dist_overall_softcalibrated@0.15']['mean']} "
          f"p3 gt={ev['total_gt']} pred={ev['n_pred_total']} matched={ev['n_pred_total']-ev['n_pred_unmatched']}")
    print(f"[eval] wrote {outp}")


def _bucket(dist, buckets):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= dist < hi:
            return i
    return None


def _greedy(gt_boxes, gt_classes, pb, pl, ps, thr):
    """soft@0.15 greedy class-aware 매칭. 반환 [(center_dist, gt_radial), …] (matched pairs)."""
    n = gt_boxes.shape[0]
    if n == 0 or pb.shape[0] == 0:
        return []
    matched = np.zeros(n, dtype=bool)
    order = np.argsort(-ps)
    out = []
    for pi in order:
        cls = int(pl[pi])
        cand = [j for j in range(n) if (not matched[j]) and int(gt_classes[j]) == cls]
        if not cand:
            continue
        d = np.linalg.norm(gt_boxes[cand, :2] - pb[pi, :2], axis=1)
        k = int(np.argmin(d))
        if d[k] <= thr:
            matched[cand[k]] = True
            gj = cand[k]
            out.append((float(d[k]), float(np.hypot(gt_boxes[gj, 0], gt_boxes[gj, 1]))))
    return out


if __name__ == "__main__":
    main()

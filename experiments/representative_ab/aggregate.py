#!/usr/bin/env python3
"""대표 A/B 집계: fold별 candidate(v3)-baseline(v2) paired delta + macro + box-level paired
bootstrap CI. pooled 수치만으로 결론내지 않고 fold/macro/paired 를 분리 출력한다.
출력: ab_summary.json, paired_delta.csv. (판정은 AB_REPORT.md 에서 사람이 종합)
사용: python3 aggregate.py [--boot 2000]
"""
import argparse, csv, json, os, random

AB = os.path.dirname(os.path.abspath(__file__))
FOLDS = [("foldC_val_scen144", "scen144"),
         ("foldB_val_scen77", "scen77"),
         ("foldA_val_scen05", "scen05")]
LOWBIN = "[0,1)"


def load(fold, role):
    p = os.path.join(AB, fold, role, "eval_metrics.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def soft15(m):
    o = (m or {}).get("overall", {}).get("softcalibrated", {}).get("0.15")
    return o or {"precision": None, "recall": None, "f1": None}


def bin_recall(m, b):
    for r in (m or {}).get("p3_filtered", {}).get("bins", []):
        if r["bin"] == b:
            return r
    return {}


def boxmap(m):
    return {(x["scen"], x["stem"], x["track_id"]): x for x in (m or {}).get("p3_boxes", [])}


def paired_bootstrap(base_boxes, cand_boxes, n_boot, rng, speed_filter=None):
    """box-level paired: 공통 (scen,stem,track_id) 로 pairing.
    delta_recall = mean(matched_cand)-mean(matched_base);
    delta_cdist  = mean(cdist_cand-cdist_base) over boxes matched by BOTH.
    반환: {n, delta_recall, ci_recall[lo,hi], n_both, delta_cdist, ci_cdist[lo,hi]}"""
    keys = [k for k in base_boxes if k in cand_boxes]
    if speed_filter:
        lo, hi = speed_filter
        keys = [k for k in keys if lo <= base_boxes[k]["rel_speed"] < hi]
    n = len(keys)
    if n == 0:
        return {"n": 0}
    mb = [1.0 if base_boxes[k]["matched"] else 0.0 for k in keys]
    mc = [1.0 if cand_boxes[k]["matched"] else 0.0 for k in keys]
    both = [k for k in keys if base_boxes[k]["matched"] and cand_boxes[k]["matched"]]
    dc = [cand_boxes[k]["center_dist"] - base_boxes[k]["center_dist"] for k in both]
    d_recall = sum(mc) / n - sum(mb) / n
    d_cdist = (sum(dc) / len(dc)) if dc else None
    br, bc = [], []
    idx = list(range(n))
    for _ in range(n_boot):
        samp = [idx[rng.randrange(n)] for _ in range(n)]
        br.append(sum(mc[i] - mb[i] for i in samp) / n)
    if dc:
        m = len(dc)
        for _ in range(n_boot):
            bc.append(sum(dc[rng.randrange(m)] for _ in range(m)) / m)
    def ci(a):
        if not a:
            return [None, None]
        s = sorted(a)
        return [round(s[int(0.025 * len(s))], 5), round(s[min(int(0.975 * len(s)), len(s) - 1)], 5)]
    return {"n": n, "delta_recall": round(d_recall, 5), "ci_recall": ci(br),
            "n_both_matched": len(both),
            "delta_cdist": (round(d_cdist, 5) if d_cdist is not None else None),
            "ci_cdist": ci(bc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    folds_out = []
    csv_rows = []
    for fold, val in FOLDS:
        base, cand = load(fold, "baseline"), load(fold, "candidate")
        if base is None or cand is None:
            print(f"[skip] {fold}: eval 미완료 (baseline={base is not None} candidate={cand is not None})")
            continue
        sb, sc = soft15(base), soft15(cand)
        d_f1 = (None if sb["f1"] is None or sc["f1"] is None else round(sc["f1"] - sb["f1"], 5))
        d_rec = (None if sb["recall"] is None or sc["recall"] is None else round(sc["recall"] - sb["recall"], 5))
        d_prec = (None if sb["precision"] is None or sc["precision"] is None else round(sc["precision"] - sb["precision"], 5))
        d_vloss = round(cand["val_loss"] - base["val_loss"], 5)
        bb, cb = boxmap(base), boxmap(cand)
        boot_all = paired_bootstrap(bb, cb, args.boot, rng)
        boot_low = paired_bootstrap(bb, cb, args.boot, rng, speed_filter=(0.0, 1.0))
        boot_mid = paired_bootstrap(bb, cb, args.boot, rng, speed_filter=(1.0, 6.0))
        p3b = {b["bin"]: {"base": bin_recall(base, b["bin"]).get("recall"),
                          "cand": bin_recall(cand, b["bin"]).get("recall"),
                          "n_gt": b["n_gt"]}
               for b in cand["p3_filtered"]["bins"]}
        # provenance 정합(anchor SHA 는 fold 안에서 baseline/candidate 동일해야)
        prov_ok = (base["anchor_full_sha256"] == cand["anchor_full_sha256"])
        fo = {
            "fold": fold, "val": val, "provenance_anchor_match": prov_ok,
            "baseline_soft15": sb, "candidate_soft15": sc,
            "delta_soft15": {"f1": d_f1, "recall": d_rec, "precision": d_prec},
            "delta_val_loss": d_vloss,
            "p3_recall_by_bin": p3b,
            "boot_overall": boot_all, "boot_low_0_1": boot_low, "boot_mid_1_6": boot_mid,
            "baseline_val_frames": base["val_frames"], "candidate_val_frames": cand["val_frames"],
            "ckpt_update_step": {"base": base.get("ckpt_update_step"), "cand": cand.get("ckpt_update_step")},
        }
        folds_out.append(fo)
        csv_rows.append([fold, val, d_f1, d_rec, d_prec, d_vloss,
                         boot_all.get("delta_recall"), boot_all.get("ci_recall"),
                         boot_all.get("delta_cdist"), boot_all.get("ci_cdist"),
                         boot_low.get("delta_recall"), boot_low.get("ci_recall")])
        print(f"[{fold}] Δf1={d_f1} Δrecall={d_rec} Δcdist(overall)={boot_all.get('delta_cdist')} "
              f"CI={boot_all.get('ci_cdist')} n={boot_all.get('n')} provOK={prov_ok} "
              f"updates(base/cand)={base.get('ckpt_update_step')}/{cand.get('ckpt_update_step')}")

    def macro(key_path):
        vals = []
        for fo in folds_out:
            v = fo
            for k in key_path:
                v = (v or {}).get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)
        return (round(sum(vals) / len(vals), 5) if vals else None), len(vals)

    macro_out = {
        "n_folds": len(folds_out),
        "delta_f1_soft15": macro(["delta_soft15", "f1"]),
        "delta_recall_soft15": macro(["delta_soft15", "recall"]),
        "delta_precision_soft15": macro(["delta_soft15", "precision"]),
        "delta_val_loss": macro(["delta_val_loss"]),
        "delta_recall_boot_overall": macro(["boot_overall", "delta_recall"]),
        "delta_cdist_boot_overall": macro(["boot_overall", "delta_cdist"]),
        "delta_cdist_boot_low_0_1": macro(["boot_low_0_1", "delta_cdist"]),
    }
    # 방향 일치성(fold 간 부호 충돌 여부)
    def sign_consistency(key_path):
        signs = set()
        for fo in folds_out:
            v = fo
            for k in key_path:
                v = (v or {}).get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                signs.add(1 if v > 0 else (-1 if v < 0 else 0))
        return sorted(signs)
    consistency = {
        "delta_f1_signs": sign_consistency(["delta_soft15", "f1"]),
        "delta_recall_boot_signs": sign_consistency(["boot_overall", "delta_recall"]),
    }
    out = {"folds": folds_out, "macro": macro_out, "fold_sign_consistency": consistency,
           "n_bootstrap": args.boot,
           "note": ("candidate = v3-train, baseline = v2-train; 모두 v3 val. delta = candidate - baseline. "
                    "고정 budget 학습(수렴 아님)이므로 성능 delta 는 방향/안정성 참고용.")}
    json.dump(out, open(os.path.join(AB, "ab_summary.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(os.path.join(AB, "paired_delta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "val", "d_f1", "d_recall", "d_precision", "d_val_loss",
                    "boot_d_recall", "boot_ci_recall", "boot_d_cdist", "boot_ci_cdist",
                    "boot_low_d_recall", "boot_low_ci_recall"])
        w.writerows(csv_rows)
    print("\n[macro]", json.dumps(macro_out, ensure_ascii=False))
    print("[sign consistency]", consistency)
    print("wrote ab_summary.json, paired_delta.csv")


if __name__ == "__main__":
    main()

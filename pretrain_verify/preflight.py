#!/usr/bin/env python3
"""
task I: 짧은 학습 전 preflight (full training 아님).

production 코드 경로(train.AutoNavModel / compute_auxiliary_detection_loss / validate /
decode_detections, morai_dataset.MoraiTemporalDataset, loss_calculator.CustomLoss)를
그대로 사용해 다음을 동일 조건에서 검증한다:
  - baseline(train v2 / val v3) vs candidate(train v3 / val v3)
  - depth ON(production-equivalent) / depth OFF(label-correction isolation) 각각 양쪽 동일 조건
  - 동일 initial checkpoint, seed, anchor hash, sampler(batch stem) 순서
  - 각 설정 20 warm-up + ~100 optimizer step
  - 기록: batch stem 순서, 전체/세부 loss, grad finite/norm, param finite,
          peak CUDA allocated/reserved, throughput, val metric, anchor hash
  - checkpoint save/resume 1회
  - candidate depth-ON 재실행 최대 loss 차이(GPU 비결정성) 측정
env: PF_WARMUP(20) PF_STEPS(100) PF_VAL_FRAMES(120) PF_QUICK(0 → 2/3/8 로 스모크)
"""
import os, sys, json, time, hashlib
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = "/home/autonav/projects/morai-3d-detection"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch

ANCHOR_DIR = os.path.join(ROOT, "anchors", "v3_train_scen05_scen77_k900")
AFULL = os.path.join(ANCHOR_DIR, "anchor_kmeans_full.npy")
AXY = os.path.join(ANCHOR_DIR, "anchor_kmeans_xy.npy")
AMETA = os.path.join(ANCHOR_DIR, "anchor_kmeans_meta.json")

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()

# ── anchor fail-fast + 모델이 versioned anchor 를 읽도록 env 주입 ──
with open(AMETA) as f:
    ameta = json.load(f)
assert ameta["anchor_full_sha256"] == sha_file(AFULL), "anchor SHA mismatch — fail-fast"
os.environ["ANCHOR_FULL_FILE"] = AFULL
os.environ["ANCHOR_XY_FILE"] = AXY

import train
from morai_dataset import MoraiTemporalDataset, morai_temporal_collate_fn
from loss_calculator import CustomLoss
from torch.utils.data import DataLoader
import eval_relative_speed as p3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = os.path.join(ROOT, "dataset")
BATCH = 4
GRAD_ACCUM = 2
CLIP = 25.0

QUICK = os.environ.get("PF_QUICK", "0") not in ("0", "", "false")
WARMUP = int(os.environ.get("PF_WARMUP", "2" if QUICK else "20"))
STEPS = int(os.environ.get("PF_STEPS", "3" if QUICK else "100"))
VAL_FRAMES = int(os.environ.get("PF_VAL_FRAMES", "8" if QUICK else "120"))


def seed_all(s=0):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_model(use_dense_depth):
    m = train.AutoNavModel(
        num_decoder_layers=6, pretrained_backbone=True,
        use_temporal_memory=False, num_temp_instances=600,
        use_grid_mask=True, use_dense_depth=use_dense_depth,
    ).to(DEVICE)
    m.freeze_backbone_bn()
    return m


def make_optimizer(model):
    backbone_ids = set(id(p) for p in model.backbone.parameters())
    other = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "lr": 1e-5, "weight_decay": 1e-3},
        {"params": other, "lr": 5e-5, "weight_decay": 1e-2},
    ])


def lr_scale(step, warmup):
    return min(1.0, (step + 1) / max(warmup, 1))


def loaders(train_gt, val_gt, load_depth):
    tr = MoraiTemporalDataset(DATASET, "train", ["scen144"], photometric_aug=True,
                              load_depth=load_depth, gt_version=train_gt)
    va = MoraiTemporalDataset(DATASET, "val", ["scen144"], load_depth=load_depth, gt_version=val_gt)
    g = torch.Generator(); g.manual_seed(0)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g,
                    collate_fn=morai_temporal_collate_fn, num_workers=0)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False,
                    collate_fn=morai_temporal_collate_fn, num_workers=0)
    return tr, va, tl, vl


def train_steps(model, tl, use_dense_depth, warmup, steps, record_losses):
    crit = CustomLoss(num_classes=2, quality_weight=0.2).to(DEVICE)
    opt = make_optimizer(model)
    base_lrs = [g["lr"] for g in opt.param_groups]
    model.train(); model.freeze_backbone_bn()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    total_updates = warmup + steps
    dl_target = total_updates * GRAD_ACCUM
    stem_order, step_losses = [], []
    n_upd = 0; n_dl = 0
    all_finite_grad = True; all_finite_param = True; max_grad_norm = 0.0
    t0 = time.time()
    it = iter(tl)
    while n_upd < total_updates:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tl); batch = next(it)
        n_dl += 1
        if len(stem_order) < 24:
            stem_order.append(list(batch["stem"]))
        images = batch["images"].to(DEVICE)
        intr = batch["intrinsics"].to(DEVICE)
        extr = batch["extrinsics"].to(DEVICE)
        ego = batch["ego_pose"].to(DEVICE)
        focal = batch["focal"].to(DEVICE) if (use_dense_depth and "focal" in batch) else None
        out = model(images, intr, extr, stems=batch["stem"], ego_poses=ego,
                    focal=focal, timestamps=batch.get("timestamp"), return_intermediate=True)
        bl, cl, bx, ql = train.compute_auxiliary_detection_loss(out, batch, crit, DEVICE, aux_weight=0.5)
        dl = images.new_tensor(0.0)
        if out.get("depth_pred") is not None:
            gtd = [g.to(DEVICE) for g in batch["gt_depth"]]
            dl = model.depth_net.loss(out["depth_pred"], gtd)
            bl = bl + dl
        (bl / GRAD_ACCUM).backward()
        if record_losses:
            step_losses.append(float(bl.item()))
        for name, t in (("cls", cl), ("box", bx), ("q", ql), ("depth", dl), ("total", bl)):
            if not torch.isfinite(t).all():
                all_finite_param = False
        should_step = (n_dl % GRAD_ACCUM == 0)
        if should_step:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            gnf = bool(torch.isfinite(gnorm))
            all_finite_grad = all_finite_grad and gnf
            max_grad_norm = max(max_grad_norm, float(gnorm))
            sc = lr_scale(n_upd, warmup)
            for grp, blr in zip(opt.param_groups, base_lrs):
                grp["lr"] = blr * sc
            opt.step(); opt.zero_grad(set_to_none=True)
            n_upd += 1
            if n_upd <= 24 or n_upd % 25 == 0:
                for p in model.parameters():
                    if not torch.isfinite(p).all():
                        all_finite_param = False
                        break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    peak_alloc = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
    peak_resv = (torch.cuda.max_memory_reserved() / (1024**3)) if torch.cuda.is_available() else 0.0
    return {
        "n_optimizer_updates": n_upd, "n_dataloader_steps": n_dl,
        "wall_s": dt, "updates_per_s": n_upd / dt if dt else 0.0,
        "dl_steps_per_s": n_dl / dt if dt else 0.0,
        "peak_cuda_alloc_gib": peak_alloc, "peak_cuda_reserved_gib": peak_resv,
        "grad_all_finite": all_finite_grad, "param_all_finite": all_finite_param,
        "max_grad_norm": max_grad_norm,
        "first_loss": step_losses[0] if step_losses else None,
        "last_loss": step_losses[-1] if step_losses else None,
        "loss_finite": bool(np.all(np.isfinite(step_losses))) if step_losses else True,
        "batch_stem_order": stem_order, "step_losses": step_losses, "opt": opt,
    }


@torch.no_grad()
def dump_val_preds(model, vl, max_frames):
    model.eval()
    if hasattr(model, "reset_temporal_memory"):
        model.reset_temporal_memory()
    preds = {"scen": [], "stem": [], "x": [], "y": [], "label": [], "score": []}
    done = 0
    for batch in vl:
        if done >= max_frames:
            break
        images = batch["images"].to(DEVICE); intr = batch["intrinsics"].to(DEVICE)
        extr = batch["extrinsics"].to(DEVICE); ego = batch["ego_pose"].to(DEVICE)
        out = model(images, intr, extr, stems=batch["stem"], ego_poses=ego,
                    timestamps=batch.get("timestamp"), return_intermediate=True)
        dc = out["det_cls"].detach().cpu(); db = out["det_box"].detach().cpu()
        dq = out["det_quality"].detach().cpu()
        for i in range(images.shape[0]):
            boxes, labels, scores = train.decode_detections(
                dc[i], db[i], det_quality=dq[i], score_thresh=0.15,
                apply_quality=True, quality_power=0.5)
            # loader stem 은 "scenario#seg/live_N" → base stem("live_N") + scenario 로 분리(하드코드 제거)
            base_stem = batch["stem"][i].split("/")[-1]
            preds["scen"].append(batch["scenario"][i]); preds["stem"].append(base_stem)
            preds["x"].append(boxes[:, 0].numpy().astype(np.float32))
            preds["y"].append(boxes[:, 1].numpy().astype(np.float32))
            preds["label"].append(labels.numpy().astype(np.int64))
            preds["score"].append(scores.numpy().astype(np.float32))
            done += 1
    return preds


def val_metric(model, vl, max_frames):
    crit = CustomLoss(num_classes=2, quality_weight=0.2).to(DEVICE)
    vloss, vm = train.validate(model, vl, crit, DEVICE, compute_metric=True,
                               recall_thr=2.0, max_frames=max_frames)
    head = None
    if vm is not None:
        soft = vm["by_mode"]["softcalibrated"].get(0.15)
        head = {"precision": soft["precision"], "recall": soft["recall"], "f1": soft["f1"]} if soft else None
    return float(vloss), head


def p3_recall(preds, val_gt, val_scens):
    # production validation filter 이후 GT(모델이 실제 채점되는 membership) + rel_speed.
    frames, n_missing_total = [], 0
    for scen in val_scens:
        fr, _audit, nm = p3.load_filtered_gt_with_relspeed(DATASET, scen, val_gt)
        frames.extend(fr)
        n_missing_total += nm
    validated = set(zip(preds["scen"], preds["stem"]))   # 실제 검증한 (scen, base_stem)
    frames = [f for f in frames if (f["scen"], f["stem"]) in validated]
    pbk = {}
    for i in range(len(preds["stem"])):
        pbk[(preds["scen"][i], preds["stem"][i])] = {
            "x": np.atleast_1d(preds["x"][i]).astype(float),
            "y": np.atleast_1d(preds["y"][i]).astype(float),
            "label": np.atleast_1d(preds["label"][i]).astype(int),
            "score": np.atleast_1d(preds["score"][i]).astype(float)}
    ev = p3.evaluate_relative_speed(frames, pbk, p3.DEFAULT_BIN_EDGES, dist_thr=2.0,
                                    require_nonempty=True)
    ev["n_relspeed_missing"] = n_missing_total
    ev["membership"] = "production_validation_filter(_load_labels_v2)"
    return ev


def run_config(name, train_gt, val_gt, use_dense_depth):
    print(f"\n===== CONFIG {name}: train={train_gt} val={val_gt} depth={'ON' if use_dense_depth else 'OFF'} =====", flush=True)
    seed_all(0)
    model = build_model(use_dense_depth)
    # 공유 init.pth 로드(depth OFF 모델은 depth_net 키가 없어 필터 로드 → 공유 파라미터는 동일)
    init_sd = torch.load(INIT_PATH, map_location=DEVICE)
    msd = model.state_dict()
    filt = {k: v for k, v in init_sd.items() if k in msd and v.shape == msd[k].shape}
    msd.update(filt); model.load_state_dict(msd)
    print(f"  init load: {len(filt)}/{len(init_sd)} params from shared init", flush=True)
    seed_all(0)  # 학습 루프 RNG(GridMask draws) 동일화
    tr, va, tl, vl = loaders(train_gt, val_gt, load_depth=use_dense_depth)
    res = train_steps(model, tl, use_dense_depth, WARMUP, STEPS, record_losses=True)
    opt = res.pop("opt")
    vloss, head = val_metric(model, vl, VAL_FRAMES)
    preds = dump_val_preds(model, vl, VAL_FRAMES)
    # P3 는 join 무결성(frame/GT=0)이면 fail-fast(ValueError). 단 model 이 아직 예측 0개인
    # 것은 학습 부족일 뿐이므로 preflight 는 crash 대신 사유를 기록한다(비침묵).
    try:
        ev = p3_recall(preds, val_gt, va.scenario_names)
    except ValueError as _e:
        ev = {"error": str(_e), "bins": [],
              "n_pred_total": int(sum(len(s) for s in preds["score"])),
              "n_pred_unmatched": 0}
    res["val_loss"] = vloss
    res["val_softcalibrated@0.15"] = head
    res["p3_recall_by_bin"] = ev
    res["anchor_sha"] = ANCHOR_SHA
    res["train_gt"] = train_gt; res["val_gt"] = val_gt; res["depth"] = use_dense_depth
    print(f"  updates={res['n_optimizer_updates']} loss {res['first_loss']:.3f}->{res['last_loss']:.3f} "
          f"gradFinite={res['grad_all_finite']} paramFinite={res['param_all_finite']} "
          f"peak={res['peak_cuda_alloc_gib']:.2f}GiB upd/s={res['updates_per_s']:.2f} "
          f"val_loss={vloss:.3f} val_soft@0.15={head}", flush=True)
    if ev.get("error"):
        print(f"  P3(filtered) FAIL-FAST recorded: {ev['error']}", flush=True)
    else:
        _rb = {b["bin"]: (b["n_gt"], b["n_matched"], round(b["recall"], 4) if b["recall"] is not None else None)
               for b in ev["bins"]}
        print(f"  P3(filtered) frames={ev['n_frames_evaluated']} gt={ev['total_gt']} "
              f"pred={ev['n_pred_total']} matched={ev['n_pred_total']-ev['n_pred_unmatched']} "
              f"bins[n_gt,n_matched,recall]={_rb}", flush=True)
    return model, opt, res


# ── shared init ──
seed_all(0)
_m0 = build_model(use_dense_depth=True)
ANCHOR_SHA = sha_bytes(_m0.det_anchors_full.detach().cpu().numpy().tobytes())
INIT_PATH = os.path.join(ROOT, "pretrain_verify", "init_model.pth")
torch.save(_m0.state_dict(), INIT_PATH)
INIT_SHA = sha_file(INIT_PATH)
del _m0
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"[init] shared init saved sha={INIT_SHA[:16]} anchor_sha={ANCHOR_SHA[:16]}", flush=True)
print(f"[cfg] WARMUP={WARMUP} STEPS={STEPS} VAL_FRAMES={VAL_FRAMES} device={DEVICE} "
      f"deformable=grid_sample_fallback(no nvcc)", flush=True)

report = {"init_sha": INIT_SHA, "anchor_sha": ANCHOR_SHA, "device": str(DEVICE),
          "warmup": WARMUP, "steps": STEPS, "val_frames": VAL_FRAMES,
          "grad_accum": GRAD_ACCUM, "configs": {}}

# ── PROD pair (depth ON): baseline vs candidate ──
mb, ob, rb = run_config("PROD_baseline_v2v3", "v2", "v3", use_dense_depth=True)
report["configs"]["PROD_baseline_v2v3"] = {k: v for k, v in rb.items() if k not in ("batch_stem_order", "step_losses")}
prod_baseline_losses = rb["step_losses"]; prod_order_b = rb["batch_stem_order"]

mc, oc, rc = run_config("PROD_candidate_v3v3", "v3", "v3", use_dense_depth=True)
report["configs"]["PROD_candidate_v3v3"] = {k: v for k, v in rc.items() if k not in ("batch_stem_order", "step_losses")}
prod_order_c = rc["batch_stem_order"]

# sampler order identical between baseline/candidate (same seed → same shuffle)
order_same = (prod_order_b == prod_order_c)
report["prod_sampler_order_identical"] = order_same
print(f"\n[order] PROD baseline vs candidate batch stem order identical: {order_same}", flush=True)

# checkpoint save/resume (candidate PROD)
ckpt = os.path.join(ROOT, "pretrain_verify", "pf_candidate_ckpt.pth")
torch.save({"model_state": mc.state_dict(), "optimizer_state": oc.state_dict(),
            "anchor_sha": ANCHOR_SHA}, ckpt)
m_res = build_model(use_dense_depth=True)
ck = torch.load(ckpt, map_location=DEVICE)
m_res.load_state_dict(ck["model_state"])
same = all(torch.equal(a.detach().cpu(), b.detach().cpu())
           for a, b in zip(mc.state_dict().values(), m_res.state_dict().values()))
report["checkpoint_save_resume_ok"] = bool(same)
print(f"[ckpt] save/resume param-identical: {same}", flush=True)

# ── ISO pair (depth OFF): baseline vs candidate ──
_, _, rib = run_config("ISO_baseline_v2v3", "v2", "v3", use_dense_depth=False)
report["configs"]["ISO_baseline_v2v3"] = {k: v for k, v in rib.items() if k not in ("batch_stem_order", "step_losses")}
_, _, ric = run_config("ISO_candidate_v3v3", "v3", "v3", use_dense_depth=False)
report["configs"]["ISO_candidate_v3v3"] = {k: v for k, v in ric.items() if k not in ("batch_stem_order", "step_losses")}

# ── nondeterminism: rerun candidate PROD, compare step losses ──
seed_all(0)
_, _, rc2 = run_config("PROD_candidate_v3v3_rerun", "v3", "v3", use_dense_depth=True)
L1 = np.array(rc["step_losses"]); L2 = np.array(rc2["step_losses"])
n = min(len(L1), len(L2))
max_diff = float(np.max(np.abs(L1[:n] - L2[:n]))) if n else None
report["nondeterminism"] = {
    "rerun_max_abs_loss_diff": max_diff,
    "note": ("GPU grid_sample/deformable fallback backward(atomicAdd)는 비결정적 → bitwise 동일 주장 안 함. "
             "재실행 최대 loss 차이로 정량화."),
    "first_loss_run1": float(L1[0]) if len(L1) else None,
    "first_loss_run2": float(L2[0]) if len(L2) else None,
}
print(f"\n[nondet] candidate PROD rerun max|Δloss|={max_diff}", flush=True)

# NaN/Inf/OOM summary
any_bad = any((not c["grad_all_finite"]) or (not c["param_all_finite"]) or (not c["loss_finite"])
              for c in report["configs"].values())
report["preflight_no_nan_inf"] = not any_bad
out = os.path.join(ROOT, "pretrain_verify", "preflight_report.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\n[done] no NaN/Inf across configs: {not any_bad} | wrote {out}", flush=True)

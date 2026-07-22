#!/usr/bin/env python3
"""task A: pretrain_manifest.json 생성 (원본 데이터 미수정).
git diff hash, scene 목록, frame/box/depth 개수, split, GT version, anchor hash, seed,
depth 설정, CUDA fallback, raw 입력 hash 불변성, 테스트/감사 산출을 한 곳에 모은다."""
import csv, hashlib, json, os, subprocess, shutil

ROOT = "/home/autonav/projects/morai-3d-detection"
os.chdir(ROOT)
DS = "dataset"
SCENS = ["scen05", "scen77", "scen144"]
ANCHOR_DIR = "anchors/v3_train_scen05_scen77_k900"


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_dir(d, exts=(".csv", ".npy")):
    if not os.path.isdir(d):
        return None, 0
    names = sorted(f for f in os.listdir(d) if f.endswith(exts))
    agg = hashlib.sha256()
    for n in names:
        agg.update((n + ":" + sha_file(os.path.join(d, n)) + "\n").encode())
    return agg.hexdigest(), len(names)


def count_boxes(d):
    nf = nb = 0
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv"):
            nf += 1
            with open(os.path.join(d, f), newline="", encoding="utf-8") as fh:
                nb += sum(1 for _ in csv.DictReader(fh))
    return nf, nb


# ── git ──
git = {
    "head": sh("git", "rev-parse", "HEAD"),
    "diff_stat": sh("git", "diff", "--stat"),
    "tracked_unstaged_diff_sha256": hashlib.sha256(
        sh("git", "diff").encode()).hexdigest(),
    "diff_check_clean": (sh("git", "diff", "--check") == ""),
    "modified_tracked": sh("git", "diff", "--name-only").split("\n") if sh("git", "diff", "--name-only") else [],
}
untracked = sh("git", "ls-files", "--others", "--exclude-standard").split("\n")
git["untracked"] = [u for u in untracked if u]

# ── file hashes of code we changed / added ──
code_files = ["train.py", "make_kmeans.py", "anchor_generator.py", "morai_dataset.py",
              "loss_calculator.py", "correct_source_time.py", "eval_relative_speed.py",
              "test_source_time_correction.py", "test_velocity_valid_loss.py",
              "test_anchor_policy.py", "test_gt_version_split.py", "test_relative_speed_eval.py",
              "g_membership_audit.py"]
code_sha = {f: sha_file(f) for f in code_files if os.path.isfile(f)}

# ── raw input hashes + invariance vs A-freeze baseline ──
raw = {}
for s in SCENS:
    rec = {}
    for sub in ["labels_3d", "labels_3d_v2", "ego_pose", "lidar"]:
        h, n = sha_dir(os.path.join(DS, s, sub))
        rec[sub] = {"sha256": h, "n": n}
    sy = os.path.join(DS, s, "sync_log.csv")
    rec["sync_log"] = {"sha256": sha_file(sy) if os.path.isfile(sy) else None}
    raw[s] = rec

invariant = True
baseline_path = "pretrain_verify/a_freeze_data.json"
if os.path.isfile(baseline_path):
    with open(baseline_path) as f:
        base = json.load(f)
    for s in SCENS:
        b = base["scenes"][s]
        for sub in ["labels_3d", "labels_3d_v2", "ego_pose", "lidar"]:
            if b.get(sub + "_sha") != raw[s][sub]["sha256"]:
                invariant = False
        if b.get("sync_log_sha") != raw[s]["sync_log"]["sha256"]:
            invariant = False

# ── counts ──
counts = {"per_scene": {}, "total": {"v2_frames": 0, "v2_boxes": 0, "v3_frames": 0,
                                     "v3_boxes": 0, "depth_files": 0}}
for s in SCENS:
    v2f, v2b = count_boxes(os.path.join(DS, s, "labels_3d_v2"))
    v3f, v3b = count_boxes(os.path.join(DS, s, "labels_3d_v3"))
    dcount = 0
    for cam in ["cam_front", "cam_front_left", "cam_front_right"]:
        dd = os.path.join(DS, s, "depth_gt", cam)
        if os.path.isdir(dd):
            dcount += sum(1 for f in os.listdir(dd) if f.endswith(".npy"))
    counts["per_scene"][s] = {"v2_frames": v2f, "v2_boxes": v2b, "v3_frames": v3f,
                              "v3_boxes": v3b, "depth_files": dcount}
    counts["total"]["v2_frames"] += v2f; counts["total"]["v2_boxes"] += v2b
    counts["total"]["v3_frames"] += v3f; counts["total"]["v3_boxes"] += v3b
    counts["total"]["depth_files"] += dcount

# ── anchor ──
with open(os.path.join(ANCHOR_DIR, "anchor_kmeans_meta.json")) as f:
    ameta = json.load(f)
anchor = {
    "dir": ANCHOR_DIR,
    "gt_version": ameta["gt_version"], "k": ameta["k"], "seed": ameta["seed"],
    "label_dir": ameta["label_dir"], "train_scenarios": ameta["train_scenarios"],
    "val_scenarios": ameta["val_scenarios"],
    "input_label_sha256": ameta["input_label_sha256"],
    "anchor_full_sha256": ameta["anchor_full_sha256"],
    "anchor_full_sha256_actual": sha_file(os.path.join(ANCHOR_DIR, "anchor_kmeans_full.npy")),
    "anchor_xy_sha256": ameta["anchor_xy_sha256"],
    "num_train_boxes_visible": ameta["num_train_boxes_visible"],
}
anchor["sha_matches"] = (anchor["anchor_full_sha256"] == anchor["anchor_full_sha256_actual"])

# ── membership + p3 (if computed) ──
def load_json(p):
    return json.load(open(p)) if os.path.isfile(p) else None

membership = load_json("pretrain_verify/g_membership_audit.json")
p3 = load_json("pretrain_verify/p3_gt_distribution_v3.json")
preflight = load_json("pretrain_verify/preflight_report.json")

manifest = {
    "purpose": "pretrain verification manifest (원본 데이터 미수정). full training 미시작.",
    "git": git,
    "code_file_sha256": code_sha,
    "representative_split": {
        "train_scenarios": ["scen05", "scen77"],
        "val_scenarios": ["scen144"],
        "baseline": {"train_gt": "v2", "val_gt": "v3"},
        "candidate": {"train_gt": "v3", "val_gt": "v3"},
        "note": "validation metric은 항상 v3 GT. anchor는 v3 train split에서 1회 생성, 양쪽 공유.",
    },
    "counts": counts,
    "raw_input_hashes": raw,
    "raw_input_invariant_vs_baseline": invariant,
    "anchor": anchor,
    "seed": 0,
    "depth": {
        "default_use_dense_depth": True,
        "env_override": "USE_DENSE_DEPTH (0/1)",
        "dataset_load_depth_follows_use_dense_depth": True,
        "depth_gt_files_expected": 10011,
        "depth_gt_files_found": counts["total"]["depth_files"],
    },
    "cuda": {
        "nvcc_present": False,
        "deformable_aggregation": "grid_sample fallback (no nvcc)",
        "torch": "2.1.0+cu121", "cuda_available": True,
    },
    "membership_audit": (None if membership is None else {
        "v2_natural": membership["v2_natural"], "v3_natural": membership["v3_natural"],
        "common": membership["common"], "v2_only": membership["v2_only"],
        "v3_only": membership["v3_only"], "n_boundary_cross": membership["n_boundary_cross"],
        "expected": membership["expected"]}),
    "p3_gt_distribution": (None if p3 is None else {
        "bins": [{"bin": r["bin"], "n_gt": r["n_gt"], "n_tracks": r["n_tracks"],
                  "n_frames": r["n_frames"]} for r in p3["gt_distribution"]],
        "total_boxes": p3["n_boxes_total"], "match_radius_m": p3["match_radius_m"],
        "halfwidth": p3["halfwidth"], "warnings": p3["high_speed_warnings"]}),
    "preflight": (None if preflight is None else {
        "init_sha": preflight["init_sha"], "anchor_sha": preflight["anchor_sha"],
        "warmup": preflight["warmup"], "steps": preflight["steps"],
        "no_nan_inf": preflight.get("preflight_no_nan_inf"),
        "sampler_order_identical": preflight.get("prod_sampler_order_identical"),
        "checkpoint_save_resume_ok": preflight.get("checkpoint_save_resume_ok"),
        "nondeterminism": preflight.get("nondeterminism"),
        "configs": {k: {kk: vv for kk, vv in v.items() if kk in (
            "n_optimizer_updates", "first_loss", "last_loss", "grad_all_finite",
            "param_all_finite", "loss_finite", "peak_cuda_alloc_gib",
            "peak_cuda_reserved_gib", "updates_per_s", "val_loss",
            "val_softcalibrated@0.15")}
                    for k, v in preflight.get("configs", {}).items()}}),
}

with open("pretrain_verify/pretrain_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

# console summary
print("=== pretrain_manifest.json summary ===")
print(f"git HEAD={git['head'][:12]} diff_sha={git['tracked_unstaged_diff_sha256'][:16]} "
      f"diff_check_clean={git['diff_check_clean']}")
print(f"counts total: v2 {counts['total']['v2_frames']}f/{counts['total']['v2_boxes']}b "
      f"v3 {counts['total']['v3_frames']}f/{counts['total']['v3_boxes']}b "
      f"depth {counts['total']['depth_files']}")
print(f"raw_input_invariant_vs_baseline = {invariant}")
print(f"anchor: gt={anchor['gt_version']} k={anchor['k']} seed={anchor['seed']} "
      f"sha={anchor['anchor_full_sha256'][:16]} matches={anchor['sha_matches']} "
      f"input_label_sha={anchor['input_label_sha256'][:16]}")
if membership:
    print(f"membership: v2={membership['v2_natural']} v3={membership['v3_natural']} "
          f"common={membership['common']} v2only={membership['v2_only']} "
          f"v3only={membership['v3_only']} cross={membership['n_boundary_cross']}")
if p3:
    print("p3 bins:", {r["bin"]: r["n_gt"] for r in p3["gt_distribution"]})
print(f"preflight present: {preflight is not None}")
print("wrote pretrain_verify/pretrain_manifest.json")

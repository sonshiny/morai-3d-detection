#!/usr/bin/env python3
"""Task A data-side freeze: input hashes, v2/v3 counts, epoch structure, existing reports."""
import os, sys, json, csv, hashlib, collections

ROOT = "/home/autonav/projects/morai-3d-detection"
DS = os.path.join(ROOT, "dataset")
SCENS = ["scen05", "scen77", "scen144"]

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_dir_csv(d):
    """Order-independent aggregate: sha256 over sorted 'name:filehash' lines."""
    if not os.path.isdir(d):
        return None, 0
    names = sorted(f for f in os.listdir(d) if f.endswith(".csv") or f.endswith(".npy"))
    agg = hashlib.sha256()
    for n in names:
        agg.update((n + ":" + sha_file(os.path.join(d, n)) + "\n").encode())
    return agg.hexdigest(), len(names)

def count_boxes(label_dir):
    nframes = nboxes = 0
    for f in sorted(os.listdir(label_dir)):
        if not f.endswith(".csv"):
            continue
        nframes += 1
        with open(os.path.join(label_dir, f), newline="", encoding="utf-8") as fh:
            nboxes += sum(1 for _ in csv.DictReader(fh))
    return nframes, nboxes

out = {"scenes": {}, "raw_input_hashes": {}, "totals": {}}
tot = collections.Counter()
for s in SCENS:
    sd = os.path.join(DS, s)
    rec = {}
    # raw input aggregate hashes (must stay invariant)
    for sub in ["labels_3d", "labels_3d_v2", "ego_pose", "lidar"]:
        h, n = sha_dir_csv(os.path.join(sd, sub))
        rec[sub + "_sha"] = h
        rec[sub + "_n"] = n
    sync = os.path.join(sd, "sync_log.csv")
    rec["sync_log_sha"] = sha_file(sync) if os.path.isfile(sync) else None
    # v2 / v3 counts
    for v, ld in [("v2", "labels_3d_v2"), ("v3", "labels_3d_v3")]:
        ldp = os.path.join(sd, ld)
        if os.path.isdir(ldp):
            nf, nb = count_boxes(ldp)
            rec[f"{v}_frames"], rec[f"{v}_boxes"] = nf, nb
            tot[f"{v}_frames"] += nf
            tot[f"{v}_boxes"] += nb
    # epoch structure from sync_log save rows
    epochs = collections.Counter()
    saves = 0
    if os.path.isfile(sync):
        with open(sync, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("action") == "save":
                    saves += 1
                    epochs[r.get("epoch_id", "")] += 1
    rec["sync_saves"] = saves
    rec["epoch_ids"] = dict(epochs)
    rec["n_epochs"] = len(epochs)
    # existing timing report
    trp = os.path.join(sd, "timing_correction_report.json")
    if os.path.isfile(trp):
        with open(trp) as fh:
            tr = json.load(fh)
        rec["existing_timing_report"] = {
            k: tr.get(k) for k in
            ["n_boxes", "p50_m", "p95_m", "max_m", "mean_m", "frac_gt_0p2m",
             "n_correction_valid", "n_correction_invalid", "n_frames_no_sync",
             "n_vel_motion_tracks", "ego_interp_conflicts", "max_interp_gap_s",
             "interp_method_counts", "frame_valid_counts"]
        }
    out["scenes"][s] = rec

out["totals"] = dict(tot)
os.makedirs(os.path.join(ROOT, "pretrain_verify"), exist_ok=True)
with open(os.path.join(ROOT, "pretrain_verify", "a_freeze_data.json"), "w") as f:
    json.dump(out, f, indent=2)

# console summary
print("=== FRAME / BOX COUNTS ===")
for s in SCENS:
    r = out["scenes"][s]
    print(f"  {s}: v2 {r.get('v2_frames')}f/{r.get('v2_boxes')}b  "
          f"v3 {r.get('v3_frames')}f/{r.get('v3_boxes')}b  "
          f"epochs={r['n_epochs']} saves={r['sync_saves']} epoch_ids={list(r['epoch_ids'])}")
print(f"  TOTAL: v2 {tot['v2_frames']}f/{tot['v2_boxes']}b  v3 {tot['v3_frames']}f/{tot['v3_boxes']}b")
print("\n=== EXISTING v3 TIMING REPORT (per scene) ===")
for s in SCENS:
    tr = out["scenes"][s].get("existing_timing_report", {})
    print(f"  {s}: n_boxes={tr.get('n_boxes')} p50={tr.get('p50_m')} p95={tr.get('p95_m')} "
          f"max={tr.get('max_m')} >0.2m={tr.get('frac_gt_0p2m')} "
          f"valid={tr.get('n_correction_valid')} invalid={tr.get('n_correction_invalid')} "
          f"noSync={tr.get('n_frames_no_sync')} maxgap={tr.get('max_interp_gap_s')}")
    print(f"        methods={tr.get('interp_method_counts')}")
print("\n=== RAW INPUT AGGREGATE HASHES ===")
for s in SCENS:
    r = out["scenes"][s]
    print(f"  {s}: labels_3d={r['labels_3d_sha'][:16] if r['labels_3d_sha'] else None} "
          f"v2={r['labels_3d_v2_sha'][:16] if r['labels_3d_v2_sha'] else None} "
          f"ego={r['ego_pose_sha'][:16] if r['ego_pose_sha'] else None} "
          f"sync={r['sync_log_sha'][:16] if r['sync_log_sha'] else None}")
print("\nwrote pretrain_verify/a_freeze_data.json")

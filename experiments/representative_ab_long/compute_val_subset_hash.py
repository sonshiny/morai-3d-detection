#!/usr/bin/env python3
"""subset150 monitoring 프레임 목록의 결정성/동일성 provenance.
train.py validate() 는 max_frames>0 일 때 (shuffle=False) val_loader 앞쪽 N 프레임만 처리한다.
baseline/candidate/seed 는 같은 val scene → **동일 순서·동일 subset** 임을 hash 로 확증한다.
GT/pred join 여부는 학습 후 eval 단계에서 확인(여기서는 subset 정체성만).
사용: python3 compute_val_subset_hash.py <val_scen> <N> <out_json> [val_gt=v3]
"""
import sys, os, json, hashlib
os.environ.setdefault("WANDB_MODE", "disabled"); os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)

def main():
    val_scen = sys.argv[1]; N = int(sys.argv[2]); out = sys.argv[3]
    val_gt = sys.argv[4] if len(sys.argv) > 4 else "v3"
    from morai_dataset import MoraiTemporalDataset, morai_temporal_collate_fn
    from torch.utils.data import DataLoader
    dataset_root = os.path.abspath(os.environ.get("DATASET_ROOT", os.path.join(ROOT, "dataset")))
    ds = MoraiTemporalDataset(dataset_root, "val", [val_scen], load_depth=False, gt_version=val_gt)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=morai_temporal_collate_fn, num_workers=0)
    stems = []
    for batch in dl:
        for s in batch["stem"]:
            stems.append(str(s))
            if N > 0 and len(stems) >= N:
                break
        if N > 0 and len(stems) >= N:
            break
    subset = stems[:N] if N > 0 else stems
    h = hashlib.sha256(("\n".join(subset)).encode()).hexdigest()
    rec = {"val_scen": val_scen, "val_gt": val_gt, "n_requested": N, "n_actual": len(subset),
           "total_val_frames": len(ds), "stems_sha256": h,
           "first_stem": subset[0] if subset else None,
           "last_stem": subset[-1] if subset else None}
    json.dump(rec, open(out, "w"), indent=2)
    print("subset_hash %s scen=%s n=%d/%d sha=%s" % (h[:16], val_scen, len(subset), len(ds), h[:16]))
    if len(subset) == 0:
        sys.exit("FATAL: subset 0 frames")

if __name__ == "__main__":
    main()

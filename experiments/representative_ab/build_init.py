#!/usr/bin/env python3
"""fold별 공유 initial checkpoint 생성.
seed 고정 + fold anchor(ANCHOR_FULL_FILE env)로 depth-OFF 모델을 구성해 state_dict 저장.
baseline(v2)/candidate(v3)가 이 동일 파일을 RESUME(warm-start)해 동일 init 에서 출발한다.
learnable init 은 seed 로 결정되어 fold 무관 동일, det_anchors_full 버퍼만 fold anchor.
사용: ANCHOR_FULL_FILE=<fold full.npy> python3 build_init.py <out.pth> <seed>"""
import os, sys, hashlib
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = "/home/autonav/projects/morai-3d-detection"
os.chdir(ROOT); sys.path.insert(0, ROOT)
import random, numpy as np, torch

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

import train
m = train.AutoNavModel(num_decoder_layers=6, pretrained_backbone=True,
                       use_temporal_memory=False, num_temp_instances=600,
                       use_grid_mask=True, use_dense_depth=False)
sd = m.state_dict()
torch.save(sd, out)
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
anc = hashlib.sha256(m.det_anchors_full.detach().cpu().numpy().tobytes()).hexdigest()
print(f"[init] saved {out} seed={seed} state_sha={h[:16]} anchor_buffer_sha={anc[:16]} "
      f"anchor_file={os.environ.get('ANCHOR_FULL_FILE')}")

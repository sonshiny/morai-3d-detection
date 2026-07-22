<!-- STATUS_BANNER -->
> # ⛔ 상태: 150-scene PRODUCTION = **BLOCKED**
> 대표 A/B 판정(`experiments/representative_ab/AB_REPORT.md`): **TECHNICAL_GO=PASS ·
> V3_PERFORMANCE_GO=INCONCLUSIVE · PRODUCTION_150_GO=NO-GO.**
> 파이프라인/재현성/안전장치는 GO 지만, 고정 300-update budget 에서 v3−v2 성능 우열이 확정되지
> 않았다(macro Δf1=−0.008, 지표·fold 방향 상충, 미수렴). **따라서 이 문서의 §14 anchor 생성 이후
> production 학습(§19)은 BLOCKED**. 해제 조건: (1) seed=1 반복 + 수렴/장기 matched budget 재학습으로
> V3_PERFORMANCE 재판정, (2) 150-scene 전체 감사·split 승인·전체 anchor·full-split preflight(§6–18),
> (3) depth 동적 동기화 정량화. §1–18(감사·전처리·검증 파이프라인)은 지금 그대로 재현·수행 가능하다.

# HANDOFF — 150-scene source-time v3 재현·전처리·검증·학습 인수인계

동료 PC 에서 이 저장소를 clone 한 뒤, **문서 순서대로** 실행하면 150-scene 데이터 감사 →
labels_3d_v3 생성 → depth_gt 생성 → 최종 split 승인 → anchor 생성 → preflight → (승인 시)
production 학습까지 재현할 수 있다. 모든 경로는 **환경변수 기반**이며 현재 PC 절대경로를
동료 PC 에 강요하지 않는다.

핵심 환경변수:
- `DATASET_ROOT` : 데이터 루트(예: `/data/dataset`). scen01…scenNNN 하위 디렉터리 포함.
- `ANCHOR_DIR`   : 최종 split 로 생성한 anchor 디렉터리.
- `RUN_DIR`      : run 별 출력 격리 디렉터리(checkpoint/log/history/run_config).

---

## 0. 현재 검증 상태와 이 문서의 전제
- 대표 3-scene(scen05/77/144)에서 v3 label 을 **정식(고정 budget) 3-fold A/B** 로 검증했다
  (`experiments/representative_ab/AB_REPORT.md`). 판정은 세 축으로 분리한다:
  **TECHNICAL_GO / V3_PERFORMANCE_GO(or INCONCLUSIVE) / PRODUCTION_150_GO(항상 NO-GO)**.
- **PRODUCTION_150_GO 는 이 인수인계 시점에도 NO-GO** 다. 이유: 150-scene 전체 anchor·preflight
  가 아직 없고, LiDAR–camera 동적 depth 동기화가 미해결이다.
- **P1 anisotropic loss 는 구현하지 않았고(NO-GO)**, 150-scene 상대속도 분포 + v3 residual 분석
  전에는 구현하지 않는다.
- 대표 A/B primary 는 **USE_DENSE_DEPTH=0(depth OFF)** 로 수행했다(아래 §11).

## 1. 목적과 검증 범위
- 목적: source-time 보정 GT(v3)를 150-scene 로 확장해 production 학습하기 위한, **재현 가능한**
  전처리·검증 파이프라인과 안전장치를 인수인계한다.
- 범위: 이 문서는 데이터 감사부터 preflight 까지를 **동료가 직접** 수행하도록 안내한다.
  실제 150 production 학습은 **최종 split 승인 + full-split preflight 통과 후**에만 시작한다.

## 2. 동료 PC 요구 환경
- OS: Linux/WSL2(Ubuntu 20.04 검증). Windows 는 WSL2 안에서 실행.
- Python 3.8+ (검증 3.8).
- PyTorch 2.1.0 + CUDA 12.1 (`torch.cuda.is_available()==True`).
- GPU: 최소 12GB(검증 환경 12GB). depth OFF batch=4 peak ≈ 6–8GiB.
- `nvcc` 유무: 없어도 됨. 없으면 deformable aggregation 이 **grid_sample fallback**(정확도 무해,
  속도 느림). 있으면 CUDA op 컴파일로 가속 가능.
- 의존성: `pip install -r requirements.txt` (torch/scipy/numpy/opencv/matplotlib 등).
- 확인:
  ```bash
  python3 -c "import torch,scipy,numpy,cv2; print(torch.__version__, torch.cuda.is_available())"
  ```

## 3. Git clone / branch / tag
```bash
git clone <origin-url> morai-3d-detection && cd morai-3d-detection
git fetch --all --tags
git checkout source-time-v3-handoff      # 인수인계 branch
# 또는 고정 태그: git checkout v3-source-time-representative-pass-v1
```
> dataset/checkpoint/anchor 바이너리는 저장소에 포함되지 않는다(§5). 코드·문서·스크립트만.

## 4. 데이터 디렉터리 구조 (DATASET_ROOT 하위)
```
$DATASET_ROOT/
  scen001/
    images/{cam_front,cam_front_left,cam_front_right}/live_NNNNNN.jpg
    lidar/live_NNNNNN.(pcd|bin|npy)
    labels_3d/live_NNNNNN.csv          # 원본(수정 금지)
    ego_pose/live_NNNNNN.csv           # 원본(수정 금지)
    sync_log.csv                       # 원본(수정 금지; epoch_id/ref_src_ts/ego_src_ts/obj_src_ts)
    # ↓ 파이프라인이 생성(versioned 산출):
    labels_3d_v2/  scene_info.json
    labels_3d_v3/  scene_info_v3.json  timing_correction_report.json
    depth_gt/{cam_front,cam_front_left,cam_front_right}/live_NNNNNN.npy
  scen002/ …
```
> `generate_depth_gt.py` 는 `<repo>/dataset` 를 기대한다. `DATASET_ROOT` 가 다르면
> `ln -s "$DATASET_ROOT" "<repo>/dataset"` 로 심볼릭 링크하라(§10).

## 5. dataset 은 Git 에 포함되지 않는다
- `dataset/`, `depth_gt`, `labels_3d_v3`, `*.pth/*.pt`, anchor `*.npy`, 대용량 PNG 는 `.gitignore`
  로 제외된다. 동료는 데이터를 **별도 채널**(사내 스토리지)로 받아 `DATASET_ROOT` 에 배치한다.
- 저장소에는 재현 코드·스크립트·문서·작은 manifest/JSON 만 있다.

## 6. 원본 데이터 사전 감사 (report-only)
```bash
export DATASET_ROOT=/data/dataset
scripts/audit_150_dataset.sh            # → pretrain_verify/audit_150.json
```
검사: 모달리티 frame 정합(labels/ego/lidar/images), 누락 file, 중복 stem/timestamp,
raw velocity 비율, epoch/teleport(epoch_id 수), nonfinite, calibration 존재.
- critical 이슈가 있으면 **정확한 scene/stem 과 함께 비-0 종료**. 해결 전 다음 단계 금지.
- teleport/interp gap 상세는 `analyze_ego_teleport.py`(참고) 로 추가 확인.

## 7. 원본 hash baseline 생성 (불변성 감시)
```bash
python3 - <<'PY'
import hashlib,os,json
root=os.environ["DATASET_ROOT"]; out={}
def shad(d):
    if not os.path.isdir(d): return None
    a=hashlib.sha256()
    for n in sorted(os.listdir(d)):
        if n.endswith(('.csv','.npy')):
            with open(os.path.join(d,n),'rb') as f: a.update((n+':'+hashlib.sha256(f.read()).hexdigest()+'\n').encode())
    return a.hexdigest()
for s in sorted(os.listdir(root)):
    sd=os.path.join(root,s)
    if not (s.startswith('scen') and os.path.isdir(sd)): continue
    out[s]={m:shad(os.path.join(sd,m)) for m in ('labels_3d','ego_pose')}
json.dump(out,open('pretrain_verify/raw_hash_baseline_150.json','w'),indent=2)
print('wrote pretrain_verify/raw_hash_baseline_150.json', len(out),'scenes')
PY
```
이후 어느 단계 후에도 이 스크립트를 재실행해 원본 hash 불변을 확인한다.

## 8. labels_3d_v3 생성
> 전제: `labels_3d_v2`/`scene_info.json` 이 이미 있어야 한다(기존 preprocess 산출).
> 없으면 `preprocess_dataset.py` 로 v2 를 먼저 생성.
```bash
DRY_RUN=1 scripts/build_v3_all.sh        # --report-only: 통계만(파일 미생성)로 미리 검토
scripts/build_v3_all.sh                   # 실제 생성(원본 미수정, versioned 산출만)
```
`correct_source_time.py` 는 epoch 단위 ego 보간(cross-epoch 차단), 경계 correction_valid=0 을
적용한다. class/dimension/frame·stem 수는 보존된다.

## 9. correction report 집계 + correction_valid=0 fail-fast
`build_v3_all.sh` 는 생성 후 모든 `timing_correction_report.json` 을 집계하고
`n_correction_invalid>0` 이면 경고 + 비-0 종료한다. v3 loader(morai_dataset)는 학습 시
correction_valid=0 프레임/박스를 **hard-fail** 하므로, 여기서 0 이 되도록 원인 해결이 선행돼야
한다(주로 epoch 경계/보간 gap; §6 참조).

## 10. depth_gt 생성 + 전수검사
```bash
[ -d "$PWD/dataset" ] || ln -s "$DATASET_ROOT" "$PWD/dataset"   # generate_depth_gt.py 는 <repo>/dataset 기대
DRY_RUN=1 scripts/build_depth_all.sh     # 실행 명령만 출력
scripts/build_depth_all.sh                # depth_gt 생성(--all) + 내부 verify
# 전수검사: depth_gt 파일 수 == labels_3d_v2 프레임 × 3(cam) 확인
python3 - <<'PY'
import os
root=os.environ["DATASET_ROOT"]; bad=[]
for s in sorted(os.listdir(root)):
    sd=os.path.join(root,s)
    if not os.path.isdir(os.path.join(sd,'depth_gt')): continue
    nlab=len([f for f in os.listdir(os.path.join(sd,'labels_3d_v2')) if f.endswith('.csv')]) if os.path.isdir(os.path.join(sd,'labels_3d_v2')) else 0
    ndep=sum(len([f for f in os.listdir(os.path.join(sd,'depth_gt',c)) if f.endswith('.npy')]) for c in os.listdir(os.path.join(sd,'depth_gt')))
    if ndep != nlab*3: bad.append((s,nlab,ndep))
print('MISMATCH:',bad if bad else 'none')
PY
```
> `depth_gt/*.npy` 의 (u,v) 는 **704×256 학습입력 해상도** 좌표계다(원본 1600×900 아님).
> 재사용 시 (1600/704, 900/256) 스케일 필요. `morai_dataset._load_depth_maps` 는 동일 stride 소비.

## 11. depth ON/OFF 정책
- **primary/권장 기본값: `USE_DENSE_DEPTH=0`(OFF).** LiDAR–camera source clock 이 안정 정렬되지
  않아(recv gap p95≈57.85ms) depth_gt 의 **동적 객체 capture-time 동기화가 미검증**이다.
- depth ON 은 코드 경로 확인용으로만 유지한다. **동기화가 정량화되기 전에는 depth ON 을
  production 권장값으로 선언하지 않는다.**

## 12. 최종 train/val/test split 생성 및 승인
- 150-scene 의 실제 split 은 **데이터 확인 없이 추정하지 않는다.** scene 특성(도심/고속/보행자
  비율, 고속 track 분포)을 감사(§6)한 뒤 사람이 결정한다.
- split 파일 예(승인 후 작성): `splits/split_v3_150.json` `{ "train":[…], "val":[…], "test":[…] }`.
- **승인 게이트**: `build_anchor_for_split.sh` 와 학습 스크립트는 `SPLIT_APPROVED=1` /
  `CONFIRM_PRODUCTION=1` 없이는 실행되지 않는다. split 승인 전 anchor·training 금지.

## 13. validation/test leakage 검사
- scene-level split 이므로 같은 scene 이 train/val/test 에 동시에 들어가지 않도록 교집합 0 확인:
  ```bash
  python3 - splits/split_v3_150.json <<'PY'
  import json,sys; d=json.load(open(sys.argv[1]))
  a,b,c=set(d['train']),set(d.get('val',[])),set(d.get('test',[]))
  assert a&b==set() and a&c==set() and b&c==set(), "split leakage!"
  print("no leakage. train/val/test =",len(a),len(b),len(c))
  PY
  ```
- track/timestamp 연속성으로 인한 인접 scene 누수 가능성도 감사에서 확인.

## 14. full train split v3 anchor 생성
```bash
export ANCHOR_DIR=anchors/v3_full_split_k900
export VAL_SCENARIOS="scenXXX,scenYYY,…"   # 승인된 val(+test) — train = 전체 - 이 목록
SPLIT_APPROVED=1 scripts/build_anchor_for_split.sh
```
K=900, seed=42, gt=v3. train box 수 ≥ K 여야 한다.

## 15. anchor meta/hash 검증
- `$ANCHOR_DIR/anchor_kmeans_meta.json` 의 `label_dir=labels_3d_v3`, `gt_version=v3`, `k=900`,
  `seed=42`, `train_scenarios`/`val_scenarios`, `input_label_sha256`, `anchor_full_sha256` 확인.
- 학습 시작 시 `train.py` 가 **SHA + split/k/seed/GT/input-label hash** 정합을 자동 검증하고
  불일치면 **GPU 이전 fail-fast** 한다(안전장치).

## 16. ⚠️ 3-scene anchor 를 150-scene 에 쓸 수 없다
- 대표 실험의 `anchors/v3_train_scen05_scen77_k900/`(3-scene, full SHA `aef60c9d…`)는 **150-scene
  production 에 절대 사용 금지**. train.py 의 anchor 정합 검사가 train_scenarios/input hash
  불일치로 이를 fail-fast 한다. 반드시 최종 split 로 §14 재생성.

## 17. 전체 150-scene loader walk
```bash
python3 - <<'PY'
import os
os.environ.setdefault("GT_VERSION","v3")
from morai_dataset import MoraiTemporalDataset
ds=MoraiTemporalDataset(os.environ["DATASET_ROOT"],"train",
        val_scenarios=os.environ["VAL_SCENARIOS"].split(","),load_depth=False,gt_version="v3")
n=0
for i in range(len(ds.items)):
    _=ds._load_labels_v2(ds.items[i]["scen_dir"], ds.items[i]["stem"]); n+=1
print("train frames walked:",n)
PY
```
correction_valid=0 이나 누락 파일이 있으면 여기서 예외(정확한 stem)로 드러난다.

## 18. full-split 짧은 preflight (100~500 step)
```bash
export ANCHOR_DIR=anchors/v3_full_split_k900
export VAL_SCENARIOS="scenXXX,…"
STEPS=400 DEPTH=0 RUN_DIR=runs/preflight_150 scripts/preflight_150.sh
```
production 이 아니다. 전체 loader + anchor 정합 + NaN/Inf/OOM/throughput 확인용.
로그에서 loss finite, peak memory, updates/sec 를 확인한다.

## 19. production training (승인 후에만)
```bash
export ANCHOR_DIR=anchors/v3_full_split_k900 VAL_SCENARIOS="scenXXX,…"
export RUN_DIR=runs/prod_v3_$(date +%Y%m%d)     # run 격리
CONFIRM_PRODUCTION=1 USE_DENSE_DEPTH=0 NUM_EPOCHS=100 scripts/train_150_v3.sh
```
`CONFIRM_PRODUCTION=1` 없이는 실행되지 않는다. 모든 산출은 `RUN_DIR` 에 격리된다.

## 20. checkpoint resume
```bash
export ANCHOR_DIR=… VAL_SCENARIOS=… RUN_DIR=runs/prod_v3_YYYYMMDD
CONFIRM_PRODUCTION=1 scripts/resume_150.sh
```
train.py 가 checkpoint 의 split/anchor/GT 정합을 검사해 **다른 run 이면 fail-fast** 한다.

## 21. validation / P3 평가
- 학습 중: train.py 가 매 epoch overall/by-class/by-distance P/R/F1@2.0m 를 로깅.
- 최종 checkpoint 전체 val + filtered production P3:
  ```bash
  python3 experiments/representative_ab/eval_run.py \
     --run-dir "$RUN_DIR" --val-scen scenXXX --val-gt v3 \
     --anchor-full "$ANCHOR_DIR/anchor_kmeans_full.npy" --ckpt last_checkpoint.pth
  ```
  (val scene 이 여러 개면 scene 별로 실행) → `eval_metrics.json`.
- P3 는 **production filter 이후 GT membership** 을 쓰고, frame/GT/pred 가 0 이면 fail-fast 한다.
  raw 전체-box 상대속도 분포 audit 는 별도(`eval_relative_speed.py --gt-only`).

## 22. monitoring / 즉시 중단 조건
학습 중 다음이면 즉시 중단·조사: NaN/Inf, OOM, anchor provenance mismatch, split mismatch,
sampler order 불일치, validation GT 가 v3 아님, P3 join 0건, 원본 hash 변경, checkpoint 충돌.
`RUN_DIR/run_config.json` 과 `training_history.csv` 로 loss/metric 추이를 감시한다.

## 23. expected output 파일과 hash
- `$ANCHOR_DIR/anchor_kmeans_{xy,full}.npy` + `anchor_kmeans_meta.json`(SHA 기록).
- `$RUN_DIR/{run_config.json, last_checkpoint.pth, best_model*.pth, training_history.csv, training_curves.png, eval_metrics.json}`.
- checkpoint 자체는 **Git 에 넣지 않는다.** run_config.json/eval_metrics.json 등 작은 산출만 공유.

## 24. GO / NO-GO 체크리스트 (150 production 시작 전)
- [ ] §6 감사 critical=0
- [ ] §7 원본 hash baseline 생성·불변 확인
- [ ] §8–9 labels_3d_v3 생성, correction_valid=0 (invalid box 0)
- [ ] §10 depth_gt 전수검사 통과(파일 수 == frames×3)
- [ ] §12 최종 split 승인(SPLIT_APPROVED), §13 leakage 0
- [ ] §14–16 최종 split anchor 생성 + meta/hash 검증, 3-scene anchor 미사용
- [ ] §17 전체 loader walk 무예외
- [ ] §18 full-split preflight: NaN/Inf/OOM 없음
- [ ] depth 정책 확인(권장 OFF)
- [ ] 위가 전부 충족돼야 §19 production 시작

## 25. 장애 진단 순서
1. anchor fail-fast → meta 의 train/val/k/seed/gt/input-hash 와 현재 env 비교(정확한 mismatch 출력됨).
2. correction_valid=0 hard-fail → 해당 scene 의 timing_correction_report.json + epoch 경계/보간 gap 확인.
3. depth 파일 누락/shape → §10 전수검사, `<repo>/dataset` symlink 확인.
4. OOM → batch/depth OFF 확인, GPU 메모리 확인.
5. loader 예외 → 정확한 stem 출력됨; 원본/모달리티 정합(§6) 재확인.
6. resume mismatch → RUN_DIR/anchor/split 가 원 run 과 동일한지 확인.
7. throughput 저하 → nvcc 없으면 grid_sample fallback(정상). 극단적으로 느리면 I/O 확인.

---
### 부록: dataset-required vs dataset-independent 테스트
- dataset 없이 통과해야 하는 **단위 테스트**(clean clone smoke): 아래로 실행하면 데이터 의존 테스트는
  자동 skip 되고, 순수 단위 테스트만 통과한다(“조용한 전체 PASS” 착시 방지 — skip 개수를 확인하라).
  ```bash
  python3 -m pytest -q test_source_time_correction.py test_velocity_valid_loss.py \
     test_relative_speed_eval.py -k "not filtered" -rs
  ```
- dataset-required **통합 테스트**: `test_gt_version_split.py`, `test_anchor_policy.py`(대표 anchor),
  `test_relative_speed_eval.py::test_filtered_gt_uses_production_membership` — `DATASET_ROOT` 에
  대표 데이터가 있어야 실행된다. skip 되면 그 사실을 로그로 남긴다.

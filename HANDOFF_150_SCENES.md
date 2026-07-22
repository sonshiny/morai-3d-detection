# HANDOFF — source-time v3 기반 150-scene temporal 학습

> 현재 상태: **코드/대표 데이터 검증 완료, 150-scene production 학습은 아직 시작 전**
> 인수인계 branch: `source-time-v3-handoff`
> 금지: 3-scene anchor 재사용, 단일 프레임 production 실행, 확인되지 않은 AMP 자동 적용,
> `MAX_STEPS_PER_EPOCH`로 production epoch 절단, dataset/checkpoint Git 추가.

이 문서는 동료 PC에서 약 150GB/150개 시나리오를 받아 원본 감사부터 SparseDrive-style
temporal production 학습까지 재현하기 위한 실행 명세다. 경로와 GPU 이름을 가정하지 않는다.

## 1. 결론과 의사결정 상태

| 축 | 현재 판정 | 의미 |
|---|---|---|
| v3 수식 정확성 | **CONFIRMED** | 동일 source-clock에서 box를 cam_front capture time으로 옮기는 수식이 물리적으로 맞음 |
| v3 구현 정확성 | **CONFIRMED** | 독립 재계산과 저장 v3가 CSV 반올림 오차 이내 일치 |
| 시간 일관성 | **IMPROVED**(차량), **EQUAL**(정적/보행자) | leave-one-out 잔차가 차량에서 명확히 감소 |
| 절대 정확도 | **UNVERIFIABLE** | v3와 독립인 t_ref 동기 oracle 부재 |
| 대표 파이프라인 | **TECHNICAL_GO** | 24개 기존 테스트와 preflight/A-B 실행 경로 통과 |
| 150-scene production | **PRECONDITIONS PENDING** | §8 체크리스트를 실제 150 scenes에서 통과해야 시작 가능 |
| P1 이방성 loss | **NO-GO** | 대표 데이터 고속 표본 부족; 구현하지 않음 |
| dense depth | **OFF 권장** | 동적 LiDAR-camera source-time 정량 검증 전에는 보조 loss에서 제외 |

중요한 해석:

- 12-run 장기 A/B는 **v3 label correctness의 필수 검증이 아니다**. 검출 성능 비교를 추가로 하고
  싶을 때만 수행하는 선택 실험이다. 직접 측정 결과는
  [V3_CORRECTNESS.md](experiments/v3_correctness/V3_CORRECTNESS.md)에 고정했다.
- pilot 3-fold A/B의 성능 우열은 `INCONCLUSIVE`였지만, 300-update 미수렴 결과를 이유로 물리적으로
  검증된 v3를 폐기하거나 7~21 GPU-day A/B를 production 선행조건으로 만들지 않는다.
- `ABSOLUTE_ACCURACY=UNVERIFIABLE`는 숨기지 않는다. 이는 구현 오류 판정이 아니라 독립 oracle의
  부재다. 가능하면 별도 소규모 oracle 재수집을 권장하지만, 현재 production의 강제 blocker는 아니다.

## 2. 지금까지 확정된 사실

대표 데이터 `scen05/scen77/scen144`, 3,337 frame, 10,516 box에서:

- `ref_src_ts`는 `/cam_front` message의 source-header timestamp다. ego/object/lidar는 같은 source
  clock의 가장 가까운 메시지로 매칭되며 receive-time fallback은 금지되어 있다.
- v3는 position, ego-frame yaw/velocity, timestamp를 `t_ref`로 다시 표현한다. dimensions, class,
  track_id, global yaw는 바꾸지 않는다.
- correction distance: mean 0.105m, median 0.077m, p95 0.31m, max 0.91m. 2.0m matching
  threshold를 넘은 box는 0개다.
- 차량 leave-one-out temporal residual: v2 0.099m → v3 0.016m, paired delta -0.083m,
  95% CI [-0.085,-0.081], v3 개선 비율 82.4%.
- 10,516 box 모두 `vel_source=raw`; motion-fill box는 0개다. 향후 motion-fill은 velocity loss에서
  마스크하는 기존 `velocity_valid` 가드를 유지한다.
- raw input은 대표 검증 전후 hash가 불변이었다. 직접 correctness 검증 중 학습/checkpoint/tag는
  생성하지 않았다.

세부 수치와 순환성 한계는 `experiments/v3_correctness/`의 보고서·JSON·read-only 분석 스크립트가
근거다. track plot PNG는 재생성 산출물이므로 Git에는 넣지 않는다.

## 3. production 학습 모드 — 반드시 temporal

이 프로젝트의 `MoraiTemporalDataset` 사용 여부와 temporal instance memory 활성화 여부는 다르다.
과거 기본값은 dataset만 temporal이고 `USE_TEMPORAL_MEMORY=0`이라 실질적으로 A0 단일 프레임이었다.
이번 production 계약은 다음과 같다.

```text
USE_TEMPORAL_MEMORY=1
STREAMING_SAMPLER=1
TEMP_GNN_MODE=gated
SEQUENCE_LENGTH=150
USE_DENSE_DEPTH=0
TRAIN_GT_VERSION=v3
VAL_GT_VERSION=v3
effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 8
```

`StreamingGroupSampler`가 batch slot별 시퀀스 순서를 보존하고 sequence 시작에서 instance bank를
reset한다. 이는 이 저장소의 **SparseDrive-style recurrent instance-bank 경로**라는 뜻이며,
SparseDrive 논문의 모든 구성과 동일하다는 주장은 아니다.

`scripts/preflight_150.sh`, `train_150_v3.sh`, `resume_150.sh`는 위 계약을 강제한다. production
wrapper는 `MAX_STEPS_PER_EPOCH=0`과 full train loader도 강제하므로 시간을 맞추려고 optimizer
update를 몰래 줄일 수 없다.

## 4. Git과 데이터 전달

```bash
git clone <origin-url> morai-3d-detection
cd morai-3d-detection
git checkout source-time-v3-handoff
git rev-parse HEAD
```

Git 포함: 코드, 테스트, 재현 스크립트, 작은 JSON/CSV, 보고서.
Git 제외: `dataset/`, `*.pth`, `*.pt`, anchor `*.npy`, run logs/checkpoints, 대량 visual-audit PNG.
직접 correctness 근거인 작은 track plot 5개는 보고서와 함께 포함한다.

데이터는 별도 스토리지에서 전달하고 다음 구조를 맞춘다.

```text
$DATASET_ROOT/scenXXX/
  images/{cam_front,cam_front_left,cam_front_right}/live_NNNNNN.jpg
  lidar/live_NNNNNN.*
  labels_3d/live_NNNNNN.csv
  ego_pose/live_NNNNNN.csv
  sync_log.csv
  labels_3d_v2/               # 기존 전처리 산출
  scene_info.json
```

환경:

```bash
export DATASET_ROOT=/data/morai/dataset
export VAL_SCENARIOS="scenXXX,scenYYY"   # 승인된 scene-level validation split
```

## 5. 150-scene 전처리 순서

### 5.1 원본 read-only 감사와 hash baseline

```bash
scripts/audit_150_dataset.sh
```

`pretrain_verify/audit_150.json`에서 critical=0인지 확인한다. 이미지 3종, LiDAR, labels, ego_pose,
sync log의 stem/timestamp 대응, nonfinite, epoch/teleport, raw velocity 비율을 확인한다. 이어서
원본 `labels_3d`와 `ego_pose` aggregate SHA-256 baseline을 별도 저장하고 모든 전처리 후 재비교한다.

### 5.2 v3 생성

```bash
DRY_RUN=1 scripts/build_v3_all.sh
FORCE=1 scripts/build_v3_all.sh
```

각 scene에 `labels_3d_v3/`, `scene_info_v3.json`, `timing_correction_report.json`이 생겨야 한다.
`correction_valid=0`, cross-epoch interpolation, 누락 frame이 하나라도 있으면 해당 stem을 출력하고
중단한다. 대표 3 scenes의 수치를 150 scenes에 그대로 가정하지 말고 전체 분포를 다시 기록한다.

### 5.3 depth

primary production은 `USE_DENSE_DEPTH=0`이므로 depth 생성은 필수가 아니다. 향후 ON 실험을 할 때만
`scripts/build_depth_all.sh`로 생성하고 파일 수/shape/nonfinite와 동적 LiDAR-camera source-time
alignment를 별도로 검증한다. 단순 로딩 성공은 동적 depth의 시간 정확성을 증명하지 않는다.

## 6. split과 full anchor

scene 단위로 train/val/test를 나누고 교집합이 0인지 확인한다. 연속 frame이나 같은 track을 frame
단위로 쪼개지 않는다.

```bash
export ANCHOR_DIR=/data/morai/anchors/v3_full_split_k900
SPLIT_APPROVED=1 scripts/build_anchor_for_split.sh
```

anchor 계약: v3 train labels only, K=900, seed=42. `anchor_kmeans_meta.json`의 train/val scenarios,
GT version, input-label aggregate hash, anchor SHA가 현재 run과 일치해야 한다. `train.py`가 GPU 모델
생성 전에 이를 fail-fast 검사한다. 대표 3-scene anchor는 150-scene에서 절대 사용하지 않는다.

## 7. 동료 RTX 5070에서 시간 측정과 최적화

### 7.0 PyTorch/CUDA 호환성부터 확인

현재 PC의 `torch==2.1.0+cu121`은 RTX 4070 SUPER에서 검증된 과거 환경일 뿐, RTX 50-series에
그대로 복사할 환경이 아니다. [PyTorch 2.7 공식 릴리스](https://pytorch.org/blog/pytorch-2-7/) 기준
Blackwell 지원과 CUDA 12.8 wheel은 2.7에서 도입됐다. 동료 PC에서는 **PyTorch 2.7+ / CUDA
runtime 12.8+ build**를 별도 환경에 설치하고,
다음 검사를 먼저 통과시킨다. 정확한 설치 버전·driver는 통과 후 `pip freeze`로 고정한다.

```bash
python3 scripts/check_gpu_environment.py
python3 -m pytest -q test_*.py
```

저장소 `requirements.txt`의 torch/cu121 세 줄을 RTX 5070에 그대로 강제 설치하지 않는다. 나머지
의존성과 새 torch/torchvision 조합의 호환성은 28 tests 및 temporal preflight로 확인한다.

### 7.1 기존 23~35일 수치를 사용하면 안 되는 이유

기존 RTX 4070 SUPER 실측 6.23 frame/s와 23~35일 추정은
`USE_TEMPORAL_MEMORY=0` 단일 프레임 경로에서 얻은 값이다. 이번 production은 instance bank,
temp-GNN, streaming sampler를 사용하므로 그 수치는 temporal 시간 예측의 근거가 아니다. 또한
“RTX 5070”만으로 desktop/laptop, VRAM, 전력 제한, CUDA op/fallback, I/O 성능을 알 수 없다.

GPU가 빨라져도 필요한 optimizer update 수가 줄어드는 것은 아니다. 따라서 **GPU 이름을 이유로
epoch/step을 축소하지 않고**, 실제 처리량을 높이고 validation 빈도를 분리한다.

### 7.2 실제 temporal benchmark

최종 150-scene split/anchor가 준비된 뒤 동료 PC에서 먼저 FP32 baseline을 실행한다.

```bash
export RUN_DIR=runs/bench_temporal_fp32_w0
STEPS=400 VAL_MON=200 BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 \
NUM_WORKERS=0 USE_AMP=0 ALLOW_TF32=0 \
scripts/benchmark_temporal_150.sh
```

산출물:

- `hardware.txt`: 실제 GPU/driver/VRAM/power limit
- `torch_gpu_check.json`: torch/CUDA/compute capability와 CUDA conv forward/backward 확인
- `run_config.json`: split/hash/temporal/numeric/loader 계약
- `throughput.jsonl`: CUDA synchronize를 포함한 train/validation 실측
- `temporal_budget.json`: 실제 loader 길이로 100 epoch와 validation cadence 1/5/10을 외삽

이 결과가 새 학습시간 기준이다. 짧은 측정이므로 thermal throttling, checkpoint I/O를 고려해 최종
운영 계획에는 10~15% 여유를 둔다.

### 7.3 최적화 적용 순서

각 후보는 새 `RUN_DIR`에서 400-step preflight를 반복하고 NaN/Inf/OOM, sampler order, peak VRAM,
loss 범위, checkpoint resume를 확인한다.

1. `NUM_WORKERS=2`, 필요 시 4: 수학 경로를 바꾸지 않는 첫 후보. 실제 I/O 개선이 없으면 0으로 복귀.
2. batch/accum: VRAM 부족 시 `BATCH_SIZE=2, GRAD_ACCUM_STEPS=4`로 effective batch 8 유지.
   batch 4가 안전하면 유지한다. 속도는 반드시 실측 비교한다.
3. `ALLOW_TF32=1`: 별도 후보. finite와 짧은 수치 비교를 통과한 경우에만 승인한다.
4. `USE_AMP=1`: custom aggregation/quality 경로 안정성 때문에 기본 OFF다. loss/gradient finite,
   checkpoint resume, 짧은 동일-seed validation 동등성을 통과하기 전 production에 쓰지 않는다.
5. per-sample forward 벡터화/CUDA op 교체는 모델 코드 변경이다. 별도 branch와 회귀 테스트 없이
   production 직전에 적용하지 않는다.

가장 빠른 후보가 아니라 **안정성 조건을 통과한 후보 중 가장 빠른 것**을 고른다. 선택한
`BATCH_SIZE`, accumulation, workers, AMP, TF32는 production과 resume에서 동일해야 한다.

### 7.4 validation 비용

학습 update를 자르지 않고 full validation만 매 5 epoch로 분리한다. `train.py`는 final epoch를 항상
검증하고, 검증을 건너뛴 epoch에도 resumable checkpoint를 저장한다. 기본 production wrapper는
`VALIDATE_EVERY_EPOCHS=5`; benchmark 결과에 따라 1/5/10 중 승인한다. early-stop patience는
validation **횟수** 기준이므로 cadence 5, patience 10은 최대 50 epoch 정체를 의미한다.

## 8. production 시작 전 필수 체크리스트

- [ ] 150-scene audit critical=0, 실제 scene/frame/box 수 기록
- [ ] raw hash baseline과 전처리 후 hash 동일
- [ ] 모든 scene의 v3 생성 성공, invalid/cross-epoch/누락 0
- [ ] 150-scene correction magnitude, dt, relative-speed 분포 기록
- [ ] 최종 scene-level split 승인, train/val/test leakage 0
- [ ] full train-split v3 anchor 생성 및 meta/hash 일치
- [ ] 전체 loader walk 성공
- [ ] temporal contract가 run_config에서 모두 참: memory/streaming/gated/v3/depth OFF
- [ ] 400-step temporal preflight: NaN/Inf/OOM 없음, peak VRAM 여유
- [ ] checkpoint 저장 후 동일 config resume와 config mismatch fail-fast 확인
- [ ] 실제 동료 PC temporal budget 생성, validation cadence와 수치 모드 승인
- [ ] 최소 checkpoint 3개 분량의 디스크 여유와 장시간 전원/냉각/로그 감시 확보

다음은 권장하지만 depth OFF v3 production의 강제 blocker는 아니다.

- 독립 t_ref-synced object oracle 또는 수치 camera/LiDAR reprojection으로 absolute accuracy 검증
- 150 scenes의 고속(`>=8m/s`) 표본으로 P1 필요성 재판정
- 12-run v2/v3 장기 A/B
- dense-depth 동적 시간 동기화 검증

## 9. production 실행

아래 값은 예시다. benchmark에서 승인한 값으로 고정한다.

```bash
export DATASET_ROOT=/data/morai/dataset
export VAL_SCENARIOS="scenXXX,scenYYY"
export ANCHOR_DIR=/data/morai/anchors/v3_full_split_k900
export RUN_DIR=/data/morai/runs/prod_v3_temporal_$(date +%Y%m%d)

CONFIRM_PRODUCTION=1 \
NUM_EPOCHS=100 VALIDATE_EVERY_EPOCHS=5 \
BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 NUM_WORKERS=2 \
USE_AMP=0 ALLOW_TF32=0 USE_DENSE_DEPTH=0 \
scripts/train_150_v3.sh
```

`NUM_EPOCHS=100`은 상한이다. epoch를 사전에 임의 축소하지 말고 full validation의 수렴, best epoch,
train/val divergence를 근거로 조기 종료한다. production wrapper는 `CONFIRM_PRODUCTION=1` 없이는
실행되지 않는다.

재개:

```bash
CONFIRM_PRODUCTION=1 \
NUM_EPOCHS=100 VALIDATE_EVERY_EPOCHS=5 \
BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 NUM_WORKERS=2 \
USE_AMP=0 ALLOW_TF32=0 USE_DENSE_DEPTH=0 \
scripts/resume_150.sh
```

batch/accum/temporal/sampler/temp-GNN/depth/AMP/TF32 또는 split/anchor/GT가 원 checkpoint와 다르면
resume은 fail-fast한다. optimizer/scaler/global update도 함께 복원한다.

## 10. 학습 중 감시와 중단 조건

즉시 중단:

- NaN/Inf, 반복 OOM, anchor/split/hash mismatch
- temporal memory 또는 streaming sampler가 false인 run_config
- v2 train/val GT가 섞임, depth가 의도와 다르게 ON
- loss 급발산, sampler 순서/sequence reset 이상
- checkpoint resume 후 epoch/update/optimizer 상태 불연속
- raw input hash 변경

매 validation에서 overall/by-class/by-distance P/R/F1, center-distance와 상대속도 구간을 확인한다.
고속 표본 수가 적으면 해당 bin의 우열을 일반화하지 않는다. 최종 checkpoint는 full validation을
별도 `eval_run.py`로 한 번 더 평가하고 frame/GT/pred join이 모두 0보다 큰지 확인한다.

## 11. 재현 테스트

```bash
python3 -m pytest -q \
  test_source_time_correction.py test_velocity_valid_loss.py \
  test_anchor_policy.py test_gt_version_split.py test_relative_speed_eval.py \
  test_temporal_production_contract.py
bash -n scripts/*.sh experiments/representative_ab_long/run_ab_long.sh
git diff --check
```

dataset이 없는 clean clone에서는 dataset-required test가 skip될 수 있다. 동료 PC에서는 150-scene
통합 audit/preflight가 unit test의 대체가 아니라 추가 gate다.

## 12. 관련 근거

- `experiments/v3_correctness/V3_CORRECTNESS.md`: v3 직접 정확성 판정
- `pretrain_verify/FINAL_REPORT.md`: 대표 데이터 pretrain 검증
- `experiments/representative_ab/AB_REPORT.md`: 300-update pilot A/B
- `experiments/representative_ab_long/LONG_AB_REPORT.md`: 미실행 장기 A/B budget(선택 실험)
- `scripts/benchmark_temporal_150.sh`: 실제 hardware temporal preflight
- `scripts/estimate_temporal_budget.py`: 측정 기반 wall-time 산출

최종 원칙은 간단하다. **label correctness는 직접 측정 결과로 채택하고, production 안전성은 실제
150-scene 입력과 실제 RTX 5070 temporal 경로에서 짧게 검증한 뒤, 학습 update는 보존하면서 I/O·수치
모드·validation cadence만 검증된 범위에서 최적화한다.**

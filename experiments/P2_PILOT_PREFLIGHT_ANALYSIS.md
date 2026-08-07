# P2 pilot 및 scratch preflight 결과 분석

- 작성일: 2026-08-07
- 분석 대상:
  - `runs/p2_pilot_control_20260730_2104`
  - `runs/p2_pilot_parity_20260730_2104`
  - `runs/p2_scratch_preflight_20260731_ext`
- 목적: 위 run의 대용량 checkpoint/prediction을 삭제하기 전에 실험 조건, 핵심 수치, 의사결정 근거와 한계를 보존한다.

## 결론

1. 2-epoch repair pilot에서 `parity` loss 계약은 동일 조건의 `legacy` control보다 명확히 우수했다. Full validation의 `final_nonms` mAP은 0.1215에서 0.5365로 +0.4150 증가했고, NMS 적용 mAP도 0.2669에서 0.5804로 +0.3135 증가했다.
2. 개선은 vehicle과 pedestrian, 근거리와 원거리, 위치 오차와 중복 예측에서 일관되게 나타났다. 특히 pedestrian non-NMS mean AP가 0.0038에서 0.4721로 회복됐다.
3. 두 pilot의 `run_config.json` 차이는 출력 경로와 `loss_contract`(`legacy` 대 `parity`)뿐이었다. 데이터 split, GT v3, anchor, seed, 초기 checkpoint, temporal/depth 설정과 2-epoch 스케줄은 같았으므로 이 비교는 loss 계약 변경의 효과를 분리한 통제 실험으로 해석할 수 있다.
4. scratch preflight는 초기 가중치 없이 parity loss/optimizer 계약으로 1,500 update를 수행했다. 3,000-frame 부분 검증에서 `final_nonms` mAP 0.1571, `final_nms` mAP 0.2416을 기록했고 loss가 초기 100 update 평균 19.73에서 마지막 100 update 평균 14.47로 감소했다. 이는 scratch 장기 학습을 시작할 근거로는 충분했지만, pilot의 26,503-frame full validation 수치와 직접적인 성능 우열 비교에는 사용할 수 없다.
5. 이 결과를 근거로 `p2_scratch_parity_20260801_30ep` 장기 학습을 선택한 판단은 타당하다.

## 공통 실험 조건

| 항목 | 값 |
|---|---:|
| Git HEAD | `f74322a516e72c33830052f60339e20142dab903` |
| train/validation GT | v3 / v3 |
| train frames | 103,964 (streaming 유효 103,960) |
| validation frames | 26,503 |
| batch / grad accumulation | 4 / 2 |
| temporal memory / streaming sampler | on / on |
| temporal GNN | gated |
| dense depth | on |
| seed | 0 |
| GPU | NVIDIA GeForce RTX 5070 |
| AMP / TF32 | off / on |
| anchor | v3 full split, k=900 |

두 pilot은 동일한 `prod_v3_depth_20260726_1332/checkpoint_epoch20.pth`에서 시작하고 optimizer는 새로 시작했다. Pilot config의 실질적 차이는 다음 하나였다.

| run | loss contract |
|---|---|
| control | `legacy` |
| parity | `parity` — focal-cost matcher, batch `num_pos` 정규화 등 공식 Stage-1 계약 |

따라서 loss의 절대 크기는 계약별 정규화와 항 구성의 차이 때문에 직접 비교하면 안 된다. 판단의 중심은 동일 평가 프로토콜의 AP, TP error, 중복 예측과 처리량이다.

## Control 대 parity pilot

### 최종 AP

평가 프로토콜은 30개 validation scene, 26,503 frames, front ROI 50 m, nuScenes-style 101-point AP였다.

| 지표 | Control | Parity | 절대 변화 |
|---|---:|---:|---:|
| final non-NMS mAP | 0.1215 | 0.5365 | +0.4150 |
| raw non-NMS mAP | 0.1229 | 0.5313 | +0.4084 |
| final NMS mAP | 0.2669 | 0.5804 | +0.3135 |
| vehicle non-NMS mean AP | 0.2392 | 0.6009 | +0.3617 |
| pedestrian non-NMS mean AP | 0.0038 | 0.4721 | +0.4683 |
| epoch 2 soft-calibrated F1 @ 0.15 | 0.4244 | 0.7175 | +0.2931 |
| epoch 2 raw F1 @ 0.25 | 0.2989 | 0.7639 | +0.4651 |

Parity의 non-NMS mAP은 control의 약 4.4배다. Control은 NMS 후에야 mAP이 0.1454 상승한 반면 parity의 NMS 이득은 0.0439였다. 즉 parity는 후처리로 중복을 정리하기 전부터 더 유용한 후보 분포를 만들었다.

### 거리별 AP@2m

| 클래스/거리 | Control | Parity | 절대 변화 |
|---|---:|---:|---:|
| vehicle 0–20 m | 0.3886 | 0.8577 | +0.4691 |
| vehicle 20–40 m | 0.3902 | 0.7636 | +0.3734 |
| vehicle 40–50 m | 0.0371 | 0.2401 | +0.2031 |
| pedestrian 0–20 m | 0.0532 | 0.9149 | +0.8616 |
| pedestrian 20–40 m | 0.0000 | 0.4461 | +0.4461 |
| pedestrian 40–50 m | 0.0000 | 0.0930 | +0.0930 |

Parity가 모든 거리 구간에서 우세하다. 다만 40–50 m는 parity에서도 특히 pedestrian AP가 0.0930으로 낮아 원거리 성능은 후속 과제로 남는다.

### TP 오차와 중복 예측

| 지표 | Control | Parity | 해석 |
|---|---:|---:|---|
| vehicle ATE-XY @2m | 1.026 m | 0.701 m | 약 31.7% 감소 |
| pedestrian ATE-XY @2m | 1.123 m | 0.787 m | 약 29.9% 감소 |
| vehicle velocity error | 1.069 | 0.785 | 감소 |
| pedestrian velocity error | 0.273 | 0.118 | 감소 |
| NMS 전 predictions/frame | 246.13 | 51.31 | 약 79.2% 감소 |
| NMS 후 predictions/frame | 20.04 | 13.86 | 감소 |
| vehicle NMS 전 preds/GT@2m | 22.94 | 4.40 | 중복 후보 대폭 감소 |
| pedestrian NMS 전 preds/GT@2m | 62.90 | 5.80 | 중복 후보 대폭 감소 |

Control의 낮은 pedestrian AP와 매우 많은 중복 후보는 legacy 계약에서 assignment/classification 신호가 현재 모델 구조와 잘 맞지 않았음을 시사한다. Parity는 위치 정밀도뿐 아니라 후보 수와 중복률까지 함께 개선했다.

### 학습 거동과 처리량

| 항목 | Control | Parity |
|---|---:|---:|
| epoch 1 train loss | 1.2777 | 13.8575 |
| epoch 2 train loss | 1.1957 | 10.9244 |
| epoch 2 val loss | 1.0850 | 10.8845 |
| train FPS epoch 1 / 2 | 6.98 / 7.05 | 6.94 / 6.98 |
| 300-update preflight clip rate | 14.7% | 100.0% |
| preflight first→last 100 mean loss | 1.354→1.266 | 49.094→34.917 |

Parity의 loss와 gradient norm은 control보다 훨씬 크고 300-update pilot preflight에서 매 update clipping됐다. 이는 수치 규모 차이이므로 AP 실패를 의미하지는 않으며, 실제 AP는 parity가 크게 우세했다. 처리량 차이는 약 1% 이내로 실질적인 속도 비용은 없었다. 다만 높은 clipping 비율은 optimizer/normalization을 parity 계약에 맞춰야 한다는 근거가 됐다.

## Scratch parity preflight

### 목적과 설정

`p2_scratch_preflight_20260731_ext`는 repair pilot처럼 기존 checkpoint를 미세조정한 실험이 아니라 다음 장기 학습을 시작하기 위한 무가중치 사전 검증이었다.

- `init_weights: null`
- `loss_contract: parity`
- `optim_contract: parity`
- focal classification prior 0.01
- 30-epoch 설정이지만 `preflight_updates: 1500`에서 중단
- AP 평가는 full validation이 아니라 3,000-frame subset

### 관측 결과

| 지표 | 값 |
|---|---:|
| 1,500-update 평균 loss | 15.7647 |
| 초기 100 update 평균 loss | 19.7273 |
| 마지막 100 update 평균 loss | 14.4660 |
| gradient clipping 비율 | 97.8% |
| 평균 positive 수 | 10.43 |
| 평균 assignment gate rate | 0.953 |
| temporal bank 4/4 활성 비율 | 96.1% |
| 평균 temporal gate 절댓값 | 0.000312 |
| 마지막 누적 bank reset | 86 |
| 평균 backbone/head grad norm | 25.69 / 43.49 |

Loss가 약 26.7% 감소했고 backbone과 head 모두 유한한 gradient를 받았다. Temporal bank는 거의 항상 4개 slot에서 활성화됐지만 zero-init gated fusion의 평균 gate 크기는 3.12e-4로 작았다. 따라서 이 시점의 성능은 temporal fusion이 충분히 성장한 최종 성능이라기보다 scratch detector가 정상적으로 학습을 시작했다는 증거로 해석해야 한다.

### 3,000-frame 부분 평가

| 지표 | 값 |
|---|---:|
| final non-NMS mAP | 0.1571 |
| final NMS mAP | 0.2416 |
| vehicle non-NMS mean AP | 0.1482 |
| pedestrian non-NMS mean AP | 0.1660 |
| vehicle / pedestrian ATE-XY | 1.017 / 0.990 m |
| vehicle AP@2m, 0–20 / 20–40 / 40–50 m | 0.4500 / 0.1252 / 0.0166 |
| pedestrian AP@2m, 0–20 / 20–40 / 40–50 m | 0.5433 / 0.0693 / 0.0000 |
| NMS 전 / 후 predictions per frame | 107.07 / 33.64 |

초기 scratch 모델은 근거리부터 학습하고 있었고 중·원거리 성능은 아직 낮았다. NMS가 mAP을 +0.0845 높인 점도 후보 중복/분류가 아직 수렴 전임을 보여준다. 이 결과의 역할은 production 성능 입증이 아니라 NaN, dead gradient, assignment 붕괴, temporal-state 미작동 없이 장기 학습이 가능함을 확인하는 것이었다.

## 의사결정과 한계

### 당시 의사결정

- Repair pilot에서 parity 계약의 명확한 우위가 확인됐다.
- Scratch preflight에서 초기 학습 안정성과 데이터/temporal 파이프라인 작동이 확인됐다.
- 따라서 기존 legacy checkpoint를 계속 미세조정하기보다 parity loss와 parity optimizer를 사용해 scratch 30-epoch run을 수행하는 방향이 선택됐다.

### 해석 한계

- Pilot은 2 epoch라 장기 수렴 비교가 아니다.
- Scratch preflight AP는 3,000 frames, pilot AP는 26,503 frames이므로 서로 직접 비교하면 안 된다.
- Pilot은 기존 production checkpoint에서 시작했고 scratch preflight는 무가중치 시작이다.
- Gradient clipping 비율이 높아 loss scale 자체를 모델 품질 지표로 사용하면 안 된다.
- 아래 원본 run 디렉터리를 삭제하면 checkpoint와 prediction dump를 이용한 재평가는 불가능하다. 이 문서는 집계 결과와 판단 근거만 보존한다.

## 삭제 전 원본 무결성 기록

| 파일 | SHA-256 |
|---|---|
| control `run_config.json` | `4381f736ac1004c512ee33c96927ee1fccfc81b5853bba68e15f134b04f7f6ce` |
| control `training_history.csv` | `00b158d51beeebec775348b461a35384c4578ce4cf3e7ef99c5855164cb0dabe` |
| control `preflight.csv` | `3758138ec7b7d6db53da6addd2f1e9d584f301b186d943fe3aeed1d1622b8e62` |
| control `control_ep2.json` | `a7ea758719952edd64c0089892cc792103c3c1cb8545eae42dc67d47af26f97f` |
| parity `run_config.json` | `8e9a936441f2e2f2aba0269137da6b58f21eae7ee6f45cb8931be9a13e90f60f` |
| parity `training_history.csv` | `b732ef0102db904a64122c96fb7d1ea56fba267ab310fce447ca1bd1fb6f0963` |
| parity `preflight.csv` | `88f79aed900003cc34f922f11a698b7940da76a7c82d971f15e2dbdb6be844c1` |
| parity `parity_ep2.json` | `aa3a42cd84694825691331c1d83fcb1f2660d53756b1f9028259ea67def881d6` |
| scratch preflight `run_config.json` | `7680ac3ca9c0c9e5fd0d7f216f40d7756eecd35a90804517bc5c8796771022d1` |
| scratch preflight `preflight.csv` | `c5644c47d39e26f60e636bfa5170a6cb881428f29b5e3c8c9c0364243e47b507` |
| scratch preflight `pf1500.json` | `a93a221cefc100782535237b86906b6fa83bb9866e17011bc01f9b594023c7b2` |

## 보존 및 삭제 상태

- 본 문서가 위 세 run의 영구 요약본이다.
- 장기 결과와 재개 checkpoint는 `runs/p2_scratch_parity_20260801_30ep`에 별도로 보존한다.
- 분석 완료 후 2026-08-07에 위 세 원본 run 디렉터리를 영구 삭제했다(합계 약 1.16 GiB).
- 삭제 후 `runs/`에는 장기 run 디렉터리와 그 로그만 남아 있다.

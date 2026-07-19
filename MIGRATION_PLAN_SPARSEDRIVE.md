# SparseDrive 인지 모듈 완전 이행 계획 (v11 로드맵)

작성일: 2026-07-07. 근거: SparseDrive 논문(arXiv:2405.19620v2) 정독 + `~/projects/SparseDrive` 공식 코드 분석 + 기존 학습 코드(v10) 분석 + 신규 dataset(scen01~60, 37,486프레임) 전수/표본 검증.

---

## 0. 최종 판정 요약

| 질문 | 판정 |
|---|---|
| 데이터셋에 필요한 정보가 다 있는가? | **예** — SparseDrive 인지 모듈(detection+tracking+temporal+depth)에 필요한 모든 원천 정보 존재. 단 **전처리 패스 1회 필수** (§2) |
| GT가 학습용으로 충분·정확한가? | **전처리 후 예** — 원본 그대로는 **불가** (트랙 ID 재사용, yaw −90° 오프셋, 속도 인코딩 버그). 전부 수치 검증된 공식으로 복구 가능 (yaw 잔차 0.1°, 속도 잔차 0.06~0.12 m/s) |
| 데이터 양이 충분한가? | **조건부 예** — 37.5k 프레임은 nuScenes 키프레임(~28k)과 유사 규모지만 다양성(60루트, 도심 1존, 2클래스)이 약점. 시뮬레이터 도메인 내 temporal 인지 학습·검증에는 충분, 일반화를 원하면 시나리오 추가 수집 권장 (§1.3) |
| 온라인 매핑까지 가능한가? | **가능하나 후순위** — MGeo 링크/횡단보도로 polyline GT 생성 가능(보도 없음). Phase 2로 분리 |
| 모션/플래닝(full e2e)까지 확장 가능한가? | **가능** — 재생성한 트랙 GT + ego_pose에서 미래 궤적 GT 유도 가능. Phase 3 |

---

## 1. 데이터셋 충분성 분석

### 1.1 보유 항목 vs SparseDrive 요구 항목

SparseDrive stage1(인지 사전학습)이 소비하는 프레임당 키 (공식 코드 `projects/configs/sparsedrive_small_stage1.py:548-569`, `nuscenes_3d_dataset.py:298-362` 기준):

| 요구 키 | 우리 데이터셋 | 상태 |
|---|---|---|
| `img` (N_cam,3,256,704) | images/cam_front·front_left·front_right, 1600×900 | ✅ (리사이즈만) |
| `projection_mat` (N_cam,4,4) = intrinsic @ ego2cam | camera_configs.py: fx=fy=1142.6, cx=800, cy=450, 마운트/자세 확정 | ✅ (계산으로 구성) |
| `timestamp` (초) | ego_pose/labels CSV, 프레임별 일치 확인 | ✅ |
| `T_global` (lidar→global 4×4) | ego_pose CSV(x,y,z,yaw) + lidar 마운트(1.92,0,1.35) | ✅ (yaw만 있는 planar pose로 구성 — 시뮬레이터라 roll/pitch≈0 가정 타당) |
| `gt_bboxes_3d` (x,y,z_center,l,w,h,yaw,vx,vy) | labels_3d CSV | ⚠️ **디코딩 필수** (§2) |
| `gt_labels_3d` | class_id 0=vehicle, 1=pedestrian | ✅ (2클래스) |
| `instance_id` (트래킹 평가용) | object_index는 **슬롯이지 ID가 아님** | ⚠️ **재생성 필수** (§2.1) |
| `gt_depth` (LiDAR 투영 sparse depth) | lidar/*.npy (28800,4) + generate_depth_gt.py 준비됨 | ⚠️ r<1.5m 필터 추가 + 캘리브 검증 후 전체 실행 (§2.4) |
| map_annos (매핑 시) | 없음 — MGeo에서 생성 가능 | Phase 2 |

**파일 무결성: 완벽.** 60개 시나리오 전체에서 6개 모달리티(ego/labels/lidar/3캠) 프레임 수 일치, 결손 0, JPEG 디코드 실패 0.

### 1.2 알려진 제약 (수집 단계 특성 — 그대로 안고 감)
- **전방 쐐기(±80°)만 라벨링** — 3전방캠 구성과 정합. 인지 범위를 nuScenes식 55m 원이 아니라 **전방 쐐기 0~60m**로 정의하면 됨. 라벨 최대 거리 66.4m, p95≈50m.
- 클래스 2종 (vehicle 45,921 / pedestrian 35,717 rows — 균형 양호).
- 빈 프레임 18.8% (7,046/37,486) — negative 샘플로 그대로 사용 (bg_weight 처리 이미 있음).
- 10Hz 공칭이나 **지터 심함**: dt p95=269ms, >300ms 갭 1,258회, 최악 2.13s — temporal 융합은 반드시 실측 timestamp 사용 (SparseDrive instance bank가 원래 timestamp 기반이라 문제없음, `max_time_interval=2s` 초과 갭은 자동 리셋).
- meta.json이 15/60개만 존재 (전부 accident 종료) — 나머지 45개 outcome 불명. 학습에는 지장 없음, 분석용 북키핑 이슈.

### 1.3 데이터 양 판정
- **규모**: 37,486 프레임 ≈ nuScenes 키프레임 28k와 유사. 단 10Hz 연속 수집이라 프레임 간 상관 높음 → 독립 샘플 관점에선 nuScenes보다 작음. 스트리밍 temporal 학습에는 고 fps가 오히려 유리(전이가 촘촘).
- **다양성이 진짜 병목**: 60개 루트 / KATRI 도심 1존 / 주간 / 2클래스 vs nuScenes 1,000씬·보스턴+싱가포르·야간·비. 시뮬레이터 인지 모듈 구현·검증 목적에는 충분. 과적합 징후(val 정체, train-val 괴리) 보이면 우선순위: ① 시나리오 추가 수집(목표 150~300, 존/교통밀도/시간대 다양화), ② photometric + GridMask 증강(§4.4)으로 버티기.
- **검증 분할**: 시나리오 단위 8개(val) / 52개(train) 권장. 빈프레임 비율·시나리오 길이 편차가 크므로 stratify해서 val에 scen15(0% 빈프레임)·scen30(51%) 같은 극단이 몰리지 않게.

---

## 2. GT 품질 판정 및 필수 수정 (Phase 0 전처리)

검증 방법: 생성기 소스(`morai_3d_label_generator.py:94-121`) 대조 + 수치 검증(scen01/15/30/55 표본). **아래 4개는 원본 그대로 쓰면 temporal/tracking/velocity 학습이 전부 오염되는 항목.**

### 2.1 트랙 ID 재생성 (심각도 HIGH)
`(object_source, object_index)`는 **슬롯 번호**다. scen01에서 8개 슬롯에 실제 물리 객체 22개 — 객체 소멸/생성 시 인덱스가 재사용되며, 그 프레임에서 위치 점프(6.98m)+치수 시그니처 교체가 동시에 발생.
**해법(검증됨)**: 복원한 global 좌표에서 greedy NN 재연결 (게이트 ~3m, 클래스 일치 + 치수 시그니처 일치 조건). 재연결 후 트랙 내 치수 편차 <5cm → 클린. 연속 생존 구간 내에서는 인덱스 98.3% 안정이므로 재연결은 소멸/생성 경계에서만 개입.

### 2.2 Yaw 디코딩 (심각도 HIGH)
저장값: `atan2(sin_yaw,cos_yaw) = (obj_global_heading − ego_heading) − 90°` (생성기 line 112가 π/2를 빼서 저장).
```
yaw_ego    = atan2(sin_yaw, cos_yaw) + π/2          # ego(=lidar) frame yaw — 학습에 쓸 값
yaw_global = yaw_ego + ego_yaw_rad                   # 검증·트랙 재연결용
```
궤적 진행방향 대비 중앙값 오차 0.1°로 검증 완료. 순진하게 atan2만 쓰면 씬에 따라 8~113° 틀어짐.

### 2.3 속도 디코딩 (심각도 HIGH — 인코딩 버그)
MORAI는 객체 **로컬**(전방,횡방향) 속도를 **km/h**로 주는데, 생성기(line 115-117)가 이를 world 벡터로 착각하고 ego yaw로 회전시켜 저장함. 저장값 = `R(−ego_yaw) @ v_obj_local_kmh`.
```
v_local  = R(+ego_yaw) @ [vx, vy]                    # 버그 역변환
v_world  = R(yaw_global) @ v_local / 3.6             # m/s
v_ego    = R(−ego_yaw_rad) @ v_world                 # ego(=lidar) frame — 학습에 쓸 값
```
잔차 0.06~0.12 m/s로 검증 완료. (속력 크기만 필요하면 `hypot(vx,vy)/3.6`은 항상 유효.)

### 2.4 기타 필수 보정
- **z는 바닥 기준** (ego 기준 ground-to-ground 오프셋): `z_center = z + h/2`. SparseDrive 규약(z_center)에 맞춰 전처리 단계에서 통일 (현재 morai_dataset.py:35-40에서 런타임 보정 중 — 전처리로 이동).
- **LiDAR 근거리 쓰레기 필터**: 포인트의 ~55%가 r<1.5m 무반사 더미. `r >= 1.5` 필터 필수 (유효 프레임당 ~12.8k pts). **generate_depth_gt.py에 현재 이 필터 없음 — 추가 필요.**
- **LiDAR 마운트 높이 검증**: 포인트클라우드상 센서 높이 ~2.0m vs camera_configs.py의 z=1.35 — body 원점이 지면 위 0.65m인지 설정 오류인지 `verify_lidar_camera_overlay.py`로 신규 데이터에서 확인 후 depth GT 전량 생성. (depth GT가 통째로 시프트되는 문제라 순서 중요.)
- **timestamp 시나리오 간 겹침**: 자체 로더는 (시나리오, 프레임) 순서를 쓰므로 무관. SparseDrive 원본 로더 이식 시에만 주의 (전역 timestamp 정렬 코드 있음, `nuscenes_3d_dataset.py:282`).

### 2.5 전처리 산출물 (신규 스크립트 `preprocess_dataset.py`)
시나리오별 `labels_3d_v2/live_NNNNNN.csv` (또는 통합 pkl):
```
frame_id, timestamp, track_id, class_id,
x, y, z_center, w, l, h, yaw_ego(rad), vx_ego, vy_ego, vz (m/s),
num_lidar_pts(옵션: 박스 내 lidar 점수 — SparseDrive는 >0 을 유효 마스크로 씀)
```
+ `scene_info.json`: 프레임별 T_global(4×4), timestamp, 카메라 3개의 intrinsic/ego2cam, 경로.
검증 게이트: 트랙별 치수 편차 <10cm, 프레임 간 위치 점프 <3m, |v_ego − Δpos/Δt| 중앙값 <0.3 m/s 를 60개 시나리오 전체에서 리포트로 출력.

---

## 3. 왜 기존 코드베이스를 확장하는가 (공식 레포 포팅 대신)

현 v10 코드는 이미 SparseDrive 디코더와 **동형**이다: 900앵커+256dim, 6층 refinement, 공식 CUDA deformable aggregation op(`sparsedrive_ops/`), quality head(centerness+yawness), K-means 앵커, focal+L1+Hungarian. 심지어 **temporal instance bank와 ego-motion 앵커 정렬 코드가 이미 작성돼 있고 꺼져 있을 뿐** (`train.py:1663 USE_TEMPORAL_MEMORY=False`, `_get_temporal_memory` train.py:652-713, `align_anchor_prev_to_current` train.py:428-474).

→ **결정: train.py 계열을 확장한다.** 공식 레포는 mmcv/mmdet3d 의존성 지옥 없이 *레퍼런스*로만 사용. 이 선택이 전이학습도 자명하게 만든다(§5) — v10 state_dict가 그대로 로드되는 구조를 유지하기 때문.

공식 레포에서 그대로 베낄 세부 구현 (파일 참조):
- Instance bank 정확한 동작: `projects/mmdet3d_plugin/models/instance_bank.py` (top-600 캐시, confidence decay 0.6, `T_temp2cur = T_global_inv(cur) @ T_global(prev)`, 앵커 투영 시 `center ← R·(center − v·Δt) + t` + yaw/vel 회전, |Δt|>2s 리셋)
- 스트리밍 샘플러: `datasets/samplers/group_in_batch_sampler.py` (배치 슬롯당 시퀀스 고정, 시간순, 시퀀스 일관 증강)
- Depth 브랜치: `models/blocks.py:264-322 DenseDepthNet` (레벨별 1×1 conv, `exp(conv(feat)) × focal / 100`, masked L1)
- 트래킹 ID 부여: `instance_bank.py:223-241 get_instance_id` (전파 인스턴스는 ID 유지, 신규는 confidence 기준 발급, 마스크 깨지면 −1)

---

## 4. 새 학습 구조 (Phase 1 — "v11 temporal")

### 4.1 데이터 파이프라인 (`morai_dataset.py` 개편 → `MoraiTemporalDataset`)
프레임당 반환:
```
img            (3, 3, 256, 704)   # 3캠 스택, ImageNet 정규화
projection_mat (3, 4, 4)          # ego→(리사이즈 반영)이미지. 기존 코드의 스케일링 로직 재사용
image_wh       (3, 2)
timestamp      float (초)
T_global       (4, 4)             # ego_pose에서 구성 (planar: x,y,z,yaw)
gt_boxes       (N, 11)            # [x,y,z_center,ln_w,ln_l,ln_h,sin(yaw_ego),cos(yaw_ego),vx_ego,vy_ego,vz]
gt_labels      (N,), gt_track_ids (N,)
depth_gt       3레벨 [(u,v,d)...]  # stride 4/8/16, 사전 생성본 로드
seq_id, is_seq_start               # 샘플러/뱅크 리셋용
```
- 카메라 가시성 필터(기존 `box_visible_in_any_camera`) 유지.
- 시나리오를 **~150프레임 서브시퀀스로 분할** (60씬 × 평균 4.3청크 ≈ 260그룹 — 배치 크기 대비 충분. SparseDrive의 `sequences_split_num` 대응).

### 4.2 스트리밍 샘플러 (신규 `StreamingGroupSampler`)
- 배치 슬롯 B개 각각에 서브시퀀스 1개를 고정, 프레임을 시간순으로 공급. 시퀀스 소진 시 다음 시퀀스 할당하고 해당 슬롯의 instance bank 리셋(`is_seq_start`).
- 에폭 셔플은 시퀀스 순서만. `shuffle=False` + 커스텀 batch_sampler로 구현.
- 증강 파라미터는 시퀀스 내 고정(사진왜곡 강도, crop 위치 등) — temporal 일관성 필수.

### 4.3 모델 변경 (`train.py` AutoNavModel)
1. **`USE_TEMPORAL_MEMORY = True`** + instance bank를 공식 semantics로 정밀화: top-600 전파, confidence decay 0.6, max_time_interval 2s, 캐시는 detach, 배치 슬롯별 독립 상태.
2. 기존 `align_anchor_prev_to_current`를 공식 `anchor_projection`과 대조 검증 — 특히 **속도 항 보정** `center ← R·(center − v·Δt) + t` 과 yaw/velocity 회전 포함 여부.
3. **velocity 학습 활성화**: `loss_calculator.py:13 REG_CHANNELS 8 → 10` (vx,vy; vz는 계속 제외 가능). SparseDrive 규약: **매칭 비용에서 velocity 가중 0, loss에서만 1.0** — 매칭 안정성 유지하면서 속도 회귀. BOX_SCALE의 vel 채널(30,30)은 m/s 스케일로 재조정(≈15).
4. **DenseDepthNet 추가**: FPN P2/P3/P4 (stride 4/8/16)에 1×1 conv 3개, `depth = exp(conv) × focal/100`, masked L1, **가중치 0.2**. 학습 시에만 실행(추론 비용 0). → 40m+ 원거리 병목(현 F1 0.085)을 직접 겨냥하는 보조 태스크.
5. **트래킹 출력**: bank에서 instance_id 발급/유지 (threshold 0.2), 추론 시 박스에 ID 부착. 학습 loss 변화 없음 (Sparse4Dv3 방식 — 트래킹 loss 자체가 없음).

### 4.4 증강 (신규 — 현재 전무)
- PhotoMetricDistortion(밝기/대비/채도) + GridMask(p=0.7) — 시퀀스 일관 적용. 도메인 다양성 부족을 보완하는 최중요 증강.
- resize-crop은 고정 스케일부터 (기존 704×256 유지). 좌우 flip은 3캠 좌우 스왑+extrinsic 미러링이 필요해 복잡 → 효과 확인 후 후순위.

### 4.5 Loss 총합
```
L = 2.0·L_cls(focal) + 2.0·L_box(L1, 10ch) + 0.2·L_quality + 0.2·L_depth
```
(aux loss 기존 방식 유지: 6층 전부, 마지막 1.0/이전 0.5)

---

## 5. 전이학습 전략 (v10 → v11)

`best_model.pth` (v10, softcal F1@0.15=0.3220)를 `strict=False`로 로드하되 키 그룹별로 명시 처리:

| state_dict 그룹 | 처리 | 근거 |
|---|---|---|
| `backbone.*` (ResNet50+FPN) | **그대로 로드** | MORAI 렌더링 도메인 적응 완료 — 최대 자산 |
| `decoder_layers.{0..5}.deformable_agg.*` | **그대로 로드** | 카메라 구성 동일(3캠), 구조 동형 |
| `decoder_layers.{i}.self_attn/ffn/각종 norm` | **그대로 로드** | 변화 없음 |
| `decoder_layers.{i}.det_decoder.cls_branch/quality_branch` | **그대로 로드** | 클래스 2종 동일 |
| `decoder_layers.{i}.det_decoder.reg_branch` | **로드 + 마지막층 vx/vy/vz 출력 행만 zero-init** | v10에서 velocity는 loss 제외라 해당 가중치 무의미 → 0에서 시작 |
| `decoder_layers.{1..5}.temp_attn/temp_anchor_encoder/temp_*_norm` | **재초기화 (사실상 신규)** | temporal off로 학습돼 gradient를 받은 적 없음 |
| `instance_feature` [900,256] | 그대로 로드 | 쿼리 임베딩 재사용 |
| `det_anchors_full` buffer [900,11] | **신규 데이터 train split에서 K-means 재생성** | 구 데이터 분포 기반. shape 동일해 교체만 하면 됨. yaw 초기값은 공식처럼 (sin,cos)=(0,1)+속도 0, 중심만 K-means |
| DenseDepthNet (신규) | 신규 초기화 | — |

**옵티마이저 그룹 (3단)**:
- backbone: lr × 0.1 (이미 적응됨 — 공식 stage2와 같은 비율)
- 기존 로드 모듈(디코더 공통부): lr × 0.5
- 신규/재초기화 모듈(temp_attn, depth head, reg velocity행, 앵커): lr × 1.0 (기준 lr 2e-4, §6)

**전이 검증 게이트 (Stage A0)**: temporal 끈 채 신규 데이터로 1 epoch 학습 → softcal F1@0.15가 빠르게 0.30+ 회복하는지 확인. 이 게이트를 통과해야 전처리(§2)가 옳다는 뜻. 통과 못하면 좌표/캘리브 디코딩부터 재점검.

**교훈 반영 (v9 사고 재발 방지)**: resume은 반드시 full checkpoint(model+optimizer+scheduler+step). `last_checkpoint.pth` 방식 유지, best 계열은 state_dict만.

---

## 6. 학습 스케줄

| 단계 | 내용 | 설정 | 기간 |
|---|---|---|---|
| **A0** 전이 sanity | temporal OFF, 신규 데이터+새 GT+velocity+depth on | batch 4×accum2, lr 2e-4, 1~2 epoch | 반나절 |
| **A1** 본학습 | temporal ON, 스트리밍 샘플러, 전 기능 | batch 슬롯 4(VRAM 따라 6), lr 2e-4 cosine, warmup 500 iter, grad clip 25, backbone lr×0.1, 30~50 epoch + early stop(patience 10) | 수일 |
| **B** (옵션) 매핑 | MGeo 링크/횡단보도 → polyline GT, map head 추가 | 공식 map branch 구조 이식 | 별도 |
| **C** (옵션) 모션/플래닝 | 재생성 트랙 GT에서 미래 궤적 GT 유도, parallel motion planner | 공식 stage2 대응 | 별도 |

참고: 공식 stage1은 batch 64로 100 epoch — 우리는 batch가 작으므로 epoch 수보다 **iteration 수 + early stop** 기준으로 운영. 데이터가 nuScenes 대비 다양성이 낮아 과적합이 먼저 옴 → val 곡선 감시.

### 평가 프로토콜 (확장)
- 기존: softcal F1@0.15 (거리 버킷 0-20/20-40/40-55m 포함) — **유지, primary**
- 추가: **mAVE**(velocity L2, TP에 대해), **AMOTA/AMOTP/IDS**(재생성 트랙 GT 기준, nuScenes 방식 단순화 구현), depth L1 (val 모니터)
- val = 시나리오 8개 고정, 시퀀스 통째 스트리밍 평가 (temporal 상태 유지한 채)

---

## 7. 실행 체크리스트 (순서 고정)

1. [x] **`verify_lidar_camera_overlay.py` 신규데이터 실행 — 완료.** scen30 cam_front/cam_front_left 오버레이 확인 결과 LiDAR 점이 차량·보행자·기둥·도로에 정확히 얹힘, 수직 밀림 없음. "지면 z=−2.0 vs config 1.35"는 body 원점이 지면 위에 있는 것일 뿐 버그 아님. **LiDAR↔카메라 상대변환 정상, depth GT 투영 신뢰 가능.**
2. [x] **`preprocess_dataset.py` 작성·실행 — 완료.** 60 시나리오 37,486프레임 → `labels_3d_v2/` + `scene_info.json` 생성(13.5초). 검증 게이트 통과: 치수편차 max 0.326m(<0.3 목표, scen09만 0.33 경미초과), 위치점프 p95 최악 1.34m(<3), 움직이는 차량 속도-변위 적분 ratio 0.85~1.32(불일치 0건). **신규 발견: 보행자 raw 속도 85% 결측(MORAI 미제공) — 정지 보행자는 v=0 정확, 걷는 보행자는 모션기반 유도(fill 로직 대기). 트랙 재연결 게이트를 frame-count가 아닌 실제 dt 기반 물리속도로 수정(>2s 갭은 identity 리셋)해 정지 보행자 과병합 해결.** 잔여 리스크: scen25/29 등에서 프레임드랍 구간 드문 트랙경계 오류(pos jump max 13~19m, p95는 ~1m라 극소수) — 트래킹 IDS에 미미한 영향.
3. [x] **`generate_depth_gt.py`에 r≥1.5m 필터 추가 + 60개 시나리오 전체 depth GT 생성 — 완료.** `labels_3d_v2/`가 있는 scen01~60 전체 처리(37,486프레임 × 3캠 = 112,458개 `.npy`). LiDAR 결측 0건. 센서 원점 기준 `sqrt(x^2+y^2+z^2) < 1.5m` 근거리 더미 520,756,573 / 1,079,596,800 pts 제거(48.24%). 생성 후 depth 파일 수가 `labels_3d_v2` 프레임 수와 전부 일치함을 검증. 검증 오버레이는 각 시나리오 `depth_gt_verify/`에 저장.
4. [x] **train/val split 확정 (52/8, stratified) + `make_kmeans.py` 재실행 — 완료.** 확정 다운로드 데이터는 `scen01~scen60`으로 고정하고, 다운로드 중인 `scen61+`는 자동 제외. Val split은 프레임/비어있지 않은 프레임/vehicle/pedestrian 분포가 전체와 거의 같도록 `scen08, scen18, scen23, scen30, scen32, scen33, scen41, scen54`로 확정. 산출물: `dataset_split_v11.json`, `anchor_kmeans_xy.npy`(900,2), `anchor_kmeans_full.npy`(900,11), `anchor_kmeans_meta.json`. `make_kmeans.py`는 `labels_3d_v2`만 읽고 train split 52개 시나리오에서 visible GT 70,251개로 앵커 재생성. Anchor z 채널은 SparseDrive 규약에 맞춰 `z_center` 사용, yaw는 `yaw_ego` 사용, velocity prior는 0으로 초기화.
5. [x] **`morai_dataset.py` → `MoraiTemporalDataset` 구현 — 완료.** `dataset_split_v11.json` 기준으로 확정 데이터 `scen01~scen60`만 사용하고 다운로드 중인 `scen61+`는 제외. 반환 계약: `images`, `intrinsics`, `extrinsics`, `projection_mat`, `T_ego2global/T_global/T_global_inv`, `timestamp`, `ego_pose`, `gt_boxes`(`z_center`, `yaw_ego`, `vx_ego/vy_ego`), `gt_labels`, `gt_track_ids`, `velocity_valid`, `gt_depth`(stride 4/8/16 dense sparse-map), `seq_id`, `seq_frame_idx`, `is_seq_start`. 기존 train loop 호환 alias(`dynamic_gt_boxes`, `dynamic_gt_labels`, `depth_gt`)도 유지. `train.py`는 Stage A0부터 `USE_TEMPORAL_DATASET=True`로 새 loader를 사용하도록 연결. 검증: train/val 프레임 수 32,504/4,982, val 시나리오 8개 고정, `labels_3d_v2` 원본 행 ↔ loader `gt_boxes` max diff 0, `T_global @ T_global_inv` 정상, depth positive map 생성 정상, 1-batch model forward + detection loss smoke test 통과.
6. [x] **`StreamingGroupSampler` 구현 — 완료.** `morai_dataset.py`에 추가. seq_id로 시퀀스 그룹화 → LPT 그리디로 B개 트랙(batch slot)에 whole 시퀀스 분배(트랙 길이 분포 결정적 → `__len__` 안정) → step마다 각 트랙에서 1프레임 emit(batch position ↔ bank slot 안정). 시퀀스 내부 프레임 순서 절대 불변, epoch마다 트랙 내 시퀀스 **순서만** reshuffle. `drop_uneven_tail=False`(기본): full coverage(ragged tail). `drop_uneven_tail=True`(bank용): 매 배치 정확히 B개+포지션 고정. `train.py`에 `USE_STREAMING_SAMPLER`(기본=`USE_TEMPORAL_MEMORY`) 플래그로 연결 — 켜지면 train은 `shuffle=True, drop_uneven_tail=True`, val은 `shuffle=False, drop_uneven_tail=False` batch_sampler 사용, epoch마다 `set_epoch()` 호출. 지금은 `USE_TEMPORAL_MEMORY=False`라 런타임 무변경(기존 shuffle DataLoader 경로 유지). 스모크 검증(`_tmp_streaming_sampler_smoke.py`) 전항목 OK: train 32,504 / val 4,982 coverage 누락·중복 0, scen61+ 혼입 0, 슬롯별 seq_frame_idx 연속 증가 + 시퀀스 경계 is_seq_start=1, drop_tail 배치 전부 B=4, epoch shuffle 시 순서 변경+len(8124) 고정. 1-batch model forward+loss+backward(`_tmp_streaming_forward_smoke.py`) 통과: 4슬롯이 서로 다른 시나리오/시퀀스에서 나오고 전부 is_seq_start=1, total loss 2.61.
7. [x] **instance bank 정밀화 + velocity loss + DenseDepthNet + 증강 — 완료 (2026-07-08).**
   - **velocity loss**: `loss_calculator.py` — `MATCH_CHANNELS=8`(매칭 비용에서 velocity 제외, SparseDrive 방식) / `REG_CHANNELS=10`(loss에 vx,vy 포함; vz는 GT가 z지터 유래라 제외). BOX_SCALE velocity를 m/s 스케일 [15,15,3]으로 조정. 단위테스트: velocity 오차 시 box loss 발생 확인.
   - **DenseDepthNet**: 공식(models/blocks.py) 이식 — FPN stride 4/8/16 레벨에 1×1 conv, `depth=exp(conv)×focal/100`, GT>0 masked L1 fp32, loss_weight 0.2 내장. 학습 시만 실행(`model_out['depth_pred']`), train loop에서 det loss에 합산+로깅(`Depth:` 항목). 생성자 기본 `use_dense_depth=False`(구 스크립트 v10 strict 로드 호환), train.py에서 `USE_DENSE_DEPTH=True`로 켬.
   - **instance bank 재작성(공식 semantics, 배치 슬롯별)**: 기존 전역 단일 캐시는 배치 내 샘플이 서로 덮어써 스트리밍과 근본 비호환이라 폐기. 새 구조: 슬롯별 `_bank[b]`(top-600 feature/anchor/confidence/ego_pose/timestamp/context, detach 저장) — ①get: `같은 context + 0<dt≤2s`에서만 전파, **velocity 모션보상 포함 정렬**(`center += v·dt` 후 rigid; align_anchor_prev_to_current에 dt 파라미터 추가, 수치검증 오차 1e-8) ②**update**: layer0(single-frame) 직후 전파 600개가 confidence 하위 600개 교체(top-300 fresh 유지) — 공식 InstanceBank.update 대응 ③layers1-5 temp_gnn cross-attn(K/V=전파 인스턴스) ④cache: 최종 confidence를 `max(prev×0.6, new)`로 융합 후 top-600 저장. 시퀀스 경계 리셋은 context+dt로 자동(같은 시나리오 연속 청크는 전파 유지 — 물리적으로 올바름).
   - **⚠️ float64 timestamp 버그 수정**: ego_pose 텐서(float32)의 유닉스 시각(1.78e9)은 분해능 ~256s라 dt가 항상 0 → temporal이 영구 비활성이었음. forward에 `timestamps`(float64) 파라미터 추가, dt는 반드시 이걸로 계산. **향후 forward 호출 시 `timestamps=batch['timestamp']` 필수** (없으면 fallback이 조용히 temporal을 죽임).
   - **증강**: PhotoMetric(brightness/contrast/sat/hue, mmdet 축약)을 `MoraiTemporalDataset(photometric_aug=True)`로 — **(seq_id, epoch) 시드 고정 → 시퀀스 내 일관**(공식 keep_consistent_seq_aug 대응), train loop에서 `train_ds.set_epoch(epoch)` 호출. GridMask(p=0.7, 공식 이식)는 model forward에서 학습 시만 배치 공통 마스크 적용.
   - **검증 완료**: `_tmp_checklist7_unit.py`(align 수치/velocity loss/photometric 일관성/gridmask 4항목 OK) + `_tmp_checklist7_temporal_smoke.py`(GPU, temporal ON 연속 10 step: step0 fresh→step1+ 활성 18/18, loss 4.25→2.74 수렴, depth loss 유한, 슬롯↔시나리오 오염 0, eval 스트리밍 OK) + synthetic 리셋 테스트(dt>2s/역순/ctx변경 리셋, 슬롯 독립, decay 0.6 융합) + temporal OFF 기본경로 무손상(구 시그니처 하위호환).
8. [x] **전이 로더 `load_v10_weights` + 3단 옵티마이저 — 완료 (2026-07-08).**
   - **`load_v10_weights(model, ckpt_path)`** (train.py): v10 `best_model.pth`(순수 state_dict, 752키)를 규칙 기반 부분전이. 결과: **로드 681 / 스킵(규칙) 71 / shape불일치 0 / 신규 6**. 스킵 규칙 = `det_anchors_full`(신규 K-means 앵커 buffer 유지 — 덮어쓰기 금지) 1개 + `temp_*`(layer1-5 temporal, 학습된 적 없어 재초기화) 70개. 신규 6 = `depth_net.depth_layers.{0,1,2}.{weight,bias}`. 로드 후 `reg_branch.6.{weight,bias}`의 velocity 행(8,9,10=vx,vy,vz)을 6개 layer 전부 zero-init.
   - **`build_transfer_param_groups(model, base_lr=2e-4)`** (train.py): 3단 그룹 — backbone(173텐서)×0.1=2e-5 / loaded(337텐서, 전이 디코더 공통부+instance_feature)×0.5=1e-4 / new(88텐서: depth_net·temp_*·reg_branch.6)×1.0=2e-4. reg_branch.6은 텐서 단위라 velocity 행만 분리 불가 → 마지막 Linear 전체를 new 그룹에(velocity 학습 우선). wd: backbone 1e-3, 나머지 1e-2.
   - **train.py 연결**: config `TRANSFER_FROM_V10`(기본 None) 플래그. 설정 시 `TRANSFER_MODE=True` → RESUME_FROM(full-checkpoint 재개) 무시하고 model에 부분전이 + 3단 옵티마이저 + wandb name `v11-transfer`. **기본 None이라 기존 재개 경로 무손상(하위호환).** Stage A0에서 `TRANSFER_FROM_V10="best_model.pth"`로 켠다.
   - **검증(`_tmp_checklist8_transfer.py`) 전항목 OK**: 키 커버리지 758/758, 대표키 전이값 일치, 앵커 신규 유지(v10값과 다르고 전이전과 동일), velocity 행 6 layer 전부 0, temp_* 70키 재초기화(모델 init 유지), 3단 그룹 겹침0·전체커버·분류정확·lr 2e-5/1e-4/2e-4, **전이 후 cls loss 1.011 < 랜덤 1.330**(전이 성공 신호).
   - **주의(Stage A0에서 관찰)**: box loss(10ch)는 전이(0.062)가 랜덤(0.034)보다 높게 시작 — (a)앵커 재생성으로 v10 offset 기준이 바뀌고 (b)velocity 행 zero-init이라 초기 velocity 오차. reg_branch.6이 new 그룹(×1.0)이라 빠르게 재학습되지만, box loss 초반 상승은 정상. 실제 검출 성능은 F1로 판정.
9. [ ] Stage A0 게이트 → 통과 시 Stage A1 — **`TRANSFER_FROM_V10="best_model.pth"`, `USE_TEMPORAL_MEMORY=False`로 1~2 epoch 학습해 softcal F1@0.15가 0.30+ 회복 확인**(전처리·전이 정합성의 최종 게이트). 통과 후 `USE_TEMPORAL_MEMORY=True`(스트리밍 샘플러 자동 활성)로 Stage A1.
10. [ ] 평가 확장 (mAVE, AMOTA/IDS)

### Phase 0 산출물 (완료)
- `preprocess_dataset.py` — 재실행 가능, `--report-only`로 검증만 가능
- `dataset/<scen>/labels_3d_v2/live_NNNNNN.csv` — 컬럼: frame_id,timestamp,track_id,class_id,x,y,z_center,w,l,h,yaw_ego,vx_ego,vy_ego,vz,yaw_global,gx,gy,sin_yaw_ego,cos_yaw_ego,vel_source
- `dataset/<scen>/scene_info.json` — 프레임별 T_ego2global(4×4, body→global), timestamp, ego pose + 시나리오 품질지표
- 원본 `labels_3d/`는 무수정 보존

## 8. 리스크

- **프레임 지터/드랍** (>300ms 갭 1,258회): bank가 timestamp 기반이라 견디지만, 갭 직후 프레임은 temporal 이득이 줄어듦 — 성능 분석 시 갭 프레임 분리 집계 권장
- **lidar 높이 캘리브**: 미해결 시 depth GT 전체 바이어스 → 체크리스트 1번이 최우선인 이유
- **트랙 재연결 오류**: 보행자 무리(scen01 frame 736의 동시 점프)처럼 게이트 내 다중 후보 시 오연결 가능 — 치수 시그니처 + 헝가리안 매칭으로 보강, 검증 리포트에서 잔여 점프 0 확인
- **flip 증강 부재**: 데이터 다양성 낮은 상태에서 좌회전/우회전 비대칭 과적합 가능 — A1 결과 보고 3캠 스왑 flip 도입 검토
- **VRAM**: temporal cross-attn(600 keys)+depth head 추가로 v10 대비 증가 — batch 4 유지, 필요시 grad checkpoint

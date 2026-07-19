# SparseDriveV2 분석 리포트 — 내 프로젝트(morai-3d-detection) 관점

작성일: 2026-07-16 · 성격: **read-only 분석** (코드 무수정, 본 리포트 1개 파일만 생성)
분석 대상 브랜치: `SparseDriveV2-main` (= GitHub `main`, NAVSIM 통합판). Bench2Drive 브랜치는 로컬에 없음 → 미분석.

라벨 규칙: **[확인됨]** = 직접 연 파일:라인 근거 있음 / **[추론]** = 근거 불충분한 판단.
경로 표기는 `projects/` 루트 기준 상대경로 + `:라인`.

---

## 0. 결정에 직결되는 요약 (먼저 읽기)

### Q1. v2가 perception(detection/tracking)을 바꿨는가?
**아니오 — 이 릴리스에는 perception head 자체가 존재하지 않는다. [확인됨]**

`SparseDriveV2-main`의 모델은 백본 + 상태인코더 + **`TrajectoryHead`(자차 궤적 계획) 단 하나**로 구성된다
(`SparseDriveV2-main/navsim/agents/sparsedrive/sparsedrive_model.py:14,26-33,47-54`).
저장소 전체 agent 트리에서 `Head` 클래스는 `TrajectoryHead` 하나뿐이며, 3D detection head·tracking/instance-memory·map head·타 에이전트 motion head가 **전부 없다** [확인됨].
따라서 "v2가 detection/tracking을 개선했다"는 명제는 **검증 불가가 아니라 해당 없음**이다 — v2(이 릴리스)는 perception을 학습하지 않고, 타 객체 정보는 오직 학습 시 PDM metric-cache GT로만 들어온다
(`.../custom_decoder.py:272-288`, `.../scorer/get_pdm_score_v2.py:89-91`).

> 가설(2단계) "v2 혁신은 planning/scoring이고 det/track head는 v1을 계승한다" →
> **앞부분 확인, 뒷부분 반증.** planning/scoring이 혁신인 것은 맞으나, det/track head를 "계승"한 게 아니라 **아예 빼버렸다**. §2 참조.

### Q2. v2의 velocity factorization이 아이디어 X와 충돌하는가?
**충돌하지 않는다 — 완전히 다른 서브시스템이다. [확인됨]**

| | 닿는 지점 | 대상 | 근거 |
|---|---|---|---|
| **아이디어 X의 velocity supervision** | detection head의 `vx,vy` 회귀 채널 (박스 8·9번), `loss_bbox` L1 | **타 차량/객체**의 속도 (perception-side) | `morai-3d-detection/decoder.py:24,32,64`, `.../loss_calculator.py:286-289` |
| **v2의 velocity-profile factorization** | `vel_vocab`(256모드 자차 속도 프로파일) + velocity imitation loss | **자차(ego)**의 미래 종방향 속도 (planning-side) | `.../sparsedrive_model.py:77-80`, `.../custom_decoder.py:254-259`, `.../sparsedrive_features.py:456` |

두 "velocity"는 **이름만 같고 지시 대상이 서로소**다. X는 "남의 차 속도를 라이다로 잘 맞추자"(인지),
v2는 "내 차가 앞으로 낼 속도 후보를 어휘로 나눠 점수 매기자"(계획). 겹칠 코드 지점이 없으므로 **중복도 상충도 없다** [확인됨].
(자세한 전수 추적은 §3.)

**한 줄 결론:** v2는 아이디어 X와 경쟁 관계가 아니다. X는 인지 head 개선, v2는 계획 head 신설이라 **직교(orthogonal)**하며, 오히려 나중에 둘을 같이 써도 상호 간섭이 없다.

---

## 1. 저장소 위치 & 구조 맵 (0·1단계)

### 1.0 위치 파악 [확인됨]
| 대상 | 경로 | 상태 |
|---|---|---|
| SparseDriveV2 repo | `SparseDriveV2-main/SparseDriveV2-main/` | 존재 (NAVSIM 통합판, main 브랜치) |
| SparseDriveV2 논문 PDF | — | **없음** (로컬엔 v1 논문 PDF만 존재) |
| SparseDrive **v1** repo | `SparseDrive/` | 존재 (mmdet3d 플러그인) |
| SparseDrive v1 논문 PDF | `SparseDrive_ EndtoEnd Autonomous Driving via Sparse Scene Representation.pdf` | 존재 (arXiv:2405.19620) |
| 내 프로젝트 | `morai-3d-detection/` | 존재 (v1 파생, 3전방캠, MORAI) |

v2 정보의 텍스트 근거는 논문 PDF 대신 `SparseDriveV2-main/README.md`(초록 포함) + 코드로 확보 [확인됨].
README 요지: 제목 *"Scoring is All You Need for End-to-End Autonomous Driving"*, 혁신 2가지 —
(1) 궤적을 **geometric path × velocity profile**로 분해한 scalable vocabulary,
(2) coarse 분해 스코어링 → 소수 조합 궤적 fine 스코어링. 벤치마크는 NAVSIM(PDMS/EPDMS)·Bench2Drive(DS/SR) — **전부 planning 벤치마크**.

### 1.1 SparseDriveV2 모듈 트리 (핵심 파일 한 줄 요약)
경로 접두: `SparseDriveV2-main/navsim/agents/sparsedrive/`

**Perception (detection/tracking/mapping): 없음.** 해당 디렉터리·클래스가 존재하지 않음 [확인됨].

**Motion/Planning (=이 저장소의 전부):**
```
sparsedrive_model.py        SparseDriveModel(백본+status+TrajectoryHead) / TrajectoryHead(path·vel·traj 3어휘 로드)  :14,57
  ├ sparsedrive_backbone.py SparseBackbone (ResNet-34 + img neck)
  ├ custom_decoder.py       CustomTransformerDecoder/Layer — path/vel/traj 3분기 + metric_heads(점수)  :18,43
  ├ blocks.py               DeformableFeatureAggregation(이미지 특징 샘플링, v1 유래) — 단 anchor가 궤적 keypoint  :22,288
  └ scorer/
      ├ get_pdm_score_v1.py / get_pdm_score_v2.py  metric_cache.pkl 병렬 로드 후 PDM 서브점수 산출  :83-85 / :89-91
      └ pdm_score_v1.py / pdm_score_v2.py           PDM 시뮬레이터(충돌/주행가능영역/TTC/진행도…)
sparsedrive_config.py       SparseDriveConfig(어휘 경로·차원·loss sigma·metric 목록)  :10
sparsedrive_features.py     Feature/TargetBuilder — 카메라 특징 + status, 그리고 자차 path/velocity/trajectory **타깃** 생성  :425,438
sparsedrive_agent.py        SparseDriveAgent(pl 래퍼) — 입력 센서 카메라만, compute_loss=loss_dict 합  :22
```
핵심: 모델은 카메라 특징을 **자차 궤적 후보(path·velocity·trajectory 어휘)** 위에 deformable attention으로 샘플링하고, 후보들을 점수화해 argmax 궤적을 낸다. 인지 예측이 파이프라인에 없음 [확인됨].

### 1.2 SparseDrive v1 모듈 트리 (대조 기준) [확인됨]
경로 접두: `SparseDrive/projects/mmdet3d_plugin/`
```
core/box3d.py                     X,Y,Z,W,L,H,SIN,COS,VX,VY,VZ = range(11)  ← 11차원 박스(속도 포함)  :1
models/detection3d/
  detection3d_head.py             Sparse4DHead (3D 검출 head, map head도 동클래스 재사용)  :28
  detection3d_blocks.py           SparseBox3DRefinementModule(output_dim=11, 속도 잔차 회귀)  :82,145
  decoder.py                      SparseBox3DDecoder(박스 디코드, VX:이후 통과)  :24
  losses.py                       SparseBox3DLoss(cls focal + box L1[속도 포함] + centerness/yawness)  :11,67
models/instance_bank.py           InstanceBank — det/map temporal 큐 + **track ID 발급**  :25,223
models/motion/
  instance_queue.py               InstanceQueue — motion/plan용 ID 매칭 히스토리 큐 + ego 큐  :15,116,181
  motion_planning_head.py         MotionPlanningHead — motion+planning 결합 head  :34
configs/sparsedrive_small_stage2.py  풀 e2e config (embed 256, det 900 anchor, num_decoder 6, load_from stage1)  :721
```

### 1.3 내 프로젝트 morai-3d-detection 관련 파일 [확인됨]
```
decoder.py            box head reg_branch → [900,11] (x,y,z,w,l,h,sin,cos,vx,vy,vz)  :24,32,64
static_decoder.py     map/polyline head (속도 없음, 20 xy점)  :65,72
temporal_decoder.py   TemporalDecoder — **미사용 스텁**(track_ids=arange(900) 자리표시자)  :4,42
anchor_generator.py   generate_anchors_full → [900,11], 속도채널 기본 0  :37-39,68
loss_calculator.py    HungarianMatcher(검출매칭, 8ch) + MapHungarianMatcher + CustomLoss(cls/bbox/quality)  :40,96,152
morai_dataset.py      GT 로드 — 박스 vx_ego,vy_ego,vz + gt_track_ids  :144,627,644
train.py              모델 클래스 + 인라인 instance memory(_bank), temporal 정렬/속도보정  :574,781,877,2082
```

---

## 2. v1 → v2 변경점 & 가설 검증 (2단계, 가장 중요)

### 2.1 파일 단위 대조 결론
v1(`SparseDrive/`)과 v2(`SparseDriveV2-main/`)는 **저장소 골격이 다르다**: v1은 mmdet3d 플러그인 기반 풀 인지+계획 스택, v2는 **NAVSIM agent 프레임워크에 얹은 계획 전용 agent**다. 따라서 "파일 대 파일 diff"가 성립하는 부분은 거의 없고, 재사용된 것은 **저수준 블록**뿐이다.

| 구성요소 | v1 | v2(this release) | 판정 |
|---|---|---|---|
| 3D detection head | `Sparse4DHead` (`detection3d_head.py:28`) | **없음** | v2가 **제거** [확인됨] |
| tracking / ID | `InstanceBank.get_instance_id` (`instance_bank.py:223`) | **없음** | v2가 **제거** [확인됨] |
| motion/plan head | `MotionPlanningHead` (`motion_planning_head.py:34`) | `TrajectoryHead`(신설, 어휘 스코어링) (`sparsedrive_model.py:57`) | **전면 재설계** [확인됨] |
| 어휘/앵커 | det/map/motion/plan k-means (박스·라인·궤적) | **path 1024 × velocity 256 → traj 1024×256** 자차 어휘 (`sparsedrive_config.py:31-40`) | v2 **신개념** [확인됨] |
| 스코어링 | HierarchicalPlanningDecoder rescore (`motion_planning_head.py:496`) | coarse 분해 top-k(`path_filter_num`,`velocity_filter_num`) → fine traj + PDM metric heads (`custom_decoder.py:193-234,270-288`) | v2 **신개념** [확인됨] |
| 재사용된 코드 | Deformable/sparse 특징 샘플링 | `DeformableFeatureAggregation`(`blocks.py:22`) — 단 anchor가 박스가 아니라 **궤적 keypoint** | **저수준만 계승** [확인됨] |

### 2.2 가설 판정
> 가설: "v2의 혁신은 planning/scoring 쪽이고, detection/tracking head는 대체로 v1을 계승한다."

- **혁신 = planning/scoring**: **확인됨.** v2의 학습 손실·모듈·어휘가 전부 자차 궤적 스코어링(imitation + PDM distillation)에 집중 (`custom_decoder.py:236-288`). README 초록도 동일 주장.
- **detection/tracking head를 계승**: **반증됨.** 이 릴리스는 det/track head를 계승한 게 아니라 **모델에서 삭제**했다. 인지는 학습 파이프라인 밖(PDM metric-cache GT)에서만 소비된다 (`custom_decoder.py:272-288`).
  - 뉘앙스: NAVSIM 벤치마크는 "인지 성능"이 아니라 "계획 궤적의 PDM 점수"를 평가하므로, v2 논문은 애초에 인지 head를 필요로 하지 않는다. 즉 **"바꿨다"가 아니라 "범위에서 뺐다"**가 정확한 서술 [확인됨].
- 한계/미확인: Bench2Drive 브랜치(로컬 없음)는 CARLA 폐루프라 별도 인지 경로가 있을 수 있음 → **미분석 [추론]**. 단 main 릴리스 기준으로는 위 판정이 확정적.

**내 프로젝트에의 함의:** 내 코드는 v1 인지 스택 파생이다. v2는 **그 인지 스택을 개선한 게 아니라 대체 목표(계획)로 갈아탄 것**이므로, "v2로 내 detection을 업그레이드"할 소스가 이 저장소엔 존재하지 않는다.

---

## 3. velocity 전수 추적 (3단계 — 아이디어 X 충돌 판정 핵심)

### 3.1 v2 쪽 velocity 전수 (전부 planning-side, ego) [확인됨]
| 위치 | 내용 | 분류 |
|---|---|---|
| `sparsedrive_config.py:32,38-40,46,49` | `velocity_anchor=velocity_256.npy`, `mode_vel=256`, `len_vel_seq=8`, `vel_time_interval=0.5`, filter/sigma | (c) 자차 계획 어휘 |
| `sparsedrive_model.py:77-80,96-100,108,113` | `vel_vocab` 로드(256모드, frozen) + `vel_pos_embed` | (c) 자차 계획 어휘 |
| `custom_decoder.py:82-108,181-187` | velocity 분기(이미지 attention→`vel_scores`) | (c) 자차 속도 후보 점수 |
| `custom_decoder.py:203-211` | velocity top-k coarse filter | (c) |
| `custom_decoder.py:254-259` | **velocity imitation loss**: `target_vel=targets["velocity"]`, CE(vel_scores, softmax(-|vel_vocab-target|)) | (c) 자차 속도 프로파일 지도 |
| `sparsedrive_features.py:53` | `ego_statuses[-1].ego_velocity` → status 입력 | (c) 자차 **현재** 속도(상태) |
| `sparsedrive_features.py:455-456,463` | `velocity = ‖Δ자차궤적‖ / 0.5` → `targets["velocity"]` (주석 "londitudinal velocity") | (c) 자차 미래 종방향 속도 타깃 |

→ v2의 velocity는 **전부 자차(ego) 계획**이다. (a) 검출 앵커 속도 채널 = 없음, (b) instance/temporal 메모리 큐 = 없음 [확인됨].

### 3.2 아이디어 X 쪽 velocity 착지점 (전부 perception-side, 타 객체) [확인됨]
| 위치 | 내용 | 분류 |
|---|---|---|
| `morai-3d-detection/decoder.py:24,32,64` | reg_branch [900,11]의 `vx,vy,vz`(8·9·10 채널) | (a) 검출 앵커 속도 채널 |
| `morai-3d-detection/anchor_generator.py:39,68` | 앵커 속도 채널(기본 0) | (a) |
| `morai-3d-detection/loss_calculator.py:11-14,286-289` | `REG_CHANNELS=10` → box L1에 `vx,vy` 포함(**현재 활성**), 매칭비용엔 제외(`MATCH_CHANNELS=8`) | (a) 타 객체 속도 지도 |
| `morai-3d-detection/train.py:574,589-591` | temporal 메모리에서 속도로 앵커 위치 보정(`center += v·dt` 후 회전) | (b) temporal 전파 특징(손실 아님, 기본 OFF) |
| `morai-3d-detection/morai_dataset.py:144` | GT `vx_ego,vy_ego,vz` 로드 | (a) GT 소스 |

### 3.3 최종 충돌 판정
- **닿는 지점이 서로 다른 서브시스템:** X = 검출 head의 타 객체 속도 회귀(`decoder.py`/`loss_calculator.py`), v2 = 자차 계획 속도 어휘(`sparsedrive_model.py`/`custom_decoder.py`). 공유 텐서·공유 모듈·공유 손실이 하나도 없다 [확인됨].
- **겹침 여부:** 없음. 두 축이 만나는 유일한 개념적 접점은 "속도라는 단어"뿐이고, 물리적 지시대상(남의 차 vs 내 차)·좌표계·손실 형태(연속 L1 회귀 vs 어휘 CE 분류)가 전부 다르다.
- **결론:** 아이디어 X의 velocity supervision과 v2의 velocity-profile factorization은 **상충하지 않고 중복하지도 않는다.** 서로 다른 문제(인지 정확도 vs 계획 커버리지)를 푼다 [확인됨].

---

## 4. 내 setup으로의 이식성 (4단계)

내 setup: 전방 3카메라 · MORAI · **detection/tracking 중심** · **planning 벤치마크 없음** · **자차 미래궤적 GT 파이프라인 없음(Phase 3 예정)**.

### 4.1 이식 **불가** (planning 인프라 전제) [확인됨]
v2의 **본질 기여 전부**가 여기 해당한다:
- factorized path×velocity 어휘 (`sparsedrive_model.py:73-89`) — 자차 미래궤적에서 k-means로 뽑은 어휘 필요(`docs/train_eval.md:26-31`).
- scalable coarse-to-fine 스코어링 (`custom_decoder.py:193-234`).
- **PDM metric distillation** (`custom_decoder.py:270-288`) — `metric_cache.pkl`(nuplan/NAVSIM 시뮬레이터가 계산한 충돌·주행가능영역·TTC·진행도 서브점수) 필수(`scorer/get_pdm_score_v2.py:89-91`).
- 근거: 이 셋은 (자차궤적 GT + PDM 시뮬레이터 + NAVSIM 데이터 캐시)가 있어야 학습된다. MORAI엔 PDM 시뮬레이터도, planning 벤치마크도 없음 → **이식 불가** [확인됨]. (MIGRATION_PLAN Phase 3에서 자차궤적 GT 자체는 유도 가능하다고 했으나, PDM 점수 라벨 생성기는 별개 문제 [추론].)

### 4.2 이식 "가능"하지만 **얻는 게 없음** [확인됨/추론]
- `DeformableFeatureAggregation`(`blocks.py:22`)·ResNet-34 백본·GridMask — 이식 가능하나 **v1 유래 기술이라 내 코드가 이미 파생**하고 있어 신규 이득 없음 [확인됨: 기술 출처 / 추론: 이득 없음].
- v2에는 **이식할 perception 개선이 존재하지 않는다** — det/track head가 없으므로(§1.1) 인지 정확도·속도추정·추적 측면에서 가져올 코드가 물리적으로 없다 [확인됨].

**요약:** v2에서 내 detection/tracking 프로젝트로 실질 이식할 수 있는 인지 개선은 **없다**. v2의 가치는 전부 계획 벤치마크 전용 인프라에 묶여 있다.

---

## 5. 아이디어 X 삽입 지점 매핑 (5단계)

검증된 아키텍처(=내 `morai-3d-detection` 코드) 기준. **v2가 아니라 내 코드에 넣는다** — v2엔 인지 head가 없어 삽입 대상 자체가 없기 때문 [확인됨].

아이디어 X 3요소별 착지점:

**(1) velocity supervision** — *이미 부분 존재, 소스만 교체/보강*
- 현재: GT 속도(`morai_dataset.py:144`) → box L1의 `vx,vy`(`loss_calculator.py:286-289`, 활성, 가중치 2.0의 10채널 중 2채널).
- X 적용: 라이다 트래커의 프레임차분 속도를 (a) GT 속도 소스로 대체/보강(`morai_dataset._label_v2_to_box`, `morai_dataset.py:128-145`) 하거나, (b) `loss_calculator.CustomLoss.forward`(`:286-293`)에 **전용 velocity 항**을 분리 신설(현재는 box L1에 파묻혀 가중 제어 불가).
- ⚠️ v2 velocity 어휘와 충돌? **아니오** — 다른 서브시스템(§3.3). flag 불필요.

**(2) tracking-consistency supervision** — *신규(현재 완전 부재)*
- 현재: `gt_track_ids`는 로드되지만(`morai_dataset.py:627,644`) **손실로 전달되지 않음** — `criterion(...)`이 `gt_classes,gt_boxes`만 받음(`train.py:1487-1493`) [확인됨].
- X 적용: ① `criterion` 호출부(`train.py:1487-1493`)에 `gt_track_ids` 전달 → ② `loss_calculator`에 track-consistency 손실 신설(같은 트랙 매칭 앵커의 특징/속도 일관성) → ③ 인라인 instance memory `_bank`(`train.py:781,877`)의 슬롯 연결을 ID 기반으로 승격. 라이다 트래커의 일관 ID가 이 지도신호의 소스가 됨.
- 주의: 현 temporal 메모리는 **슬롯 위치 기반**이고 track ID를 안 나름(`train.py`), 기본 OFF(`USE_TEMPORAL_MEMORY` 기본 '0', `train.py:2082`). `temporal_decoder.py`의 `TemporalDecoder`는 **미사용 스텁**(`:42` track_ids=arange 자리표시자)이라 여기 넣지 말 것.

**(3) motion-prior supervision** — *부분 인프라 존재(전파용), 지도신호는 신규*
- 현재: 속도 기반 앵커 전파는 있음(`train.py:574,589-591`)이나 이는 **추론용 전파**지 손실이 아님, 그리고 기본 OFF.
- X 적용: 트래킹 궤적에서 유도한 motion prior를 temporal 정렬 손실로 추가(`loss_calculator` 신설 항 + `train.py` temporal 경로).

**라이다 트래클릿 소스의 위치 (중요, [추론]):**
- `morai-3d-detection` 파이썬 내에는 라이다 트래커·Hungarian 트래킹·Kalman/SORT가 **전혀 없다**(검출/맵용 Hungarian 매칭만 존재) [확인됨].
- MIGRATION_PLAN §2.1의 "greedy NN 재연결"은 트랙 GT 복원용 룰베이스 재연결 로직으로 아이디어 X의 원시 형태에 해당(단 라이다 아닌 GT global 좌표 기반) [확인됨].
- 실제 **룰베이스 라이다 트래커**는 이 저장소 밖 `HL_FMA/Main/src/Lidar/IMMFilter/IMMFilter.cpp`(C++ IMM 필터) 계열로 보이며, X는 그 트래클릿을 학습 라벨로 끌어와야 함 → **크로스-프로젝트 라벨 브리지가 별도 과제** [추론].

**v2와의 충돌/중복 flag:** 없음. 삽입 지점 (1)(2)(3) 어느 것도 v2의 velocity 어휘·스코어러와 코드/텐서를 공유하지 않는다. 향후 v2식 계획 head를 별도로 붙이더라도, 자차 속도 어휘 ↔ 타 객체 속도 지도는 독립 유지된다 [확인됨].

---

## 6. 학습 설정 diff (6단계)

### 6.1 v2 training config 요약 [확인됨]
근거: `SparseDriveV2-main/scripts/training/sparsedrive_navsimv1.sh`, `sparsedrive_navsimv2.sh`, `.../config/common/agent/sparsedrive_agent.yaml`, `sparsedrive_config.py`.

| 항목 | 값 | 벤치마크 특화? |
|---|---|---|
| stage 구성 | **단일 스테이지** (v1의 stage1 인지사전학습+stage2 e2e 구조 없음) | — |
| epoch | `max_epochs=10` | |
| lr | `1e-4` (`sparsedrive_agent.yaml:15`) | |
| batch / workers | 16 / 16 (prefetch 4) | |
| split | `train_test_split=navtrain`, cache `data_cache_navtrain` | ✅ NAVSIM 전용 |
| trajectory sampling | `time_horizon=4s, interval=0.5s` (num_poses=8) | ✅ 계획 지평 |
| backbone | ResNet-34 (`bkb_path=ckpt/resnet34.bin`) | |
| **손실 항** | `path_loss`(imi, σ=4.0) · `velocity_loss`(imi, σ=4.0) · `traj_loss`(imi, σ=4.0) · `{metric}_loss`(PDM distill, ×5.0) | `metric`·PDM은 ✅ 벤치마크 특화 |
| metric 목록 | v1: no_collision·drivable·driving_dir·ttc·comfort·progress / v2: +traffic_light·lane_keeping·history_comfort (`sparsedrive_config.py:58-59`, `.sh` 오버라이드) | ✅ NAVSIM v1/v2 |
| 점수 공식 | `custom_decoder.py:292-312` (v1/v2 서로 다른 곱·가중합) | ✅ 벤치마크 특화 |
| 선행 캐싱 | dataset caching + **metric caching(PDM)** + anchor 클러스터링 필수 (`docs/train_eval.md:11-31`) | ✅ 시뮬레이터 필요 |

### 6.2 내 프로젝트 학습과의 diff [확인됨/추론]
- 스테이지: v2 단일 vs 내 코드 env 기반 모드 분리(step/fast/resume-transfer, `train.py`) — 목적이 달라 직접 대응 안 됨 [추론].
- 손실 철학: v2 = **어휘 CE + PDM distillation**(계획), 내 코드 = **focal cls + box L1(속도 포함) + centerness/yawness quality**(인지) — 겹치는 손실 항 없음 [확인됨: 내 손실 `loss_calculator.py:293`].
- 벤치마크 특화로 **이식 불가** 표시 항목: `navtrain/navtest` split, `metric_cache`/PDM 손실, path/velocity/trajectory 어휘 클러스터링, trajectory_sampling 지평 — 전부 §4.1과 동일하게 MORAI에 부재.

---

## 부록 A. 확인됨 vs 추론 요약표
| 핵심 주장 | 라벨 | 대표 근거 |
|---|---|---|
| v2 릴리스에 detection/tracking/map head 없음 | 확인됨 | `sparsedrive_model.py:14,26-33` (유일 head=TrajectoryHead) |
| v2 velocity = 자차 계획 속도 프로파일 | 확인됨 | `custom_decoder.py:254-259`, `sparsedrive_features.py:456` |
| v2 인지는 PDM metric-cache GT로만 소비 | 확인됨 | `custom_decoder.py:272-288`, `scorer/get_pdm_score_v2.py:89-91` |
| 내 코드 velocity 지도 = 타 객체(perception), 현재 활성 | 확인됨 | `loss_calculator.py:286-293`, `decoder.py:24,32` |
| 아이디어 X ↔ v2 velocity 무충돌·무중복 | 확인됨 | §3.3 (서로소 서브시스템) |
| 내 코드에 라이다 트래커/트래클릿/track-ID 손실 없음 | 확인됨 | `train.py:1487-1493`(criterion 미전달), 트래커 grep 무결과 |
| v2에서 이식할 perception 개선 없음 | 확인됨 | §1.1, §4.2 (인지 head 부재) |
| v2 core는 planning 벤치마크 인프라 전제라 이식 불가 | 확인됨 | `docs/train_eval.md:11-31`, `custom_decoder.py:270-288` |
| Bench2Drive 브랜치의 인지 경로 유무 | 추론(미분석) | 로컬에 브랜치 없음 |
| 라이다 트래클릿 소스가 HL_FMA IMMFilter 계열 | 추론 | 외부 C++ 트래커, 크로스-프로젝트 라벨 브리지 필요 |

## 부록 B. 실제로 연 파일 (인용 근거)
- V2: `sparsedrive_model.py`, `custom_decoder.py`, `sparsedrive_config.py`, `sparsedrive_features.py`, `sparsedrive_agent.py`, `README.md`, `docs/train_eval.md`, `scripts/training/*.sh`, `config/.../sparsedrive_agent.yaml` (+ 하위 agent가 연 `blocks.py`, `sparsedrive_backbone.py`, `scorer/*`)
- v1: (하위 agent) `core/box3d.py`, `detection3d_head.py`, `losses.py`, `decoder.py`, `instance_bank.py`, `motion/instance_queue.py`, `motion_planning_head.py`, `configs/sparsedrive_small_stage2.py`
- morai: `decoder.py`, `loss_calculator.py`, `temporal_decoder.py`, `MIGRATION_PLAN_SPARSEDRIVE.md` (+ 하위 agent가 연 `morai_dataset.py`, `anchor_generator.py`, `static_decoder.py`, `train.py`, `inference.py`, `camera_configs.py`)

*(본 리포트는 판정·분석 전용. 코드 수정 제안은 포함하지 않는다.)*

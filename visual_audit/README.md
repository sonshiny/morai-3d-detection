# MORAI 3D-detection Visual Audit

생성 스크립트: `visual_audit/*.py` (원본 데이터는 읽기 전용, 어떤 라벨/pose/scene_info도 수정하지 않음).

이미지 총계: depth=9, v2_vs_v3=40, boundary_50m=13, track_temporal=9 (합계 71).

좌표 규약: 카메라 투영은 `camera_configs.py`/`visualize_camera_proj.py`와 동일 (depth=cam_x, u=fx·(-cam_y)/depth+cx, v=fy·(-cam_z)/depth+cy). BEV는 ego 원점, x=전방, y=좌.

**중요:** depth_gt의 (u,v)는 학습 입력 좌표계(704×256)에서 생성됨 (`generate_depth_gt.py` → `scale_intrinsic_for_input`). 원본 1600×900 이미지 위에 겹치기 위해 (u,v)를 (1600/704, 900/256)배 확대해 정합함.


## 1. depth/ — depth GT 오버레이 (9장)

각 scene×카메라 1프레임: 좌=원본, 우=depth_gt scatter(색=depth, colorbar). 이동객체(rel_speed>1)가 있는 프레임을 선택.

| scene | camera | stem | depth_pts | movers(rel>1) | max_rel(m/s) | mover in cam | file |
|---|---|---|---|---|---|---|---|
| scen05 | cam_front | live_000255 | 2782 | 4 | 4.1 | 4 | `depth/scen05_cam_front_live_000255_depth.png` |
| scen05 | cam_front_left | live_000338 | 2661 | 3 | 4.1 | 3 | `depth/scen05_cam_front_left_live_000338_depth.png` |
| scen05 | cam_front_right | live_000481 | 3586 | 2 | 3.8 | 2 | `depth/scen05_cam_front_right_live_000481_depth.png` |
| scen77 | cam_front | live_000972 | 3397 | 5 | 2.2 | 5 | `depth/scen77_cam_front_live_000972_depth.png` |
| scen77 | cam_front_left | live_001330 | 4094 | 3 | 5.9 | 1 | `depth/scen77_cam_front_left_live_001330_depth.png` |
| scen77 | cam_front_right | live_000456 | 3572 | 3 | 3.8 | 3 | `depth/scen77_cam_front_right_live_000456_depth.png` |
| scen144 | cam_front | live_000211 | 2980 | 6 | 3.4 | 6 | `depth/scen144_cam_front_live_000211_depth.png` |
| scen144 | cam_front_left | live_000198 | 2576 | 6 | 2.7 | 6 | `depth/scen144_cam_front_left_live_000198_depth.png` |
| scen144 | cam_front_right | live_001390 | 4128 | 4 | 5.5 | 4 | `depth/scen144_cam_front_right_live_001390_depth.png` |

## 2. v2_vs_v3/ — v2→v3 보정 감사 (카메라 투영 + BEV) (40장)

v2=빨강, v3=초록, 흰 화살표=v2→v3 변위, 파랑=객체 속도, 노랑=ego 속도. 패널: 카메라투영 | BEV(50m 링) | BEV 확대(±6m).

| scene | frame | track | class | corr_dist(m) | rel/obj/ego (m/s) | obj_dt/ego_dt (ms) | buckets | cats | cam | file |
|---|---|---|---|---|---|---|---|---|---|---|
| scen77 | 468 | 6 | pedestrian | 0.909 | 3.96/0.00/3.96 | 81/81 | top20 | pedestrian | cam_front | `v2_vs_v3/scen77_f468_t6_cd0.909.png` |
| scen77 | 187 | 3 | pedestrian | 0.823 | 3.74/0.00/3.74 | 96/96 | top20 | pedestrian | cam_front | `v2_vs_v3/scen77_f187_t3_cd0.823.png` |
| scen144 | 205 | 5 | pedestrian | 0.798 | 2.91/0.00/2.91 | 100/100 | top20 | pedestrian | cam_front | `v2_vs_v3/scen144_f205_t5_cd0.798.png` |
| scen77 | 439 | 5 | pedestrian | 0.793 | 3.84/0.00/3.84 | 67/67 | top20 | pedestrian | cam_front_right | `v2_vs_v3/scen77_f439_t5_cd0.793.png` |
| scen05 | 466 | 8 | pedestrian | 0.767 | 3.78/0.00/3.78 | 73/72 | top20 | pedestrian | cam_front | `v2_vs_v3/scen05_f466_t8_cd0.767.png` |
| scen05 | 100 | 1 | vehicle | 0.765 | 6.56/3.05/3.50 | 119/118 | top20 | oncoming | cam_front | `v2_vs_v3/scen05_f100_t1_cd0.765.png` |
| scen05 | 474 | 8 | pedestrian | 0.757 | 3.82/0.00/3.82 | 78/77 | top20 | pedestrian | cam_front | `v2_vs_v3/scen05_f474_t8_cd0.757.png` |
| scen77 | 1198 | 11 | vehicle | 0.753 | 1.84/4.11/3.76 | 65/64 | top20 | same_dir_vehicle | cam_front | `v2_vs_v3/scen77_f1198_t11_cd0.753.png` |
| scen05 | 115 | 1 | vehicle | 0.751 | 7.56/3.87/3.69 | 98/98 | top20 | oncoming | cam_front_left | `v2_vs_v3/scen05_f115_t1_cd0.751.png` |
| scen77 | 1201 | 11 | vehicle | 0.750 | 1.59/4.11/3.82 | 63/62 | top20 | same_dir_vehicle | cam_front | `v2_vs_v3/scen77_f1201_t11_cd0.750.png` |
| scen77 | 178 | 3 | pedestrian | 0.730 | 3.87/0.00/3.87 | 70/70 | top20 | pedestrian | cam_front_right | `v2_vs_v3/scen77_f178_t3_cd0.730.png` |
| scen05 | 118 | 1 | vehicle | 0.716 | 7.55/3.86/3.69 | 93/93 | top20 | oncoming | cam_front_left | `v2_vs_v3/scen05_f118_t1_cd0.716.png` |
| scen77 | 463 | 6 | pedestrian | 0.716 | 3.85/0.00/3.85 | 66/66 | top20 | pedestrian | cam_front | `v2_vs_v3/scen77_f463_t6_cd0.716.png` |
| scen77 | 187 | 0 | vehicle | 0.682 | 3.09/4.16/3.74 | 96/96 | top20 | - | cam_front_right | `v2_vs_v3/scen77_f187_t0_cd0.682.png` |
| scen05 | 465 | 8 | pedestrian | 0.653 | 3.80/0.00/3.80 | 39/59 | top20 | pedestrian | cam_front | `v2_vs_v3/scen05_f465_t8_cd0.653.png` |
| scen05 | 135 | 2 | vehicle | 0.650 | 6.01/2.86/3.84 | 108/107 | top20 | oncoming | cam_front | `v2_vs_v3/scen05_f135_t2_cd0.650.png` |
| scen05 | 449 | 7 | pedestrian | 0.632 | 3.96/0.00/3.96 | 85/85 | top20 | pedestrian | cam_front_right | `v2_vs_v3/scen05_f449_t7_cd0.632.png` |
| scen77 | 175 | 3 | pedestrian | 0.615 | 3.82/0.00/3.82 | 61/61 | top20 | pedestrian | cam_front_right | `v2_vs_v3/scen77_f175_t3_cd0.615.png` |
| scen77 | 150 | 2 | vehicle | 0.612 | 8.13/4.11/4.02 | 76/76 | top20 | oncoming | cam_front_left | `v2_vs_v3/scen77_f150_t2_cd0.612.png` |
| scen05 | 135 | 0 | vehicle | 0.610 | 5.65/4.08/3.84 | 108/107 | top20 | - | cam_front_right | `v2_vs_v3/scen05_f135_t0_cd0.610.png` |
| scen05 | 128 | 0 | vehicle | 0.400 | 5.67/4.09/3.82 | 75/75 | mid_0.2_0.4 | - | cam_front_right | `v2_vs_v3/scen05_f128_t0_cd0.400.png` |
| scen05 | 472 | 8 | pedestrian | 0.400 | 3.84/0.00/3.84 | 38/37 | mid_0.2_0.4 | pedestrian | cam_front | `v2_vs_v3/scen05_f472_t8_cd0.400.png` |
| scen77 | 86 | 1 | vehicle | 0.399 | 5.00/1.72/3.29 | 76/76 | mid_0.2_0.4 | - | cam_front | `v2_vs_v3/scen77_f86_t1_cd0.399.png` |
| scen77 | 1226 | 11 | vehicle | 0.399 | 0.33/4.13/3.91 | 44/86 | mid_0.2_0.4 | stationary,same_dir_vehicle | cam_front | `v2_vs_v3/scen77_f1226_t11_cd0.399.png` |
| scen77 | 555 | 6 | pedestrian | 0.399 | 4.15/0.00/4.15 | 97/97 | mid_0.2_0.4 | pedestrian | cam_front | `v2_vs_v3/scen77_f555_t6_cd0.399.png` |
| scen77 | 1205 | 11 | vehicle | 0.398 | 1.28/4.11/3.72 | 28/42 | mid_0.2_0.4 | same_dir_vehicle | cam_front | `v2_vs_v3/scen77_f1205_t11_cd0.398.png` |
| scen144 | 286 | 4 | pedestrian | 0.397 | 3.65/0.00/3.65 | 102/102 | mid_0.2_0.4 | pedestrian | cam_front_left | `v2_vs_v3/scen144_f286_t4_cd0.397.png` |
| scen05 | 119 | 2 | vehicle | 0.397 | 5.74/2.13/3.75 | 74/74 | mid_0.2_0.4 | - | cam_front | `v2_vs_v3/scen05_f119_t2_cd0.397.png` |
| scen05 | 119 | 0 | vehicle | 0.395 | 5.60/4.10/3.75 | 74/74 | mid_0.2_0.4 | - | cam_front_right | `v2_vs_v3/scen05_f119_t0_cd0.395.png` |
| scen144 | 540 | 6 | pedestrian | 0.395 | 4.10/0.00/4.10 | 102/102 | mid_0.2_0.4 | pedestrian | cam_front | `v2_vs_v3/scen144_f540_t6_cd0.395.png` |
| scen05 | 0 | 0 | vehicle | 0.000 | 0.00/0.00/0.00 | 55/55 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen05_f0_t0_cd0.000.png` |
| scen05 | 0 | 1 | vehicle | 0.000 | 0.00/0.00/0.00 | 55/55 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen05_f0_t1_cd0.000.png` |
| scen05 | 0 | 2 | vehicle | 0.000 | 0.00/0.00/0.00 | 55/55 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen05_f0_t2_cd0.000.png` |
| scen77 | 0 | 1 | vehicle | 0.000 | 0.00/0.00/0.00 | 50/49 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen77_f0_t1_cd0.000.png` |
| scen77 | 0 | 2 | vehicle | 0.000 | 0.00/0.00/0.00 | 50/49 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen77_f0_t2_cd0.000.png` |
| scen77 | 742 | 7 | vehicle | 0.000 | 0.19/1.44/1.25 | -6/-7 | small_lt0.05 | stationary,same_dir_vehicle | cam_front | `v2_vs_v3/scen77_f742_t7_cd0.000.png` |
| scen77 | 1008 | 9 | pedestrian | 0.000 | 0.04/0.00/0.04 | -12/-13 | small_lt0.05 | pedestrian,stationary | cam_front | `v2_vs_v3/scen77_f1008_t9_cd0.000.png` |
| scen144 | 0 | 0 | vehicle | 0.000 | 0.00/0.00/0.00 | 27/27 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen144_f0_t0_cd0.000.png` |
| scen144 | 0 | 1 | vehicle | 0.000 | 0.00/0.00/0.00 | 27/27 | small_lt0.05 | stationary | cam_front | `v2_vs_v3/scen144_f0_t1_cd0.000.png` |
| scen144 | 688 | 9 | pedestrian | 0.000 | 0.00/0.00/0.00 | 7/7 | small_lt0.05 | pedestrian,stationary | cam_front | `v2_vs_v3/scen144_f688_t9_cd0.000.png` |

## 3. boundary_50m/ — 50m 경계 감사 (13장)

`pretrain_verify/g_membership_audit.json`의 13개 경계교차 박스(12 v2_only + 1 v3_only). 50m 링 기준 v2 중심(빨강)·v3 중심(초록)과 radial 값 표기.

| # | scene | frame | track | kind | v2_radial | v2_side | v3_radial | v3_side | file |
|---|---|---|---|---|---|---|---|---|---|
| 1 | scen05 | 65 | 2 | v2_only | 49.980 | inside | 50.225 | OUTSIDE | `boundary_50m/01_scen05_f65_t2_v2_only.png` |
| 2 | scen05 | 191 | 3 | v2_only | 49.867 | inside | 50.114 | OUTSIDE | `boundary_50m/02_scen05_f191_t3_v2_only.png` |
| 3 | scen05 | 227 | 4 | v2_only | 49.626 | inside | 50.029 | OUTSIDE | `boundary_50m/03_scen05_f227_t4_v2_only.png` |
| 4 | scen05 | 280 | 6 | v2_only | 49.972 | inside | 50.272 | OUTSIDE | `boundary_50m/04_scen05_f280_t6_v2_only.png` |
| 5 | scen144 | 243 | 5 | v2_only | 49.931 | inside | 50.073 | OUTSIDE | `boundary_50m/05_scen144_f243_t5_v2_only.png` |
| 6 | scen144 | 543 | 7 | v2_only | 49.922 | inside | 50.010 | OUTSIDE | `boundary_50m/06_scen144_f543_t7_v2_only.png` |
| 7 | scen144 | 931 | 10 | v2_only | 49.974 | inside | 50.060 | OUTSIDE | `boundary_50m/07_scen144_f931_t10_v2_only.png` |
| 8 | scen144 | 1013 | 14 | v2_only | 49.764 | inside | 50.019 | OUTSIDE | `boundary_50m/08_scen144_f1013_t14_v2_only.png` |
| 9 | scen77 | 675 | 8 | v2_only | 49.840 | inside | 50.004 | OUTSIDE | `boundary_50m/09_scen77_f675_t8_v2_only.png` |
| 10 | scen77 | 934 | 10 | v2_only | 49.937 | inside | 50.178 | OUTSIDE | `boundary_50m/10_scen77_f934_t10_v2_only.png` |
| 11 | scen77 | 1333 | 11 | v2_only | 49.967 | inside | 50.027 | OUTSIDE | `boundary_50m/11_scen77_f1333_t11_v2_only.png` |
| 12 | scen77 | 1337 | 13 | v2_only | 49.993 | inside | 50.037 | OUTSIDE | `boundary_50m/12_scen77_f1337_t13_v2_only.png` |
| 13 | scen144 | 924 | 10 | v3_only | 50.012 | OUTSIDE | 49.995 | inside | `boundary_50m/13_scen144_f924_t10_v3_only.png` |

## 4. track_temporal/ — 트랙 시계열 감사 (9장)

scene별 mean corr_dist 상위 3개 트랙. 패널: v2/v3 x | v2/v3 y | corr_dist | rel_speed. 빨간 점선=플래그된 점프(>25m/s & >3m). reversals는 위치/보정 진동 지표.

| scene | track | rank | n_frames | mean_corr | median_corr | flagged_jumps | rev x/y/corr | file |
|---|---|---|---|---|---|---|---|---|
| scen05 | 8 | 1 | 100 | 0.223 | 0.172 | none | 1/0/46 | `track_temporal/scen05_t8_rank1_mcd0.223.png` |
| scen05 | 3 | 2 | 140 | 0.206 | 0.196 | none | 0/0/58 | `track_temporal/scen05_t3_rank2_mcd0.206.png` |
| scen05 | 1 | 3 | 126 | 0.205 | 0.167 | none | 0/0/44 | `track_temporal/scen05_t1_rank3_mcd0.205.png` |
| scen77 | 3 | 1 | 92 | 0.201 | 0.169 | none | 1/0/50 | `track_temporal/scen77_t3_rank1_mcd0.201.png` |
| scen77 | 11 | 2 | 165 | 0.182 | 0.146 | none | 1/1/74 | `track_temporal/scen77_t11_rank2_mcd0.182.png` |
| scen77 | 5 | 3 | 102 | 0.161 | 0.130 | none | 1/0/54 | `track_temporal/scen77_t5_rank3_mcd0.161.png` |
| scen144 | 5 | 1 | 160 | 0.204 | 0.188 | none | 1/1/66 | `track_temporal/scen144_t5_rank1_mcd0.204.png` |
| scen144 | 3 | 2 | 92 | 0.202 | 0.182 | none | 1/1/36 | `track_temporal/scen144_t3_rank2_mcd0.202.png` |
| scen144 | 4 | 3 | 117 | 0.199 | 0.192 | none | 1/0/48 | `track_temporal/scen144_t4_rank3_mcd0.199.png` |

## 관찰 포인트 (사용자 확인 필요)

아래는 **사람이 눈으로 직접 확인**해야 하는 항목입니다. 시각화는 수치 검증을 **대체하지 않습니다**.
본 산출물은 어떤 항목에 대해서도 **PASS를 선언하지 않습니다** — 관찰 목록일 뿐입니다.

### depth (1)
- depth 오버레이(우측)가 정적 구조물(도로 경계/차선/건물 외벽)과 픽셀 단위로 정합하는지. 근거리=보라, 원거리=빨강 순서가 지형과 맞는지.
- 이동 객체(보행자/차량) 표면에 depth 포인트가 실제로 얹히는지, 아니면 배경으로 새는지.
- (주의) depth_gt는 704×256 학습 좌표계로 생성되어 있어, 여기서는 (1600/704, 900/256)배로 확대해 겹쳤습니다. 확대 정합이 맞는지 육안 확인 필요.

### v2_vs_v3 (2)
- v3(초록) 박스가 이동 객체를 v2(빨강)보다 **이동/보정 방향으로** 옮겼는지, 그리고 그 방향이 파랑(객체 속도)·노랑(ego 속도)과 물리적으로 일관적인지.
- 흰 화살표(v2→v3 변위) 크기가 corr_dist 및 obj_dt/ego_dt(latency)와 상식적으로 비례하는지.
- **정지 객체(obj_speed≈0)인데 corr_dist가 큰 사례**(예: `scen77_f439_t5`, 보행자 obj_speed=0 이지만 0.79m 보정)가 ego 재투영으로 설명되는지, 과보정은 아닌지.
- 카메라 투영에서 초록/빨강 큐보이드가 실제 객체 외곽을 감싸는지(측면 카메라 포함), 심하게 어긋나면 투영/외부파라미터 문제일 수 있음.

### boundary_50m (3)
- 50m 경계 박스가 링 안/밖으로 갈리는 것이 맞는지: 12개 v2_only는 v2가 안(≤50)·v3가 밖(>50), 1개 v3_only는 반대인지.
- v2·v3 중심 차이가 sub-decimeter 수준인데 그 미세 이동만으로 필터 membership이 바뀌는 경계 민감성이 학습에 의미가 있는지.

### track_temporal (4)
- corr가 큰 track에 **물리적으로 불가능한 순간이동/진동**이 없는지: 위치 패널(x,y)의 v2/v3 궤적이 매끄러운지(현재 플래그된 점프 0건).
- corr_dist 패널의 **프레임 간 진동**(reversals corr 36~74)이 상대속도(rel_speed)와 연동되는 정상적 latency 효과인지, 아니면 라벨 노이즈인지.
- rel_speed 곡선이 궤적(멀어짐/다가옴)과 일관적인지.

### 공통
- 본 감사는 시각적 정합성만 확인합니다. 정량 검증(수치 기반 GT 정확도/보정 통계)은 별도 스크립트로 수행해야 하며, 여기서는 **어떤 PASS/합격 판정도 내리지 않습니다.**

# MORAI GT 라벨링/동기화 검증 현황 인계서

작성 기준일: 2026-07-17  
확인 범위: `/home/autonav/projects/morai-3d-detection` 및 `/home/autonav` 아래에 현재 존재하는 코드, 로그, 데이터, 이미지, NPZ, bag 파일  
작성 원칙: 이 문서는 기존 산출물만 열어 정리했다. 코드 수정, 데이터 생성, 재처리 결과는 포함하지 않았다.

## 요약

1. **검증 목적:** source timestamp 기반 시간동기화, ego 좌표계 GT 변환, 카메라/라이다 투영 정합, temporal track/ego pose, GT box 내부 라이다 점 수, 동적 객체 간 z-buffer 가림 처리가 코드와 기존 검증 파이프라인의 목적이다.
2. **완료된 범위:** 원본 geometry bag의 epoch-safe sync, 차량/보행자 3D box 투영, 라이다-카메라 투영, scen02의 191프레임 처리 및 sidecar 생성, 10분 live sync 무결성은 근거 산출물이 현재 존재한다.
3. **미검증/부분 범위:** 실제 동적 객체가 겹치는 장면의 visibility 판정, 정적 구조물 가림, visibility weight의 실제 loss 반영, 현재 부재한 scen01~150 원본 처리 데이터의 재검증은 완료 근거가 없다.
4. **기존 데이터로 가능한 범위:** scen02, live_sync_test/scen01, geom_verify/out 및 과거 로그로 동기화·투영·라이다 점 수·sidecar 형식·기존 품질 통계를 확인할 수 있다.
5. **새 bag의 역할:** 4개 overlap bag은 기존 scen02에 없는 실제 vehicle-vehicle, vehicle-pedestrian, 다중 겹침에서 z-buffer visibility와 그 장면의 sync/라이다 점 수를 검증하기 위한 원본이며, 현재는 raw bag만 존재한다.

| 검증항목 | 상태 | 근거 산출물 위치 | 기존데이터로 가능? | 새 bag 필요? |
|---|---|---|---|---|
| source timestamp/epoch-safe 동기화 | 완료 | `geom_verify/out/replay_metrics.txt:2-49`, `visibility_test_dataset/scen02/sync_log.csv:1-219` | 예 | 아니오 |
| 10분 라이브 저장 동기화·파일 연속성 | 완료 | `/home/autonav/live_sync_test/scen01/sync_log.csv:1-5655`와 동일 scene의 5,643개씩 존재하는 `ego_pose/`, `labels_3d/`, `lidar/`, 각 camera 이미지 | 예 | 아니오 |
| ego 좌표 변환·1.28 m 기준점 보정 | 완료 | `morai_3d_live.py:171-214`, `geom_verify/out/step6_trace.txt:4-77` | 예 | 아니오 |
| 차량/보행자 3D GT box 카메라 투영 | 완료 | `geom_verify/out/final_bag_summary.txt:60-72`, `final_front_contact_sheet.jpg`, `final_ped_front_contact.jpg`, scen02 `_geom_debug_overlay/` | 예 | 아니오 |
| 라이다-카메라 투영 정합 | 완료 | `geom_verify/out/final_bag_summary.txt:75-88`, `geom_verify/out/step6_trace.txt:72-77` | 예 | 아니오 |
| temporal track/ego pose 라벨 생성 | 부분 | scen02 `scene_info.json:1`, `preprocess_scen61_150.log:93-96` | scen02는 예, 과거 전체 scene은 제한적 | 아니오 |
| GT oriented box 내부 라이다 점 수 | 완료 | `generate_occlusion_gt.py:53-105`, scen02 `occlusion/*.npz` 191개, `_tmp_occlusion_gen.log:1484-1495` | 예 | 겹침 장면별 검증에는 예 |
| 동적 GT box z-buffer 알고리즘 | 부분 | `generate_visibility_gt.py:247-342`, scen02 `_visibility_contact/_synthetic_occlusion.jpg` | synthetic 및 비겹침만 가능 | 실제 겹침 검증에는 예 |
| 실제 동적 객체 간 가림 판정 | 미검증 | scen02 `visibility/*.npz`에는 projected camera별 `visible_ratio < 1` 사례가 없음; 새 bag의 처리 산출물은 없음 | 아니오 | 예 |
| 벽·건물 등 정적 가림 판정 | 미검증 | 미지원 범위가 `generate_visibility_gt.py:4-18`에 명시됨 | 아니오 | 현재 4개 bag만으로 완료된다는 근거 없음 |
| visibility filtering/weighting loader | 부분 | `morai_dataset.py:426-451`, `morai_dataset.py:572-620`, `morai_dataset.py:735-793` | loader 출력 확인은 가능 | 아니오 |
| sparse depth GT 데이터셋 | 부분 | 구현은 `generate_depth_gt.py:4-21`, `generate_depth_gt.py:73-120`; scen02에는 `depth_gt/` 없음 | 구현 확인만 가능 | 미확인 |
| 과거 scen01~150 처리 데이터 전체 재검증 | 미검증 | 현재 `scen150` 및 해당 dataset tree 없음; 과거 로그는 `_tmp_occlusion_gen.log:1484-1495`만 존재 | 로그 확인만 가능 | 아니오 |
| 새 overlap bag 4개의 처리 결과 | 미검증 | `/home/autonav/occlusion_bags/*.bag` 4개만 존재; 대응 processed scenario, sync log, NPZ, contact sheet 없음 | 아니오 | 해당 bag 자체가 필요 원본 |

## 1. 수집·처리 데이터의 검증 목적

### 1.1 시간동기화

- **확인됨:** `morai_sync.py`는 receive time이 아니라 ROS header source time으로 camera 기준 nearest sample을 선택하고, source clock rollback을 epoch로 분리하도록 정의한다 (`morai_sync.py:3-21`, `morai_sync.py:31-68`, `morai_sync.py:208-221`).
- **확인됨:** epoch jump, confirm hold, source monotonicity, topic별 허용 시간차 gate와 저장 여부 판정이 구현되어 있다 (`morai_sync.py:246-318`).
- **확인됨:** live 저장기는 동기화된 camera, lidar, ego, object를 동일 stem으로 저장하고 `sync_log.csv`를 기록한다 (`morai_3d_live.py:462-518`).

### 1.2 ego 좌표계 GT와 카메라 투영

- **확인됨:** world 좌표의 객체 중심을 ego 좌표로 변환하고 MORAI 기준점 보정 1.28 m를 적용하는 코드가 있다 (`morai_3d_live.py:171-214`).
- **확인됨:** 객체 크기, yaw, ROI, camera visibility를 계산해 GT CSV를 저장한다 (`morai_3d_live.py:721-826`).
- **확인됨:** 저장된 GT를 카메라 영상에 overlay하는 경로가 있다 (`morai_3d_live.py:543-590`).

### 1.3 temporal track와 ego pose

- **확인됨:** 기존 frame-local ID를 temporal track ID로 복구하고 ego/global transform을 기록하는 것이 전처리 목적에 명시되어 있다 (`preprocess_dataset.py:3-41`).
- **확인됨:** track segment 생성·병합·재할당과 `labels_3d_v2`, `scene_info.json` 출력 코드가 있다 (`preprocess_dataset.py:206-307`, `preprocess_dataset.py:381-433`).

### 1.4 GT box 내부 라이다 점 수

- **확인됨:** 라이다 점을 body 좌표로 변환한 뒤 GT의 oriented 3D box 내부 점을 세는 로직이 있다 (`generate_occlusion_gt.py:4-22`, `generate_occlusion_gt.py:41-105`).
- **확인됨:** 결과는 원본 label을 변경하지 않고 frame별 sidecar로 저장된다 (`generate_occlusion_gt.py:125-169`).

### 1.5 동적 겹침과 visibility

- **확인됨:** 모든 동적 GT box 표면을 camera별로 rasterize하고 depth가 가장 가까운 객체를 owner로 선택해 projected/visible pixel ratio를 계산한다 (`generate_visibility_gt.py:247-330`).
- **확인됨:** 라이다 점 수는 visibility 판정값이 아니라 보조 필드로만 결합된다 (`generate_visibility_gt.py:20-28`, `generate_visibility_gt.py:334-342`).
- **확인됨:** 이 구현은 동적 GT 객체 간 가림만 다루며 벽, 건물, 가드레일, 식생 등 static occluder는 다루지 않는다고 코드에 명시되어 있다 (`generate_visibility_gt.py:4-18`).

### 1.6 sparse depth GT

- **확인됨:** 라이다 점을 각 camera에 투영하고 pixel z-buffer로 sparse depth를 만드는 별도 구현이 있다 (`generate_depth_gt.py:4-21`, `generate_depth_gt.py:73-120`).
- **미확인:** 이 sparse depth가 visibility owner 판정에 배선되었다는 코드 또는 산출물 근거는 없다. `generate_visibility_gt.py:4-18`의 범위상 static depth는 현재 visibility 판정에 포함되지 않는다.

## 2. 완료된 검증

### 2.1 원본 bag geometry·동기화 검증

- **확인됨:** 사용한 bag은 `/home/autonav/geom_drive_all_20260710_180341.bag`으로 기록되어 있다 (`geom_verify/out/final_bag_summary.txt:2-4`, `geom_verify/out/replay_metrics.txt:2`).
- **확인됨:** bag integrity, drive validity, clock, transport, ego-object sync, multicamera sync, unique ID track, dynamic wireframe, LiDAR depth, side-camera geometry가 PASS로 기록되어 있다 (`geom_verify/out/final_bag_summary.txt:75-88`).
- **확인됨:** 차량 wireframe의 수치 오차는 median 0.428 px, p95 1.701 px로 기록되어 있고 front/left/right contact sheet 위치가 함께 기록되어 있다 (`geom_verify/out/final_bag_summary.txt:60-72`).
- **확인됨:** 실제 근거 이미지는 `geom_verify/out/final_front_contact_sheet.jpg`, `final_left_contact_sheet.jpg`, `final_right_contact_sheet.jpg`, `final_ped_front_contact.jpg`, `final_ped_left_contact.jpg`, `final_ped_right_contact.jpg`에 존재한다. `final_front_contact_sheet.jpg`와 `final_ped_front_contact.jpg`를 직접 열어 차량·보행자 box overlay가 영상에 표시된 것을 확인했다.
- **확인됨:** 4개 NPC의 body 위치, 1.28 m offset 적용 전후, 카메라 투영점과 라이다 투영 샘플이 trace에 남아 있다 (`geom_verify/out/step6_trace.txt:4-77`).

### 2.2 epoch-safe replay

- **확인됨:** replay 방식 A/B/C의 지표와 C 방식의 `saved=191`, `drop=27`, source regression 0, cross-epoch 결합 0이 기록되어 있다 (`geom_verify/out/replay_metrics.txt:3-17`).
- **확인됨:** ego/object/left/right/lidar의 selected source dt와 gate 판정이 기록되어 있다 (`geom_verify/out/replay_metrics.txt:19-22`).
- **확인됨:** epoch jump/confirm hold drop과 glitch 구간이 기록되어 있으며 전체 필수 조건이 PASS이다 (`geom_verify/out/replay_metrics.txt:24-43`).

### 2.3 처리 완료 scen02

- **확인됨:** `/home/autonav/visibility_test_dataset/scen02`에 `ego_pose/`, `labels_3d/`, `labels_3d_v2/`, `lidar/`, `occlusion/`, `visibility/`가 각각 191개, 3-camera image가 총 573개 존재한다.
- **확인됨:** `scene_info.json`은 `n_frames=191`, `n_tracks=10`과 dimension spread, position jump, velocity residual, segment 통계를 기록한다 (`/home/autonav/visibility_test_dataset/scen02/scene_info.json:1`).
- **확인됨:** `labels_3d_v2` 총 682 box와 대응하는 `occlusion/*.npz`, `visibility/*.npz` 191개가 존재한다. 기존 NPZ를 직접 읽은 집계에서 visibility level은 visible 681, invisible 1이며, camera에 projected된 1,033개 객체-camera 조합 중 `visible_ratio < 1`인 조합은 0개였다.
- **확인됨:** geometry overlay 30개가 `/home/autonav/visibility_test_dataset/scen02/_geom_debug_overlay/`에 있고, `live_000000.jpg`를 직접 열어 차량·보행자 box overlay를 확인했다.
- **확인됨:** visibility contact sheet 4개와 synthetic image 1개가 `/home/autonav/visibility_test_dataset/scen02/_visibility_contact/`에 있다. `_synthetic_occlusion.jpg`를 직접 열어 synthetic near box의 ratio 1.00, far box의 ratio 0.00 표시를 확인했다.

### 2.4 live sync soak 데이터

- **확인됨:** `/home/autonav/live_sync_test/scen01/sync_log.csv`에는 header 포함 5,655줄이 있으며 5,643 save, 11 drop 행이 존재한다 (`/home/autonav/live_sync_test/scen01/sync_log.csv:1-5655`).
- **확인됨:** 동일 scene에 `ego_pose`, `labels_3d`, `lidar`, 각 camera 이미지가 각각 5,643 frame stem에 대응해 존재한다.
- **확인됨:** 기존 sync log 직접 집계에서 source timestamp 역행 0, save epoch 집합 `{0}`, drop reason은 11건 모두 `sync_gap`이다.
- **확인됨:** 기존 sync log 직접 집계의 p95/max selected dt는 ego 4/50 ms, object 4/48 ms, left 4/35 ms, right 3/29 ms, lidar 61/75 ms이다 (`/home/autonav/live_sync_test/scen01/sync_log.csv:1-5655`).

### 2.5 과거 scen01~150 처리 기록

- **확인됨:** 과거 occlusion sidecar 생성 로그에는 150 scenes, 95,672 frames, 123,656 boxes, missing lidar 0, zero-point box 92,179개(74.5%)와 write 완료가 기록되어 있다 (`_tmp_occlusion_gen.log:1484-1495`).
- **확인됨:** scen61~150 temporal 전처리 로그는 총 58,186 frames, 399 tracks, max dimension spread 0.283, worst position jump p95 1.850, worst velocity residual p95 4.427 m/s를 기록한다 (`preprocess_scen61_150.log:93-96`).
- **미확인:** 현재 `/home/autonav` 아래에는 대응하는 scen01~150 처리 데이터 tree와 `scen150` 디렉터리가 없다. 따라서 과거 로그의 개별 frame/label/sidecar를 현재 원본과 대조하는 검증은 할 수 없다.

## 3. 미검증 또는 부분 검증

### 3.1 실제 동적 겹침

- **부분:** z-buffer rasterizer 구현과 synthetic 겹침 이미지는 존재한다 (`generate_visibility_gt.py:247-330`, `/home/autonav/visibility_test_dataset/scen02/_visibility_contact/_synthetic_occlusion.jpg`).
- **미검증:** scen02의 projected 객체-camera 조합에는 `visible_ratio < 1`인 실제 사례가 없다. 따라서 실제 vehicle-vehicle, vehicle-pedestrian, 3-layer overlap에서 앞 객체가 뒤 객체의 visible pixel을 제거하는 결과는 scen02로 검증되지 않았다.
- **미검증:** `/home/autonav/occlusion_bags`의 4개 bag에 대응하는 processed scenario, `sync_log.csv`, visibility NPZ, contact sheet는 없다.

### 3.2 정적 구조물 가림

- **미검증:** static occluder가 알고리즘 범위 밖임이 명시되어 있다 (`generate_visibility_gt.py:4-18`).
- **미확인:** depth/semantic/instance segmentation 또는 map mesh가 현재 visibility 판정에 입력된 근거는 없다.

### 3.3 visibility의 학습 loss 반영

- **확인됨:** visibility sidecar opt-in 설정, filter/weight 계산, sample 및 temporal collate 출력까지 구현되어 있다 (`morai_dataset.py:426-451`, `morai_dataset.py:572-620`, `morai_dataset.py:735-793`).
- **미확인:** 출력된 visibility weight가 실제 loss 계산에서 소비된다는 실행 로그, 학습 결과 또는 확인 가능한 연결 코드 근거는 없다.

### 3.4 temporal 품질의 과거 전체 데이터 판정

- **부분:** scen02의 품질 통계는 현재 존재한다 (`/home/autonav/visibility_test_dataset/scen02/scene_info.json:1`).
- **부분:** scen61~150 로그에서 dimension spread와 position jump 수치는 기록되어 있으나 velocity residual p95는 기록된 목표 0.5 m/s보다 큰 4.427 m/s이다 (`preprocess_scen61_150.log:93-96`).
- **미검증:** 현재 과거 scene 데이터가 없으므로 scen01~150 전체를 frame 단위로 재확인할 수 없다.

### 3.5 sparse depth GT

- **부분:** 생성 코드는 존재한다 (`generate_depth_gt.py:4-21`, `generate_depth_gt.py:73-120`).
- **미확인:** `generate_depth_scen61_150.log`는 존재하지만 현재 확인 환경에서 텍스트가 정상적으로 판독되지 않아 완료 수치와 판정을 근거로 사용할 수 없다.
- **확인됨:** 현재 scen02에는 `depth_gt/` 산출물 디렉터리가 없다.

## 4. 기존 처리 완료 데이터만으로 확인 가능한 항목

| 기존 산출물 | 지금 확인 가능한 사실 | 한계 |
|---|---|---|
| `visibility_test_dataset/scen02/sync_log.csv` | 원본 glitch 구간의 epoch 분리, save/drop, selected dt, monotonicity | 새로운 겹침 장면을 포함하지 않음 |
| scen02 `labels_3d/`, `ego_pose/`, `labels_3d_v2/`, `scene_info.json` | ego 변환 결과, frame/track 수, temporal 품질 통계, label schema | 과거 scen01~150 전체를 대신하지 않음 |
| scen02 `lidar/` + `occlusion/*.npz` | 각 oriented GT box 내부 라이다 점 수와 682 box 대응 관계 | 라이다 점 수만으로 카메라 가림의 원인을 확정할 수 없음 |
| scen02 `visibility/*.npz` | camera별 projected/visible pixel, best camera, level, auxiliary lidar count | 실제 projected 동적 겹침 사례가 없음 |
| scen02 `_geom_debug_overlay/`, `_visibility_contact/` | 실제 scene의 box overlay와 synthetic z-buffer 표시 | synthetic은 실제 센서 장면 검증이 아님 |
| `live_sync_test/scen01/sync_log.csv`와 저장 파일 | 10분 live save/drop, source monotonicity, dt, stem별 파일 대응 | ego가 주행·객체 겹침을 포함했다는 근거 없음 |
| `geom_verify/out/` | 원본 bag의 차량/보행자 투영, multicamera, 라이다 depth, epoch-safe replay 결과 | 새 overlap bag의 결과가 아님 |
| `_tmp_occlusion_gen.log`, `preprocess_scen61_150.log` | 과거 150-scene occlusion 생성 총계와 scen61~150 temporal 총계 | 현재 원본 scene 디렉터리가 없어 개별 산출물 대조 불가 |
| `_tmp_verify_out/` | scen08/18/23/32/41/51의 과거 camera/BEV 검증 이미지가 현재 존재 | 대응 원본 scene과 텍스트 판정이 없어 전체 데이터 완료 근거로 사용할 수 없음 |

## 5. 새 overlap bag 4개의 필요 범위

현재 존재하는 raw bag은 다음 4개다.

1. `/home/autonav/occlusion_bags/01_vehicle_vehicle_overlap.bag`
2. `/home/autonav/occlusion_bags/02_vehicle_pedestrian_overlap.bag`
3. `/home/autonav/occlusion_bags/02_ve2hicle_pedestrian_overlap.bag` (`02-2`에 해당하는 파일명으로 확인됨)
4. `/home/autonav/occlusion_bags/03_three_layer_overlap.bag`

### 기존 데이터로 확인되는 것

- **확인됨:** 기준 geometry bag에서 좌표 변환, 1.28 m offset, 차량/보행자 camera projection, multicamera geometry, 라이다-camera projection이 검증되었다 (`geom_verify/out/final_bag_summary.txt:60-88`).
- **확인됨:** scen02에서 epoch-safe sync 결과, 191프레임 label/pose/lidar/image 대응, occlusion/visibility sidecar 형식과 GT box 내부 라이다 점 수가 확인된다 (`geom_verify/out/replay_metrics.txt:8-49`, `/home/autonav/visibility_test_dataset/scen02/scene_info.json:1`).
- **확인됨:** synthetic 두 box에 대한 z-buffer owner 분리는 이미지로 존재한다 (`/home/autonav/visibility_test_dataset/scen02/_visibility_contact/_synthetic_occlusion.jpg`).

### 새 bag이 있어야 확인되는 것

- **미검증:** 실제 영상의 vehicle-vehicle overlap에서 앞 vehicle과 뒤 vehicle의 `visible_px`, `visible_ratio`, level이 거리 순서에 맞게 분리되는지.
- **미검증:** 실제 vehicle-pedestrian overlap에서 크기와 class가 다른 box 표면의 z-buffer owner가 맞는지.
- **미검증:** 3-layer overlap에서 중간 객체가 앞 객체에는 가려지고 뒤 객체는 다시 가리는 다중 depth 순서가 맞는지.
- **미검증:** 위 실제 overlap 장면에서 camera별 visibility와 GT box 내부 `num_lidar_pts`가 같은 synchronized frame/stem으로 생성되는지.
- **미검증:** source clock rollback이 포함된 overlap bag이 있다면 epoch-safe drop/hold가 overlap frame을 다른 epoch 데이터와 결합하지 않는지. 각 bag의 처리 `sync_log.csv`가 없으므로 현재 판정할 수 없다.

### 경계

- 새 bag은 이미 완료 근거가 있는 기준 좌표 변환·기본 투영·원본 geometry bag 동기화를 입증하는 데 필수인 데이터가 아니다.
- 새 bag은 기존 scen02에 존재하지 않는 **실제 동적 겹침 사례**의 visibility 결과를 입증하는 데 필요한 원본이다.
- 현재 4개 bag은 raw 파일로만 존재하므로, 이 문서에서 해당 겹침 검증 상태는 모두 **미검증**이다.

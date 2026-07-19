# morai-3d-detection 현재 상태 리포트

- 조사 기준일: 2026-07-17
- 조사 범위: 현재 working tree(기존 수정 파일과 미추적 파일 포함). 코드 실행이나 수정 없이 소스·메타데이터·기존 로그만 열어 확인했다.
- 판정 기준: `확인됨`은 요청 기능과 실행 경로가 확인된 경우, `부분`은 핵심은 있으나 기본 비활성·미배선·검증 범위 제한이 있는 경우, `미구현`은 저장소 전체 `*.py` 검색에서도 대응 구현이 없는 경우, `미확인`은 열어 확인할 근거가 없는 경우다.
- 주의: 현재 저장소 경로에는 `dataset/` 디렉터리가 존재하지 않았다. 따라서 실제 `.csv/.npy/.npz/.jpg` 산출물 자체의 현존 여부는 **미확인**이며, 아래 데이터 규모·실행 완료 판단은 열린 `dataset_split_v11.json`과 기존 실행 로그에 한정한다. 디렉터리 부재는 인용할 파일 라인이 없으므로 추측으로 보완하지 않았다.

## 요약

### 최우선 3문항

1. **가림(occlusion) 필터링이 구현되어 있는가? — 부분적으로 예.**
   - 현재 학습의 기본 활성 필터는 **ray-casting이 아니라 3D 박스 내부 LiDAR 포인트 수**다. `OCCLUSION_MIN_PTS=1`이 기본이며 1점 미만 GT를 train/val에서 제외한다(`train.py:2108-2112`, `train.py:2280-2285`, `morai_dataset.py:601-611`).
   - 3카메라의 동적 객체 상호 가림을 owner z-buffer로 계산하는 `best_visible_ratio` 파이프라인도 존재하지만(`generate_visibility_gt.py:247-330`), 기본 임계값은 0이라 비활성이다(`morai_dataset.py:435-449`). 정적 구조물 가림은 이 방식으로 해결하지 못한다고 명시돼 있다(`generate_visibility_gt.py:9-18`).
   - `box_visible_in_any_camera()`는 투영 박스가 화면과 교차하는지만 보며(`morai_dataset.py:176-210`), 그 자체는 가림 판정이 아니다.

2. **시간 동기화는 어느 수준까지 완성됐는가? — 핵심 경로는 구현·회귀검증 완료, 예외 처리는 일부 남음.**
   - `cam_front` source header를 기준으로 Ego/Object는 50 ms, 좌·우 카메라와 LiDAR는 100 ms 이내 최근접 메시지를 선택한다(`morai_sync.py:30-48`, `morai_sync.py:207-219`). 150 ms settle, epoch jump 감지, hold-3, timestamp 단조 증가, 실패 프레임 drop과 sidecar 기록까지 구현됐다(`morai_sync.py:58-66`, `morai_sync.py:182-201`, `morai_sync.py:245-314`, `morai_3d_live.py:440-460`).
   - 동일 bag A/B/C 회귀에서 C(epoch-safe)는 timestamp 역행 0, cross-epoch 결합 0, 모든 허용 오차 PASS였다(`geom_verify/out/replay_metrics.txt:8-16`, `geom_verify/out/replay_metrics.txt:33-43`).
   - 단, 라이브 CLI의 `max_sync_gap` 값은 보관·출력만 되고 실제 synchronizer 생성에는 전달되지 않는다(`morai_3d_live.py:275-278`, `morai_3d_live.py:299-306`, `morai_3d_live.py:362-367`). 또한 ref가 아닌 토픽의 `source_ts=None`을 `_nearest()`가 건너뛰는 방어는 없어 해당 예외는 부분 상태다(`morai_sync.py:79-88`, `morai_sync.py:162-180`).

3. **occupancy 관련 코드가 전무한가? — 전무하지는 않지만 학습 head/loss는 없다.**
   - 검출 박스를 BEV 4채널 costmap으로 rasterize하며 채널 0을 occupancy score로 쓰는 후처리는 존재한다(`costmap.py:40-88`). 추론 코드도 이를 호출해 `.npy/.png`를 저장한다(`inference.py:319-321`).
   - 그러나 `AutoNavModel`의 활성 head는 detection/quality와 학습용 depth뿐이며(`train.py:721-768`), occupancy head나 occupancy loss는 저장소 전체 `*.py` 검색에서 확인되지 않았다. 따라서 **occupancy 학습 구조는 미구현**이다.

| 항목 | 상태 | 핵심 근거 파일 |
|---|---|---|
| A. 데이터 수집·동기화 | 부분 | `morai_sync.py:30-48, 182-219, 245-314`; `morai_3d_live.py:299-324, 398-518`; `geom_verify/out/replay_metrics.txt:33-43` |
| B. GT 라벨링 | 부분 | `morai_3d_live.py:104-136, 661-824`; `preprocess_dataset.py:31-37, 367-430`; `generate_depth_gt.py:73-120`; `generate_occlusion_gt.py:74-105`; `generate_visibility_gt.py:247-330` |
| C. 라이다 처리 | 부분 | `camera_configs.py:93-105`; `verify_lidar_camera_overlay.py:89-110, 230-238`; `geom_verify/step_final_all.py:131-149` |
| D. 학습 구조 | 부분 | `decoder.py:6-66`; `train.py:459-508, 596-768, 1464-1504`; `loss_calculator.py:6-14, 152-294` |
| E. 평가 | 부분 | `eval_distance.py:58-158`; `train.py:1036-1045, 1384-1461, 1706-1710, 1788-1802, 1858-1882`; `eval_distance_log.txt:21-32` |

## A. 데이터 수집·동기화

### A-1. 같은 시각으로 맞추는 방식 — 확인됨

- 대상은 3카메라(`/cam_front`, `/cam_front_left`, `/cam_front_right`), `/lidar3D`, `/Ego_topic`, `/Object_topic`이다(`morai_3d_live.py:35-43`, `morai_3d_live.py:329-358`).
- `EpochGlitchSynchronizer(mode="epoch")`가 `cam_front`를 reference로 삼는다(`morai_3d_live.py:298-306`). 각 콜백은 source-header와 receive time을 함께 push하며, 콜백 간 상태·저장을 `RLock`으로 직렬화한다(`morai_3d_live.py:382-438`).
- reference 프레임을 즉시 확정하지 않고 150 ms 기다린 뒤, 각 필수 토픽의 source-header 최근접 항목을 고른다(`morai_sync.py:61-66`, `morai_sync.py:177-195`, `morai_sync.py:207-219`). 저장 timestamp는 `cam_front` source header다(`morai_3d_live.py:462-483`).
- epoch jump 0.5 s 감지 시 버퍼를 정리하고 3개의 연속 단조 reference를 확인하며, 이전 저장 timestamp 이하의 reference는 버린다(`morai_sync.py:53-59`, `morai_sync.py:253-298`).
- 기존 검증 결과에서 source-header 최근접 오차는 Ego/Object 최대 11 ms, left 54 ms, right 95 ms, LiDAR 71 ms였다(`geom_verify/out/replay_metrics.txt:19-22`). 최종 검증 리포트도 Ego/Object sync와 multicam sync를 PASS로 기록한다(`geom_verify/out/final_bag_summary.txt:75-88`).

### A-2. 동기화 실패/프레임 드롭 — 부분

- reference source header 부재, epoch jump/confirm hold, timestamp 역행, 토픽별 허용 오차 초과를 각각 `drop`으로 반환한다(`morai_sync.py:245-306`). 라이브 수집기는 drop 이유를 집계하고 `sync_log.csv` sidecar에 즉시 기록한다(`morai_3d_live.py:319-324`, `morai_3d_live.py:440-460`, `morai_3d_live.py:516-518`). 종료 시 pending reference를 flush하고 saved/drop 요약을 출력한다(`morai_3d_live.py:593-630`).
- 회귀 로그에서 C 모드는 191프레임 저장, 27프레임 drop이며 이유는 `epoch_jump` 21, `epoch_confirm_hold` 6이었다(`geom_verify/out/replay_metrics.txt:8-9`, `geom_verify/out/replay_metrics.txt:24-30`). 이는 의도된 글리치 구간 폐기다.
- 미완 지점: `max_sync_gap` CLI가 실제 `TOPIC_TOL`에 연결되지 않고, non-reference 토픽의 `None` header는 `_nearest()` 산술 전에 걸러지지 않는다. 따라서 정상 MORAI header가 존재하는 검증 경로는 완료됐지만 비정상 header 예외까지 완결됐다고 볼 수 없어 `부분`이다.
- 별도의 레거시 bag 생성기 `morai_3d_label_generator.py`는 header 우선/없으면 bag time으로 fallback하고, cam_front 기준 Ego/Object만 `max_sync_gap` 최근접 매칭한다(`morai_3d_label_generator.py:57-91`, `morai_3d_label_generator.py:199-210`). 현재 라이브 6토픽 epoch-safe 경로와 동일한 수준은 아니다.

## B. GT 라벨링 현황

### B-1. 확인된 GT/보조 GT 종류

| 종류 | 상태 | 생성 내용과 파일 | 근거 |
|---|---|---|---|
| 원본 동적 3D GT | 확인됨 | `labels_3d/<stem>.csv`: vehicle/pedestrian class, ego-frame 3D box `(x,y,z,w,l,h,sin_yaw,cos_yaw)`, `(vx,vy,vz)`, object source/index | `morai_3d_live.py:104-126`, `morai_3d_live.py:661-824` |
| Ego pose/time | 확인됨 | `ego_pose/<stem>.csv`: frame/time, ego xyz, heading/yaw | `morai_3d_live.py:128-136`, `morai_3d_live.py:644-659` |
| 보정 3D GT + track ID | 확인됨 | `labels_3d_v2/<stem>.csv`: corrected center-z/yaw/velocity, `track_id`, global 위치·yaw, velocity source. `scene_info.json`: frame별 `T_ego2global` | `preprocess_dataset.py:31-37`, `preprocess_dataset.py:206-307`, `preprocess_dataset.py:367-430` |
| velocity GT | 확인됨 | v2의 `vx_ego, vy_ego, vz`; raw가 사실상 0인 이동 트랙은 위치 중심차분으로 보완하고 `vel_source=motion` 기록 | `preprocess_dataset.py:121-145`, `preprocess_dataset.py:149-194`, `preprocess_dataset.py:399-418` |
| sparse depth GT | 확인됨 | `depth_gt/{cam_front,cam_front_left,cam_front_right}/<stem>.npy`, 각 행 `[u,v,depth]`, 동일 픽셀 최근접 점만 유지 | `generate_depth_gt.py:18-20`, `generate_depth_gt.py:73-120`, `generate_depth_gt.py:189-238` |
| LiDAR 가시성 보조 GT | 확인됨 | `occlusion/<stem>.npy`: `[track_id,num_lidar_pts,radial_dist]` | `generate_occlusion_gt.py:15-21`, `generate_occlusion_gt.py:74-105`, `generate_occlusion_gt.py:125-169` |
| 카메라 가림/절단 보조 GT | 부분 | `visibility/<stem>.npz`: 카메라별 projected/visible pixel과 ratio, best camera/ratio, truncation, visibility level, num_lidar_pts. 생성·로딩은 있으나 기본 비활성이고 현재 산출물 자체는 미확인 | `generate_visibility_gt.py:30-43`, `generate_visibility_gt.py:348-397`, `morai_dataset.py:435-449, 582-592` |
| 정적 맵 polyline GT | 부분 | 레거시 `labels_static/<stem>.txt`: lane/crosswalk/road-boundary를 ego-frame 20점 polyline으로 생성. 현재 `AutoNavModel` 학습에는 연결되지 않음 | `mgeo_to_static_labels.py:49-99`, `mgeo_to_static_labels.py:163-241`; `train.py:721-768` |

- 메타데이터에는 scen01~150, 총 95,672프레임, 123,656박스(vehicle 70,040 / pedestrian 53,616)가 기록돼 있다(`dataset_split_v11.json:4-154`, `dataset_split_v11.json:310-334`). 이는 열린 메타데이터의 기록이며 현재 `dataset/` 실파일 재검증은 아니다.

### B-2. depth GT 파이프라인 — 부분(생성·학습 연결 확인, 현 산출물 미확인)

- LiDAR 원점을 body로 옮긴 뒤 704×256 학습 좌표계의 세 카메라에 투영하고, 화면 밖 점을 제거한 후 픽셀당 가장 가까운 depth 한 개를 남긴다(`generate_depth_gt.py:58-120`). LiDAR가 없는 stem은 명시적으로 skip한다(`generate_depth_gt.py:212-232`).
- 검증 단계는 임의 GT 프레임에 depth 점과 3D wireframe을 겹쳐 `depth_gt_verify/*.jpg`로 저장하도록 구현돼 있다(`generate_depth_gt.py:260-305`). 기존 실행 로그에는 scen150까지 실행되고 scen150의 LiDAR 결측이 0/910으로 남아 있다(`generate_depth_scen61_150.log:5519-5525`).
- temporal loader는 이 sparse GT를 stride 4/8/16의 3카메라 depth map으로 변환한다(`morai_dataset.py:650-683`). 학습에서는 `DenseDepthNet`의 유효 GT 픽셀 masked L1(weight 0.2)을 detection loss에 더한다(`train.py:459-508`, `train.py:2716-2722`). 실제 학습 로그에도 150개 시나리오 loader와 nonzero Depth loss가 확인된다(`train_a1a_depth.log:28-29`, `train_a1a_depth.log:42-50`).
- 따라서 생성 코드→loader→aux head/loss→실제 학습 실행까지는 확인됐다. 다만 현재 `dataset/`가 없어 생성된 `.npy`와 overlay를 다시 열어 검증하지 못했으므로 전체 상태는 `부분`이다.

### B-3. 가림(occlusion) — 부분

- **기본 활성 방식:** LiDAR를 body로 변환하고 yaw 역회전한 3D 박스 내부 점을 센다(`generate_occlusion_gt.py:41-50`, `generate_occlusion_gt.py:74-105`). 학습 기본값 1이므로 `num_lidar_pts==0`인 GT가 제외된다(`train.py:2108-2112`, `morai_dataset.py:601-611`).
- 기존 전체 생성 로그는 150개 시나리오/95,672프레임/123,656박스에 sidecar를 썼고, 74.5%가 0점이었다고 기록한다(`_tmp_occlusion_gen.log:1484-1495`). 즉 구현·실행은 확인됐으나, 기본 필터가 매우 큰 비율의 GT를 제거하는 설정이라는 점은 별도 품질 검토가 필요하다.
- **추가 opt-in 방식:** 3D box surface를 세 카메라에 rasterize하고 가까운 객체가 픽셀 owner가 되는 z-buffer로 동적 상호 가림률을 구한다(`generate_visibility_gt.py:247-330`). `VISIBILITY_MIN_RATIO>0`이면 필터가 동작한다(`morai_dataset.py:612-624`).
- **한계:** z-buffer는 정적 벽/건물/가드레일/식생을 모른다(`generate_visibility_gt.py:15-18`). ray-casting 구현은 저장소 전체 `*.py` 검색에서 확인되지 않았다. LiDAR point count는 정적 가림의 간접 신호일 뿐 카메라 ray-casting은 아니다.
- `VISIBILITY_LOSS_WEIGHTING`은 loader가 `visibility_weight`를 만들지만(`morai_dataset.py:619-630`, `morai_dataset.py:745-747`), 현재 `compute_auxiliary_detection_loss()`는 이 값을 criterion에 전달하지 않는다(`train.py:1464-1497`). 따라서 **visibility loss weighting은 미배선**이다.

## C. 라이다 처리

### C-1. 좌표 변환 — 부분

- LiDAR→body는 회전 없이 센서 translation `[1.92,0,1.35]`를 적용하는 `LIDAR_TO_BODY`다(`camera_configs.py:93-105`). `lidar_to_body_h()`가 실제 homogeneous 변환을 수행한다(`verify_lidar_camera_overlay.py:105-110`).
- body→카메라는 카메라별 extrinsic으로 변환한 뒤 `X_cam`을 depth로 사용해 `(u,v)`를 계산한다(`camera_configs.py:8-17`, `verify_lidar_camera_overlay.py:89-102`). 실제 point cloud 투영 경로는 `verify_lidar_camera_overlay.py:230-238`, depth GT 재사용 경로는 `generate_depth_gt.py:73-120`이다.
- BEV 쪽에는 detection box corner/임의 ego XY를 grid로 바꾸는 `ego_xy_to_grid()`가 있고 costmap에서 사용된다(`costmap.py:12-37`, `costmap.py:66-88`). 그러나 **raw LiDAR point를 BEV tensor로 만드는 별도 rasterizer/encoder는 확인되지 않았다**. 현재 학습은 카메라 FPN feature 기반이다(`train.py:721-768`, `train.py:933-944`).

### C-2. voxel화 — 미구현

- 저장소 전체 `*.py`에서 `voxel`, `voxelize`, `HardVoxel`, `DynamicVoxel`, `PointPillars` 대응 심볼을 검색했으나 구현을 확인하지 못했다. 인용할 실제 파일/라인이 없으므로 경로를 만들지 않는다.

### C-3. 지면 검출/분리 — 부분

- RANSAC/plane fitting 기반의 수집·전처리·학습용 지면 분리는 저장소 전체 `*.py` 검색에서 확인되지 않았다.
- 다만 기하 검증 스크립트의 `lidar_surface()`가 ROI 내 body-z 10 percentile을 임시 ground로 잡고 `ground+0.15 m` 위 점만 남기는 휴리스틱은 있다(`geom_verify/step_final_all.py:131-149`). 이는 검증용 국소 필터이며 일반 지면 segmentation 모듈은 아니다.

## D. 학습 구조

### D-1. 모델 head 구성 — 부분

- `AutoNavModel`은 ResNet50-FPN, 900 anchor instance, 6개의 `SparseRefinementDecoderLayer`로 구성된다(`train.py:721-772`). 각 refinement layer는 self-attention→multi-view deformable aggregation→FFN→`FFNDecoder` refinement를 수행한다(`train.py:596-715`).
- 각 detection head는 foreground 2-class logits(vehicle/pedestrian), 11채널 box `(x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz)`, 2채널 quality `(centerness,yawness)`를 낸다(`decoder.py:6-42`, `decoder.py:61-66`). 여섯 layer 모두 출력하며 최종과 중간 출력을 함께 반환한다(`train.py:970-1025`).
- depth aux head는 FPN stride 4/8/16 각각의 1×1 conv이며 학습 시에만 실행된다(`train.py:459-489`, `train.py:938-944`). 현재 기본 설정은 temporal dataset과 함께 depth를 켠다(`train.py:2077-2107`).
- temporal instance bank/temp-GNN 코드는 있으나 실행 기본값은 `USE_TEMPORAL_MEMORY=0`, 즉 기본 학습은 Stage A0 단일프레임이다(`train.py:2078-2100`).
- `StaticMapDecoder` 파일은 class/polyline head를 정의하지만(`static_decoder.py:41-73`), `AutoNavModel`에는 포함되지 않는다(`train.py:721-768`). 정적 맵 head는 현 학습 경로에서 비활성이다.

### D-2. `loss_calculator.py` 기준 활성/비활성 손실

| 손실/항목 | 상태 | 근거 |
|---|---|---|
| sigmoid focal classification | 활성 | 전 채널 0을 background로 쓰며 foreground 수로 정규화(`loss_calculator.py:17-37`, `loss_calculator.py:270-280`) |
| Hungarian class cost | 활성 | sigmoid class probability cost(`loss_calculator.py:76-86`) |
| Hungarian bbox cost | 활성, velocity 제외 | `MATCH_CHANNELS=8`; 위치/크기/yaw까지만 matching(`loss_calculator.py:11-14`, `loss_calculator.py:82-86`) |
| bbox L1 | 활성 | `REG_CHANNELS=10`에 scale-normalized L1(`loss_calculator.py:282-289`) |
| velocity `vx,vy` 회귀 | **활성** | 채널 8,9가 `REG_CHANNELS=10`에 포함(`loss_calculator.py:6-14`, `loss_calculator.py:286-289`) |
| velocity `vz` 회귀 | **마스킹/비활성** | 11번째 채널(ch10)은 REG 범위 밖이며 z jitter 때문에 제외한다고 명시(`loss_calculator.py:11-14`) |
| centerness quality BCE | 활성 | matched anchor target과 BCE(`loss_calculator.py:179-215`) |
| yawness quality BCE | 활성 | yaw cosine 부호 target, 내부 weight 0.5(`loss_calculator.py:183-224`) |
| unmatched/background quality | 마스킹 | `bg_quality_weight=0.0`이 기본이라 unmatched weight 0(`loss_calculator.py:152-170`, `loss_calculator.py:179-188`) |
| detection total | 활성 | `2*cls + 2*bbox + 0.2*quality`(`loss_calculator.py:291-294`, `train.py:2331-2333`) |
| decoder auxiliary loss | 활성 | 마지막 layer 1.0, 이전 layer 0.5 후 weight 합 정규화(`train.py:1464-1504`, `train.py:2077`) |
| dense depth masked L1 | 활성 | GT>0/finite pred만, weight 0.2; detection loss에 합산(`train.py:491-508`, `train.py:2716-2722`) |
| map loss | 미배선 | `MapHungarianMatcher`는 존재(`loss_calculator.py:96-149`)하지만 현재 train loss 호출은 detection criterion뿐(`train.py:1464-1497`) |
| visibility-weighted loss | 미배선 | batch에는 값이 있으나 loss 호출에서 소비하지 않음(`morai_dataset.py:745-747`, `train.py:1482-1493`) |
| occupancy loss | 미구현 | 대응 head/criterion 없음; `costmap.py` 후처리만 존재 |

### D-3. 학습 실행 진척 — 부분

- 전체 150 시나리오 설정의 로그에서 epoch 1 학습·검증과 checkpoint 저장까지 실행됐다(`train_all_scenes.log:2330-2341`). 로그는 epoch 2 step 1170에서 끝나며 100 epoch 완료 기록은 없다(`train_all_scenes.log:2343-2461`).
- depth/occlusion이 켜진 별도 실험도 loader와 nonzero loss까지 실행됐으나 `KeyboardInterrupt`로 중단된 기록이 있다(`train_a1a_depth.log:164-196`, `train_a1a_depth.log:223-251`). 따라서 구조의 end-to-end 실행은 확인됐지만 장기 학습 완료는 `부분`이다.

### D-4. occupancy — 학습은 미구현

- 존재: 검출 결과를 BEV occupancy/risk costmap으로 만드는 추론 후처리(`costmap.py:40-88`, `inference.py:319-321`).
- 부재: occupancy GT loader, occupancy prediction head, occupancy loss. 따라서 “관련 코드가 전무”는 아니지만 “occupancy 학습”은 미구현이다.

## E. 평가

### E-1. `eval_distance.py`와 거리 구간 평가 — 부분

- `eval_distance.py`는 optimizer/backward 없이 checkpoint를 로드하고 `validate(..., compute_metric=True)`를 1회 호출한다(`eval_distance.py:58-118`). softcalibrated@0.15 전체와 거리 구간별 결과를 출력한다(`eval_distance.py:120-158`).
- 구간은 ego 방사거리 `[0,20)`, `[20,40)`, `[40,55)`이며 55 m 이상은 집계하지 않는다(`train.py:1036-1045`). TP/FN은 GT 거리, FP는 prediction 거리로 버킷팅하고 class-aware greedy 2 m matching을 한다(`train.py:1384-1461`). `validate()`가 매 프레임 이 함수를 실제 호출해 집계한다(`train.py:1706-1710`, `train.py:1788-1802`, `train.py:1858-1882`).
- 기존 `eval_distance_log.txt`에는 실제 CUDA 평가 결과가 세 구간 모두 기록돼 있어 **거리 구간별 평가가 실행된 이력은 확인됨**이다(`eval_distance_log.txt:21-32`).

### E-2. 현재 v11과의 정합성 — 부분/미완

- 현재 스크립트는 레거시 `MoraiDataset`/`morai_collate_fn`, 저장소 루트 `best_model.pth`, `use_temporal_memory=False`를 고정한다(`eval_distance.py:32-55`, `eval_distance.py:69-104`). 반면 현재 학습 기본은 `MoraiTemporalDataset`과 `labels_3d_v2 + scene_info + depth_gt`, occlusion filter를 쓰며(`train.py:2077-2112`, `train.py:2278-2285`), 기본 checkpoint 경로도 `checkpoints/v11_transfer/best_model.pth`다(`train.py:2119-2126`).
- 기존 eval 로그는 val이 scen43~47이었다(`eval_distance_log.txt:2-6`), 현재 split 메타데이터의 val은 scen08/18/23/30/32/33/41/54다(`dataset_split_v11.json:300-308`). 따라서 기존 실행 로그는 거리 계산 동작을 증명하지만 **현재 v11 데이터·필터·checkpoint 조합을 재평가한 결과는 아니다**.
- 현재 `dataset/`가 없고 요청이 read-only였으므로 이 스크립트를 재실행하지 않았다. 결론: 거리 버킷 로직과 과거 실행은 확인됐으나 current-v11 평가 entry point 정리는 미완이라 `부분`이다.

## 남은 미완 항목 요약

1. 정적 구조물까지 포함하는 카메라 occlusion(ray-casting/map mesh/depth/segmentation) — **미구현**.
2. z-buffer visibility filter의 기본 학습 활성화와 `visibility_weight` loss 배선 — **미완**.
3. raw LiDAR BEV encoder/voxelization — **미구현**.
4. RANSAC/plane-fitting 기반 일반 지면 분리 — **미구현**; 검증용 percentile 휴리스틱만 존재.
5. occupancy prediction head/loss — **미구현**; 검출 후 costmap만 존재.
6. `eval_distance.py`의 current v11 dataset/filter/checkpoint 경로 통일 및 재실행 — **미완**.
7. live `--max_sync_gap`의 실제 synchronizer 배선과 non-reference header 부재 방어 — **미완**.
8. 전체 100 epoch 학습 완료 — **미확인/미완**; 열린 로그는 epoch 2 도중까지다.

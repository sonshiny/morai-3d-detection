# OCCLUSION_MIN_PTS 거리별 GT 제거 영향

## 판정 요약

현재 저장소에는 `dataset/`과 `occlusion/*.npy` 실파일이 없으므로 아래 수치는 `_tmp_occlusion_gen.log`의 전체 집계를 사용한 **로그 기반 추정**이다. 로그의 제거율은 소수점 한 자리로 반올림되어 있어 threshold 1/3/5의 박스 수에는 `약(≈)`을 붙였다.

### Q1. 현재 기본 필터값에서 원거리 GT 제거율

코드 기본값은 `OCCLUSION_MIN_PTS=1`이며 `num_lidar_pts < 1`, 즉 0점 GT를 제거한다.

| 거리 구간 | 필터 전 GT | 살아남는 GT | 제거 GT | 제거율 |
|---|---:|---:|---:|---:|
| `[20,40)m` | 53,381 | ≈14,466 | ≈38,915 | 72.9% |
| 대체 구간 `[40,∞)m` | 23,348 | ≈3,315 | ≈20,033 | 85.8% |

기존 로그에는 `[40,55)m`와 `[55,∞)m`가 분리되어 있지 않아 허용된 대체 구간 `[40,∞)m`를 사용했다.

### Q2. threshold=0일 때 원거리 GT 배수

| 거리 구간 | 현재 threshold=1 생존 GT | threshold=0 GT | threshold=0 / 현재 |
|---|---:|---:|---:|
| `[20,40)m` | ≈14,466 | 53,381 | ≈3.69배 |
| 대체 구간 `[40,∞)m` | ≈3,315 | 23,348 | ≈7.04배 |

### Q3. train/val 및 `eval_distance.py` 필터 적용 여부

- **기본 학습의 train과 val에는 같은 필터값 1이 적용된다.** `train.py`가 기본값을 1로 읽고([`train.py:2108-2112`](train.py#L2108)), 기본 `MoraiTemporalDataset` 경로에서 train과 val 생성자 양쪽에 동일한 `OCCLUSION_MIN_PTS`를 전달한다([`train.py:2078`](train.py#L2078), [`train.py:2278-2285`](train.py#L2278)). 실제 학습 로그에도 `min_lidar_pts=1 (train+val GT에서 ... drop)`이 기록돼 있다([`train_shuffle_nobank.log:21`](train_shuffle_nobank.log#L21)).
- 필터는 `labels_3d_v2` 행을 batch GT에 추가하기 전에 `0 <= npts < self.occlusion_min_pts`인 행을 `continue`로 제외한다([`morai_dataset.py:601-611`](morai_dataset.py#L601)). `train.py`의 validation 거리 지표는 이 batch의 `dynamic_gt_boxes`와 `dynamic_gt_labels`를 사용한다([`train.py:1742-1743`](train.py#L1742), [`train.py:1788-1802`](train.py#L1788)). 따라서 **`train.py` 내부 val 거리별 recall은 필터 적용 후 GT 기준**이다.
- **별도 `eval_distance.py`의 거리별 recall에는 OCCLUSION_MIN_PTS 필터가 적용되지 않는다.** 이 스크립트는 occlusion 인자가 없는 레거시 `MoraiDataset`을 import하고([`eval_distance.py:31-39`](eval_distance.py#L31)), val dataset도 `MoraiDataset`으로 생성하면서 `occlusion_min_pts`를 전달하지 않는다([`eval_distance.py:89-100`](eval_distance.py#L89)). `MoraiDataset` 생성자에는 해당 인자가 없고([`morai_dataset.py:237-243`](morai_dataset.py#L237)), 라벨 로딩 시 카메라 화면 교차 필터만 적용한다([`morai_dataset.py:284-302`](morai_dataset.py#L284)). 따라서 기존 `eval_distance.py`/`eval_distance_log.txt`의 원거리 recall은 **이 occlusion 필터로 걸러진 뒤의 값이 아니다**.

## 거리 × threshold 집계

필터 조건은 `num_lidar_pts < threshold`이다. threshold 0은 필터 없음이다.

| 거리 구간 | threshold | 전체 GT | 살아남는 GT | 제거 GT | 제거율 |
|---|---:|---:|---:|---:|---:|
| `[0,20)m` | 0 | 46,927 | 46,927 | 0 | 0.0% |
| `[0,20)m` | 1 | 46,927 | ≈13,703 | ≈33,224 | 70.8% |
| `[0,20)m` | 3 | 46,927 | ≈13,233 | ≈33,694 | 71.8% |
| `[0,20)m` | 5 | 46,927 | ≈12,717 | ≈34,210 | 72.9% |
| `[20,40)m` | 0 | 53,381 | 53,381 | 0 | 0.0% |
| `[20,40)m` | 1 | 53,381 | ≈14,466 | ≈38,915 | 72.9% |
| `[20,40)m` | 3 | 53,381 | ≈11,797 | ≈41,584 | 77.9% |
| `[20,40)m` | 5 | 53,381 | ≈9,769 | ≈43,612 | 81.7% |
| 대체 `[40,∞)m` | 0 | 23,348 | 23,348 | 0 | 0.0% |
| 대체 `[40,∞)m` | 1 | 23,348 | ≈3,315 | ≈20,033 | 85.8% |
| 대체 `[40,∞)m` | 3 | 23,348 | ≈1,424 | ≈21,924 | 93.9% |
| 대체 `[40,∞)m` | 5 | 23,348 | ≈957 | ≈22,391 | 95.9% |

## 데이터 및 산출 방법

- 현재 `dataset/`: 없음
- 현재 `occlusion/` 디렉터리: 0개
- 현재 `occlusion/*.npy`: 0개
- 현재 실파일 기준 시나리오: 0개
- 기존 생성 로그 기준: 150개 시나리오, 95,672프레임, 123,656박스, LiDAR 결측 프레임 0개
- 사용 로그: [`_tmp_occlusion_gen.log:1484-1495`](_tmp_occlusion_gen.log#L1484)
- 로그 버킷: `[0,20)m`, `[20,40)m`, `[40,∞)m`
- 로그 임계값별 수치: `npts < 1`, `< 3`, `< 5`, `< 10`의 제거율
- 추정 제거 수: `round(전체 GT × 로그 제거율 / 100)`
- threshold 1/3/5의 정확한 정수 제거 수: 로그에 제거율이 소수점 한 자리로만 남아 있어 산출 불가
- 클래스별 집계: dataset의 `labels_3d*` 실파일이 없고 로그에 클래스 분해가 없어 산출 불가
- 집계 스크립트: [`analyze_occlusion_filter_impact.py`](analyze_occlusion_filter_impact.py)

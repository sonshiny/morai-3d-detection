# V3 (source-time correction) label correctness — direct-measurement diagnosis

> **범위**: v3 label 의 *정확성*을 **학습 없이** 직접 측정으로 판정한다. 성능(F1/detection center-distance)은
> 다루지 않는다. corr_dist·residual 은 label-공간 거리이며 모델 예측 지표가 아니다.
> 데이터: scen05/scen77/scen144 (10,516 box, 3,337 frame). 학습·checkpoint·GPU A/B 미실행.
>
> ## 4-축 판정 (rule 11 — 가장 약한 축을 숨기지 않음)
> | 축 | 판정 | 근거 등급 |
> |---|---|---|
> | **FORMULA_CORRECTNESS** | **CONFIRMED** | 코드+대수 분해(rule3)+음성대조(rule8)+clock-domain(rule4) |
> | **IMPLEMENTATION_CORRECTNESS** | **CONFIRMED** | 독립 재계산이 저장 v3 와 CSV 반올림 내 일치(grade C, rule5) |
> | **TEMPORAL_CONSISTENCY** | **IMPROVED** (차량); EQUAL (준정지/보행자) | leave-one-out(grade B, rule6-7) |
> | **ABSOLUTE_ACCURACY** | **UNVERIFIABLE** | grade-A 독립 oracle 부재; grade-B 기준이 v3 입력과 동일 소스(순환) |
>
> **전체 V3_CORRECTNESS**: v3 의 **수식과 구현은 정확**하며, label 시각 t_ref 에서 **움직이는 차량의
> 시간 정합(temporal consistency)을 측정 가능하게 개선**한다. 그러나 **절대 정확도는 독립 검증 불가**다 —
> 유일하게 이용 가능한 기준 궤적이 v3 를 만든 것과 **동일한 MORAI Object 스트림**에서 나오기 때문이다
> (v2 yaw 순환 함정과 같은 부류). 따라서 가장 약한 축인 **ABSOLUTE_ACCURACY=UNVERIFIABLE** 를 명시한다.

---

## 1. v3 보정 로직 (코드에서 읽은 확정 사실)
[FACT — `correct_source_time.py`, `preprocess_dataset.py`, `morai_sync.py`]

- **시간 기준 (gate, item#1)**: `ref_src_ts` = **`/cam_front`(전방 카메라) 메시지의 source-header 시각** t_ref.
  `morai_sync.py:36 REF_TOPIC="/cam_front"`; 다른 스트림(ego/object/lidar)은 그 t_ref 에 **source-header 가
  가장 가까운** 메시지로 선택(`_select_by_header`), receive-time fallback 은 **금지**(`ref_no_source_header`→drop,
  line 298-300). 라벨 stem 도 cam_front 기준(`build_frame_groups.py`). → 시간 기준은 **cam_front source clock**.
- **문제**: 저장 박스 기하는 sync-매칭된 Ego/Object 메시지(각각 t_e=`ego_src_ts`, t_o=`obj_src_ts`)로
  계산됐으나 label 시각·이미지는 t_ref 다. 즉 `r_stored = R(-yaw_e(t_e))·(p_o(t_o) - p_e(t_e))` 인데
  정답(=이미지 시각)은 `R(-yaw_e(t_ref))·(p_o(t_ref) - p_e(t_ref))`.
- **보정 대상 필드**: position(ego-frame x,y,z_center 및 world gx,gy), yaw(yaw_ego, ego_yaw 를 t_ref 로 재표현),
  velocity(vx_ego,vy_ego 를 t_ref ego frame 으로 회전), timestamp(=t_ref). **dimensions/class/track_id 는 불변.**
- **시간 보간**: ego pose 를 (t_e, ego_state) 표에서 x/y/z 선형·yaw unwrap+선형으로 t_ref 로 보간;
  epoch/segment 경계 넘는 보간 금지; 경계·과대 gap(>0.5s)·과대 외삽(>0.2s)은 `correction_valid=0`.
- **object 전파**: `vel_source="raw"`(MORAI 속도 제공) → `p_o(t_ref)=p_o(t_o)+v_world·(t_ref-t_o)`;
  `"motion"`(위치차분 유도, 주로 보행자) → object self-motion **전파 안 함**(ego frame 변화만).
- **v2 대비 차이**: v2 는 동일 decode 규약이나 **박스를 t_o 기하 그대로 두고 t_ref 로 라벨링**(재타이밍 없음);
  v3 는 위 물리식으로 t_ref 로 재타이밍한다. (본 데이터셋은 `_fill_velocity_from_motion` 결과 **motion 트랙 0개**
  → 전 박스 `vel_source="raw"`.)

## 2. Clock domain 검증 (rule 4) — [FACT]
ref/ego/obj source ts 는 **동일 clock domain**:
- 동일 base(같은 epoch, ~1.784e9), **전부 단조증가**(ref/ego/obj n_backward=0), ~10Hz 균일(frame_dt med≈102ms),
  **discontinuity 0**, **단일 epoch('0')**.
- 오프셋: ego−ref med +26~41ms, obj−ref med +25~40ms(미래 편향, 문서 주장과 일치), **obj−ego med 0ms**
  (67–75% 정확히 동일 → MORAI ego+object 동시 발행).
- **recv clock 은 별개·잡음**: recv−src med +65~390ms, 스프레드 ±1s. → source ts 를 recv 로 대체하지 않음(코드도 금지).
- **판정**: clock domain 동일 → **보정 수식 적용 가능**(REFUTED/BLOCKED 아님).

## 3. 수식 분해(rule 3) + 구현 재현(grade C, rule 5) — [FACT]
박스별 `(v3_rel − v2_rel)` 를 세 항의 벡터합으로 **정확 분해**:
`term_egorot = (R(-yaw_ref)−R(-yaw_te))·(p_o(t_o)−p_e(t_e))` + `term_egotrans = R(-yaw_ref)·(p_e(t_e)−p_e(t_ref))`
+ `term_objvel = R(-yaw_ref)·v_world·(t_ref−t_o)`.
- **분해 검증**: `max|(objvel+egotrans+egorot) − (v3−v2)_xy| = 9e-5 m`(전 10,516 box) — CSV 반올림 내 일치.
- **구현 재현**: v2+ego보간+timing 으로 v3 를 **독립 재계산** → 저장 v3 와 오차 **≤1.1e-4 m**
  (x,y,gx,gy,vx_ego,vy_ego), yaw_ego 오차 0. = CSV 4-소수 반올림. → **구현이 수식을 정확히 실현.**
- 순환성(rule 5): 이 재계산은 **같은 프레임·같은 입력**을 쓰므로 IMPLEMENTATION 검증일 뿐,
  independent accuracy 가 아니다(등급 C).

## 4. 불변량(rule 9) + 음성 대조(rule 8) — [FACT]
- **불변량**: 10,516 joined box 에서 dimensions(w,l,h)·class_id **불일치 0**, track_id 집합 v2↔v3 **완전 동일**
  (only-v2=0, only-v3=0), **|yaw_global(v2)−yaw_global(v3)|max = 0**. (dims/class/track/global-yaw 불변 확인.)
- **음성 대조**(각 항이 물리대로 동작):
  - (b) 정지객체+정지ego: corr_dist mean 0.008m (≈0)
  - (c) world-static 객체: global 위치 drift mean 2e-5m (≈0 — 정지 객체를 허위로 옮기지 않음)
  - (d) 이동객체+정지ego: `|disp − objvel_term|` mean 0.012m (변위를 object-velocity 항이 설명; 잔차=ego 잔여)
  - (e) 정지객체+이동ego: objvel_term mean 2e-5m (≈0 — 보정이 ego rigid transform 으로만 설명)

## 5. 보정 **크기**(task 2) — [FACT] · rule 1: 크기는 accuracy 가 아님
`corr_dist = |v3 − 원본 decode|` (ego-frame 변위). **이 값의 크기로 CONFIRMED/REFUTED 하지 않는다.**
- overall: mean **0.105m**, median 0.077m, p95 0.31m, **max 0.91m** (`magnitude_stats.json`).
- scene별: scen05 mean 0.19m(p95 0.42) > scen77 0.097 ≈ scen144 0.092.
- frac>0.2m = 16.3%, frac>0.5m = 0.68%, **frac ≥ 2.0m(matcher thr) = 0** (max/2.0m = 0.45).
- object world speed: mean 2.0, p95 4.2, max 4.3 m/s. |t_ref−t_o|: mean 34ms, p95 76ms, max 121ms.
- 해석[FACT]: corr_dist ≈ (object speed)×(dt) + ego motion 항으로 설명되며 항상 sub-threshold. track 시계열
  플롯 5개(`plots/`)에서 속도↑ → corr_dist↑, 정지구간 → ~0 을 확인.

## 6. Leave-one-out 시간 정합 (grade B, rule 6-7) — [FACT+해석]
기준 궤적 = **대상 프레임 제외**, 같은 track 의 이전/이후 **raw world 위치(MORAI object)**를 같은 epoch·
gap≤0.5s 안에서 t_ref 로 선형보간(양측 이웃 필수, primary). v2(=t_o 위치를 t_ref 로 라벨) 와 v3(=전파) 를
**동일 t_ref world 기준**으로 비교(rule 2). primary box = 10,426.
- **차량(raw)**: residual v2 **0.099m → v3 0.016m**; paired Δ(v3−v2) **−0.083m, 95%CI[−0.085,−0.081]**,
  v3<v2 **82.4%**. dt 증가 시 개선 증가([0.05,0.20)s: v2 0.204 → v3 0.015, Δ−0.189). 속도·거리 구간 모두 개선.
- **보행자/준정지**: v2≈v3≈0 (본 데이터 객체는 저속·정지 다수, v3 는 이들 world 위치를 옮기지 않음) → EQUAL.
- **const-velocity 모델 오차(rule 7)**: 0.5·|a|·dt² ≈ median 0.0006m, p95 0.014m — v3 잔차(0.016m)는
  이 모델 floor + 기준 보간 잡음 수준. 즉 v3 잔차는 버그가 아니라 등속 가정의 이론적 한계 근방.
- **순환성 경고(rule 4/5/10)**: 이 기준은 v3 가 소비하는 **동일 MORAI object 스트림**의 이웃 표본이다.
  대상 프레임을 제외했으므로 grade **B(temporal-consistency evidence)** 이나, **grade-A 독립 GT 가 아니다.**
  timestamp 자체가 틀렸다면 v3 와 기준이 함께 틀려 이 검사로 잡히지 않는다.

## 7. 독립 기준 등급 (rule 10) + 순환성 — [FACT/판정]
| 등급 | 정의 | 본 데이터 가용성 | 결론 |
|---|---|---|---|
| A | v3 에 안 쓰인 정확한 t_ref oracle | **없음** (Object 는 t_o 에서만 표본; t_ref 동기 object 로그 부재) | 절대정확도 확증 불가 |
| B | 대상 제외 이웃 source 표본(LOO) | 있음(§6) | **temporal consistency 개선** 근거 |
| C | 같은 프레임·입력 재계산 | 있음(§3) | **구현 정확** 근거 |
| D | camera/LiDAR 육안 정렬 | `visual_audit/v2_vs_v3/`(cam_front 이미지는 t_ref) | 보조 시각 증거(수치 아님) |
- **순환성 명시**: MORAI Object position/velocity 는 v3 **생성 입력**이다. 이를 기준으로 한 residual 검증은
  independent accuracy 로 **무효**(rule 4/5) — B(시간정합)·C(구현)로만 인정. A 부재 → **ABSOLUTE_ACCURACY=UNVERIFIABLE**.
- D(cam_front 이미지가 t_ref 라 v3 가 더 잘 정렬될 것)는 보조적이며, 본 판정의 수치 근거로 쓰지 않는다.

## 8. 판정 근거 요약 (rule 11)
- **FORMULA_CORRECTNESS = CONFIRMED**: clock domain 적용가능(§2), 3-항 분해가 정확(§3), 음성대조가 물리대로(§4).
- **IMPLEMENTATION_CORRECTNESS = CONFIRMED**: 독립 재계산이 저장 v3 와 반올림 내 일치, 불변량 보존(§3-4).
- **TEMPORAL_CONSISTENCY = IMPROVED**(차량, grade B): LOO residual 0.099→0.016m, Δ−0.083 CI 명확·방향 일치;
  준정지/보행자 EQUAL(§6). (grade B 이므로 "정합 개선"이지 "절대정확" 아님.)
- **ABSOLUTE_ACCURACY = UNVERIFIABLE**: grade-A 독립 oracle 부재, grade-B 기준이 v3 입력과 동일 소스(순환)(§7).

## 9. 추측/미검증 (facts 와 분리) — [SPECULATION/UNVERIFIED]
- MORAI Object position/velocity 가 실제 시뮬레이터 GT 라면 §6 개선은 v3 가 t_ref 궤적을 정확 재구성함을 뜻한다.
  그러나 **그 가정 자체를 본 데이터로 증명할 수 없다**(A 부재).
- timestamp 할당(어느 메시지가 어느 프레임에 매칭됐는지)의 절대 정확성은 검증 범위 밖(clock domain 일관성만 확인).
- 고속(>6 m/s) 차량 표본 0 → 고속 영역 일반화 불가.
- 절대정확도 확정에는 **grade-A 신호**가 필요: (i) t_ref 동기 object oracle 재수집, 또는
  (ii) cam_front(t_ref) 재투영 IoU·LiDAR 점군 정합의 정량화(현재 D 등급, 수치화 안 됨).

## 재현
`experiments/v3_correctness/{v3_lib,v3_stage1,v3_stage2,v3_stage3}.py` (read-only), `magnitude_stats.json`,
`track_plots_manifest.json`, `plots/*.png`. 저장소 root에서 실행:
`DATASET_ROOT=/path/to/dataset V3_SCENARIOS=scen05,scen77,scen144 python3 experiments/v3_correctness/v3_stage1.py`
(stage2/stage3도 동일 환경; dataset 필요, 학습·checkpoint·GPU 불필요).

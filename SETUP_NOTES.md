# morai-3d-detection 재현 노트

새 머신으로 이전 시 참고. 마지막 확인: 2026-07-02 (기존 WSL 환경 기준)

---

## 1. 환경 버전 (확인된 값)

| 항목 | 값 |
|---|---|
| OS (WSL 배포판) | Ubuntu 20.04.6 LTS (focal) |
| Python | 3.8.10 |
| PyTorch | 2.1.0+cu121 |
| PyTorch가 요구하는 CUDA | 12.1 |
| ROS | Noetic |
| GPU | NVIDIA GeForce RTX 4060 (8GB) |
| 확인 당시 드라이버 | 595.97 (지원 CUDA 13.2, 12.1과 하위호환 정상 작동) |

**새 머신 설치 순서 권장:**
1. WSL2 + Ubuntu 20.04 설치
2. NVIDIA 드라이버 설치 (CUDA 12.1 이상을 지원하는 버전이면 됨 — 반드시 595.97일 필요 없음)
3. ROS Noetic 설치
4. `pip install -r requirements.txt` (아래 참고)

**requirements.txt**: `~/backup_env/requirements.txt`에 `pip freeze`로 생성됨 (torch==2.1.0+cu121 포함, 전체 패키지 버전 고정). 코드와 함께 GitHub에 커밋할 것.

> 참고: `pip install --break-system-packages` 필요 여부는 새 머신의 Python 관리 방식(system Python vs venv)에 따라 다를 수 있음 — 설치 시 확인.

---

## 2. GPU 메모리 참고

확인 시점 `nvidia-smi`: 7815MiB / 8188MiB 사용 중 (v10 학습 진행 중, PID 552228 `/python3.8`). RTX 4060 8GB 한계에 근접한 상태 — 새 머신에서도 동일 GPU면 배치 크기 등 현재 설정 그대로 유지 가능. GPU 업그레이드 시 배치 크기 상향 검토 가능.

---

## 3. 네트워크 설정 (IP는 새 머신에서 재확인 필요)

### gRPC (MORAI Windows ↔ WSL2)
- 포트: `7789`
- WSL2 → Windows 접근 주소 예시(구 머신): `172.31.96.1:7789`
- **새 머신에서는 이 IP가 반드시 바뀜.** WSL 안에서 `ip route | grep default`의 첫 번째 IP가 대개 Windows 호스트 주소.

### rosbridge (Foxglove / 시각화용)
- WSL 쪽 실행: `roslaunch rosbridge_server rosbridge_websocket.launch`
- 접속 주소: `ws://localhost:9090` (Windows에서 Foxglove Studio로 접속)
- MORAI 센서 JSON 내 `RosBridgeServerUrl`도 동일하게 `ws://127.0.0.1:9090`로 맞춰져 있어야 함 (`sensor_config_final_v2.json` 참고)

### ROS Master
- `ROS_MASTER_URI=http://localhost:11311` (기본값, 구 머신 기준 확인됨)
- `roscore`가 반드시 먼저 떠 있어야 함 — 안 뜨면 rviz/rostopic이 `Connection refused`

---

## 4. 모델 가중치(.pth) 백업 상태

**보관 위치:** `Y:\AIM_2026\개인 폴더\손재호` (NAS 네트워크 드라이브)

**백업 완료 (2026-07-02, sha256 대조 검증됨):** 13개 파일 전부 무결성 확인 완료.

| 파일 | 비고 |
|---|---|
| `best_model_backup_v9_0.2558.pth` | v9 안전 앵커. softcalibrated F1=0.2558. 147,909,038 bytes (메모리상 기록된 147,938,726 bytes와 ~29KB 차이 — 파일명 변경/재저장 과정에서 발생한 것으로 추정, 로드 가능 여부는 미확인이니 여유 있을 때 `torch.load`로 열어서 state_dict 정상인지 확인 권장) |
| `best_model_epoch001_softcalibrated_f1_015_0.3220.pth` | v10 첫 결과, v8 기준선(0.2960) 초과 |
| `best_model_raw_f1_025.pth`, `best_model_raw_f1_025_epoch003_..._0.1114.pth` | raw F1 기준 체크포인트 |
| `best_model_val_loss.pth`, `best_model_val_loss_epoch003_val_loss_1.2374.pth` | val loss 기준 체크포인트 |
| `checkpoint_epoch10/20/30/40.pth` | 주기적 체크포인트 |
| `best_model.pth`, `morai_autonav_weights.pth` | 범용 명명 체크포인트 — 어느 실험 버전인지 이름만으로 불명확, 필요시 내부 메타데이터(epoch, 저장 시각) 확인 권장 |
| `last_checkpoint.pth` | **학습 진행 중 계속 갱신되는 파일.** 백업 시점 스냅샷일 뿐이므로, v10 학습 종료 시 또는 머신 이전 직전에 반드시 재백업할 것 |

**체크섬 원본:** `~/pth_checksums.txt` (WSL 내 위치, sha256sum 기준)

**중요 — 완전 체크포인트 vs state_dict 여부 미확인:** 어떤 `.pth`가 옵티마이저/LR스케줄러 상태를 포함한 완전 체크포인트이고 어떤 게 순수 state_dict인지 파일명만으로는 구분 안 됨. v9 실패 원인(state_dict만 저장된 걸 재개해서 옵티마이저/LR 리셋)이 재발하지 않으려면, 새 머신에서 이어 학습할 체크포인트를 고를 때 반드시 내용물을 열어서 확인할 것.

---

## 5. 데이터셋

라이다 미포함 scen43~47 데이터셋은 이번 백업 범위에서 제외(사용자 판단, 별도 필요 시 재수집 가능). MGeo/시나리오 러너 자산도 백업 대상에서 제외(더 이상 다루지 않기로 결정).

---

## 6. wandb

프로젝트 `morai-3d-detection` (entity: `ssonjh14-inha-university`)는 더 이상 로그 분석에 사용하지 않음 — 대신 로컬 `train_log_vN.txt`를 Claude에게 전달해 분석하는 방식으로 전환. 별도 백업 불필요.

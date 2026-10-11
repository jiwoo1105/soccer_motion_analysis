# 상세 사용법

기본 설치와 영상 한 편, 여러 편 실행은 [README](../README.md)를 참고합니다. 아래 명령은 저장소 루트에서 가상환경을 활성화한 뒤 실행합니다. 입력 예시는 자신의 파일로 바꾸고, 매번 새로운 출력 폴더를 지정합니다.

## 설치 환경

Python 3.11을 사용합니다. Windows에서는 다음과 같이 가상환경을 만들고 PowerShell에서 활성화한 뒤 README의 패키지 설치 명령을 실행합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

`requirements-sam2.txt`는 원 연구에서 사용한 SAM2 커밋과 추가 의존성을 설치합니다. CPU/MPS에서 선택적 CUDA 확장 빌드를 생략하려면 다음과 같이 설치합니다.

```bash
SAM2_BUILD_CUDA=0 python -m pip install -r requirements-sam2.txt
```

확장을 생략하면 일부 마스크 후처리가 달라질 수 있습니다. [SAM2 공식 설치 안내](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)를 참고합니다. 환경별 실제 실행 범위는 [검증 기록](validation.md)에 구분하며, 기존 YOLO11 ROI 실행 기록을 SAM2의 플랫폼 검증으로 해석하지 않습니다.

## 모델 확인과 오프라인 가중치

기본 `scripts/setup_models.py`는 공식 MediaPipe Pose Heavy, YOLOv8n, SAM2.1 Hiera Small 파일을 내려받아 SHA-256을 확인합니다. 분석 명령 자체는 모델을 다운로드하지 않습니다.

```bash
# 다운로드 없이 설치된 모델 확인
python scripts/setup_models.py --check

# 이미 준비한 공식 파일로 설치하고 체크섬 확인
python scripts/setup_models.py \
  --pose-source /path/to/pose_landmark_heavy.tflite \
  --yolo-source /path/to/yolov8n.pt \
  --sam2-source /path/to/sam2.1_hiera_small.pt
```

이 명령은 모델 파일의 오프라인 설치용입니다. Python 패키지도 별도로 준비해야 완전히 오프라인으로 설치할 수 있습니다. SAM2 파일은 [공식 체크포인트 안내](https://github.com/facebookresearch/sam2#download-checkpoints)의 Small 모델을 사용합니다.

분석 시 `--model`은 YOLO 가중치, `--sam2-checkpoint`는 로컬 SAM2 체크포인트를 지정합니다. 기본 설치 경로를 명시한 실행은 다음과 같습니다.

```bash
python analyze_video.py \
  --video "input/my_dribble.mp4" \
  --output "output/my_dribble_v2" \
  --ball-backend yolov8_sam2 \
  --model "models/yolov8n.pt" \
  --sam2-checkpoint "sam2_checkpoints/sam2.1_hiera_small.pt" \
  --device cpu
```

다른 위치에 준비했다면 두 경로를 실제 파일로 바꿉니다. SAM2 경로는 Small 구조를 사용하므로 다른 크기의 체크포인트를 대체 파일로 쓰지 않습니다. 다른 학습 가중치의 결과가 기본 모델과 같다고 보장하지 않습니다. 모델 파일과 원본 영상은 Git에 포함하지 않으며, 모델과 패키지는 각 배포처의 라이선스를 따릅니다.

## 실행 장치

`--device cpu`가 기본입니다. 지원되는 Apple Silicon에서는 `--device mps`, CUDA 환경에서는 `--device 0` 또는 `--device cuda:0`을 지정합니다. 이 옵션은 YOLO 공 검출과 SAM2 추적에 적용하며 MediaPipe 실행 장치와 별개입니다. GPU별 결과 일치는 보장하지 않습니다.

## 가중치와 점수 설정

```bash
python analyze_video.py \
  --video "input/my_dribble.mp4" \
  --output "output/weights_30_60_10" \
  --weights 0.3 0.6 0.1
```

순서는 **헤드업, 상체, 어깨**입니다. 음수가 없어야 하고 합은 1이어야 하며 자동 정규화하지 않습니다. 변경한 비중은 보고서에 기록합니다. 정답 등급에 맞춰 비중을 자동 학습하지 않으며, 계산 불가 항목의 비중이 0이어도 총점은 `N/A`입니다.

`--calibration`은 점수와 측정 설정 JSON을 지정합니다. 기본값은 `configs/scoring.json`이며 현재 프로필은 `2026-10-11-sam2-touch16`, 스키마는 2입니다. 고정된 환산값의 출처와 어깨 ±16 창 선택의 한계는 [계산 방법](methods.md), [공 추적 출처 감사](touch-source-audit.md)를 참고합니다. 설정을 바꾼 결과를 기본 프로필의 결과와 혼동하지 않습니다.

## 선수 영역 지정

고정된 영역으로 관절 추출 범위를 제한할 수 있습니다. 좌표는 원본 영상의 픽셀 단위 `x y 너비 높이`입니다. 아래는 1920×1080 영상에서의 예시입니다.

```bash
python analyze_video.py \
  --video "input/my_dribble.mp4" \
  --output "output/player_roi" \
  --roi 400 80 1100 950
```

`--roi`는 한 영상에만 사용할 수 있습니다. 선수 전신이 전체 구간 동안 영역 안에 들어와야 합니다. 자동 인물 식별이나 추적 ID 기능은 없으므로 다른 선수가 영역에 들어오면 검수가 필요합니다. 이 옵션은 선수의 관절 추출 영역이며, 공 처리 경로를 YOLO11 ROI로 바꾸는 옵션은 아닙니다.

## 장면 수와 글꼴

```bash
python analyze_video.py --video "input/my_dribble.mp4" \
  --output "output/eight_frames" --frames 8 \
  --font "/path/to/NotoSansCJK-Regular.ttc"
```

장면 수는 기본 6장, 선택 범위는 2~12장입니다. 원본 시간 순서의 장면을 나열하며 배경을 합성하지 않습니다. 시스템의 한글 글꼴을 찾으면 한글 표시를 사용하고, 없으면 영문 표시로 대체합니다. 한글 파일명에는 `--font`로 한글 지원 TTF/TTC를 지정할 수 있습니다. 글꼴은 자동 다운로드하지 않습니다.

그림의 노란 선은 어깨 중앙에서 눈 중앙으로 향하는 벡터, 파란 선은 무릎–엉덩이–어깨, 빨간 선은 어깨 벡터입니다. 선은 2D 표시이며 실제 각도와 속도는 추정 3D 좌표로 계산합니다. HTML을 공유할 때는 같은 폴더의 `panorama.png`도 함께 전달합니다.

## 선택적 YOLO11 ROI 대안

YOLO11s 발목 ROI는 **원래 논문의 공 추적 방식과 별개인 대안**입니다. 프레임별 발목 영역에서 공 후보를 검출하며 SAM2를 사용하지 않습니다. 이 경로만 쓸 환경은 `requirements.txt`로 설치할 수 있고 SAM2 추가 패키지는 필요하지 않습니다.

```bash
python scripts/setup_models.py --ball-backend yolo11_roi
python analyze_video.py \
  --video "input/my_dribble.mp4" \
  --output "output/yolo11_roi" \
  --ball-backend yolo11_roi \
  --model "models/yolo11s.pt"
```

가중치를 따로 준비했다면 `--model /path/to/yolo11s.pt`로 지정합니다. 오프라인 설치는 `setup_models.py --ball-backend yolo11_roi`에 `--pose-source`와 `--yolo-source`를 추가합니다. 이 경로의 공 자료는 `tracks.json` 대신 `candidates.json`에 저장됩니다.

터치 중심이 달라질 수 있으므로 SAM2와 점수가 같다고 가정하지 않습니다. 보고서와 캐시의 `ball_source`에 선택한 경로를 기록합니다. 과거 ROI 기반 헤드업 표와 어깨 실험을 SAM2 결과로 해석하지 않습니다. 현재 헤드업 환산 경계도 과거 ROI 탐색의 운영값을 고정한 것으로, 새 SAM2 자료에 다시 맞추거나 검증한 보정값이 아닙니다.

## 캐시로 다시 계산

이전 실행의 `clips/clip-...` 폴더를 지정하면 같은 원본의 관절과 공 자료를 재사용할 수 있습니다. 그림 생성을 위해 원본 영상도 필요합니다.

```bash
python analyze_video.py \
  --video "input/my_dribble.mp4" \
  --reuse-cache "output/my_dribble/clips/clip-01-실제해시12자리" \
  --output "output/reweighted" \
  --weights 0.3 0.6 0.1
```

실제 생성된 캐시 폴더명으로 바꿉니다. 한 번에 영상 한 편만 재사용할 수 있습니다. 원본과 캐시 해시, FPS, 프레임 번호, 해상도, ROI를 확인하며 ROI 캐시는 같은 `--roi`를 지정해야 합니다. 예전 실험 캐시를 자동 검색하거나 호환되지 않는 자료를 임의 변환하지 않습니다.

공 처리 경로도 일치해야 합니다. SAM2 캐시는 기본 경로로, YOLO11 ROI 캐시는 `--ball-backend yolo11_roi`로 재사용합니다. 경로 표기가 없는 예전 캐시는 ROI로 취급하며 SAM2 결과로 해석하지 않습니다. `--model`이나 `--sam2-checkpoint`를 지정하면 해당 가중치 해시도 대조합니다.

모델 없는 캐시 재계산 환경은 `requirements-evaluation.txt`로 설치할 수 있습니다. 재사용 결과에는 `cache_replay: true`가 기록되며 새 추론 결과로 표시하지 않습니다. 새 출력에는 보고서와 파노라마를 만들고, 원본 캐시 폴더를 복제하지는 않습니다.

## 테스트와 실행 기록

```bash
python -m pip install -r requirements-evaluation.txt
python -m pytest -q
```

모델 없는 자동 테스트와 실제 영상 추론은 서로 다른 검증입니다. 이 명령은 실행 안내이며 새 통과 기록을 뜻하지 않습니다. 과거 테스트 횟수와 예시 점수는 [검증 기록](validation.md)의 해당 날짜와 버전에 한정합니다. 저장된 11편 연구의 창 선택 결과도 새 CLI 추론 검증을 대신하지 않습니다.

계산 정의는 [계산 방법](methods.md), 이전 실행 경로와의 차이는 [이전 코드와의 차이](migration.md), 입력 자료의 출처는 [공 추적 출처 감사](touch-source-audit.md)에 정리했습니다.

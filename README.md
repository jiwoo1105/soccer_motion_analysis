<div align="center">

# ⚽ Soccer Dribble Motion Analysis

**단안 카메라 기반 축구 드리블 동작 정량 평가 시스템**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)](https://mediapipe.dev/)
[![SAM2](https://img.shields.io/badge/SAM2-Meta-purple.svg)](https://github.com/facebookresearch/sam2)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Depth Anything V2](https://img.shields.io/badge/Depth_Anything_V2-ViT--L-red.svg)](https://github.com/DepthAnything/Depth-Anything-V2)

</div>

---

## 📖 목차

- [프로젝트 소개](#-프로젝트-소개)
- [시스템 파이프라인](#-시스템-파이프라인)
- [사용 모델](#-사용-모델)
- [공 터치 감지](#-공-터치-감지)
- [평가 지표](#-평가-지표)
- [결과 예시](#-결과-예시)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [참고 자료](#-참고-자료)

---

## 🎯 프로젝트 소개

스마트폰 **1대**로 촬영한 드리블 영상만으로, 선수의 **헤드업 각도**, **어깨 회전**, **골반 회전**을 자동으로 측정하고 정량 평가하는 시스템입니다.

### 왜 이 프로젝트인가?

- 기존 드리블 평가는 코치의 **주관적 판단**에 의존
- 모션 캡쳐 장비는 **비용이 높고** 현장 적용이 어려움
- **카메라 1대 + AI**만으로 정량적 자세 평가 가능

### Demo

<!-- 데모 영상/GIF 추가 -->
```
[터치 감지 trace 영상 GIF 추가 예정]
```

<!-- 스켈레톤 오버레이 스크린샷 추가 -->
| 스켈레톤 + 공 추적 | 터치 감지 |
|:---:|:---:|
| ![skeleton](<!-- 스크린샷 경로 -->) | ![touch](<!-- 스크린샷 경로 -->) |

---

## 🔄 시스템 파이프라인

```
📹 드리블 영상 (스마트폰 1대)
    │
    ├─ ① YOLOv8 ──→ 첫 20프레임에서 발목에 가장 가까운 공 감지
    │                  │
    ├─ ② SAM2 ────→ 초기 공 좌표를 받아 전체 영상 mask 추적
    │                  → 프레임별 공 중심(x, y) + 반지름(r)
    │
    ├─ ③ Mediapipe Pose ──→ 33개 관절 2D/3D 좌표 추출
    │
    ├─ ④ 터치 감지 ──→ 공 X 궤적 방향 전환 + 발목 근접
    │                    → 터치 프레임 + 어느 발인지 판별
    │
    ├─ ⑤ 평가 지표 측정
    │     ├─ 헤드업: 터치 ±8프레임 head angle range
    │     ├─ 어깨 회전: 터치 구간 ±8프레임 median 비교
    │     └─ 골반 회전: 터치 구간 ±8프레임 median 비교
    │
    └─ ⑥ 출력
          ├─ 정량 평가 결과 (headup°, shoulder°, pelvis°)
          └─ trace 영상 (터치 시점 표시)
```

---

## 🤖 사용 모델

| 모델 | 역할 | 입력 → 출력 |
|:---:|:---:|:---|
| **YOLOv8** | 초기 공 감지 | 영상 프레임 → 공 바운딩 박스 |
| **SAM2** | 공 전체 추적 | 초기 좌표 + 영상 → 프레임별 공 mask |
| **Mediapipe Pose** | 관절 좌표 추출 | 영상 → 33개 관절 2D/3D 좌표 |
| **Depth Anything V2** | 깊이 추정 | 영상 프레임 → pixel별 depth map |

### YOLOv8 + SAM2 공 추적

```
YOLOv8이 첫 20프레임에서 공 후보 감지
    ↓
Mediapipe 발목 좌표와 가장 가까운 공 선택 (드리블 중인 공)
    ↓
SAM2에 초기 좌표 전달 → 전체 영상 mask 기반 추적
    ↓
가림(occlusion) 상황에서도 안정적 추적
```

<!-- SAM2 공 추적 결과 스크린샷 -->
```
[SAM2 공 추적 결과 이미지 추가 예정]
```

### Mediapipe Pose

```
33개 landmark 추출 (프레임별)
    ├─ image landmarks: 2D 좌표 (0-1 normalized) → 터치 판별에 사용
    └─ world landmarks: 3D 좌표 (골반 기준 상대좌표) → 회전 측정에 사용
```

### Depth Anything V2

```
영상 프레임 → ViT-L 기반 depth map 추출
    → pixel별 상대적 깊이값
    → 공/발목의 깊이 비교 가능 (향후 터치 정확도 향상에 활용)
```

<!-- Depth map 시각화 -->
| 원본 프레임 | Depth Map |
|:---:|:---:|
| ![original](<!-- 경로 -->) | ![depth](<!-- 경로 -->) |

---

## ⚽ 공 터치 감지

### 핵심 아이디어

드리블 = 공을 **좌 ↔ 우 반복 이동**. 공의 X 좌표가 방향 전환하는 순간 = 터치.

```
공 X 좌표 시계열:

        ● peak(극대)          ● peak(극대)
       / \                  / \
      /   \                /   \
     /     \              /     \
    /       \            /       \
  ●/    T1   \● trough  /   T3   \●
              (극소) T2             T4

극대(●) = 공이 →에서 ←로 방향 전환 = 터치
극소(●) = 공이 ←에서 →로 방향 전환 = 터치
```

### 처리 과정

1. **SAM2 공 X 좌표** 시계열 수집
2. **Savitzky-Golay filter** (window=11)로 노이즈 제거
3. **find_peaks**로 극대/극소 감지 (prominence=10, distance=8)
4. 전환 지점에서 **발목과 공의 2D 거리 < 200px** 검증
5. 가까운 발 = 터치한 발 (left / right)

### 가정

- 한 방향에 터치 **1개만** 존재 (오른발 터치 → 왼발 터치 → 오른발 터치)
- 중간에 추가 터치 없음

<!-- 터치 감지 trace 영상 스크린샷 -->
| Touch #1 | Touch #2 | Touch #3 |
|:---:|:---:|:---:|
| ![t1](<!-- 경로 -->) | ![t2](<!-- 경로 -->) | ![t3](<!-- 경로 -->) |

---

## 📊 평가 지표

### 1. 헤드업 (Head-up Angle Range)

> 터치 시점 ±8프레임에서 머리를 얼마나 들었다 숙였다 하는가

```
측정: 어깨 중앙 → 눈 중앙 벡터와 수직축(Y)의 각도
평가: 구간 내 (max - min) = range
최종: 모든 터치의 range 평균
의미: 값이 클수록 → 주변을 많이 살핌 (좋음)
```

<!-- 헤드업 측정 시각화 -->
```
[헤드업 벡터 시각화 이미지 추가 예정]
```

### 2. 어깨 회전 (Shoulder Rotation)

> 터치 구간에서 어깨가 얼마나 회전했는가

```
측정: R_SHOULDER(12) - L_SHOULDER(11) 벡터의 XZ 방향각
방법: T1 ±8프레임 median vs T2 ±8프레임 median 비교
      → |median 차이| = 회전각
보정: vec_x에 savgol filter(window=21) 적용 (X 노이즈 안정화)
의미: 값이 클수록 → 상체를 많이 활용 (좋음)
```

### 3. 골반 회전 (Pelvis Rotation)

> 어깨와 동일한 방식, R_HIP(24) - L_HIP(23) 벡터 사용

```
의미: 값이 클수록 → 하체 회전이 활발 (좋음)
```

### X 노이즈 문제와 해결

Mediapipe world_landmarks의 **X 좌표가 프레임마다 급변**하는 문제가 있었습니다.

| 문제 | 원인 | 해결 |
|:---:|:---:|:---|
| 1프레임에 ~38° 점프 | 개별 landmark X 위치 불안정 | vec_x에 **savgol filter** 적용 |
| 절대 각도 부정확 | Z >> X라 arctan2가 X에 과민 | 터치 ±8프레임 **median**으로 이상치 무시 |
| 스파이크 잔존 | savgol로도 완전 제거 불가 | 여러 터치 구간 **평균**으로 상쇄 |

<!-- XZ 평면 회전 비교 그래프 -->
```
[어깨 회전 XZ 비교 그래프 추가 예정]
```

---

## 📈 결과 예시

### 정량 평가 테이블

```
영상              헤드업    어깨회전   골반회전   터치수
──────────────────────────────────────────────────
인,인 3-1          9.9°    54.0°    52.7°      4
인,인 3-2          8.4°    54.4°    66.7°      4
인,인 5-1          8.7°    23.7°    39.9°      4
인,인 7-1          8.5°    46.7°    62.0°      5
인,인 7-2         11.0°    38.0°    50.5°      8
인,인 9-1         11.5°    52.4°    69.1°      6
```

### 출력물

| 출력 | 설명 |
|:---|:---|
| **정량 평가 결과** | headup(°), shoulder(°), pelvis(°), touch_count |
| **touch trace 영상** | 터치 시점 "TOUCH #N" 표시된 영상 (.mp4) |
| **depth map** | Depth Anything V2 pixel별 깊이 시각화 |
| **스켈레톤 영상** | 33개 관절 + 공 추적 오버레이 영상 |

<!-- 결과 그래프/영상 스크린샷 -->
| trace 영상 | depth map |
|:---:|:---:|
| ![trace](<!-- 경로 -->) | ![depth](<!-- 경로 -->) |

---

## 🔧 설치 방법

### 요구사항

- **Python**: 3.9+
- **OS**: macOS (Apple Silicon MPS 지원), Linux
- **RAM**: 8GB 이상 권장
- **저장공간**: ~2GB (모델 체크포인트 포함)

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/jiwoo1105/soccer_motion_analysis.git
cd soccer_motion_analysis

# 2. 패키지 설치
pip install mediapipe opencv-python numpy scipy matplotlib

# 3. SAM2 설치 (Python 3.11 필요)
pip3.11 install sam2

# 4. Depth Anything V2 설치
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2_repo

# 5. 모델 체크포인트 다운로드
mkdir -p checkpoints
# Depth Anything V2 Large (1.2GB)
wget -O checkpoints/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

# YOLOv8 (자동 다운로드됨)
# SAM2 체크포인트 (sam2_checkpoints/ 디렉토리에 배치)
```

---

## 🚀 사용 방법

### 1. 영상 준비

```bash
input/
  └── in_in/
      ├── 인,인 3-1.MOV
      ├── 인,인 5-1.MOV
      └── ...
```

### 2. 실행

```bash
# 전체 영상 분석 (터치 감지 + 평가 지표 + trace 영상)
python extract_depth_metrics.py

# 단일 영상 trace 확인
python -c "
from extract_depth_metrics import process
r = process('input/in_in/인,인 3-1.MOV', save_trace=True)
print(r)
"
```

### 3. 결과 확인

```bash
output/
  └── videos/
      ├── touch_trace_인,인 3-1.mp4     # 터치 시점 표시 영상
      ├── touch_trace_인,인 5-1.mp4
      └── ...
```

---

## 🗂️ 프로젝트 구조

```
soccer_motion_analysis/
│
├── extract_depth_metrics.py      # 🎯 메인 실행 (터치 감지 + 평가)
├── config.py                     # ⚙️ Mediapipe 설정
│
├── core/                         # 🧠 핵심 모듈
│   ├── pose_extractor.py         # Mediapipe 포즈 추출
│   └── depth_estimator.py        # Depth Anything V2 래퍼
│
├── analysis/                     # 📐 분석 모듈
│   ├── ball_motion_analyzer.py   # 공 추적 + 터치 감지 (기존)
│   ├── head_pose_analyzer.py     # 헤드업 각도 분석
│   └── cycle_analyzer.py         # 사이클 기반 분석
│
├── visualization/                # 📊 시각화 모듈
│   ├── skeleton_drawer.py        # 스켈레톤 + 공 + 터치 표시
│   └── cycle_plotter.py          # 사이클 그래프
│
├── utils/                        # 🛠️ 유틸리티
│   └── math_utils.py             # 벡터/각도 계산
│
├── sam2_ball_tracker.py          # SAM2 공 추적 스크립트
├── depth_anything_v2_repo/       # Depth Anything V2 모델
├── checkpoints/                  # 모델 체크포인트
├── sam2_checkpoints/             # SAM2 체크포인트
│
├── input/                        # 📂 입력 영상
└── output/                       # 📂 출력 결과
    ├── videos/                   # trace 영상
    └── graphs/                   # 분석 그래프
```

---

## 📚 참고 자료

### 사용 기술

| 기술 | 용도 | 링크 |
|:---|:---|:---|
| MediaPipe Pose | 33개 관절 3D 포즈 추정 | [mediapipe.dev](https://mediapipe.dev/) |
| SAM2 | 세그멘테이션 기반 공 추적 | [GitHub](https://github.com/facebookresearch/sam2) |
| YOLOv8 | 실시간 공 감지 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| Depth Anything V2 | 단안 깊이 추정 | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| OpenCV | 영상 처리 | [opencv.org](https://opencv.org/) |
| SciPy | 신호 처리 (savgol, find_peaks) | [scipy.org](https://scipy.org/) |

### 관련 연구

- [SoccerNet-Depth (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Leduc_SoccerNet-Depth_a_Scalable_Dataset_for_Monocular_Depth_Estimation_in_Sports_CVPRW_2024_paper.pdf) — 축구 영상 특화 depth estimation
- [Where Is The Ball (CVPR 2025)](https://arxiv.org/abs/2506.05763) — 단안 2D 추적에서 3D 공 궤적 복원

---

## 📧 Contact

**개발자**: PARK JIWOO

- GitHub: [github.com/jiwoo1105](https://github.com/jiwoo1105)
- Email: psskej00@daum.net

---

<div align="center">

Made with ⚽ by PARK JIWOO

</div>

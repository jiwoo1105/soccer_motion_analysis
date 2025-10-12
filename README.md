# Soccer Motion Analysis

축구 드리블 영상의 3D 모션 분석 및 비교 시스템

## 📁 프로젝트 구조

```
soccer_motion_analysis/
├── main.py                          # 메인 실행 파일
├── config.py                        # 설정 및 상수
├── requirements.txt                 # 패키지 의존성
│
├── core/                           # 핵심 모듈
│   ├── pose_extractor.py          # MediaPipe 3D 포즈 추출
│   ├── video_processor.py         # 비디오 읽기/쓰기
│   └── coordinate_system.py       # 좌표계 변환/정규화
│
├── analysis/                       # 분석 모듈
│   ├── segment_analyzer.py        # Body Segment 각도 분석
│   ├── joint_analyzer.py          # 관절 각도 분석
│   └── comparison.py              # 특성 기반 비교
│
├── utils/                         # 유틸리티
│   └── math_utils.py              # 수학 함수
│
├── visualization/                 # 시각화
│   ├── skeleton_drawer.py         # 스켈레톤 그리기
│   └── angle_plotter.py           # 각도 그래프
│
└── output/                        # 출력 결과
    ├── data/                      # CSV 데이터
    ├── reports/                   # 텍스트 리포트
    └── plots/                     # 그래프 이미지
```

## 🚀 시작하기

### 1. 패키지 설치

```bash
cd soccer_motion_analysis
pip install -r requirements.txt --break-system-packages
```

### 2. 실행

```bash
python main.py
```

## 📊 분석 내용

### 측정 항목

**Body Segments (분절 각도)**
- Trunk (몸통): 어깨-엉덩이 각도
- Thigh (허벅지): 엉덩이-무릎 각도
- Shank (정강이): 무릎-발목 각도
- Foot (발): 발뒤꿈치-발끝 각도

**Joints (관절 각도)**
- Knee (무릎): hip-knee-ankle
- Hip (고관절): shoulder-hip-knee
- Ankle (발목): knee-ankle-foot

### 비교 방식

프레임별 비교가 아닌 **특성 기반 비교**:
- 평균 각도
- 최대/최소 각도
- 가동 범위 (ROM)
- 표준편차
- 백분위수

## 📈 출력 결과

### 1. 비교 테이블 (CSV)
`output/data/comparison_table.csv`

각 관절/분절의 평균, 범위, 차이 등을 테이블로 정리

### 2. 텍스트 리포트
`output/reports/comparison_report.txt`

- 드리블 스타일 분류
- 주요 발견사항
- 개선 제안

### 3. 시각화 그래프
`output/plots/`

- `*_comparison.png`: 시계열 비교 그래프
- `*_distribution.png`: 각도 분포 히스토그램

## 🎯 사용 예시

### 기본 사용

```python
from core.pose_extractor import PoseExtractor
from analysis.segment_analyzer import SegmentAnalyzer
from analysis.comparison import MotionComparison

# 1. 포즈 추출
extractor = PoseExtractor()
poses = extractor.extract_from_video("video.mp4")

# 2. 각도 계산
analyzer = SegmentAnalyzer()
angles = [analyzer.calculate_all_segments(p.world_landmarks) for p in poses]

# 3. 비교
comparator = MotionComparison(motion1_data, motion2_data)
comparison = comparator.compare_characteristics()
```

### 설정 변경

`config.py`에서 다양한 설정 조정 가능:
- MediaPipe 모델 복잡도
- 신뢰도 임계값
- 시각화 색상/스타일
- 해석 임계값

## 📝 결과 해석

### 무릎 각도 예시

```
Left Knee Mean: 145° (Video 1) vs 152° (Video 2)
→ Video 2가 7° 더 큼 (더 높은 자세)
```

### 스타일 분류

- **Low stance (<145°)**: 낮은 자세, 공격적
- **High stance (≥145°)**: 높은 자세, 안정적

### 가동 범위 (ROM)

```
Left Knee ROM: 45° (Video 1) vs 40° (Video 2)
→ Video 1이 더 다이나믹한 움직임
```

## ⚠️ 주의사항

1. **영상 품질**: 사람이 명확히 보이는 영상 필요
2. **카메라 각도**: 측면 또는 정면에서 촬영된 영상 권장
3. **조명**: 충분한 조명이 있는 환경
4. **가림**: 사람이 가려지지 않아야 함

## 🔧 문제 해결

### "No poses detected"
- 영상에서 사람이 제대로 보이지 않음
- MediaPipe 신뢰도 설정을 낮춤 (config.py)

### 각도 값이 이상함
- 좌표계 정규화 확인
- World landmarks 사용 확인

## 📚 참고 문헌

이 프로젝트는 다음 연구를 참고했습니다:
- Dribbling determinants in sub-elite youth soccer players (2015)
- Biomechanical characteristics for identifying cutting direction (2021)
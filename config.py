# Legacy pipeline configuration. Current scoring uses experiments/2026-09-20/calibration.json and scoring/current_metrics.py.
# config.py
"""축구 드리블 스킬 평가 시스템 설정"""

MEDIAPIPE_CONFIG = {
    'model_complexity': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
}

BALL_DETECTION_CONFIG = {
    'model_name': 'yolov8n.pt',
    'confidence': 0.3,
    'max_distance_threshold': 150.0,
    'history_size': 5,
    'smoothing_alpha': 0.3,
    # 꼬깔콘 필터링은 코드에 내장 (색상 + 크기 자동 필터링)
}

# 공 움직임 분석 설정 (거리×속도 결합 신호 기반 터치 감지)
BALL_MOTION_CONFIG = {
    'min_distance_between_touches': 10,  # 터치 간 최소 프레임 거리 (30fps 기준 0.33초)
    'peak_prominence': 0.1,              # 결합 신호 민감도 (0~1 범위)
    'max_touch_distance': 150,           # 터치로 인정할 최대 발-공 거리 (px)
}

# MediaPipe Landmark 인덱스
LANDMARK_INDICES = {
    'NOSE': 0,
    'LEFT_EYE': 2,
    'RIGHT_EYE': 5,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}

# 드리블 스킬 평가 설정
SKILL_EVALUATION = {
    # 각 기준별 가중치 (합계 = 1.0)
    'weights': {
        'head_up': 0.15,              # 1. 헤드업 (시선)
        'torso_coordination': 0.40,   # 2. 상체-하체 협응성 (어깨+골반)
        'trunk_lean': 0.15,           # 3. 상체 낮춤
        'knee_flexion': 0.20,         # 4. 무릎 구부림
        'ankle_agility'  : 0.10         # 5. 발목 민첩성
    },

    # 1. 헤드업: 고개 각도 (수평 기준)
    'head_up': {
        'optimal_range': (-10, 10),   # 최적: 수평 ±10도 → 10점
        'worst': 30,                  # 최악: ±30도 벗어나면 → 0점
    },

    # 2. 상체-하체 협응성: 어깨와 골반이 함께 공 방향으로 회전
    'torso_coordination': {
        # 'optimal_rom': (140, 170),    # 최적 회전 범위: 140-170도 → 10점 (ROM 사용 안함)
        # 'worst_rom': 80,              # 최악: 80도 미만 → 0점 (ROM 사용 안함)
        'optimal_correlation': 0.95,  # 최적 상관관계: 0.95 이상 → 10점 (까다롭게)
        'worst_correlation': 0.0,     # 최악: 0.0 이하 → 0점
    },

    # 3. 상체 낮춤: 전방 경사각
    'trunk_lean': {
        'optimal_range': (70, 85),   # 최적: 70-85도 → 10점
        'worst': 30,                 # 최악: ±30도 벗어나면 → 0점
    },

    # 4. 무릎 구부림: 각도 + ROM
    'knee_flexion': {
        'optimal_angle': (120, 135),  # 최적 준비자세: 120-135도 → 10점
        'worst_angle': 40,            # 최악: ±40도 벗어나면 → 0점
        'optimal_rom': (90, 120),     # 최적 ROM: 90-120도 → 10점
        'worst_rom': 40,              # 최악: ±40도 벗어나면 → 0점
    },

    # 5. 발목 민첩성: 속도 + 가속도
    'ankle_agility': {
        'optimal_speed': (0.5, 2.0),  # 최적 속도: 0.5-2.0 m/s → 10점
        'worst_speed': 3.0,           # 최악: ±3.0 m/s 벗어나면 → 0점
        'optimal_acceleration': 33.0,  # 최적 가속도: 33.0 m/s² 이상 → 10점 (기준 영상: 33.95)
        'worst_acceleration': 0.0,    # 최악: 0.0 m/s² → 0점
    }
}

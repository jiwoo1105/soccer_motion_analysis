# analysis/head_pose_analyzer.py
"""머리 자세 분석 모듈 - head_vector 각도 계산"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from utils.math_utils import angle_with_vertical


@dataclass
class HeadPoseData:
    """머리 자세 분석 결과"""
    frame_numbers: np.ndarray           # 분석된 프레임 번호들
    head_angles: np.ndarray             # 각 프레임의 머리 각도 (degrees)
    mean_angle: float                   # 평균 머리 각도
    sum_squared_deviations: float       # Σ(각도 - 평균)²
    touch_frames: list                  # 터치 순간 프레임 번호들 (그래프 표시용)
    touch_window_data: list = None      # 터치 전후 ±8프레임 각도 변화율 데이터
    touch_window_mean_angle: float = None  # 터치 윈도우 전체 각도 평균
    touch_window_mean_range: float = None  # 터치별 (최대-최소) 평균

    def __str__(self):
        result = []
        result.append("="*70)
        result.append("머리 자세 분석 결과")
        result.append("="*70)
        result.append(f"총 분석 프레임 수: {len(self.frame_numbers)}")
        result.append(f"평균 머리 각도: {self.mean_angle:.2f}°")
        if len(self.touch_frames) > 0:
            result.append(f"터치 횟수: {len(self.touch_frames)}")
        if self.touch_window_mean_range is not None:
            result.append(f"mean_range (터치별 각도 변화폭 평균): {self.touch_window_mean_range:.2f}°")
        result.append("="*70)
        return "\n".join(result)


class HeadPoseAnalyzer:
    """머리 자세 분석 - head_vector 각도 계산"""

    # 랜드마크 인덱스
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self, min_visibility_threshold: float = 0.5):
        """
        Args:
            min_visibility_threshold: 최소 랜드마크 신뢰도 (0.5 권장)
        """
        self.min_visibility_threshold = min_visibility_threshold

    def calculate_head_angle(self, pose_frame) -> Optional[float]:
        """
        단일 프레임에서 머리 각도 계산

        Args:
            pose_frame: PoseFrame 객체 (world_landmarks 사용)

        Returns:
            머리 각도 (degrees) 또는 None (신뢰도 낮을 경우)

        계산 방법:
            1. 눈 중앙 = (왼쪽 눈 + 오른쪽 눈) / 2
            2. 어깨 중앙 = (왼쪽 어깨 + 오른쪽 어깨) / 2
            3. head_vector = 눈 중앙 - 어깨 중앙
            4. 각도 = angle_with_vertical(head_vector)
        """
        # Visibility 체크
        required_landmarks = [self.LEFT_EYE, self.RIGHT_EYE,
                             self.LEFT_SHOULDER, self.RIGHT_SHOULDER]
        visibilities = [pose_frame.visibility[idx] for idx in required_landmarks]

        if any(v < self.min_visibility_threshold for v in visibilities):
            return None

        # world_landmarks 사용 (3D 실제 좌표)
        world_landmarks = pose_frame.world_landmarks

        # 눈 중앙 계산
        left_eye = world_landmarks[self.LEFT_EYE]
        right_eye = world_landmarks[self.RIGHT_EYE]
        eye_center = (left_eye + right_eye) / 2

        # 어깨 중앙 계산
        left_shoulder = world_landmarks[self.LEFT_SHOULDER]
        right_shoulder = world_landmarks[self.RIGHT_SHOULDER]
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # head_vector: 어깨에서 눈으로의 벡터 (머리 방향)
        head_vector = eye_center - shoulder_center

        # 수직선 대비 각도 계산
        angle = angle_with_vertical(head_vector)

        return angle

    def analyze(self, pose_frames, ball_motion_data=None):
        """
        PoseFrame 리스트에서 머리 자세 분석

        Args:
            pose_frames: PoseFrame 객체 리스트
            ball_motion_data: BallMotionData 객체 (옵션, 터치 프레임 표시용)

        Returns:
            HeadPoseData 객체 또는 None (유효한 프레임이 없을 경우)
        """
        if len(pose_frames) == 0:
            return None

        # 각 프레임의 머리 각도 계산
        valid_data = []
        for pf in pose_frames:
            angle = self.calculate_head_angle(pf)
            if angle is not None:
                valid_data.append({
                    'frame_number': pf.frame_number,
                    'angle': angle
                })

        if len(valid_data) == 0:
            return None

        # 데이터 추출
        frame_numbers = np.array([d['frame_number'] for d in valid_data])
        head_angles = np.array([d['angle'] for d in valid_data])

        # 평균 각도 계산
        mean_angle = np.mean(head_angles)

        # 편차 제곱합 계산: Σ(각도 - 평균)²
        deviations = head_angles - mean_angle
        sum_squared_deviations = np.sum(deviations ** 2)

        # 터치 프레임 추출 (그래프 표시용)
        touch_frames = []
        if ball_motion_data:
            touch_frames = ball_motion_data.touch_frames

        # 터치 전후 ±8프레임 각도 변화율 계산
        touch_window_data, touch_window_mean_angle, touch_window_mean_range = \
            self._compute_touch_window(frame_numbers, head_angles, touch_frames)

        return HeadPoseData(
            frame_numbers=frame_numbers,
            head_angles=head_angles,
            mean_angle=mean_angle,
            sum_squared_deviations=sum_squared_deviations,
            touch_frames=touch_frames,
            touch_window_data=touch_window_data,
            touch_window_mean_angle=touch_window_mean_angle,
            touch_window_mean_range=touch_window_mean_range,
        )

    def _compute_touch_window(self, frame_numbers: np.ndarray, head_angles: np.ndarray,
                               touch_frames: list, window: int = 8):
        """각 터치 전후 ±window 프레임의 머리 각도 변화율 계산

        Args:
            frame_numbers: 분석된 프레임 번호 배열
            head_angles: 각 프레임의 머리 각도 배열
            touch_frames: 터치 프레임 번호 리스트
            window: 터치 전후 프레임 수 (기본값 8)

        Returns:
            tuple: (touch_window_data, touch_window_mean_angle, touch_window_mean_range)

            touch_window_data: list of dict [
                {
                    'touch_num': int,       # 터치 번호 (1-based)
                    'touch_frame': int,     # 터치 프레임 번호
                    'rel_frames': list,     # 상대 프레임 인덱스 (-5 ~ +5)
                    'angles': list,         # 각 상대 프레임의 머리 각도
                    'rates': list,          # 각 상대 프레임의 변화율 (deg/frame)
                    'angle_range': float,   # 해당 터치 윈도우의 최대-최소 각도
                }
            ]
            touch_window_mean_angle: 전체 터치 윈도우 각도 평균
            touch_window_mean_range: 터치별 (최대-최소)의 평균 → 점수 계산 기준
            touch_window_mean_range: 터치별 변화폭 평균

        점수 기준:
            각 터치 윈도우(±8프레임)에서 max - min = 각도 변화폭
            이를 모든 터치에 대해 평균 → mean_range
            mean_range가 작을수록 안정적인 헤드업 자세 → 높은 점수
              0°   → 10점  (변화 없음 = 머리 고정)
              30°+ →  0점  (변화폭 큼 = 머리 흔들림)
            score = max(0, (30 - mean_range) / 30 * 10)
        """
        angle_map = {int(f): float(a) for f, a in zip(frame_numbers, head_angles)}
        result = []
        all_window_angles = []
        per_touch_ranges = []

        for i, tf in enumerate(touch_frames):
            tf = int(tf)
            rel_frames = []
            angles = []

            for rel in range(-window, window + 1):
                frame = tf + rel
                if frame in angle_map:
                    rel_frames.append(rel)
                    angles.append(angle_map[frame])

            if len(angles) < 3:
                continue

            all_window_angles.extend(angles)

            # 이 터치 윈도우의 각도 변화폭 (최대 - 최소)
            angle_range = float(max(angles) - min(angles))
            per_touch_ranges.append(angle_range)

            # 중앙 차분으로 변화율 계산 (deg/frame)
            rates = list(np.gradient(angles, rel_frames))

            result.append({
                'touch_num': i + 1,
                'touch_frame': tf,
                'rel_frames': rel_frames,
                'angles': angles,
                'rates': rates,
                'angle_range': angle_range,
            })

        # 전체 윈도우 통계
        if all_window_angles and per_touch_ranges:
            touch_window_mean_angle = float(np.mean(all_window_angles))
            touch_window_mean_range = float(np.mean(per_touch_ranges))
        else:
            touch_window_mean_angle = None
            touch_window_mean_range = None

        return result, touch_window_mean_angle, touch_window_mean_range

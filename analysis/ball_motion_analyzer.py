# analysis/ball_motion_analyzer.py
"""공의 움직임 분석 및 터치 순간 감지 모듈

터치 감지 방식:
- 공의 2D 위치와 양 발목의 2D 위치를 비교
- 거리가 가장 가까워지는 순간(극소값) = 터치
- find_peaks로 거리 극소값 감지

3D 방향 계산:
- 터치 순간 공의 Z 좌표 = 터치한 발목의 Z 좌표
- 연속 터치 간의 3D 방향 벡터 계산
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.signal import find_peaks
from dataclasses import dataclass, field


class BallKalmanFilter:
    """2D 공 추적용 Kalman Filter (위치 + 반지름 동시 추적)

    State:  [x, y, vx, vy, r, vr]  (위치 + 속도 + 반지름 + 반지름 변화율)
    Obs:    [x, y, r]               (YOLO 감지 위치 + 반지름)

    - 감지된 프레임: predict → update (실제값으로 보정)
    - 감지 안 된 프레임: predict만 (속도/변화율 기반 예측)

    반지름은 등속 변화 모델 (vr 일정) 사용.
    공은 카메라와의 거리가 천천히 변하므로 반지름도 완만하게 변함.
    """

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 5.0,
                 radius_process_noise: float = 0.5, radius_measurement_noise: float = 3.0):
        dt = 1.0
        # 상태 전이 행렬: [x, y, vx, vy, r, vr]
        self.F = np.array([[1, 0, dt, 0,  0,  0 ],
                           [0, 1,  0, dt, 0,  0 ],
                           [0, 0,  1,  0, 0,  0 ],
                           [0, 0,  0,  1, 0,  0 ],
                           [0, 0,  0,  0, 1,  dt],
                           [0, 0,  0,  0, 0,  1 ]], dtype=float)
        # 관측 행렬: x, y, r만 관측
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0]], dtype=float)
        # 프로세스 노이즈: 위치/속도 vs 반지름 분리 설정
        self.Q = np.diag([process_noise, process_noise,
                          process_noise, process_noise,
                          radius_process_noise, radius_process_noise])
        # 측정 노이즈: 위치 vs 반지름 분리 설정
        self.R = np.diag([measurement_noise, measurement_noise,
                          radius_measurement_noise])
        self.P = np.eye(6) * 100.0
        self.x = np.zeros(6)

    def initialize(self, x: float, y: float, r: float = 18.0) -> None:
        self.x = np.array([x, y, 0.0, 0.0, r, 0.0])
        self.P = np.eye(6) * 100.0

    def predict(self) -> np.ndarray:
        """예측 후 [x, y, r] 반환"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # 반지름은 양수 유지
        self.x[4] = max(self.x[4], 4.0)
        return np.array([self.x[0], self.x[1], self.x[4]])

    def update(self, measurement: Tuple[float, float, float]) -> np.ndarray:
        """측정값 [x, y, r]로 보정 후 [x, y, r] 반환"""
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[4] = max(self.x[4], 4.0)
        return np.array([self.x[0], self.x[1], self.x[4]])


# MediaPipe 랜드마크 인덱스
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass
class TouchEvent:
    """단일 터치 이벤트 정보"""
    frame_number: int
    timestamp: float
    ball_2d: Tuple[float, float]  # 공의 2D 위치 (x, y)
    ball_3d: Tuple[float, float, float]  # 공의 3D 위치 (x, y, z) - z는 발목에서 가져옴
    touching_foot: str  # 'left' or 'right'
    ankle_2d: Tuple[float, float]  # 터치한 발목의 2D 위치
    ankle_z: float  # 터치한 발목의 Z 좌표 (world_landmarks)
    distance: float  # 공과 발목 사이의 2D 거리
    shoulder_direction: Tuple[float, float, float]  # 어깨 라인 방향 (right - left)
    pelvis_direction: Tuple[float, float, float]    # 골반 라인 방향 (right - left)

    def __str__(self):
        return (f"Frame {self.frame_number}: {self.touching_foot}발 터치, "
                f"공 위치=({self.ball_2d[0]:.0f}, {self.ball_2d[1]:.0f}), "
                f"거리={self.distance:.1f}px")


@dataclass
class TouchDirection:
    """두 터치 사이의 3D 방향 정보"""
    from_touch: TouchEvent
    to_touch: TouchEvent
    direction_2d: Tuple[float, float]  # (dx, dy)
    direction_3d: Tuple[float, float, float]  # (dx, dy, dz) - 공 이동 벡터 (A)
    distance_2d: float
    distance_3d: float
    shoulder_rotation_angle: float  # XZ 평면 어깨 회전각 (degrees)
    pelvis_rotation_angle: float    # XZ 평면 골반 회전각 (degrees)

    def __str__(self):
        return (f"Frame {self.from_touch.frame_number} → {self.to_touch.frame_number}: "
                f"어깨 회전={self.shoulder_rotation_angle:.1f}°, "
                f"골반 회전={self.pelvis_rotation_angle:.1f}°")


@dataclass
class BallMotionData:
    """공의 움직임 분석 결과"""
    frame_numbers: np.ndarray
    positions: np.ndarray  # 공의 2D 위치
    touch_frames: List[int]
    touch_count: int
    # 터치 관련
    touch_events: List[TouchEvent] = field(default_factory=list)
    touch_directions: List[TouchDirection] = field(default_factory=list)
    ankle_distances: Optional[np.ndarray] = None      # 매 프레임 공-발목 거리
    interp_positions: Optional[dict] = field(default_factory=dict)   # {frame_num: (x,y,r)} SAM2 미감지 → 선형 보간 프레임
    all_ball_positions: Optional[dict] = field(default_factory=dict) # {frame_num: (x,y,r)} 전체 프레임 (SAM2 감지 + 선형 보간 통합)

    def __str__(self):
        result = []
        result.append("="*70)
        result.append("공 움직임 분석 결과")
        result.append("="*70)
        result.append(f"총 프레임 수: {len(self.frame_numbers)}")
        result.append(f"터치 횟수: {self.touch_count}")
        result.append(f"터치 프레임: {self.touch_frames}")

        if self.touch_events:
            result.append("\n" + "-"*50)
            result.append("터치 이벤트 상세:")
            result.append("-"*50)
            for i, touch in enumerate(self.touch_events):
                result.append(f"  [{i+1}] {touch}")

        if self.touch_directions:
            result.append("\n" + "-"*50)
            result.append("터치 구간별 상체 회전량 (XZ 평면):")
            result.append("-"*50)
            for i, direction in enumerate(self.touch_directions):
                result.append(f"  Frame {direction.from_touch.frame_number} → {direction.to_touch.frame_number}:")
                result.append(f"    어깨 회전: {direction.shoulder_rotation_angle:.1f}°")
                result.append(f"    골반 회전: {direction.pelvis_rotation_angle:.1f}°")

            if len(self.touch_directions) > 0:
                avg_shoulder = sum(d.shoulder_rotation_angle for d in self.touch_directions) / len(self.touch_directions)
                avg_pelvis = sum(d.pelvis_rotation_angle for d in self.touch_directions) / len(self.touch_directions)
                rotation_score = (avg_shoulder + avg_pelvis) / 2
                result.append(f"\n  평균 어깨 회전: {avg_shoulder:.1f}°")
                result.append(f"  평균 골반 회전: {avg_pelvis:.1f}°")
                result.append(f"  상체 회전 점수: {rotation_score:.1f}°")

        result.append("="*70)
        return "\n".join(result)


class BallMotionAnalyzer:
    """공의 움직임 분석 및 터치 순간 감지

    터치 감지 알고리즘 (거리 기반):
    1. 매 프레임마다 공과 양 발목의 2D 거리 계산
    2. 거리가 극소인 지점(find_peaks) = 터치 순간
    3. 터치 순간의 공 Z좌표는 터치한 발목의 Z좌표로 대체
    4. 연속 터치 간 3D 방향 벡터 계산
    """

    def __init__(self,
                 min_distance_between_touches: int = 5,
                 peak_prominence: float = 15.0,
                 max_touch_distance: float = 150.0,
                 **kwargs):

        self.min_distance_between_touches = min_distance_between_touches
        self.peak_prominence = peak_prominence
        self.max_touch_distance = max_touch_distance

    def _fill_gaps_rts(self, det_frame_nums: np.ndarray,
                       det_positions: np.ndarray,
                       all_frames: np.ndarray,
                       det_radii: Optional[np.ndarray] = None):
        """RTS (Rauch-Tung-Striebel) Smoother로 전체 프레임 위치·반지름 스무딩

        오프라인 처리이므로 Forward + Backward 두 패스 모두 가능.

        Forward pass (일반 Kalman):
          - 감지 프레임: predict → update
          - 미감지 프레임: predict only
          - 각 프레임의 x_filt, P_filt, x_pred, P_pred 저장

        Backward pass (RTS):
          - 마지막 프레임부터 역방향으로 미래 정보를 반영해 과거 추정 개선
          - G[t] = P_filt[t] @ F^T @ inv(P_pred[t+1])   (스무더 게인)
          - x_s[t] = x_filt[t] + G[t] @ (x_s[t+1] - x_pred[t+1])
          - P_s[t] = P_filt[t] + G[t] @ (P_s[t+1] - P_pred[t+1]) @ G[t]^T

        결과: lag 없는 최적 스무딩 (Kalman forward만 했을 때보다 항상 더 정확)
        """
        N = len(all_frames)
        init_r = float(det_radii[0]) if det_radii is not None else 18.0

        kf = BallKalmanFilter(process_noise=2.0, measurement_noise=5.0)
        kf.initialize(float(det_positions[0, 0]), float(det_positions[0, 1]), r=init_r)

        det_lookup = {int(f): i for i, f in enumerate(det_frame_nums)}

        # Forward pass: 상태·공분산 저장
        x_filt = np.zeros((N, 6))    # 보정된 상태 (predict+update)
        P_filt = np.zeros((N, 6, 6)) # 보정된 공분산
        x_pred = np.zeros((N, 6))    # 예측 상태 (update 전)
        P_pred = np.zeros((N, 6, 6)) # 예측 공분산 (update 전)

        for i, frame in enumerate(all_frames):
            frame = int(frame)

            # predict
            xp = kf.F @ kf.x
            Pp = kf.F @ kf.P @ kf.F.T + kf.Q
            x_pred[i] = xp
            P_pred[i] = Pp
            kf.x = xp
            kf.P = Pp

            # update (감지 프레임만)
            if frame in det_lookup:
                idx = det_lookup[frame]
                r   = float(det_radii[idx]) if det_radii is not None else 18.0
                z   = np.array([det_positions[idx, 0], det_positions[idx, 1], r])
                inn = z - kf.H @ kf.x
                S   = kf.H @ kf.P @ kf.H.T + kf.R
                K   = kf.P @ kf.H.T @ np.linalg.inv(S)
                kf.x = kf.x + K @ inn
                kf.P = (np.eye(6) - K @ kf.H) @ kf.P

            x_filt[i] = kf.x
            P_filt[i] = kf.P

        # Backward pass (RTS)
        x_smooth = x_filt.copy()
        P_smooth = P_filt.copy()

        for i in range(N - 2, -1, -1):
            # P_pred[i+1]이 singular할 경우 대비해 pinv 사용
            G = P_filt[i] @ kf.F.T @ np.linalg.pinv(P_pred[i + 1])
            x_smooth[i] = x_filt[i] + G @ (x_smooth[i + 1] - x_pred[i + 1])
            P_smooth[i] = P_filt[i] + G @ (P_smooth[i + 1] - P_pred[i + 1]) @ G.T
            x_smooth[i, 4] = max(x_smooth[i, 4], 4.0)  # 반지름 양수 유지

        all_positions = x_smooth[:, :2]
        all_radii     = np.maximum(x_smooth[:, 4], 4.0)

        return all_positions, all_radii

    def _get_ankle_2d_position(self, landmarks: np.ndarray,
                               frame_width: int, frame_height: int,
                               ankle_idx: int) -> Tuple[float, float]:
        """랜드마크에서 발목의 2D 픽셀 좌표 추출"""
        x = landmarks[ankle_idx][0] * frame_width
        y = landmarks[ankle_idx][1] * frame_height
        return (x, y)

    def _calculate_distance(self, pos1: Tuple[float, float],
                           pos2: Tuple[float, float]) -> float:
        """두 점 사이의 2D 거리 계산"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def analyze(self, pose_frames) -> Optional[BallMotionData]:
        """PoseFrame 리스트에서 공의 움직임 분석

        Args:
            pose_frames: PoseFrame 객체 리스트

        Returns:
            BallMotionData: 분석 결과
        """
        # 공이 감지된 프레임만 필터링
        ball_frames = [(pf.frame_number, pf.ball_position, pf.timestamp, pf)
                      for pf in pose_frames
                      if pf.ball_position is not None]

        if len(ball_frames) < 5:
            return None

        det_frame_nums  = np.array([f[0] for f in ball_frames])
        det_positions   = np.array([f[1] for f in ball_frames])
        timestamps      = np.array([f[2] for f in ball_frames])
        pose_frame_list = [f[3] for f in ball_frames]

        # 감지 프레임의 공 반지름 추출 (bbox 기반)
        det_radii = np.array([
            (pf.ball_bbox[2] - pf.ball_bbox[0] + pf.ball_bbox[3] - pf.ball_bbox[1]) / 4.0
            if pf.ball_bbox is not None else 18.0
            for pf in pose_frame_list
        ])

        # ── 1. RTS Smoother로 전체 프레임 스무딩 ────────────────────────
        # SAM2 감지 위치를 입력으로, 미감지 구간은 칼만 예측으로 채움
        all_frames = np.arange(det_frame_nums[0], det_frame_nums[-1] + 1)

        all_positions, all_radii = self._fill_gaps_rts(
            det_frame_nums, det_positions, all_frames, det_radii
        )

        # ── 2. 감지 프레임 위치 추출 ─────────────────────────────────────
        det_indices_in_all = np.searchsorted(all_frames, det_frame_nums)
        smoothed_positions = all_positions[det_indices_in_all]

        # ── 4. 발목-공 거리 계산 (감지 프레임만 가능) → 전체로 보간 ──────
        ankle_distances = self._calculate_all_ankle_distances(smoothed_positions, pose_frame_list)
        ankle_dist_full = np.zeros((len(all_frames), 2))
        for col in range(2):
            ankle_dist_full[:, col] = np.interp(all_frames, det_frame_nums, ankle_distances[:, col])

        # ── 6. 터치 감지 (radius 극값 + 발-공 거리 검증) ──────────────
        touch_all_indices, touch_feet = self._detect_touches_by_radius(
            all_radii, ankle_dist_full
        )
        touch_frame_nums = all_frames[touch_all_indices]

        # ── 7. 각 터치에 대해 가장 가까운 감지 프레임 → TouchEvent 생성 ──
        touch_det_indices = [
            int(np.argmin(np.abs(det_frame_nums - tf))) for tf in touch_frame_nums
        ]

        touch_events = self._create_touch_events(
            touch_det_indices, touch_feet,
            smoothed_positions, pose_frame_list,
            det_frame_nums, timestamps,
            override_frame_numbers=touch_frame_nums,
        )
        touch_directions = self._calculate_touch_directions(touch_events)

        # 전체 프레임 위치 dict → (x, y, r), SAM2 감지·선형 보간 구분 없이 통합
        detected_set = set(det_frame_nums.tolist())
        all_ball_positions_dict = {
            int(all_frames[i]): (float(all_positions[i, 0]),
                                  float(all_positions[i, 1]),
                                  float(all_radii[i]))
            for i in range(len(all_frames))
        }
        # 보간 프레임만 따로 유지 (시각화에서 주황 원 구분용)
        interp_positions_dict = {
            f: v for f, v in all_ball_positions_dict.items()
            if f not in detected_set
        }

        return BallMotionData(
            frame_numbers=det_frame_nums,
            positions=smoothed_positions,
            touch_frames=list(touch_frame_nums),
            touch_count=len(touch_frame_nums),
            touch_events=touch_events,
            touch_directions=touch_directions,
            ankle_distances=ankle_distances,
            interp_positions=interp_positions_dict,
            all_ball_positions=all_ball_positions_dict,
        )

    def _detect_touches_by_radius(self, all_radii: np.ndarray,
                                    ankle_distances: np.ndarray) -> tuple:
        """SAM2 공 반지름 극값 + 발-공 거리로 터치 감지

        로직:
          1. radius 스무딩 (savgol)
          2. radius 극대/극소 = 공이 드리블 양 끝에 도달한 시점 (후보)
          3. 후보 중 발-공 거리 < threshold 인 것만 터치로 인정
          4. distance=20으로 너무 가까운 후보 제거 (NMS 역할)
        """
        # 극대값(카메라쪽 끝) + 극소값(반대쪽 끝) = 터치 후보
        peak_idx, _   = find_peaks( all_radii, distance=20, prominence=0.5)
        trough_idx, _ = find_peaks(-all_radii, distance=20, prominence=0.5)

        # 합쳐서 시간순 정렬
        candidates = sorted(set(peak_idx.tolist() + trough_idx.tolist()))

        # 합친 후에도 최소 20프레임 간격 적용 (NMS)
        min_gap = 20
        filtered_candidates = []
        for p in candidates:
            if not filtered_candidates or (p - filtered_candidates[-1]) >= min_gap:
                filtered_candidates.append(p)

        # 발-공 거리 검증
        min_dist = np.min(ankle_distances, axis=1)
        valid_peaks = []
        touch_feet  = []
        for p in filtered_candidates:
            if p < len(min_dist) and min_dist[p] < self.max_touch_distance:
                valid_peaks.append(p)
                touch_feet.append('left' if ankle_distances[p, 0] < ankle_distances[p, 1] else 'right')

        return valid_peaks, touch_feet

    def _calculate_all_ankle_distances(self, ball_positions: np.ndarray,
                                       pose_frame_list: List) -> np.ndarray:
        """모든 프레임에서 공과 양 발목의 거리 계산

        Returns:
            (N, 2) 배열: [:, 0] = 왼발 거리, [:, 1] = 오른발 거리
        """
        N = len(ball_positions)
        distances = np.full((N, 2), np.inf)

        for i, pf in enumerate(pose_frame_list):
            ball_pos = tuple(ball_positions[i])

            # 왼쪽 발목
            left_ankle_2d = self._get_ankle_2d_position(
                pf.landmarks, pf.frame_width, pf.frame_height, LEFT_ANKLE
            )
            distances[i, 0] = self._calculate_distance(ball_pos, left_ankle_2d)

            # 오른쪽 발목
            right_ankle_2d = self._get_ankle_2d_position(
                pf.landmarks, pf.frame_width, pf.frame_height, RIGHT_ANKLE
            )
            distances[i, 1] = self._calculate_distance(ball_pos, right_ankle_2d)

        return distances

    def _create_touch_events(self, touch_indices: List[int],
                            touch_feet: List[str],
                            ball_positions: np.ndarray,
                            pose_frame_list: List,
                            frame_numbers: np.ndarray,
                            timestamps: np.ndarray,
                            override_frame_numbers: Optional[np.ndarray] = None) -> List[TouchEvent]:
        """터치 이벤트 객체 생성

        Args:
            override_frame_numbers: 보간 프레임에서 찾은 실제 터치 프레임 번호 배열.
                                    None이면 frame_numbers[touch_idx] 사용.
        """
        events = []

        for i, (touch_idx, foot) in enumerate(zip(touch_indices, touch_feet)):
            pf = pose_frame_list[touch_idx]
            ball_2d = tuple(ball_positions[touch_idx])

            # 터치한 발목의 정보
            ankle_idx = LEFT_ANKLE if foot == 'left' else RIGHT_ANKLE

            ankle_2d = self._get_ankle_2d_position(
                pf.landmarks, pf.frame_width, pf.frame_height, ankle_idx
            )

            # 발목의 Z 좌표 (world_landmarks에서)
            ankle_z = pf.world_landmarks[ankle_idx][2]

            # 공의 3D 위치: (ball_x, ball_y, ankle_z)
            ball_3d = (ball_2d[0], ball_2d[1], ankle_z)

            # 공-발목 거리
            distance = self._calculate_distance(ball_2d, ankle_2d)

            # 어깨 라인 방향 계산 (right_shoulder - left_shoulder)
            left_shoulder = pf.world_landmarks[LEFT_SHOULDER]
            right_shoulder = pf.world_landmarks[RIGHT_SHOULDER]
            shoulder_direction = (
                float(right_shoulder[0] - left_shoulder[0]),
                float(right_shoulder[1] - left_shoulder[1]),
                float(right_shoulder[2] - left_shoulder[2])
            )

            # 골반 라인 방향 계산 (right_hip - left_hip)
            left_hip = pf.world_landmarks[LEFT_HIP]
            right_hip = pf.world_landmarks[RIGHT_HIP]
            pelvis_direction = (
                float(right_hip[0] - left_hip[0]),
                float(right_hip[1] - left_hip[1]),
                float(right_hip[2] - left_hip[2])
            )

            actual_frame_num = (
                int(override_frame_numbers[i])
                if override_frame_numbers is not None
                else int(frame_numbers[touch_idx])
            )

            event = TouchEvent(
                frame_number=actual_frame_num,
                timestamp=float(timestamps[touch_idx]),
                ball_2d=ball_2d,
                ball_3d=ball_3d,
                touching_foot=foot,
                ankle_2d=ankle_2d,
                ankle_z=ankle_z,
                distance=distance,
                shoulder_direction=shoulder_direction,
                pelvis_direction=pelvis_direction
            )
            events.append(event)

        return events

    def _xz_rotation_angle(self, v1_xz: np.ndarray, v2_xz: np.ndarray) -> float:
        """XZ 평면에서 두 벡터 사이 회전각도 (degrees)

        Args:
            v1_xz: 프레임1의 XZ 방향벡터 (2D)
            v2_xz: 프레임2의 XZ 방향벡터 (2D)

        Returns:
            회전각도 (0 ~ 180도)
        """
        norm1 = np.linalg.norm(v1_xz)
        norm2 = np.linalg.norm(v2_xz)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_a = np.dot(v1_xz, v2_xz) / (norm1 * norm2)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _calculate_touch_directions(self,
                                   touch_events: List[TouchEvent]) -> List[TouchDirection]:
        """연속 터치 간의 어깨·골반 XZ 평면 회전각 계산

        터치 1 → 터치 2:
          어깨 회전각 = XZ 투영 어깨벡터1 vs 어깨벡터2 사이 각도
          골반 회전각 = XZ 투영 골반벡터1 vs 골반벡터2 사이 각도
        """
        directions = []

        for i in range(len(touch_events) - 1):
            touch1 = touch_events[i]
            touch2 = touch_events[i + 1]

            # 2D 방향
            dx = touch2.ball_2d[0] - touch1.ball_2d[0]
            dy = touch2.ball_2d[1] - touch1.ball_2d[1]
            direction_2d = (dx, dy)
            distance_2d = np.sqrt(dx**2 + dy**2)

            # 3D 방향 (Z는 발목에서 가져옴) = 공 이동 벡터 (A)
            dz = touch2.ankle_z - touch1.ankle_z
            direction_3d = (dx, dy, dz)
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)

            # XZ 평면 어깨 회전각 계산
            sh1_xz = np.array([touch1.shoulder_direction[0], touch1.shoulder_direction[2]])
            sh2_xz = np.array([touch2.shoulder_direction[0], touch2.shoulder_direction[2]])
            shoulder_rotation_angle = self._xz_rotation_angle(sh1_xz, sh2_xz)

            # XZ 평면 골반 회전각 계산
            pe1_xz = np.array([touch1.pelvis_direction[0], touch1.pelvis_direction[2]])
            pe2_xz = np.array([touch2.pelvis_direction[0], touch2.pelvis_direction[2]])
            pelvis_rotation_angle = self._xz_rotation_angle(pe1_xz, pe2_xz)

            direction = TouchDirection(
                from_touch=touch1,
                to_touch=touch2,
                direction_2d=direction_2d,
                direction_3d=direction_3d,
                distance_2d=distance_2d,
                distance_3d=distance_3d,
                shoulder_rotation_angle=shoulder_rotation_angle,
                pelvis_rotation_angle=pelvis_rotation_angle,
            )
            directions.append(direction)

        return directions

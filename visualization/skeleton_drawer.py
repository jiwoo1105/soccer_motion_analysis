# visualization/skeleton_drawer.py
"""
=============================================================================
스켈레톤 시각화
=============================================================================

이 모듈은 MediaPipe 포즈 landmark를 비디오 프레임에 그립니다.

주요 기능:
1. 33개 landmark 점 그리기
2. landmark 간 연결선 그리기
3. 텍스트 오버레이 (비디오 이름 등)

MediaPipe 연결 구조:
- 몸통: 어깨-어깨, 어깨-엉덩이, 엉덩이-엉덩이
- 팔: 어깨-팔꿈치-손목
- 다리: 엉덩이-무릎-발목-발뒤꿈치-발끝

활용:
>>> drawer = SkeletonDrawer(color=(0, 255, 0))  # 초록색
>>> frame_with_skeleton = drawer.draw_skeleton(frame, landmarks)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class SkeletonDrawer:
    """
    MediaPipe 포즈를 영상에 그리는 클래스

    역할:
    1. Normalized landmarks (0-1)를 픽셀 좌표로 변환
    2. 관절 점과 연결선을 OpenCV로 그리기
    3. 텍스트 오버레이 추가

    좌표 변환:
        landmarks[i][0] (0-1) → x_pixel (0-width)
        landmarks[i][1] (0-1) → y_pixel (0-height)
    """

    # MediaPipe 연결선 정의 (landmark 인덱스 쌍)
    # 각 튜플은 (시작점, 끝점) 인덱스
    CONNECTIONS = [
        # 몸통 (Torso)
        (11, 12),  # 왼쪽 어깨 - 오른쪽 어깨
        (11, 23),  # 왼쪽 어깨 - 왼쪽 엉덩이
        (12, 24),  # 오른쪽 어깨 - 오른쪽 엉덩이
        (23, 24),  # 왼쪽 엉덩이 - 오른쪽 엉덩이

        # 왼쪽 팔 (Left Arm)
        (11, 13),  # 어깨 - 팔꿈치
        (13, 15),  # 팔꿈치 - 손목

        # 오른쪽 팔 (Right Arm)
        (12, 14),  # 어깨 - 팔꿈치
        (14, 16),  # 팔꿈치 - 손목

        # 왼쪽 다리 (Left Leg)
        (23, 25),  # 엉덩이 - 무릎
        (25, 27),  # 무릎 - 발목
        (27, 29),  # 발목 - 발뒤꿈치
        (29, 31),  # 발뒤꿈치 - 발끝

        # 오른쪽 다리 (Right Leg)
        (24, 26),  # 엉덩이 - 무릎
        (26, 28),  # 무릎 - 발목
        (28, 30),  # 발목 - 발뒤꿈치
        (30, 32),  # 발뒤꿈치 - 발끝
    ]

    def __init__(self, color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 point_radius: int = 4):
        """
        SkeletonDrawer 초기화

        Args:
            color: 선 색상 (BGR 형식)
                  예: (0, 255, 0) = 초록색
                      (255, 0, 0) = 파란색
                      (0, 0, 255) = 빨간색
            thickness: 선 두께 (픽셀)
            point_radius: 점 반지름 (픽셀)

        참고: OpenCV는 BGR 순서 사용 (RGB 아님!)

        사용 예시:
            >>> # 파란색 스켈레톤
            >>> drawer1 = SkeletonDrawer(color=(255, 0, 0))
            >>> # 빨간색 스켈레톤, 더 굵게
            >>> drawer2 = SkeletonDrawer(color=(0, 0, 255), thickness=3)
        """
        self.color = color
        self.thickness = thickness
        self.point_radius = point_radius

    def draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        프레임에 스켈레톤 그리기

        처리 과정:
        1. 연결선 그리기: CONNECTIONS에 정의된 순서대로
        2. 관절점 그리기: 모든 33개 landmark
        3. 원본 프레임은 보존하고 복사본에 그림

        Args:
            frame: BGR 이미지 (OpenCV 형식)
                  shape: (height, width, 3)
            landmarks: (33, 3) normalized landmarks [0-1 범위]
                      landmarks[i] = [x, y, z]
                      - x, y는 0-1로 정규화됨
                      - z는 깊이 (여기서는 사용 안 함)

        Returns:
            np.ndarray: 스켈레톤이 그려진 이미지

        주의:
            - landmarks의 x, y가 0-1 범위가 아니면 화면 밖에 그려질 수 있음
            - 원본 frame은 수정되지 않음 (복사본 반환)

        예시:
            >>> drawer = SkeletonDrawer()
            >>> result = drawer.draw_skeleton(frame, landmarks)
            >>> cv2.imshow("Skeleton", result)
        """
        # STEP 1: 프레임 크기 가져오기
        h, w, _ = frame.shape

        # STEP 2: 원본 보존을 위해 복사
        output = frame.copy()

        # STEP 3: 연결선 그리기
        for start_idx, end_idx in self.CONNECTIONS:
            # 시작점과 끝점 landmark 가져오기
            start_point = landmarks[start_idx]  # [x, y, z] (0-1 범위)
            end_point = landmarks[end_idx]

            # Normalized coordinates (0-1)를 픽셀 좌표로 변환
            # x * width = 픽셀 x좌표
            # y * height = 픽셀 y좌표
            start_pixel = (int(start_point[0] * w), int(start_point[1] * h))
            end_pixel = (int(end_point[0] * w), int(end_point[1] * h))

            # OpenCV로 선 그리기
            cv2.line(output, start_pixel, end_pixel, self.color, self.thickness)

        # STEP 4: 관절점 그리기
        for landmark in landmarks:
            # 픽셀 좌표로 변환
            pixel = (int(landmark[0] * w), int(landmark[1] * h))

            # 원(circle) 그리기
            # -1: 채워진 원
            cv2.circle(output, pixel, self.point_radius, self.color, -1)

        return output

    def draw_angle_text(self, frame: np.ndarray,
                       text: str,
                       position: Tuple[int, int],
                       color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        프레임에 텍스트 그리기 (배경 박스 포함)

        용도:
        - 비디오 이름 표시
        - 각도 값 표시
        - 프레임 번호 표시 등

        특징:
        - 검은색 배경 박스로 가독성 향상
        - 흰색 텍스트 (기본값)

        Args:
            frame: BGR 이미지
            text: 표시할 텍스트
            position: (x, y) 텍스트 시작 위치 (픽셀)
            color: 텍스트 색상 (BGR)

        Returns:
            np.ndarray: 텍스트가 추가된 이미지

        예시:
            >>> drawer = SkeletonDrawer()
            >>> frame = drawer.draw_angle_text(
            >>>     frame, "Soccer 1", (10, 30)
            >>> )
            >>> frame = drawer.draw_angle_text(
            >>>     frame, "Knee: 152.3°", (10, 60), color=(0, 255, 0)
            >>> )
        """
        # STEP 1: 원본 보존을 위해 복사
        output = frame.copy()

        # STEP 2: 폰트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # STEP 3: 텍스트 크기 측정 (배경 박스 크기 계산용)
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # STEP 4: 배경 박스 그리기 (검은색)
        # 텍스트보다 약간 큰 박스
        cv2.rectangle(output,
                     (position[0] - 5, position[1] - text_h - 5),  # 좌상단
                     (position[0] + text_w + 5, position[1] + 5),  # 우하단
                     (0, 0, 0),  # 검은색
                     -1)  # 채워진 사각형

        # STEP 5: 텍스트 그리기
        cv2.putText(output, text, position, font, font_scale, color, thickness)

        return output

    def draw_head_vector(self, frame: np.ndarray,
                         landmarks: np.ndarray,
                         color: Tuple[int, int, int] = (0, 0, 255),
                         thickness: int = 3) -> np.ndarray:
        """헤드업 측정 기준 벡터를 빨간 막대로 시각화

        어깨 중앙 → 눈 중앙 벡터를 영상에 직접 그려서
        머리 각도 측정 기준을 직관적으로 보여줌.

        랜드마크 인덱스:
            2  = LEFT_EYE
            5  = RIGHT_EYE
            11 = LEFT_SHOULDER
            12 = RIGHT_SHOULDER

        Args:
            frame: BGR 이미지
            landmarks: (33, 3) normalized landmarks [0-1 범위]
            color: 선 색상 (BGR), 기본값 빨간색
            thickness: 선 두께

        Returns:
            np.ndarray: 헤드 벡터가 그려진 이미지
        """
        h, w, _ = frame.shape
        output = frame.copy()

        LEFT_EYE, RIGHT_EYE = 2, 5
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

        # 눈 중앙 픽셀
        eye_center_x = (landmarks[LEFT_EYE][0] + landmarks[RIGHT_EYE][0]) / 2 * w
        eye_center_y = (landmarks[LEFT_EYE][1] + landmarks[RIGHT_EYE][1]) / 2 * h

        # 어깨 중앙 픽셀
        shoulder_center_x = (landmarks[LEFT_SHOULDER][0] + landmarks[RIGHT_SHOULDER][0]) / 2 * w
        shoulder_center_y = (landmarks[LEFT_SHOULDER][1] + landmarks[RIGHT_SHOULDER][1]) / 2 * h

        pt_shoulder = (int(shoulder_center_x), int(shoulder_center_y))
        pt_eye      = (int(eye_center_x),      int(eye_center_y))

        # 어깨 → 눈 방향으로 화살표 형태의 선
        cv2.line(output, pt_shoulder, pt_eye, color, thickness, cv2.LINE_AA)

        # 양 끝점 강조 (작은 원)
        cv2.circle(output, pt_shoulder, 5, color, -1)
        cv2.circle(output, pt_eye,      5, color, -1)

        return output

    def draw_body_direction_vectors(self, frame: np.ndarray,
                                     landmarks: np.ndarray,
                                     world_landmarks: np.ndarray = None,
                                     color: Tuple[int, int, int] = (255, 100, 0),
                                     thickness: int = 4) -> np.ndarray:
        """어깨·골반 라인 + R→L 방향 화살표 (협응성 분석 기준)

        어깨 라인: left_shoulder ──→ right_shoulder
        골반 라인: left_hip      ──→ right_hip
        화살표 방향: R_HIP - L_HIP 벡터 (오른쪽이 끝점)
        """
        h, w, _ = frame.shape
        output = frame.copy()

        L_SH, R_SH   = 11, 12
        L_HIP, R_HIP = 23, 24

        def to_px(idx):
            return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))

        pt_lsh  = to_px(L_SH)
        pt_rsh  = to_px(R_SH)
        pt_lhip = to_px(L_HIP)
        pt_rhip = to_px(R_HIP)

        for pt_l, pt_r, label in [(pt_lsh, pt_rsh, 'S'), (pt_lhip, pt_rhip, 'P')]:
            # 라인 끝(오른쪽)에 화살표 (L → R 방향)
            cv2.arrowedLine(output, pt_l, pt_r, color, thickness, cv2.LINE_AA, tipLength=0.25)

            # 양 끝점 강조
            cv2.circle(output, pt_l, 5, color, -1)
            cv2.circle(output, pt_r, 5, color, -1)

            # 라벨 (S=어깨, P=골반) — 라인 중앙 위에
            cx = (pt_l[0] + pt_r[0]) // 2
            cy = (pt_l[1] + pt_r[1]) // 2
            cv2.putText(output, label, (cx - 8, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return output

    def draw_trunk_angle(self, frame: np.ndarray,
                         landmarks: np.ndarray,
                         trunk_angle: Optional[float] = None,
                         color: Tuple[int, int, int] = (255, 255, 255),
                         thickness: int = 2) -> np.ndarray:
        """상체 측정 기준 연결선 표시 (엉덩이→어깨, 엉덩이→무릎, 흰색)

        랜드마크 인덱스:
            11=LEFT_SHOULDER,  12=RIGHT_SHOULDER
            23=LEFT_HIP,       24=RIGHT_HIP
            25=LEFT_KNEE,      26=RIGHT_KNEE

        Args:
            frame:       BGR 이미지
            landmarks:   (33, 3) normalized landmarks [0-1 범위]
            trunk_angle: 미사용 (호환성 유지용)
            color:       연결선 색상
            thickness:   연결선 두께

        Returns:
            np.ndarray: 연결선이 그려진 이미지
        """
        h, w, _ = frame.shape
        output = frame.copy()

        L_SH, R_SH   = 11, 12
        L_HIP, R_HIP = 23, 24
        L_KN, R_KN   = 25, 26

        def to_px(idx):
            return (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))

        for sh_idx, hip_idx, kn_idx in [(L_SH, L_HIP, L_KN), (R_SH, R_HIP, R_KN)]:
            pt_sh  = to_px(sh_idx)
            pt_hip = to_px(hip_idx)
            pt_kn  = to_px(kn_idx)
            cv2.line(output, pt_hip, pt_sh, color, thickness, cv2.LINE_AA)
            cv2.line(output, pt_hip, pt_kn, color, thickness, cv2.LINE_AA)
            cv2.circle(output, pt_hip, 4, color, -1)

        return output

    def draw_ball_bbox(self, frame: np.ndarray,
                      ball_bbox: Optional[Tuple[int, int, int, int]],
                      color: Tuple[int, int, int] = (0, 255, 255),
                      thickness: int = 3) -> np.ndarray:
        """
        프레임에 공 바운딩 박스 그리기

        Args:
            frame: BGR 이미지
            ball_bbox: (x1, y1, x2, y2) 바운딩 박스 좌표, None이면 그리지 않음
            color: 박스 색상 (BGR), 기본값 노란색
            thickness: 선 두께

        Returns:
            np.ndarray: 바운딩 박스가 그려진 이미지
        """
        if ball_bbox is None:
            return frame

        output = frame.copy()
        x1, y1, x2, y2 = ball_bbox

        # 바운딩 박스 사각형 그리기
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        return output

    def draw_ball_interpolated(self, frame: np.ndarray,
                               position: Tuple[int, int],
                               radius: int = 18) -> np.ndarray:
        """보간된 공 위치 표시 (감지 안 된 프레임)

        점선 느낌의 얇은 주황색 원으로 표시해 실제 감지와 구분.
        """
        output = frame.copy()
        cx, cy = int(position[0]), int(position[1])
        # 주황색 얇은 원 (보간 추정 위치)
        cv2.circle(output, (cx, cy), radius, (0, 165, 255), 2)
        # 중심 점
        cv2.circle(output, (cx, cy), 3, (0, 165, 255), -1)
        return output

    def draw_ball_trajectory(self, frame: np.ndarray,
                            trajectory: List[Tuple[int, int]],
                            color: Tuple[int, int, int] = (255, 200, 0),
                            thickness: int = 2) -> np.ndarray:
        """
        프레임에 공의 궤적 그리기

        Args:
            frame: BGR 이미지
            trajectory: [(x, y), ...] 공의 이동 경로 좌표 리스트
            color: 궤적 색상 (BGR), 기본값 하늘색
            thickness: 선 두께

        Returns:
            np.ndarray: 궤적이 그려진 이미지
        """
        if len(trajectory) < 2:
            return frame

        output = frame.copy()

        # 연속된 점들을 선으로 연결
        for i in range(len(trajectory) - 1):
            pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
            pt2 = (int(trajectory[i+1][0]), int(trajectory[i+1][1]))
            cv2.line(output, pt1, pt2, color, thickness)

        return output

    def draw_touch_highlight(self, frame: np.ndarray,
                            foot_position: Optional[Tuple[int, int]],
                            text: str = "TOUCH!") -> np.ndarray:
        """
        터치 순간 하이라이트 표시

        Args:
            frame: BGR 이미지
            foot_position: (x, y) 터치한 발의 발끝-뒤꿈치 중앙 위치, None이면 원 미표시
            text: 표시할 텍스트

        Returns:
            np.ndarray: 하이라이트가 그려진 이미지
        """
        output = frame.copy()
        _, w, _ = frame.shape

        # 터치한 발 중앙에 빨간 원 그리기
        if foot_position is not None:
            center = (int(foot_position[0]), int(foot_position[1]))
            cv2.circle(output, center, 20, (0, 0, 255), 3)

        # "TOUCH!" 텍스트 표시 (화면 상단)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 4  # 굵게 표시
        text_color = (0, 0, 255)  # 빨간색

        # 텍스트 크기 측정
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # 화면 중앙 상단에 배치
        text_x = (w - text_w) // 2
        text_y = 60

        # 배경 박스
        cv2.rectangle(output,
                     (text_x - 10, text_y - text_h - 10),
                     (text_x + text_w + 10, text_y + 10),
                     (255, 255, 255), -1)

        # 텍스트
        cv2.putText(output, text, (text_x, text_y), font,
                   font_scale, text_color, thickness)

        return output

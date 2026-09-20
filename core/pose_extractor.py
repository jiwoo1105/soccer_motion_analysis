# core/pose_extractor.py
"""
=============================================================================
MediaPipe를 사용한 3D 포즈 추출
=============================================================================

이 모듈은 비디오에서 사람의 3D 포즈(관절 위치)를 추출하는 핵심 기능을 제공합니다.

주요 기능:
1. MediaPipe Pose 모델을 사용한 33개 랜드마크 추출
2. 각 프레임마다 2가지 좌표계 제공:
   - landmarks: 이미지 좌표 (0-1로 정규화, 화면 상 위치)
   - world_landmarks: 추정 3D 좌표 (미터 단위, 추정 공간 위치)
3. visibility 점수로 각 랜드마크의 신뢰도 제공

MediaPipe Pose의 33개 랜드마크:
    0: NOSE (코)
    1-10: 얼굴 (눈, 귀 등)
    11-12: SHOULDER (어깨)
    13-16: ARM (팔, 팔꿈치, 손목)
    23-24: HIP (엉덩이)
    25-26: KNEE (무릎)
    27-28: ANKLE (발목)
    29-30: HEEL (발뒤꿈치)
    31-32: FOOT_INDEX (발끝)

데이터 플로우:
    비디오 파일
        ↓
    각 프레임 읽기
        ↓
    MediaPipe Pose 모델
        ↓
    PoseFrame 객체 (landmarks + visibility)
        ↓
    List[PoseFrame] 반환
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

try:
    from .ball_detector import BallDetector
    BALL_DETECTION_AVAILABLE = True
except ImportError:
    BALL_DETECTION_AVAILABLE = False
    BallDetector = None


@dataclass
class PoseFrame:
    """
    단일 프레임의 포즈 데이터를 담는 데이터 클래스

    Attributes:
        frame_number: 프레임 번호 (0부터 시작)
        timestamp: 비디오 내 시간 (초)
        landmarks: (33, 3) 정규화된 좌표 [0-1 범위]
                   - landmarks[i][0]: x 좌표 (좌우, 0=왼쪽, 1=오른쪽)
                   - landmarks[i][1]: y 좌표 (상하, 0=위, 1=아래)
                   - landmarks[i][2]: z 좌표 (깊이, 카메라 기준)
        world_landmarks: (33, 3) 추정 3D 좌표 (미터 단위)
                        - 엉덩이 중심을 원점으로 하는 추정 공간 좌표
                        - 기존 각도 계산에서 사용하는 추정 좌표
        visibility: (33,) visibility 점수 [0-1]
                   - 1.0: 매우 확신함
                   - 0.0: 가려져서 보이지 않음
                   - 0.5 이하: 신뢰도 낮음
        ball_position: (x, y) 공의 2D 픽셀 좌표, 탐지 실패 시 None
                      - x: 이미지 내 x 좌표 (픽셀)
                      - y: 이미지 내 y 좌표 (픽셀)
        ball_bbox: (x1, y1, x2, y2) 공의 바운딩 박스, 탐지 실패 시 None
                  - x1, y1: 박스 왼쪽 위 모서리
                  - x2, y2: 박스 오른쪽 아래 모서리
        frame_width: 프레임 이미지 너비 (픽셀)
        frame_height: 프레임 이미지 높이 (픽셀)

    사용 예시:
        >>> pose_frame = pose_frames[0]
        >>> left_knee = pose_frame.world_landmarks[25]  # 왼쪽 무릎
        >>> if pose_frame.visibility[25] > 0.5:
        >>>     print(f"무릎 위치: {left_knee}")
        >>> if pose_frame.ball_position:
        >>>     print(f"공 위치: {pose_frame.ball_position}")
    """
    frame_number: int
    timestamp: float
    landmarks: np.ndarray  # (33, 3) - normalized coordinates
    world_landmarks: np.ndarray  # (33, 3) - real world coordinates in meters
    visibility: np.ndarray  # (33,) - visibility scores
    ball_position: Optional[tuple] = None  # (x, y) - ball 2D position in pixels
    ball_bbox: Optional[tuple] = None  # (x1, y1, x2, y2) - ball bounding box
    frame_width: int = 0  # frame width in pixels
    frame_height: int = 0  # frame height in pixels


class PoseExtractor:
    """
    MediaPipe Pose를 사용한 3D 포즈 추출 클래스

    역할: 비디오 파일을 입력받아 모든 프레임에서 포즈를 추출

    MediaPipe Pose 설정:
    - static_image_mode: False (비디오 모드, 트래킹 활성화)
    - model_complexity: 0/1/2 (2가 가장 정확하지만 느림)
    - min_detection_confidence: 최초 감지 신뢰도 임계값
    - min_tracking_confidence: 프레임 간 추적 신뢰도 임계값
    - smooth_landmarks: True (떨림 방지)
    """

    def __init__(self,
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 detect_ball: bool = True,
                 ball_detector_config: Optional[Dict] = None):
        """
        PoseExtractor 초기화

        Args:
            model_complexity: 모델 복잡도 (0, 1, 2)
                             0: 가장 빠름, 낮은 정확도
                             1: 중간
                             2: 가장 정확함, 느림 (축구 분석에 권장)
            min_detection_confidence: 최소 감지 신뢰도 (0.0-1.0)
                                     사람을 처음 찾을 때의 임계값
            min_tracking_confidence: 최소 추적 신뢰도 (0.0-1.0)
                                    이미 찾은 사람을 계속 따라갈 때의 임계값
            detect_ball: 공 탐지 활성화 여부 (YOLOv8 사용)
            ball_detector_config: 공 탐지 설정 딕셔너리

        사용 예시:
            >>> extractor = PoseExtractor(model_complexity=2, detect_ball=True)
            >>> pose_frames = extractor.extract_from_video("soccer.mp4")
        """
        # MediaPipe Pose 모듈 초기화
        self.mp_pose = mp.solutions.pose

        # Pose 객체 생성 (실제 추론을 수행하는 모델)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # 비디오 모드 (프레임 간 연속성 활용)
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,  # 사람 분할 마스크 비활성화 (속도 향상)
            smooth_landmarks=True  # 랜드마크 스무딩 활성화 (떨림 감소)
        )

        # 공 탐지 초기화
        self.detect_ball = detect_ball
        self.ball_detector = None
        if detect_ball:
            if BALL_DETECTION_AVAILABLE:
                if ball_detector_config:
                    self.ball_detector = BallDetector(**ball_detector_config)
                else:
                    self.ball_detector = BallDetector()
            else:
                print("Warning: BallDetector not available. Install ultralytics: pip install ultralytics")
                self.detect_ball = False

    def extract_from_video(self, video_path: str) -> List[PoseFrame]:
        """
        비디오에서 모든 프레임의 포즈 추출

        처리 과정:
        1. 비디오 파일 열기
        2. 각 프레임 읽기
        3. BGR → RGB 변환 (MediaPipe 요구사항)
        4. MediaPipe로 포즈 추정
        5. 성공한 프레임만 PoseFrame 객체로 저장
        6. 모든 PoseFrame을 리스트로 반환

        Args:
            video_path: 비디오 파일 경로 (예: "input/soccer1.mp4")

        Returns:
            List[PoseFrame]: 포즈가 감지된 프레임들의 리스트
                            - 비디오 총 프레임 수보다 적을 수 있음
                              (사람이 안 보이거나 감지 실패한 프레임 제외)

        예시:
            >>> extractor = PoseExtractor()
            >>> pose_frames = extractor.extract_from_video("input/soccer1.mp4")
            >>> print(f"총 {len(pose_frames)}개 프레임에서 포즈 감지")
            >>> # 첫 번째 프레임의 왼쪽 무릎 위치
            >>> left_knee = pose_frames[0].world_landmarks[25]
        """
        # STEP 1: 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)

        # 비디오 메타정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이

        # 결과 저장용 리스트
        pose_frames = []
        frame_number = 0

        print(f"Extracting poses from {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Size: {frame_width}x{frame_height}")

        # STEP 2: 진행 상황 표시를 위한 progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            # STEP 3: 비디오의 모든 프레임 순회
            while cap.isOpened():
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    break  # 비디오 끝

                # STEP 4: BGR → RGB 변환
                # OpenCV는 BGR, MediaPipe는 RGB 사용
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # STEP 5: MediaPipe로 포즈 추정
                # results 객체에 추정 결과가 담김
                results = self.pose.process(frame_rgb)

                # STEP 6: 포즈가 성공적으로 감지되었는지 확인
                if results.pose_landmarks and results.pose_world_landmarks:
                    # 타임스탬프 계산 (초 단위)
                    timestamp = frame_number / fps

                    # MediaPipe 결과를 numpy 배열로 변환
                    landmarks = self._extract_landmarks(results.pose_landmarks)
                    world_landmarks = self._extract_landmarks(results.pose_world_landmarks)
                    visibility = self._extract_visibility(results.pose_landmarks)

                    # 공 탐지 (활성화된 경우)
                    ball_position = None
                    ball_bbox = None
                    if self.detect_ball and self.ball_detector:
                        ball_result = self.ball_detector.detect_ball_with_box(frame)
                        if ball_result:
                            # (center_x, center_y, x1, y1, x2, y2)
                            ball_position = (ball_result[0], ball_result[1])
                            ball_bbox = (ball_result[2], ball_result[3], ball_result[4], ball_result[5])

                    # PoseFrame 객체 생성
                    pose_frame = PoseFrame(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        landmarks=landmarks,
                        world_landmarks=world_landmarks,
                        visibility=visibility,
                        ball_position=ball_position,
                        ball_bbox=ball_bbox,
                        frame_width=frame_width,
                        frame_height=frame_height
                    )

                    # 리스트에 추가
                    pose_frames.append(pose_frame)

                frame_number += 1
                pbar.update(1)  # progress bar 업데이트

        # STEP 7: 비디오 캡처 객체 해제
        cap.release()

        # 공 탐지 통계
        if self.detect_ball:
            ball_detected_count = sum(1 for pf in pose_frames if pf.ball_position is not None)
            print(f"Successfully extracted {len(pose_frames)} frames with poses")
            print(f"Ball detected in {ball_detected_count}/{len(pose_frames)} frames ({100*ball_detected_count/len(pose_frames):.1f}%)")
        else:
            print(f"Successfully extracted {len(pose_frames)} frames with poses")

        return pose_frames

    def _extract_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        MediaPipe 랜드마크 객체를 numpy 배열로 변환 (내부 헬퍼 함수)

        MediaPipe는 랜드마크를 특수한 객체로 반환하는데,
        이를 계산하기 쉬운 numpy 배열로 변환합니다.

        Args:
            pose_landmarks: MediaPipe의 pose_landmarks 또는 pose_world_landmarks

        Returns:
            np.ndarray: (33, 3) 배열
                       - 33개 랜드마크
                       - 각 랜드마크는 (x, y, z) 좌표

        동작:
            MediaPipe 랜드마크 객체:
            pose_landmarks.landmark[0].x  # 코의 x 좌표
            pose_landmarks.landmark[0].y  # 코의 y 좌표
            pose_landmarks.landmark[0].z  # 코의 z 좌표

            변환 후:
            landmarks[0] = [x, y, z]  # numpy 배열
        """
        # 리스트 컴프리헨션으로 모든 랜드마크를 [x, y, z] 리스트로 변환
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in pose_landmarks.landmark
        ])
        return landmarks

    def _extract_visibility(self, pose_landmarks) -> np.ndarray:
        """
        Visibility 점수를 numpy 배열로 추출 (내부 헬퍼 함수)

        각 랜드마크가 얼마나 확실하게 보이는지 점수를 추출합니다.
        - 1.0: 매우 확실함
        - 0.5: 애매함
        - 0.0: 가려짐 또는 보이지 않음

        Args:
            pose_landmarks: MediaPipe의 pose_landmarks

        Returns:
            np.ndarray: (33,) 배열, 각 랜드마크의 visibility 점수

        활용:
            >>> if visibility[27] < 0.5:  # 왼쪽 발목
            >>>     print("발목이 가려져 있음, 이 데이터는 신뢰할 수 없음")
        """
        visibility = np.array([
            lm.visibility
            for lm in pose_landmarks.landmark
        ])
        return visibility

    def extract_from_frame(self, frame: np.ndarray) -> Optional[PoseFrame]:
        """
        단일 프레임에서 포즈 추출

        용도: 실시간 스트림이나 이미지에서 포즈 추출
        (주로 extract_from_video를 사용하고, 이 함수는 특수한 경우에만 사용)

        Args:
            frame: BGR 이미지 (OpenCV 형식)

        Returns:
            PoseFrame or None: 포즈가 감지되면 PoseFrame, 아니면 None

        예시:
            >>> import cv2
            >>> frame = cv2.imread("image.jpg")
            >>> pose_frame = extractor.extract_from_frame(frame)
            >>> if pose_frame:
            >>>     print("포즈 감지 성공!")
        """
        # BGR → RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 포즈 추정
        results = self.pose.process(frame_rgb)

        # 포즈가 감지되었는지 확인
        if results.pose_landmarks and results.pose_world_landmarks:
            # numpy 배열로 변환
            landmarks = self._extract_landmarks(results.pose_landmarks)
            world_landmarks = self._extract_landmarks(results.pose_world_landmarks)
            visibility = self._extract_visibility(results.pose_landmarks)

            # 공 탐지
            ball_position = None
            ball_bbox = None
            if self.detect_ball and self.ball_detector:
                ball_result = self.ball_detector.detect_ball_with_box(frame)
                if ball_result:
                    ball_position = (ball_result[0], ball_result[1])
                    ball_bbox = (ball_result[2], ball_result[3], ball_result[4], ball_result[5])

            # PoseFrame 객체 반환
            return PoseFrame(
                frame_number=0,
                timestamp=0.0,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                visibility=visibility,
                ball_position=ball_position,
                ball_bbox=ball_bbox
            )

        # 포즈 감지 실패
        return None

    def __del__(self):
        """
        객체 소멸자 - 리소스 정리

        PoseExtractor 객체가 삭제될 때 MediaPipe 모델도 정리합니다.
        GPU 메모리나 기타 리소스를 해제하기 위함입니다.

        자동으로 호출되므로 직접 사용할 필요 없음.
        """
        if hasattr(self, 'pose'):
            self.pose.close()

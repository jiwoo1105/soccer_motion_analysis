"""
특정 프레임의 world_landmarks를 3D로 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# MediaPipe POSE_CONNECTIONS (33개 keypoint 연결 쌍)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]


def plot_3d_pose(world_landmarks: np.ndarray,
                 frame_index: int,
                 video_stem: str,
                 save_path: str) -> None:
    """
    world_landmarks (33, 3) numpy 배열을 3D로 시각화하여 저장

    Args:
        world_landmarks: (33, 3) 미터 단위 3D 좌표
        frame_index: 프레임 번호 (제목 표시용)
        video_stem: 영상 파일명 stem (제목 표시용)
        save_path: 저장 경로
    """
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection='3d')

    xs = world_landmarks[:, 0]
    ys = -world_landmarks[:, 1]  # y축 반전 (위가 양수)
    zs = world_landmarks[:, 2]

    # 연결선 그리기
    for i, j in POSE_CONNECTIONS:
        ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]],
                color='black', linewidth=2)

    # keypoint 점 그리기
    ax.scatter(xs, zs, ys, color='red', s=30, zorder=5)

    ax.set_xlabel('X  (좌우)', fontsize=9)
    ax.set_ylabel('Z  (깊이, 카메라방향)', fontsize=9)
    ax.set_zlabel('Y  (상하)', fontsize=9)
    ax.set_title(f'{video_stem}  |  Frame {frame_index}', fontsize=10)

    # 실제 데이터 범위 비율로 박스 크기 설정 (왜곡 없음)
    x_range = xs.max() - xs.min()
    z_range = zs.max() - zs.min()
    y_range = ys.max() - ys.min()
    ax.set_box_aspect([x_range, z_range, y_range])

    # 정면에서 보는 시점
    ax.view_init(elev=10, azim=-80)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D 포즈 저장: {save_path}")

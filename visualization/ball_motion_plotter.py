# visualization/ball_motion_plotter.py
"""
=============================================================================
공 움직임 그래프 생성 모듈
=============================================================================

이 모듈은 공의 움직임 데이터를 시각화하여 그래프로 저장합니다.

주요 기능:
1. 공-발목 거리 그래프 (터치 감지 결과)
2. 2D 궤적 그래프 (터치 지점 및 방향)
3. 3D 방향 벡터 시각화

활용:
>>> plotter = BallMotionPlotter()
>>> plotter.plot_motion(motion_data, save_path="output/graphs/ball_motion.png")
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path


class BallMotionPlotter:
    """
    공의 움직임을 그래프로 시각화하는 클래스
    """

    def __init__(self, use_korean_font: bool = True):
        """
        BallMotionPlotter 초기화

        Args:
            use_korean_font: 한글 폰트 사용 여부
        """
        if use_korean_font:
            self._setup_korean_font()

    def _setup_korean_font(self):
        """한글 폰트 설정 (맥OS용)"""
        try:
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.rcParams['axes.unicode_minus'] = False
        except Exception:
            print("Warning: 한글 폰트 설정 실패. 기본 폰트를 사용합니다.")

    def plot_motion(self, motion_data, save_path: Optional[str] = None):
        """
        공-발목 거리 그래프 시각화 (터치 감지 결과)

        Args:
            motion_data: BallMotionData 객체
            save_path: 그래프 저장 경로 (None이면 화면에 표시만)
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        fig.suptitle('공 움직임 분석 (거리 기반 터치 감지)', fontsize=16, fontweight='bold')

        if motion_data.ankle_distances is not None:
            self._plot_ankle_distance(ax, motion_data)
        else:
            ax.text(0.5, 0.5, '분석 데이터 없음', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def _plot_ankle_distance(self, ax, motion_data):
        """
        공-발목 거리 그래프 그리기

        Args:
            ax: matplotlib axis
            motion_data: BallMotionData 객체
        """
        frames = motion_data.frame_numbers
        distances = motion_data.ankle_distances

        # 왼발, 오른발 거리
        ax.plot(frames, distances[:, 0], 'b-', linewidth=1.2, alpha=0.7, label='왼발 거리')
        ax.plot(frames, distances[:, 1], 'r-', linewidth=1.2, alpha=0.7, label='오른발 거리')

        # 최소 거리
        min_dist = np.min(distances, axis=1)
        ax.plot(frames, min_dist, 'k--', linewidth=1.5, alpha=0.5, label='최소 거리')

        # 터치 임계값 표시 (기본값 100px)
        ax.axhline(y=100, color='orange', linestyle=':', linewidth=2, label='터치 임계값 (100px)')

        # 터치 이벤트 표시
        if motion_data.touch_events:
            for i, touch in enumerate(motion_data.touch_events):
                tf = touch.frame_number
                dist = touch.distance

                color = 'blue' if touch.touching_foot == 'left' else 'red'
                marker = 'o' if touch.touching_foot == 'left' else 's'

                ax.scatter([tf], [dist], color=color, s=150, zorder=5,
                          marker=marker, edgecolors='black', linewidths=1.5)
                ax.annotate(f'{i+1}', (tf, dist), textcoords="offset points",
                           xytext=(0, -15), ha='center', fontsize=9, fontweight='bold')

        # 축 레이블 및 제목
        ax.set_xlabel('프레임 번호', fontsize=11)
        ax.set_ylabel('공-발목 거리 (pixels)', fontsize=11)
        ax.set_title('공과 발목 사이의 거리 (극소값 = 터치)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # y축 범위 제한
        valid_mask = min_dist < 500
        if np.any(valid_mask):
            ax.set_ylim(0, min(300, np.max(min_dist[valid_mask]) * 1.2))

    def plot_trajectory_2d(self, motion_data, save_path: Optional[str] = None):
        """
        공의 2D 궤적을 평면에 그리기

        Args:
            motion_data: BallMotionData 객체
            save_path: 그래프 저장 경로
        """
        _, ax = plt.subplots(figsize=(10, 8))

        positions = motion_data.positions
        x = positions[:, 0]
        y = positions[:, 1]

        # 궤적 플롯 (색상 그라디언트: 시간 순서)
        scatter = ax.scatter(x, y, c=motion_data.frame_numbers,
                           cmap='viridis', s=30, alpha=0.6)
        ax.plot(x, y, 'b-', linewidth=1, alpha=0.3)

        # 터치 지점 표시 (발 구분)
        if motion_data.touch_events:
            left_plotted = False
            right_plotted = False

            for i, touch in enumerate(motion_data.touch_events):
                tx, ty = touch.ball_2d
                color = 'blue' if touch.touching_foot == 'left' else 'red'
                marker = 'o' if touch.touching_foot == 'left' else 's'

                # 레이블은 각 발당 한 번만
                label = None
                if touch.touching_foot == 'left' and not left_plotted:
                    label = '왼발 터치'
                    left_plotted = True
                elif touch.touching_foot == 'right' and not right_plotted:
                    label = '오른발 터치'
                    right_plotted = True

                ax.scatter([tx], [ty], color=color, s=200, marker=marker,
                          zorder=5, edgecolors='white', linewidths=2, label=label)
                ax.annotate(f'{i+1}', (tx, ty), textcoords="offset points",
                           xytext=(8, 8), ha='left', fontsize=10, fontweight='bold')

            # 터치 간 방향 화살표 표시
            for direction in motion_data.touch_directions:
                x1, y1 = direction.from_touch.ball_2d
                x2, y2 = direction.to_touch.ball_2d

                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color='green',
                                          lw=2, mutation_scale=15))
        else:
            # 기존 방식
            touch_frames = motion_data.touch_frames
            frames = motion_data.frame_numbers
            for tf in touch_frames:
                if tf in frames:
                    touch_idx = np.where(frames == tf)[0][0]
                    ax.scatter([x[touch_idx]], [y[touch_idx]], color='red', s=200,
                              marker='*', zorder=5, label='터치 지점')

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('프레임 번호', fontsize=11)

        # 축 레이블 및 제목
        ax.set_xlabel('X 좌표 (pixels)', fontsize=12)
        ax.set_ylabel('Y 좌표 (pixels)', fontsize=12)
        ax.set_title('공의 2D 궤적 및 터치 지점', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # y축 반전 (이미지 좌표계는 위가 0)
        ax.invert_yaxis()

        plt.tight_layout()



        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)










            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"궤적 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_3d_direction(self, motion_data, save_path: Optional[str] = None):
        """
        터치 간 3D 방향 벡터 시각화

        Args:
            motion_data: BallMotionData 객체
            save_path: 그래프 저장 경로
        """
        if not motion_data.touch_directions:
            print("Warning: 터치 방향 데이터가 없습니다.")
            return

        fig = plt.figure(figsize=(12, 5))

        # 1. 3D 방향 벡터 (dx, dy, dz) 막대 그래프
        ax1 = fig.add_subplot(1, 2, 1)

        n_directions = len(motion_data.touch_directions)
        indices = np.arange(n_directions)
        width = 0.25

        dx_values = [d.direction_3d[0] for d in motion_data.touch_directions]
        dy_values = [d.direction_3d[1] for d in motion_data.touch_directions]
        dz_values = [d.direction_3d[2] * 100 for d in motion_data.touch_directions]  # Z는 스케일 조정

        ax1.bar(indices - width, dx_values, width, label='Δx (pixels)', color='red', alpha=0.7)
        ax1.bar(indices, dy_values, width, label='Δy (pixels)', color='green', alpha=0.7)
        ax1.bar(indices + width, dz_values, width, label='Δz (×100)', color='blue', alpha=0.7)

        ax1.set_xlabel('터치 구간', fontsize=11)
        ax1.set_ylabel('방향 성분', fontsize=11)
        ax1.set_title('터치 간 3D 방향 벡터', fontsize=12, fontweight='bold')
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f'{i+1}→{i+2}' for i in range(n_directions)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)

        # 2. 터치 순서 및 발 정보
        ax2 = fig.add_subplot(1, 2, 2)

        text_lines = ["터치 이벤트 상세:\n"]
        for i, touch in enumerate(motion_data.touch_events):
            foot_kor = '왼발' if touch.touching_foot == 'left' else '오른발'
            text_lines.append(f"[{i+1}] Frame {touch.frame_number}: {foot_kor}")
            text_lines.append(f"    위치: ({touch.ball_2d[0]:.0f}, {touch.ball_2d[1]:.0f})")
            text_lines.append(f"    Z좌표: {touch.ankle_z:.3f}m")
            text_lines.append(f"    거리: {touch.distance:.1f}px\n")

        if motion_data.touch_directions:
            text_lines.append("\n3D 방향 (터치 간):\n")
            for i, direction in enumerate(motion_data.touch_directions):
                dx, dy, dz = direction.direction_3d
                text_lines.append(f"[{i+1}→{i+2}] ({dx:.1f}, {dy:.1f}, {dz:.3f})")

        ax2.text(0.05, 0.95, '\n'.join(text_lines), transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        ax2.set_title('터치 이벤트 요약', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"3D 방향 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_coordination(self, motion_data, save_path: Optional[str] = None):
        """
        상체-하체 협응성 시각화 (어깨/골반 정렬도)

        Args:
            motion_data: BallMotionData 객체
            save_path: 그래프 저장 경로
        """
        if not motion_data.touch_directions:
            print("Warning: 터치 방향 데이터가 없습니다.")
            return

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle('상체 회전량 분석 (XZ 평면)', fontsize=16, fontweight='bold')

        # 1. 어깨/골반 회전각 막대 그래프
        ax1 = fig.add_subplot(1, 2, 1)

        n_directions = len(motion_data.touch_directions)
        indices = np.arange(n_directions)
        width = 0.35

        shoulder_angles = [d.shoulder_rotation_angle for d in motion_data.touch_directions]
        pelvis_angles = [d.pelvis_rotation_angle for d in motion_data.touch_directions]

        bars1 = ax1.bar(indices - width/2, shoulder_angles, width,
                       label='어깨 회전', color='royalblue', alpha=0.8)
        bars2 = ax1.bar(indices + width/2, pelvis_angles, width,
                       label='골반 회전', color='darkorange', alpha=0.8)

        for bar, val in zip(bars1, shoulder_angles):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, pelvis_angles):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax1.set_xlabel('터치 구간', fontsize=11)
        ax1.set_ylabel('회전각 (degrees)', fontsize=11)
        ax1.set_title('터치 구간별 회전량', fontsize=12, fontweight='bold')
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f'{i+1}→{i+2}' for i in range(n_directions)])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 상세 정보
        ax2 = fig.add_subplot(1, 2, 2)

        text_lines = ["상체 회전 상세:\n"]
        text_lines.append("="*40 + "\n")

        for i, direction in enumerate(motion_data.touch_directions):
            from_f = direction.from_touch.frame_number
            to_f = direction.to_touch.frame_number
            text_lines.append(f"[{i+1}→{i+2}] Frame {from_f} → {to_f}")
            text_lines.append(f"  어깨 회전: {direction.shoulder_rotation_angle:.1f}°")
            text_lines.append(f"  골반 회전: {direction.pelvis_rotation_angle:.1f}°")
            text_lines.append("")

        if n_directions > 0:
            avg_shoulder = sum(shoulder_angles) / n_directions
            avg_pelvis = sum(pelvis_angles) / n_directions
            rotation_score = (avg_shoulder + avg_pelvis) / 2

            text_lines.append("="*40)
            text_lines.append(f"\n평균 어깨 회전: {avg_shoulder:.1f}°")
            text_lines.append(f"평균 골반 회전: {avg_pelvis:.1f}°")
            text_lines.append(f"\n상체 회전 점수: {rotation_score:.1f}°")

        ax2.text(0.05, 0.95, '\n'.join(text_lines), transform=ax2.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax2.axis('off')
        ax2.set_title('상체 회전 상세', fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"협응성 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

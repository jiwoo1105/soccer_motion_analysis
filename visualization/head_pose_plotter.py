# visualization/head_pose_plotter.py
"""
=============================================================================
머리 자세 그래프 생성 모듈
=============================================================================

이 모듈은 머리 각도 데이터를 시각화하여 그래프로 저장합니다.

주요 기능:
1. 프레임별 머리 각도 변화 그래프
2. 터치 순간 표시
3. 평균값 가이드 라인
4. 헤드업 기준 각도 표시 (설정된 경우)

활용:
>>> plotter = HeadPosePlotter()
>>> plotter.plot_head_angle(head_pose_data, save_path="output/graphs/head_pose.png")
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import Optional
from pathlib import Path


class HeadPosePlotter:
    """
    머리 자세를 그래프로 시각화하는 클래스
    """

    def __init__(self, use_korean_font: bool = True):
        """
        HeadPosePlotter 초기화

        Args:
            use_korean_font: 한글 폰트 사용 여부
        """
        # 한글 폰트 설정
        if use_korean_font:
            self._setup_korean_font()

    def _setup_korean_font(self):
        """한글 폰트 설정 (맥OS용)"""
        try:
            # macOS의 AppleGothic 폰트 사용
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        except:
            # 폰트 설정 실패 시 기본 폰트 사용
            print("Warning: 한글 폰트 설정 실패. 기본 폰트를 사용합니다.")

    def plot_head_angle(self, head_pose_data, save_path: Optional[str] = None):
        """
        머리 각도 그래프 시각화

        Args:
            head_pose_data: HeadPoseData 객체
            save_path: 그래프 저장 경로 (None이면 화면에 표시만)
        """
        # 그래프 생성
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        fig.suptitle('머리 각도 분석 (Head Pose Angle)', fontsize=16, fontweight='bold')

        # 머리 각도 그래프
        self._plot_angle_over_time(ax, head_pose_data)

        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 저장 또는 표시
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 머리 각도 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def _plot_angle_over_time(self, ax, head_pose_data):
        """
        프레임별 머리 각도 변화 그래프 그리기

        Args:
            ax: matplotlib axis
            head_pose_data: HeadPoseData 객체
        """
        frames = head_pose_data.frame_numbers
        angles = head_pose_data.head_angles

        # 각도 플롯 (메인 라인)
        ax.plot(frames, angles, 'b-', linewidth=2, label='머리 각도', alpha=0.7)

        # 평균값 가이드 라인
        mean_angle = head_pose_data.mean_angle
        ax.axhline(y=mean_angle, color='green', linestyle='--',
                  linewidth=1.5, alpha=0.6,
                  label=f'평균 ({mean_angle:.1f}°)')

        # 터치 순간 표시
        if len(head_pose_data.touch_frames) > 0:
            touch_frames = head_pose_data.touch_frames
            touch_angles = []

            # 터치 프레임의 각도 찾기
            for tf in touch_frames:
                idx = np.where(frames == tf)[0]
                if len(idx) > 0:
                    touch_angles.append(angles[idx[0]])
                else:
                    # 가장 가까운 프레임 찾기
                    closest_idx = np.argmin(np.abs(frames - tf))
                    touch_angles.append(angles[closest_idx])

            ax.scatter(touch_frames, touch_angles, color='red',
                      s=150, zorder=5, marker='o',
                      edgecolors='darkred', linewidths=2,
                      label=f'터치 순간 ({len(touch_frames)}회)')

            # 터치 순간에 수직 점선 추가
            for tf in touch_frames:
                ax.axvline(x=tf, color='red', linestyle=':',
                          linewidth=1, alpha=0.3)

        # 축 레이블 및 제목
        ax.set_xlabel('프레임 번호', fontsize=12, fontweight='bold')
        ax.set_ylabel('머리 각도 (degrees)', fontsize=12, fontweight='bold')
        ax.set_title('프레임별 머리 각도 변화 (head_vector)', fontsize=13, fontweight='bold', pad=10)

        # 그리드
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # 범례
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # 통계 정보 텍스트 박스
        stats_text = f'평균: {head_pose_data.mean_angle:.2f}°\n'
        stats_text += f'편차 제곱합: {head_pose_data.sum_squared_deviations:.2f}'
        if head_pose_data.touch_window_mean_range is not None:
            stats_text += f'\n터치 변화폭 평균: {head_pose_data.touch_window_mean_range:.2f}°'
        # 텍스트 박스 위치 (왼쪽 상단)
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def plot_touch_window_rate(self, head_pose_data, save_path: Optional[str] = None):
        """터치 전후 ±8프레임 머리 각도 변화율 그래프

        각 터치별로 서브플롯을 만들어:
        - 상단: 머리 각도 (degrees)
        - 하단: 각도 변화율 (deg/frame)
        X축: 터치 시점 기준 상대 프레임 (-5 ~ +5)

        Args:
            head_pose_data: HeadPoseData 객체
            save_path: 저장 경로 (None이면 화면 표시)
        """
        data = head_pose_data.touch_window_data
        if not data:
            print("Warning: 터치 윈도우 데이터가 없습니다.")
            return

        n = len(data)
        fig, axes = plt.subplots(2, n, figsize=(max(5 * n, 8), 8), squeeze=False)

        # 전체 요약 제목
        title = '터치 시점 전후 ±8프레임 머리 각도 변화율'
        if head_pose_data.touch_window_mean_range is not None:
            title += f'    |    평균 변화폭: {head_pose_data.touch_window_mean_range:.1f}°'
        fig.suptitle(title, fontsize=13, fontweight='bold')

        for col, wd in enumerate(data):
            rel         = wd['rel_frames']
            ang         = wd['angles']
            rates       = wd['rates']
            tf          = wd['touch_frame']
            tnum        = wd['touch_num']
            angle_range = wd.get('angle_range', max(ang) - min(ang))

            # ── 상단: 머리 각도 ──────────────────────────────────────────
            ax_top = axes[0][col]
            ax_top.plot(rel, ang, 'b-o', linewidth=2, markersize=5, label='머리 각도')
            ax_top.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='터치')
            ax_top.set_title(
                f'Touch #{tnum}  (Frame {tf})\n변화폭: {angle_range:.1f}°',
                fontsize=10, fontweight='bold'
            )
            ax_top.set_ylabel('각도 (deg)', fontsize=10)
            ax_top.set_xticks(rel)
            ax_top.grid(True, alpha=0.3)
            ax_top.legend(fontsize=8)

            # 0° 기준선
            ax_top.axhline(y=0, color='gray', linewidth=0.8, linestyle=':')

            # ── 하단: 변화율 ─────────────────────────────────────────────
            ax_bot = axes[1][col]
            colors = ['tomato' if r > 0 else 'steelblue' for r in rates]
            ax_bot.bar(rel, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax_bot.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
            ax_bot.axhline(y=0, color='black', linewidth=0.8)
            ax_bot.set_xlabel('터치 기준 상대 프레임', fontsize=10)
            ax_bot.set_ylabel('변화율 (deg/frame)', fontsize=10)
            ax_bot.set_title('각도 변화율', fontsize=10)
            ax_bot.set_xticks(rel)
            ax_bot.grid(True, alpha=0.3, axis='y')

            # 색 범례
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='tomato',    alpha=0.8, label='각도 증가 (↑)'),
                Patch(facecolor='steelblue', alpha=0.8, label='각도 감소 (↓)'),
            ]
            ax_bot.legend(handles=legend_elements, fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 터치 윈도우 변화율 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_angle_distribution(self, head_pose_data, save_path: Optional[str] = None):
        """
        머리 각도 분포 히스토그램

        Args:
            head_pose_data: HeadPoseData 객체
            save_path: 그래프 저장 경로
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('머리 각도 분포 분석', fontsize=16, fontweight='bold')

        angles = head_pose_data.head_angles

        # 히스토그램
        ax1.hist(angles, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(head_pose_data.mean_angle, color='red',
                   linestyle='--', linewidth=2, label=f'평균 ({head_pose_data.mean_angle:.1f}°)')
        ax1.set_xlabel('머리 각도 (degrees)', fontsize=11)
        ax1.set_ylabel('빈도', fontsize=11)
        ax1.set_title('각도 분포 히스토그램', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 박스 플롯
        ax2.boxplot([angles], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('머리 각도 (degrees)', fontsize=11)
        ax2.set_title('각도 분포 박스플롯', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 저장 또는 표시
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 각도 분포 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_combined(self, head_pose_data, save_path: Optional[str] = None):
        """
        머리 각도 종합 분석 그래프 (시계열 + 분포)

        Args:
            head_pose_data: HeadPoseData 객체
            save_path: 그래프 저장 경로
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 메인 타이틀
        fig.suptitle('머리 각도 종합 분석 리포트', fontsize=18, fontweight='bold')

        # 1. 시계열 그래프 (상단 전체)
        ax1 = fig.add_subplot(gs[0:2, :])
        self._plot_angle_over_time(ax1, head_pose_data)

        # 2. 히스토그램 (하단 왼쪽)
        ax2 = fig.add_subplot(gs[2, 0])
        angles = head_pose_data.head_angles
        ax2.hist(angles, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(head_pose_data.mean_angle, color='red',
                   linestyle='--', linewidth=2, label=f'평균')
        ax2.set_xlabel('머리 각도 (degrees)', fontsize=10)
        ax2.set_ylabel('빈도', fontsize=10)
        ax2.set_title('각도 분포', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. 터치 순간 각도 바 차트 (하단 오른쪽)
        ax3 = fig.add_subplot(gs[2, 1])
        if len(head_pose_data.touch_frame_angles) > 0:
            touch_nums = list(range(1, len(head_pose_data.touch_frame_angles) + 1))
            touch_angles = [angle for _, angle in head_pose_data.touch_frame_angles]

            colors = ['green' if angle <= head_pose_data.mean_angle else 'orange'
                     for angle in touch_angles]

            ax3.bar(touch_nums, touch_angles, color=colors, alpha=0.7, edgecolor='black')
            ax3.axhline(head_pose_data.mean_angle, color='red',
                       linestyle='--', linewidth=1.5, label='평균')
            ax3.set_xlabel('터치 번호', fontsize=10)
            ax3.set_ylabel('머리 각도 (degrees)', fontsize=10)
            ax3.set_title('터치 순간 각도', fontsize=11, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, '터치 데이터 없음',
                    ha='center', va='center', fontsize=12)
            ax3.set_title('터치 순간 각도', fontsize=11, fontweight='bold')

        # 저장 또는 표시
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 종합 분석 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

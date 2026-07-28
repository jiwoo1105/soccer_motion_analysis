# visualization/dribble_cycle_plotter.py
"""드리블 사이클 방향 분석 그래프 (PPT용)"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from analysis.dribble_cycle_analyzer import DribbleCycleData


def plot_dribble_cycles(data: DribbleCycleData,
                        fps: float = 30.0,
                        save_path: str = None) -> None:
    frames = data.frame_nums
    times  = frames / fps

    BG       = '#0F1117'
    PANEL    = '#1A1D27'
    TEXT     = '#E8EAF0'
    GRID     = '#2A2D3A'
    CYAN     = '#00D4FF'
    BLUE_DIM = '#1E4A7A'
    NEAR_C   = '#FF6B6B'   # 카메라 쪽으로 (반지름↑)
    FAR_C    = '#4FC3F7'   # 카메라 반대로 (반지름↓)
    PEAK_C   = '#FF6B6B'
    TROUGH_C = '#FFB347'

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.facecolor': PANEL,
        'figure.facecolor': BG,
        'text.color': TEXT,
        'axes.labelcolor': TEXT,
        'xtick.color': TEXT,
        'ytick.color': TEXT,
        'axes.edgecolor': GRID,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
    })

    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(1, 1, figure=fig, top=0.88, bottom=0.10, left=0.07, right=0.97)
    ax  = fig.add_subplot(gs[0])

    # ── 제목 ──────────────────────────────────────────────────────
    fig.text(0.5, 0.945,
             'Dribble Cycle Analysis  ·  Ball Radius Change (90° Side View)',
             ha='center', fontsize=16, fontweight='bold', color=TEXT)
    fig.text(0.5, 0.915,
             f'Cycles: {data.cycle_count}    |    '
             f'Near (↑ toward camera): {data.near_count}    '
             f'Far (↓ away from camera): {data.far_count}',
             ha='center', fontsize=11, color='#8899AA')

    # ── half-cycle 구간 색상 ──────────────────────────────────────
    for hc in data.half_cycles:
        t1 = hc.start_frame / fps
        t2 = hc.end_frame   / fps
        c  = NEAR_C if hc.direction == 'NEAR' else FAR_C
        ax.axvspan(t1, t2, alpha=0.12, color=c, zorder=0)

    # ── 반지름 곡선 ───────────────────────────────────────────────
    ax.fill_between(times, data.radii_smooth, alpha=0.07, color=CYAN)
    ax.plot(times, data.radii_raw,    color=BLUE_DIM, lw=1,   alpha=0.5)
    ax.plot(times, data.radii_smooth, color=CYAN,     lw=2.5)

    # ── peaks / troughs ───────────────────────────────────────────
    for pf in data.peak_frames:
        t   = pf / fps
        idx = int(np.searchsorted(frames, pf))
        if idx < len(data.radii_smooth):
            ax.scatter(t, data.radii_smooth[idx],
                       color=PEAK_C, s=100, zorder=6,
                       edgecolors='white', linewidths=0.8)

    for tf in data.trough_frames:
        t   = tf / fps
        idx = int(np.searchsorted(frames, tf))
        if idx < len(data.radii_smooth):
            ax.scatter(t, data.radii_smooth[idx],
                       color=TROUGH_C, s=100, zorder=6, marker='v',
                       edgecolors='white', linewidths=0.8)

    # ── half-cycle 라벨 ───────────────────────────────────────────
    r_max = data.radii_smooth.max()
    r_min = data.radii_smooth.min()
    r_rng = r_max - r_min

    for hc in data.half_cycles:
        mid_t  = (hc.start_frame + hc.end_frame) / 2 / fps
        c      = NEAR_C if hc.direction == 'NEAR' else FAR_C
        symbol = '↑' if hc.direction == 'NEAR' else '↓'
        label  = f'{symbol} {abs(hc.r_change):.1f}px'
        y_pos  = r_max + r_rng * 0.06
        ax.text(mid_t, y_pos, label,
                ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=c)

    # ── 방향 전환 시점 프레임 번호 (peak/trough 아래) ─────────────
    for pf in data.peak_frames:
        t   = pf / fps
        idx = int(np.searchsorted(frames, pf))
        if idx < len(data.radii_smooth):
            ax.text(t, data.radii_smooth[idx] - r_rng * 0.04,
                    f'f{pf}', ha='center', va='top',
                    fontsize=7.5, color=PEAK_C, alpha=0.85)

    for tf in data.trough_frames:
        t   = tf / fps
        idx = int(np.searchsorted(frames, tf))
        if idx < len(data.radii_smooth):
            ax.text(t, data.radii_smooth[idx] + r_rng * 0.04,
                    f'f{tf}', ha='center', va='bottom',
                    fontsize=7.5, color=TROUGH_C, alpha=0.85)

    # ── 범례 ─────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], color=CYAN, lw=2.5, label='Ball Radius (smooth)'),
        plt.scatter([], [], color=PEAK_C,   s=80, label='Peak  — ball closest to camera'),
        plt.scatter([], [], color=TROUGH_C, s=80, marker='v', label='Trough — ball farthest'),
        mpatches.Patch(color=NEAR_C, alpha=0.4, label=f'NEAR ↑  toward camera  ({data.near_count})'),
        mpatches.Patch(color=FAR_C,  alpha=0.4, label=f'FAR  ↓  away from camera  ({data.far_count})'),
    ]
    ax.legend(handles=legend_handles, fontsize=9,
              loc='lower right', framealpha=0.3, edgecolor=GRID)

    ax.set_xlabel('Time  (s)', fontsize=12)
    ax.set_ylabel('Ball Radius  (px)', fontsize=12)
    ax.set_ylim(r_min - r_rng * 0.1, r_max + r_rng * 0.28)
    ax.grid(True, axis='x', alpha=0.35)
    ax.grid(True, axis='y', alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=BG)
        print(f"  드리블 사이클 그래프 저장: {save_path}")
    else:
        plt.show()
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)

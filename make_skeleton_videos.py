# -*- coding: utf-8 -*-
"""Legacy 시각화: 기존 SAM2 캐시에서 스켈레톤 영상을 일괄 생성한다.

현재 ROI 후보/evaluate.py 평가 결과를 렌더링하는 경로가 아니다.
출력: output/videos/legacy_skeleton_<영상명>.mp4 (재실행 시 해당 영상 교체).

main.py는 영상 1편을 하드코딩하고 포즈 추출 + SAM2를 처음부터 돌린다(편당 10분+).
캐시(`output/scoring_cache`)의 포즈와 기존 공 추적을 읽어서 렌더링한다.
캐시에 남아 있는 공 추적이 정확하거나 최신이라는 보장은 없다.

실행:
    python3.9 make_skeleton_videos.py              # 인,인 전체
    python3.9 make_skeleton_videos.py 3-1 기준1     # 일부만
    python3.9 make_skeleton_videos.py --in-out     # 인,아웃 전체
"""
import os
import sys
import glob
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def nfc(s):
    """macOS는 파일명을 NFD로 돌려준다. 한글 비교 전에 반드시 NFC로 합칠 것."""
    return unicodedata.normalize('NFC', s)


# 2026-09-17 영상 검토: 다른 사람 추적 / 스켈레톤 불량으로 현재 코호트에서 제외.
SKIP = ('인,인 6-1', '인,인 7-3')


def build_ball_motion(pose_frames, ball_tracks, touch_frames):
    """create_skeleton_video가 기대하는 BallMotionData를 캐시에서 조립한다.

    SAM2가 놓친 프레임은 선형 보간해서 채우되, 보간했다는 사실을 따로 표시한다
    (영상에서 점선/다른 색으로 구분되도록).
    """
    from analysis.ball_motion_analyzer import BallMotionData

    frames = sorted(pf.frame_number for pf in pose_frames)
    known = {int(k): tuple(v) for k, v in ball_tracks.items()}
    ks = sorted(known)

    all_pos, interp = {}, {}
    if ks:
        xs = np.array([known[k][0] for k in ks])
        ys = np.array([known[k][1] for k in ks])
        rs_ = np.array([known[k][2] for k in ks])
        for f in frames:
            if f in known:
                all_pos[f] = known[f]
            elif ks[0] <= f <= ks[-1]:
                p = (float(np.interp(f, ks, xs)), float(np.interp(f, ks, ys)),
                     float(np.interp(f, ks, rs_)))
                all_pos[f] = p
                interp[f] = p

    ordered = sorted(all_pos)
    return BallMotionData(
        frame_numbers=np.array(ordered),
        positions=np.array([[all_pos[f][0], all_pos[f][1]] for f in ordered])
        if ordered else np.empty((0, 2)),
        touch_frames=list(touch_frames),
        touch_count=len(touch_frames),
        interp_positions=interp,
        all_ball_positions=all_pos,
    )


def render(video_path):
    stem = Path(video_path).stem
    label = nfc(stem)
    if label in SKIP:
        print(f'  [건너뜀] {label} — 다른 사람 추적 / 스켈레톤 불량')
        return False

    from scoring.extraction_cache import load, is_cached
    from scoring.touch_filter import filter_touches
    from extract_depth_metrics import detect_touches_by_ball_direction
    from analysis.head_pose_analyzer import HeadPoseAnalyzer
    from analysis.trunk_pose_analyzer import TrunkPoseAnalyzer
    from main import create_skeleton_video

    if not is_cached(stem):
        print(f'  [캐시 없음] {label}')
        return False

    pose_frames, ball_tracks = load(stem)
    touches = detect_touches_by_ball_direction(pose_frames, ball_tracks,
                                               max_foot_dist=200)
    valid, _ = filter_touches(touches, pose_frames, ball_tracks)
    touch_frames = [t.frame_number for t in valid]

    ball = build_ball_motion(pose_frames, ball_tracks, touch_frames)
    head = HeadPoseAnalyzer().analyze(pose_frames)
    trunk = TrunkPoseAnalyzer().analyze(pose_frames)

    cov = len(ball_tracks) / max(len(pose_frames), 1) * 100
    print(f'  [LEGACY] {label}: 포즈 {len(pose_frames)}f, 공 {len(ball_tracks)}f({cov:.0f}%), '
          f'터치 {len(touch_frames)}개', flush=True)

    create_skeleton_video(video_path, pose_frames, ball,
                          touch_frames_override=touch_frames,
                          output_filename=f'legacy_skeleton_{stem}.mp4',
                          head_data=head, trunk_data=trunk)
    return True


def main():
    argv = sys.argv[1:]
    # zsh 대화형 셸은 기본적으로 '#'을 주석으로 처리하지 않는다
    # (interactive_comments 미설정). 명령 뒤에 한글 주석을 붙이면 그게 통째로
    # 인자로 들어와 "대상 0편"이 된다. '#' 이후는 무시한다.
    if '#' in argv:
        argv = argv[:argv.index('#')]
    argv = [a for a in argv if not a.startswith('#')]

    folder = 'in_out' if '--in-out' in argv else 'in_in'
    prefix = '인,아웃' if folder == 'in_out' else '인,인'
    args = [a for a in argv if not a.startswith('-')]

    # 한글 glob 패턴은 NFD/NFC 불일치로 매칭에 실패한다. ASCII 패턴만 쓴다.
    paths = sorted(glob.glob(f'input/{folder}/*.MOV'))
    available = [nfc(Path(p).stem).replace(f'{prefix} ', '') for p in paths]

    if not paths:
        print(f'input/{folder}/ 에 .MOV 파일이 없다. '
              f'프로젝트 루트에서 실행하고 있는지 확인하라 (현재: {os.getcwd()})')
        return

    if args:
        paths = [p for p in paths
                 if any(nfc(Path(p).stem) == f'{prefix} {a}' for a in args)]
        if not paths:
            print(f'"{" ".join(args)}" 와 맞는 영상이 없다.')
            print(f'  쓸 수 있는 이름: {", ".join(available)}')
            return

    print(f'[LEGACY SAM2 시각화] 대상 {len(paths)}편 ({folder}); 현재 ROI 평가와 별개')
    ok = 0
    for p in paths:
        try:
            ok += bool(render(p))
        except Exception as e:
            print(f'  [{nfc(Path(p).stem)}] 오류: {e}', flush=True)
    print(f'\n완료 {ok}/{len(paths)}   → output/videos/')


if __name__ == '__main__':
    main()

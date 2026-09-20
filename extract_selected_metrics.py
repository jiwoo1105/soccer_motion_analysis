"""
선택 영상들만 3개 평가지표 추출:
1. 상체각도 (mean_angle)
2. 헤드업 (mean_range)
3. 어깨골반 회전 ((avg shoulder + avg pelvis) / 2)
"""

import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
from core.pose_extractor import PoseExtractor
from analysis.ball_motion_analyzer import BallMotionAnalyzer
from analysis.head_pose_analyzer import HeadPoseAnalyzer
from analysis.trunk_pose_analyzer import TrunkPoseAnalyzer
import config

VIDEOS = [
    "input/in_in/인,인 9-1.MOV",
    "input/in_in/인,인 기준1.MOV",
    "input/in_in/인,인 7-2.MOV",
    "input/in_in/인,인 7-1.MOV",
    "input/in_in/인,인 5-1.MOV",
    "input/in_in/인,인 3-2.MOV",
    "input/in_in/인,인 3-1.MOV",
]


def run_sam2(video_path, pose_frames):
    try:
        ankle_positions = {}
        for pf in pose_frames:
            if pf.frame_number >= 20:
                break
            fw, fh = pf.frame_width, pf.frame_height
            lm = pf.landmarks
            ankle_positions[pf.frame_number] = [
                [float(lm[27][0] * fw), float(lm[27][1] * fh)],
                [float(lm[28][0] * fw), float(lm[28][1] * fh)],
            ]
        ankle_json = json.dumps(ankle_positions)
        sam2_script = f"""
import sys, json
sys.path.insert(0, '{os.path.abspath(".")}')
from sam2_ball_tracker import run_sam2_tracking
ankle_raw = json.loads('{ankle_json.replace(chr(39), chr(92)+chr(39))}')
ankle_positions = {{int(k): v for k, v in ankle_raw.items()}}
tracks = run_sam2_tracking('{video_path}', ankle_positions=ankle_positions)
result = {{str(k): list(v) for k, v in tracks.items()}}
print(json.dumps(result))
"""
        result = subprocess.run(['python3.11', '-c', sam2_script],
                                capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            json_line = [l for l in result.stdout.strip().split('\n') if l.startswith('{')][-1]
            tracks = {int(k): tuple(v) for k, v in json.loads(json_line).items()}
            frame_map = {pf.frame_number: pf for pf in pose_frames}
            for frame_idx, (cx, cy, r) in tracks.items():
                if frame_idx in frame_map:
                    pf = frame_map[frame_idx]
                    pf.ball_position = (cx, cy)
                    r = max(8, r)
                    pf.ball_bbox = (cx - r, cy - r, cx + r, cy + r)
    except Exception as e:
        print(f"    SAM2 스킵: {e}")


def process(video_path):
    stem = Path(video_path).stem
    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence'],
        detect_ball=False,
    )
    pose_frames = extractor.extract_from_video(video_path)
    if len(pose_frames) == 0:
        return None

    run_sam2(video_path, pose_frames)

    # 1. 상체각도
    trunk_data = TrunkPoseAnalyzer(min_visibility_threshold=0.5).analyze(pose_frames)
    mean_angle = trunk_data.mean_angle if trunk_data else None

    # 2. 공 터치
    ball_data = BallMotionAnalyzer(
        min_distance_between_touches=config.BALL_MOTION_CONFIG['min_distance_between_touches'],
        peak_prominence=config.BALL_MOTION_CONFIG['peak_prominence'],
        max_touch_distance=config.BALL_MOTION_CONFIG['max_touch_distance'],
    ).analyze(pose_frames)

    # 3. 헤드업
    head_data = HeadPoseAnalyzer(min_visibility_threshold=0.5).analyze(pose_frames, ball_data)
    mean_range = head_data.touch_window_mean_range if head_data else None

    # 4. 어깨골반 회전
    rotation_avg = None
    if ball_data and ball_data.touch_directions:
        dirs = ball_data.touch_directions
        avg_sh = sum(d.shoulder_rotation_angle for d in dirs) / len(dirs)
        avg_pe = sum(d.pelvis_rotation_angle for d in dirs) / len(dirs)
        rotation_avg = (avg_sh + avg_pe) / 2

    return {
        'video': stem,
        'mean_angle': mean_angle,
        'mean_range': mean_range,
        'rotation_avg': rotation_avg,
        'touch_count': ball_data.touch_count if ball_data else 0,
    }


def main():
    print(f"\n{'영상':<20} {'상체각도':>8} {'mean_range':>10} {'회전평균':>8} {'터치':>4}")
    print("-" * 58)

    results = []
    for v in VIDEOS:
        print(f"\n처리 중: {Path(v).name}")
        r = process(v)
        if r:
            results.append(r)
            ma = f"{r['mean_angle']:.1f}°" if r['mean_angle'] else "N/A"
            mr = f"{r['mean_range']:.1f}°" if r['mean_range'] else "N/A"
            ra = f"{r['rotation_avg']:.1f}°" if r['rotation_avg'] else "N/A"
            print(f"  → {r['video']:<20} {ma:>8} {mr:>10} {ra:>8} {r['touch_count']:>4}")

    print("\n" + "=" * 58)
    print(f"\n{'영상':<20} {'상체각도':>8} {'mean_range':>10} {'회전평균':>8} {'터치':>4}")
    print("-" * 58)
    for r in results:
        ma = f"{r['mean_angle']:.1f}°" if r['mean_angle'] else "N/A"
        mr = f"{r['mean_range']:.1f}°" if r['mean_range'] else "N/A"
        ra = f"{r['rotation_avg']:.1f}°" if r['rotation_avg'] else "N/A"
        print(f"{r['video']:<20} {ma:>8} {mr:>10} {ra:>8} {r['touch_count']:>4}")


if __name__ == "__main__":
    main()

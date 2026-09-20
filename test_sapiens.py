"""
Legacy Sapiens experiment: explicit execution requires the optional local package.
Sapiens 테스트: 3-1 영상 1프레임에서 pose(308 keypoints) + depth 추출
"""


def main():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sapiens_inference'))

    import cv2
    import torch
    import numpy as np
    from sapiens_inference.pose import SapiensPoseEstimation, SapiensPoseEstimationType
    from sapiens_inference.depth import SapiensDepth, SapiensDepthType, draw_depth_map

    # MPS 지원
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 프레임 읽기
    video_path = "input/in_in/인,인 3-1.MOV"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    print(f"Frame: {frame.shape}")

    # 2. Pose 추출 (0.3B - 가장 작은 모델로 테스트)
    print("\nPose 모델 로딩 중...")
    pose_estimator = SapiensPoseEstimation(
        SapiensPoseEstimationType.POSE_ESTIMATION_03B,
        device=device,
        dtype=torch.float32
    )
    print("Pose 추출 중...")
    pose_img, keypoints_list = pose_estimator(frame)

    if keypoints_list:
        kps = keypoints_list[0]  # 첫 번째 사람
        print(f"\n감지된 키포인트 수: {len(kps)}")

        # 우리가 필요한 키포인트 확인
        need = ['left_eye', 'right_eye', 'left_shoulder', 'right_shoulder',
                'left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        print(f"\n{'키포인트':<20} {'x':>8} {'y':>8} {'conf':>8}")
        print("-" * 48)
        for name in need:
            found = False
            for k, v in kps.items():
                if name.replace('_', ' ') in k.lower() or name.replace('_', '') in k.lower().replace(' ', '').replace('_', ''):
                    print(f"{k:<20} {v[0]:>8.1f} {v[1]:>8.1f} {v[2]:>8.3f}")
                    found = True
                    break
            if not found:
                # 전체 키포인트에서 검색
                for k, v in kps.items():
                    if any(part in k.lower() for part in name.split('_')):
                        print(f"{k:<20} {v[0]:>8.1f} {v[1]:>8.1f} {v[2]:>8.3f}  ← {name}?")
                        break

        # 전체 키포인트 이름 출력
        print(f"\n전체 키포인트 목록 ({len(kps)}개):")
        for i, (name, (x, y, conf)) in enumerate(kps.items()):
            if conf > 0.3:
                print(f"  {i:>3}: {name:<30} ({x:.0f}, {y:.0f}) conf={conf:.3f}")
    else:
        print("사람 감지 실패")

    # 3. Depth 추출
    print("\nDepth 모델 로딩 중...")
    depth_estimator = SapiensDepth(
        SapiensDepthType.DEPTH_03B,
        device=device,
        dtype=torch.float32
    )
    print("Depth 추출 중...")
    depth_map = depth_estimator(frame)
    print(f"Depth map shape: {depth_map.shape}, range: {depth_map.min():.2f} ~ {depth_map.max():.2f}")

    # 4. 시각화 저장
    os.makedirs('output/graphs', exist_ok=True)
    cv2.imwrite('output/graphs/sapiens_pose_test.png', pose_img)
    depth_vis = draw_depth_map(depth_map)
    cv2.imwrite('output/graphs/sapiens_depth_test.png', depth_vis)
    print("\n저장 완료: output/graphs/sapiens_pose_test.png, sapiens_depth_test.png")


if __name__ == "__main__":
    main()

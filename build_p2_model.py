"""
Legacy YOLO P2 model-building experiment; run explicitly to write a checkpoint.
- yolov8-p2.yaml 아키텍처 (ultralytics 내장)
- yolov8n.pt 가중치 전이
- 저장: yolov8s-p2-transfer.pt
"""


def main():
    from ultralytics import YOLO

    # 1. P2 yaml로 모델 생성 + 기존 가중치 전이
    print("yolov8-p2 모델 생성 + yolov8n.pt 가중치 전이 중...")
    model = YOLO("yolov8s-p2.yaml").load("yolov8n.pt")

    # 2. 저장
    model.save("yolov8s-p2-transfer.pt")
    print("저장 완료: yolov8s-p2-transfer.pt")
    print("\n이제 config.py에서 model_name을 'yolov8s-p2-transfer.pt'로 변경하세요.")


if __name__ == "__main__":
    main()

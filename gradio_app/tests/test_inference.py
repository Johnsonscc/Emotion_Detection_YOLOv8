import cv2
import sys
from core.inference import EmotionInference


def test_inference():
    # 加载模型
    model = EmotionInference("best.pt")

    # 测试图像集
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),  # 随机图像
        cv2.imread("test.jpg"),  # 实际图像
        np.zeros((640, 640, 3), dtype=np.uint8)  # 全黑图像
    ]

    for img in test_images:
        detections = model.predict(img)
        print(f"检测结果: [{len(detections)}] 个目标")
        for det in detections:
            print(f"- {det['label']} | 置信度: {det['conf']:.2f} | 坐标: {det['bbox']}")


if __name__ == "__main__":
    test_inference()

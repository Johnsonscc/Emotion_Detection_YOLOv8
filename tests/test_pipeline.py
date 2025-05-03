import cv2
import numpy as np
from core.inference import EmotionDetector


def test_error_handling():
    detector = EmotionDetector("models/yolov8s-emo.pt")

    # 测试用例
    cases = [
        ("正常图像", "test_images/happy.jpg"),
        ("灰度图像", np.random.rand(224, 224)),
        ("空输入", None),
        ("错误维度", np.random.rand(224, 224, 5))
    ]

    for name, case in cases:
        print(f"\n测试: {name}")
        if isinstance(case, str):
            img = cv2.imread(case)
        else:
            img = case

        try:
            results = detector.predict(img)
            print(f"结果数量: {len(results)}")
        except Exception as e:
            print(f"测试失败: {str(e)}")
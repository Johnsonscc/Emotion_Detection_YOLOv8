import numpy as np
from core.inference import EmotionDetector


def test_error_conditions():
    detector = EmotionDetector("models/yolov8s-emo.pt")

    # 测试空输入
    assert detector.predict(None) == []

    # 测试错误维度输入
    gray_img = np.random.rand(224, 224)  # 二维灰度图
    assert isinstance(detector.predict(gray_img), list)

    # 测试无表情图像
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    results = detector.predict(blank)
    assert isinstance(results, list)  # 可能为空但不报错
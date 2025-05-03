import unittest
import cv2
import numpy as np
from pathlib import Path
from emotion import EmotionInference


class TestEmotionInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path("models/yolov8s-emo.pt")
        cls.test_img_dir = Path("test_images")
        cls.detector = EmotionInference(model_path=str(cls.model_path))

        # 生成测试用假数据
        cls.valid_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        cls.gray_image = cv2.cvtColor(cls.valid_image, cv2.COLOR_BGR2GRAY)
        cls.small_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    def test_01_model_loading(self):
        """验证模型正确加载"""
        self.assertIsNotNone(self.detector.model)
        print("✅ 模型加载验证通过")

    def test_02_normal_prediction(self):
        """标准输入测试"""
        results = self.detector.predict(self.valid_image)
        self.assertIsInstance(results, list)
        for res in results:
            self.assertIn(res['label'], self.detector.class_names)
            self.assertTrue(0 <= res['conf'] <= 1)
            self.assertEqual(len(res['bbox']), 4)
        print("✅ 正常预测流程验证完成")

    def test_03_edge_cases(self):
        """边界条件测试"""
        test_cases = [
            (np.zeros((0, 0, 3), np.uint8), "空图像"),
            (np.random.rand(100, 100, 2), "无效通道数"),
            (self.gray_image, "灰度图输入"),
            (self.small_image, "极小分辨率")
        ]

        for img, desc in test_cases:
            with self.subTest(desc):
                results = self.detector.predict(img)
                self.assertTrue(isinstance(results, list),
                                f"{desc} 测试失败")
        print("✅ 边界条件测试覆盖完成")

    def test_04_output_validation(self):
        """模型输出验证"""
        # 生成测试样本的模型原始输出
        input_tensor = torch.randn(1, 3, 640, 640).to(self.detector.device)
        with torch.no_grad():
            raw_output = self.detector.model(input_tensor)

        # 验证输出张量属性
        self.assertEqual(raw_output.shape[-1], 6,
                         "输出应包含6个参数 (x1,y1,x2,y2,conf,cls)")
        print("✅ 模型输出结构验证通过")

    def test_05_performance_benchmark(self):
        """性能基准测试"""
        import time
        warmup_runs = 5
        test_runs = 20

        # GPU预热
        for _ in range(warmup_runs):
            _ = self.detector.predict(self.valid_image)

        # 正式测试
        timings = []
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = self.detector.predict(self.valid_image)
            timings.append(time.perf_counter() - start)

        avg_time = sum(timings) / len(timings) * 1000  # 转毫秒
        print(f"⏱️ 平均处理时间: {avg_time:.1f}ms")
        self.assertLess(avg_time, 250, "性能不达标")
        print("✅ 性能基准测试完成")


if __name__ == '__main__':
    unittest.main(verbosity=2)

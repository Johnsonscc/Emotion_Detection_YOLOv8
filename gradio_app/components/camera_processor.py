import cv2
import numpy as np
from typing import Optional, Dict


class CameraProcessor:
    """摄像头处理流水线"""

    def __init__(self, config: Dict):
        self.config = config
        self.cap = None

    def _process_camera(self, frame: np.ndarray) -> np.ndarray:
        """处理摄像头输入"""
        if frame is None:
            print("⚠️ 收到空帧! 请检查摄像头输入")
            return None

        print(f"摄像头数据统计 → 形状: {frame.shape}, 类型: {frame.dtype}, 最大值: {frame.max()}")

        try:
            processed = self.camera.process_frame(frame)
            if processed is None:
                print("🛑 摄像头预处理返回空值")
                return None

            detections = self.model.predict(processed)
            return self.renderer.render(processed, detections)

        except Exception as e:
            print(f"🔴 摄像头处理致命错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.renderer.error_image()

    def process_frame(self, frame):
        try:
            processed = self._preprocess(frame)

            # 确保图像是三通道（BGR），如果是灰度图转换为 BGR
            if processed.ndim == 2 or processed.shape[-1] != 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            # 确保是 NumPy 数组，不是 Tensor
            if isinstance(processed, torch.Tensor):
                processed = processed.permute(1, 2, 0).cpu().numpy()

            detections = self.model.predict(processed)

            return detections
        except Exception as e:
            print(f"处理错误: {e}")
            return None

    def _auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """自动白平衡算法"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        a = a - ((avg_a - 128) * 1.1)
        b = b - ((avg_b - 128) * 1.1)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
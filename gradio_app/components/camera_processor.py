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

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """处理单帧图像 (兼容Gradio 3.x的numpy输入)"""
        if frame is None or frame.size == 0:
            return None

        try:
            # 新版本Gradio可能直接传入BGR格式
            if frame.shape[2] == 3:  # 确保是彩色图像
                if np.max(frame) > 1:  # 处理0-255范围的图像
                    frame = frame.astype(np.uint8)
                else:  # 处理0-1范围的图像
                    frame = (frame * 255).astype(np.uint8)

            # 统一转换为BGR格式
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 继续原有处理流程...
            return cv2.resize(frame, tuple(self.config["target_size"]))
        except Exception as e:
            print(f"[CameraProcessor] 帧处理失败: {str(e)}")
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
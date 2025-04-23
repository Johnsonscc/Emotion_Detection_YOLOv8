import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    def __init__(self, input_size: Tuple[int, int] = (640, 640)):
        self.input_size = input_size

    def resize(self, img: np.ndarray) -> np.ndarray:
        """仅用于非YOLOv8模型的resize"""
        return cv2.resize(img, self.input_size)

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        完整的图像预处理流程：
        1. BGR转RGB
        2. Resize
        3. 归一化 [0,255] -> [0,1]
        4. 通道重排 HWC -> CHW

        参数:
            img: 输入BGR图像 (H,W,3)

        返回:
            processed_img: 预处理后的图像 (3,H,W)
        """
        # 输入校验
        if img is None or img.size == 0:
            raise ValueError("输入图像不能为空")

        # 1. BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Resize (保持宽高比)
        h, w = img.shape[:2]
        scale = min(self.input_size[1] / h, self.input_size[0] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img_rgb, (new_w, new_h))

        # 3. 归一化
        normalized = resized.astype(np.float32) / 255.0

        # 4. 通道重排 HWC -> CHW
        return np.transpose(normalized, (2, 0, 1))

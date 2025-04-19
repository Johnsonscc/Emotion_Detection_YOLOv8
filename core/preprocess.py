import cv2
import numpy as np
from typing import Tuple


class ImagePreprocessor:
    def __init__(self,
                 target_size: Tuple[int, int] = (640, 640),
                 normalize: bool = True):
        """
        图像预处理模块

        参数:
            target_size: 目标输入尺寸 (H,W)
            normalize: 是否执行归一化
        """
        self.target_size = target_size
        self.normalize = normalize

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        完整预处理流水线：
        1. BGR转RGB
        2. 调整尺寸并保持比例填充
        3. 归一化
        4. 转换为torch tensor
        """
        # 颜色空间转换
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 尺寸调整
        resized_img = self._letterbox(rgb_img)

        # 归一化
        if self.normalize:
            resized_img = resized_img.astype(np.float32) / 255.0

        # 维度调整 (H,W,C) -> (C,H,W)
        tensor_img = resized_img.transpose(2, 0, 1)
        tensor_img = np.expand_dims(tensor_img, axis=0)  # 添加batch维度

        return torch.from_numpy(tensor_img).to(self.device)

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        """自适应的尺寸调整方法"""
        h, w = self.target_size
        ih, iw = img.shape[:2]

        # 计算缩放比例
        scale = min(w / iw, h / ih)  # 选择较小比例尺寸
        nw, nh = int(iw * scale), int(ih * scale)

        # 调整图像尺寸
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 创建空画布
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)

        # 计算填充位置
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # 将调整后的图像放入画布中心
        canvas[dy:dy + nh, dx:dx + nw] = resized

        return canvas
